"""
server/main.py

Production Federated Learning Server.

Implements:
  - GET  /model            — Clients fetch current global weights.
  - POST /update           — Clients push local weight updates.
  - GET  /metrics          — Dashboard: evaluation & aggregation history.
  - GET  /clients          — Dashboard: recent client update attempts.
  - POST /simulate         — Fire a synthetic local training run (for testing).
  - POST /api/dataset/upload — Upload a dataset file or URL and trigger background training.
  - GET  /api/model/weights/{client_id} — Download per-client trained weights.

Architecture:
  - ML logic (aggregation, evaluation, detection) is fully modular.
  - DB logic is isolated inside SupabaseManager.
  - asyncio.Lock guards all global model mutations.
"""

import os
import io
import sys
import ssl
import asyncio
import logging
import shutil
import urllib.request
from urllib.parse import urlparse
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ── local server modules ────────────────────────────────────────────────────
from model import CNNModel
from detection import detect_update
from aggregator import fedavg, dp_trimmed_mean
from storage import SupabaseManager

# ── experiments (one directory up) ──────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.evaluation import evaluate_model

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Production Asynchronous Federated Learning")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global mutable state  (all writes go through model_lock)
# ---------------------------------------------------------------------------

global_model: CNNModel = CNNModel()
model_lock: asyncio.Lock = asyncio.Lock()
storage: SupabaseManager = None

# Pending client updates waiting to be aggregated
# Each element: (client_id: str, weights: Dict[str, Tensor])
pending_updates: List[tuple] = []

# Minimum number of updates before aggregation fires
MIN_UPDATE_QUEUE: int = int(os.getenv("MIN_AGGREGATE_SIZE", "1"))

# Differential Privacy settings  (set to "false" / 0 to disable)
DP_ENABLED:    bool  = os.getenv("DP_ENABLED", "true").lower() == "true"
DP_CLIP_NORM:  float = float(os.getenv("DP_CLIP_NORM", "10.0"))
DP_NOISE_MULT: float = float(os.getenv("DP_NOISE_MULT", "1.0"))

# Validation loader (loaded once at startup)
val_loader: DataLoader = None

# Active version trackers
current_version_id:  int = None
current_version_num: int = 0


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """
    Runs on server boot.
    1. Initialises Supabase storage.
    2. Loads the latest global model checkpoint if one exists.
    3. Loads the MNIST validation set for evaluation.
    """
    global storage, val_loader, current_version_id, current_version_num

    # ── storage ──────────────────────────────────────────────────────────────
    try:
        storage = SupabaseManager()
        logger.info("SupabaseManager initialised.")
    except Exception as exc:
        logger.error(f"Failed to initialise SupabaseManager: {exc}")

    # ── model ─────────────────────────────────────────────────────────────────
    if storage:
        latest = storage.get_latest_version()
        if (
            latest["version_num"] > 0
            and latest["file_path"]
            and os.path.exists(latest["file_path"])
        ):
            try:
                global_model.load_state_dict(
                    torch.load(latest["file_path"], weights_only=True)
                )
                current_version_num = latest["version_num"]
                current_version_id  = latest["id"]
                logger.info(f"Loaded global model: Version {current_version_num}")
            except Exception as exc:
                logger.warning(
                    f"Could not load checkpoint ({exc}). Starting fresh."
                )
                _init_fresh_model()
        else:
            _init_fresh_model()

    # ── validation dataset ───────────────────────────────────────────────────
    ssl._create_default_https_context = ssl._create_unverified_context
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    val_dataset = datasets.MNIST(
        "../data", train=False, download=True, transform=transform
    )
    val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)
    logger.info("MNIST validation set loaded.")


def _init_fresh_model():
    """Helper: save a brand-new random model at the next available version number."""
    global current_version_num, current_version_id
    try:
        # Find the highest version already in the DB so we never duplicate
        rows = (
            storage.supabase.table("model_versions")
            .select("version_num")
            .order("version_num", desc=True)
            .limit(1)
            .execute()
        )
        if rows.data:
            current_version_num = rows.data[0]["version_num"] + 1
        else:
            current_version_num = 1
    except Exception:
        current_version_num = 1

    current_version_id = storage.save_model_version(
        global_model.get_weights(), current_version_num
    )
    logger.info(f"Initialised fresh global model (Version {current_version_num}).")



# ---------------------------------------------------------------------------
# Core aggregation logic
# ---------------------------------------------------------------------------

async def _aggregate_and_update(updates: List[tuple]) -> Dict[str, float]:
    """
    Aggregates a batch of accepted client updates, applies the result to the
    global model, evaluates it, and persists everything to storage.

    Thread-safety: caller must hold model_lock.

    Args:
        updates: List of (client_id, state_dict) tuples.

    Returns:
        Evaluation metrics dict {"loss": float, "accuracy": float}.
    """
    global current_version_num, current_version_id

    weight_dicts = [u[1] for u in updates]
    n = len(weight_dicts)

    logger.info(f"Aggregating {n} update(s) …")

    # ── aggregation  ──────────────────────────────────────────────────────────
    if DP_ENABLED and n > 1:
        aggregated = dp_trimmed_mean(
            weight_dicts,
            trim_ratio=float(os.getenv("TRIM_RATIO", "0.2")),
            clip_norm=DP_CLIP_NORM,
            noise_multiplier=DP_NOISE_MULT,
        )
    else:
        # FedAvg — works correctly even with a single update
        aggregated = fedavg(weight_dicts)

    # ── apply to global model ─────────────────────────────────────────────────
    global_model.set_weights(aggregated)
    current_version_num += 1

    # ── evaluate ─────────────────────────────────────────────────────────────
    metrics = evaluate_model(global_model, val_loader)

    # ── persist ───────────────────────────────────────────────────────────────
    new_vid = storage.save_model_version(aggregated, current_version_num)
    storage.log_aggregation(
        new_vid,
        accepted=n,
        rejected=0,
        method="DP-TrimmedMean" if (DP_ENABLED and n > 1) else "FedAvg",
    )
    storage.log_evaluation(new_vid, metrics["loss"], metrics["accuracy"])
    current_version_id = new_vid

    logger.info(
        f"Round {current_version_num} complete — "
        f"Loss: {metrics['loss']:.4f} | Acc: {metrics['accuracy']:.4f}"
    )
    return metrics


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/model")
async def get_global_model():
    """
    Clients call this to download the current global model weights.
    Returns a binary .pt file as an octet-stream.
    """
    async with model_lock:
        weights = global_model.get_weights()

    buf = io.BytesIO()
    torch.save(weights, buf)
    buf.seek(0)

    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        buf,
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=global_model.pt"},
    )


@app.get("/metrics")
async def get_metrics():
    """Evaluation & aggregation history for the dashboard."""
    if not storage:
        raise HTTPException(500, "Storage uninitialized.")

    evals = (
        storage.supabase.table("evaluation_metrics")
        .select("*")
        .order("version_id", desc=True)
        .limit(20)
        .execute()
    )
    aggs = (
        storage.supabase.table("aggregation_logs")
        .select("*")
        .order("version_id", desc=True)
        .limit(20)
        .execute()
    )
    return {
        "current_version": current_version_num,
        "evaluations":     evals.data,
        "aggregations":    aggs.data,
        "pending_queue_size": len(pending_updates),
    }


@app.get("/clients")
async def get_clients():
    """Recent client update stream for the dashboard."""
    if not storage:
        raise HTTPException(500, "Storage uninitialized.")
    try:
        rows = (
            storage.supabase.table("client_updates")
            .select("*")
            .order("created_at", desc=True)
            .limit(50)
            .execute()
        )
        return {"data": rows.data}
    except Exception as exc:
        logger.error(f"Failed to fetch clients: {exc}")
        return {"data": []}


# ---------------------------------------------------------------------------
# POST /update  — main FL update handler
# ---------------------------------------------------------------------------

@app.post("/update")
async def receive_update(
    background_tasks: BackgroundTasks,
    client_id: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Receives a client's locally-trained weight file (.pt) and processes it:

    1. Deserialise the .pt byte stream into a state dict.
    2. Run Byzantine detection against the current global model.
    3. If REJECT → log and return immediately.
    4. If ACCEPT →
         a. Append to pending_updates queue.
         b. If queue ≥ MIN_UPDATE_QUEUE, aggregate immediately in background.
         c. Trigger model update, evaluation, and DB logging.

    All model mutations are performed inside asyncio.Lock.
    """
    global pending_updates

    # ── deserialise ───────────────────────────────────────────────────────────
    try:
        raw = await file.read()
        buf = io.BytesIO(raw)
        client_weights: Dict[str, torch.Tensor] = torch.load(
            buf, weights_only=True, map_location="cpu"
        )
    except Exception as exc:
        raise HTTPException(400, f"Invalid weight file: {exc}")

    # ── snapshot global weights for detection (no lock needed — read only) ────
    async with model_lock:
        global_weights = global_model.get_weights()
        curr_vid       = current_version_id

    # ── malicious detection ───────────────────────────────────────────────────
    status, reason, norm_val, dist_val = detect_update(client_weights, global_weights)

    # Log this attempt regardless of outcome
    if storage:
        storage.log_client_update(
            curr_vid, client_id, status, norm_val, dist_val, reason
        )

    if status == "REJECT":
        logger.warning(f"REJECTED [{client_id}]: {reason}")
        return {"status": "REJECT", "reason": reason}

    # ── ACCEPT — queue and potentially aggregate ──────────────────────────────
    async with model_lock:
        pending_updates.append((client_id, client_weights))
        q_len = len(pending_updates)
        logger.info(
            f"ACCEPTED [{client_id}] — queue {q_len}/{MIN_UPDATE_QUEUE}"
        )

        if q_len >= MIN_UPDATE_QUEUE:
            # Drain the queue atomically inside the lock
            batch = list(pending_updates)
            pending_updates.clear()

            # Run aggregation as a background task
            # (cannot await inside lock — schedule via BackgroundTasks)
            background_tasks.add_task(_aggregate_batch, batch)

    return {"status": "ACCEPT", "message": "Update queued for aggregation."}


def _aggregate_batch(batch: List[tuple]):
    """
    Synchronous wrapper so BackgroundTasks can schedule async aggregation.
    Creates a new event loop for the async call.
    """
    import asyncio as _asyncio
    loop = _asyncio.new_event_loop()
    try:
        loop.run_until_complete(_run_aggregation(batch))
    finally:
        loop.close()


async def _run_aggregation(batch: List[tuple]):
    """Acquires the lock and runs aggregation for the given batch."""
    async with model_lock:
        await _aggregate_and_update(batch)


# ---------------------------------------------------------------------------
# Dataset upload & per-client weight serving
# ---------------------------------------------------------------------------

def train_client_background(client_id: str, file_path: str):
    """
    Background task: trains the CNN on the real uploaded dataset (CSV or otherwise)
    using the real csv_trainer module, then saves weights to the client directory.
    """
    import asyncio as _asyncio
    logger.info(f"[BG Train] Starting for client {client_id} on {file_path}")

    try:
        # Resolve paths
        backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)

        from clients.csv_trainer import train_on_csv

        # Fetch current global weights safely
        async def _fetch_weights():
            async with model_lock:
                return global_model.get_weights()

        gw = _asyncio.run(_fetch_weights())

        saved_path = train_on_csv(
            client_id=client_id,
            csv_path=file_path,
            global_weights=gw,
            epochs=int(os.getenv("BG_TRAIN_EPOCHS", "3")),
        )

        if saved_path:
            logger.info(f"[BG Train] ✅ Weights saved to {saved_path}")
        else:
            logger.warning(f"[BG Train] ⚠️  Training produced no weights for {client_id}.")

    except Exception as exc:
        logger.error(f"[BG Train] ❌ Failed for client {client_id}: {exc}", exc_info=True)


@app.get("/api/model/weights/{client_id}")
async def get_client_weights(client_id: str):
    """Download the locally-trained .pt file for a specific client."""
    client_dir   = os.path.join(os.path.dirname(__file__), "..", "data", f"client_{client_id}")
    weights_path = os.path.join(client_dir, f"weights_{client_id}.pt")

    if os.path.exists(weights_path):
        return FileResponse(
            weights_path,
            filename=f"weights_{client_id}.pt",
            media_type="application/octet-stream",
        )
    raise HTTPException(404, "Weights not found — training may still be in progress.")


@app.post("/api/dataset/upload")
async def upload_dataset(
    background_tasks: BackgroundTasks,
    client_id: str      = Form(...),
    file: UploadFile    = File(None),
    dataset_url: str    = Form(None),
):
    """
    Accepts a dataset either as a direct upload or as a URL.
    Saves it to data/client_{id}/ and fires background local training.
    """
    if not file and not dataset_url:
        raise HTTPException(
            400, "Provide either a file upload or a dataset_url."
        )

    client_dir = os.path.join(
        os.path.dirname(__file__), "..", "data", f"client_{client_id}"
    )
    os.makedirs(client_dir, exist_ok=True)

    file_path = ""

    if file:
        file_path = os.path.join(client_dir, file.filename or "dataset.csv")
        try:
            # Stream-write in 1 MB chunks to avoid loading the whole file into RAM
            CHUNK = 1024 * 1024  # 1 MB
            with open(file_path, "wb") as out:
                while True:
                    chunk = await file.read(CHUNK)
                    if not chunk:
                        break
                    out.write(chunk)
        except Exception as exc:
            raise HTTPException(500, f"Failed to save file: {exc}")
        logger.info(f"Dataset '{file.filename}' saved for client {client_id} (streaming).")


    elif dataset_url:
        parsed   = urlparse(dataset_url)
        filename = os.path.basename(parsed.path) or "dataset.csv"
        file_path = os.path.join(client_dir, filename)
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode    = ssl.CERT_NONE
            req = urllib.request.Request(
                dataset_url, headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, context=ctx) as resp, \
                 open(file_path, "wb") as out:
                shutil.copyfileobj(resp, out)
        except Exception as exc:
            raise HTTPException(500, f"Failed to download from URL: {exc}")
        logger.info(f"Dataset downloaded from {dataset_url} for client {client_id}.")

    background_tasks.add_task(train_client_background, client_id, file_path)
    return {"message": "Dataset received. Background training started."}


# ---------------------------------------------------------------------------
# Simulation helper  (fires a synthetic round for demo / testing)
# ---------------------------------------------------------------------------

class SimulationRequest(BaseModel):
    client_name:          str
    is_malicious:         bool  = False
    malicious_multiplier: float = 50.0


@app.post("/simulate")
async def run_local_simulation(
    background_tasks: BackgroundTasks,
    req: SimulationRequest,
):
    """
    Kicks off a real PyTorch local training run in a background thread,
    then calls /update internally with the resulting weights.
    Malicious clients inject large random noise before submission.
    """

    def _worker(name: str, malicious: bool, mult: float, vid: int):
        logger.info(f"[Sim] Starting MOCK-{name}")

        db_client_id = storage.register_client(f"MOCK-{name}")

        # Fetch current global weights synchronously
        import asyncio as _as
        gw = _as.run(
            _async_get_weights()
        )

        local_model = CNNModel()
        local_model.set_weights(gw)
        local_model.train()

        optimizer = torch.optim.SGD(local_model.parameters(), lr=0.01, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()

        for _ in range(5):
            inputs  = torch.randn(16, 1, 28, 28)
            targets = torch.randint(0, 10, (16,))
            optimizer.zero_grad()
            loss = criterion(local_model(inputs), targets)
            loss.backward()
            optimizer.step()

        client_weights = {
            k: v.cpu().detach().clone()
            for k, v in local_model.state_dict().items()
        }

        if malicious:
            for key in client_weights:
                client_weights[key] += torch.randn_like(client_weights[key]) * mult

        # Detect & queue
        status, reason, norm_val, dist_val = detect_update(client_weights, gw)
        if storage:
            storage.log_client_update(vid, db_client_id, status, norm_val, dist_val, reason)

        if status == "ACCEPT":
            _as.run(_enqueue_and_maybe_aggregate(db_client_id, client_weights))

        logger.info(f"[Sim] MOCK-{name} → {status}")

    background_tasks.add_task(
        _worker,
        req.client_name,
        req.is_malicious,
        req.malicious_multiplier,
        current_version_id,
    )
    return {"status": "Simulation started in background"}


async def _async_get_weights() -> Dict[str, torch.Tensor]:
    async with model_lock:
        return global_model.get_weights()


async def _enqueue_and_maybe_aggregate(client_id: str, weights: Dict[str, torch.Tensor]):
    global pending_updates
    async with model_lock:
        pending_updates.append((client_id, weights))
        if len(pending_updates) >= MIN_UPDATE_QUEUE:
            batch = list(pending_updates)
            pending_updates.clear()
            await _aggregate_and_update(batch)


# ---------------------------------------------------------------------------
# Admin config  (hot-update runtime settings without restart)
# ---------------------------------------------------------------------------

class ConfigUpdate(BaseModel):
    dp_enabled:       bool  = True
    dp_clip_norm:     float = 10.0
    dp_noise_mult:    float = 1.0
    min_update_queue: int   = 1


@app.get("/admin/config")
async def get_config():
    return {
        "dp_enabled":       DP_ENABLED,
        "dp_clip_norm":     DP_CLIP_NORM,
        "dp_noise_mult":    DP_NOISE_MULT,
        "min_update_queue": MIN_UPDATE_QUEUE,
    }


@app.post("/admin/config")
async def set_config(cfg: ConfigUpdate):
    global DP_ENABLED, DP_CLIP_NORM, DP_NOISE_MULT, MIN_UPDATE_QUEUE
    DP_ENABLED       = cfg.dp_enabled
    DP_CLIP_NORM     = cfg.dp_clip_norm
    DP_NOISE_MULT    = cfg.dp_noise_mult
    MIN_UPDATE_QUEUE = cfg.min_update_queue
    logger.info(
        f"Config updated: DP={DP_ENABLED}, C={DP_CLIP_NORM}, "
        f"σ={DP_NOISE_MULT}, Q={MIN_UPDATE_QUEUE}"
    )
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Model registry  (Dashboard: version list + global model download)
# ---------------------------------------------------------------------------

@app.get("/versions")
async def get_versions():
    """List all stored global model versions."""
    if not storage:
        raise HTTPException(500, "Storage uninitialized.")
    try:
        rows = (
            storage.supabase.table("model_versions")
            .select("*")
            .order("version_num", desc=True)
            .limit(50)
            .execute()
        )
        return {"data": rows.data}
    except Exception as exc:
        logger.error(f"Failed to fetch versions: {exc}")
        return {"data": []}


@app.get("/model/download")
async def download_global_model(version_id: str = None):
    """Download a specific (or latest) global model checkpoint."""
    if not storage:
        raise HTTPException(500, "Storage uninitialized.")
    try:
        if version_id:
            row = (
                storage.supabase.table("model_versions")
                .select("file_path")
                .eq("id", version_id)
                .single()
                .execute()
            )
            file_path = row.data.get("file_path") if row.data else None
        else:
            latest    = storage.get_latest_version()
            file_path = latest.get("file_path")

        if file_path and os.path.exists(file_path):
            return FileResponse(
                path=file_path,
                filename=os.path.basename(file_path),
                media_type="application/octet-stream",
            )
        raise HTTPException(404, "Model file not found on disk.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Download error: {exc}")
