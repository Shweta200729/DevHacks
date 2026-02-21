"""
server/main.py

Production Asynchronous Federated Learning Server.

All configurable parameters come from config.settings (FLSettings).
No magic numbers, no hardcoded dataset names, no hardcoded model shapes.

Endpoints:
  GET  /model              — Clients fetch current global weights.
  POST /update             — Clients push locally-trained weight updates.
  GET  /metrics            — Dashboard: evaluation & aggregation history.
  GET  /clients            — Dashboard: recent client update attempts.
  GET  /versions           — Dashboard: model version registry.
  GET  /model/download     — Download a saved global model checkpoint.
  POST /simulate           — Fire a synthetic FL round (for testing/demo).
  POST /admin/config       — Hot-update runtime settings.
  GET  /admin/config       — Read current runtime settings.
  POST /api/dataset/upload — Upload a CSV dataset and trigger background training.
  GET  /api/model/weights/{client_id} — Download per-client trained weights.

Architecture:
  - All ML logic (aggregation, detection, evaluation) is fully modular.
  - DB logic is isolated inside SupabaseManager.
  - asyncio.Lock guards all global model mutations.
  - Model architecture is determined at runtime from data shape (no CNNModel import).
"""

import io
import os
import sys
import ssl
import asyncio
import logging
import shutil
import urllib.request
from urllib.parse import urlparse
from typing import Dict, Any, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

import torch
from torch.utils.data import DataLoader

# ── Configuration (single source of truth) ─────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import FLSettings, settings as cfg

# ── Model factory ────────────────────────────────────────────────────────────
from models.model_factory import build_model, FLModel

# ── Universal dataset loader ─────────────────────────────────────────────────
from data.dataset_loader import load_dataset

# ── Server-side ML modules ────────────────────────────────────────────────────
from server.detection import detect_update
from server.aggregator import aggregate
from server.storage import SupabaseManager

# ── Evaluation ────────────────────────────────────────────────────────────────
from experiments.evaluation import evaluate_model

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Asynchronous Federated Learning Server",
    description="Dataset-agnostic, model-agnostic FL server.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global mutable state  (all writes go through model_lock)
# ---------------------------------------------------------------------------

global_model:    Optional[FLModel]     = None
model_lock:      asyncio.Lock          = asyncio.Lock()
storage:         Optional[SupabaseManager] = None
val_loader:      Optional[DataLoader]  = None
pending_updates: List[Tuple[str, Dict[str, torch.Tensor]]] = []

# Runtime-mutable config mirror — updated via POST /admin/config
# Points to the singleton; only the DP/queue fields are hot-swappable.
runtime_cfg: FLSettings = cfg   # replaced by ConfigUpdate POST

current_version_id:  Optional[str] = None
current_version_num: int            = 0


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """
    Runs on server boot:
    1. Initialise Supabase storage.
    2. Load dataset (to learn input_shape & num_classes).
    3. Build global model from those shapes.
    4. Restore latest checkpoint if one exists.
    5. Load validation DataLoader.
    """
    global storage, global_model, val_loader, current_version_id, current_version_num

    # ── 1. Storage ────────────────────────────────────────────────────────────
    try:
        storage = SupabaseManager()
        logger.info("SupabaseManager initialised.")
    except Exception as exc:
        logger.error(f"SupabaseManager init failed: {exc}")

    # ── 2 & 3. Dataset → model ────────────────────────────────────────────────
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        logger.info(f"Loading dataset '{cfg.DATASET_NAME}' …")
        ds_result = load_dataset(cfg)
        input_shape = ds_result["input_shape"]
        num_classes = ds_result["num_classes"]
        val_loader  = ds_result["val_dataloader"]
    except Exception as exc:
        logger.error(f"Dataset load failed: {exc}. Falling back to MNIST defaults.")
        # Absolute fallback — only if env has a misconfigured dataset
        from torchvision import datasets as tv_datasets, transforms
        _transform  = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))])
        _val_ds     = tv_datasets.MNIST(cfg.DATASET_ROOT, train=False,
                                        download=True, transform=_transform)
        val_loader  = DataLoader(_val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
        input_shape = (1, 28, 28)
        num_classes = 10

    global_model = build_model(
        input_shape  = input_shape,
        num_classes  = num_classes,
        hidden_dims  = cfg.hidden_dims(),
    )
    logger.info(f"Global model built: {global_model}")

    # ── 4. Restore checkpoint from DB ────────────────────────────────────────
    if storage:
        latest = storage.get_latest_version()
        if latest["version_num"] > 0 and latest["file_path"] and os.path.exists(latest["file_path"]):
            try:
                ckpt = torch.load(latest["file_path"], weights_only=True, map_location="cpu")
                global_model.set_weights(ckpt)
                current_version_num = latest["version_num"]
                current_version_id  = latest["id"]
                logger.info(f"Restored checkpoint: Version {current_version_num}")
            except Exception as exc:
                logger.warning(f"Checkpoint load failed ({exc}). Starting fresh.")
                await _init_fresh_model()
        else:
            await _init_fresh_model()


async def _init_fresh_model():
    """Saves the freshly-built model as Version N+1 in the DB."""
    global current_version_num, current_version_id
    if not storage:
        current_version_num = 1
        return
    try:
        rows = (
            storage.supabase.table("model_versions")
            .select("version_num")
            .order("version_num", desc=True)
            .limit(1)
            .execute()
        )
        current_version_num = (rows.data[0]["version_num"] + 1) if rows.data else 1
    except Exception:
        current_version_num = 1
    current_version_id = storage.save_model_version(
        global_model.get_weights(), current_version_num
    )
    logger.info(f"Fresh global model saved as Version {current_version_num}.")


# ---------------------------------------------------------------------------
# Core aggregation pipeline
# ---------------------------------------------------------------------------

async def _aggregate_and_update(updates: List[Tuple[str, Dict]]) -> Dict[str, float]:
    """
    Called inside model_lock.
    Aggregates client updates, applies result to global model, evaluates, persists.

    Args:
        updates: List of (client_id, state_dict) pairs.

    Returns:
        {'loss': float, 'accuracy': float}
    """
    global current_version_num, current_version_id

    weight_dicts = [w for _, w in updates]
    n = len(weight_dicts)

    logger.info(f"[Aggregation] Running on {n} update(s) …")

    aggregated, method_name = aggregate(weight_dicts, cfg=runtime_cfg)

    global_model.set_weights(aggregated)
    current_version_num += 1

    metrics = evaluate_model(global_model, val_loader, device=cfg.DEVICE)

    if storage:
        new_vid = storage.save_model_version(aggregated, current_version_num)
        storage.log_aggregation(new_vid, accepted=n, rejected=0, method=method_name)
        storage.log_evaluation(new_vid, metrics["loss"], metrics["accuracy"])
        current_version_id = new_vid

    logger.info(
        f"[Round {current_version_num}] method={method_name} | "
        f"loss={metrics['loss']:.4f} | acc={metrics['accuracy']:.4f}"
    )
    return metrics


def _run_aggregation_sync(batch: List[Tuple]):
    """Synchronous wrapper for BackgroundTasks."""
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_run_aggregation_async(batch))
    finally:
        loop.close()


async def _run_aggregation_async(batch: List[Tuple]):
    async with model_lock:
        await _aggregate_and_update(batch)


# ---------------------------------------------------------------------------
# Core endpoints
# ---------------------------------------------------------------------------

@app.get("/model")
async def get_global_model():
    """Clients download current global model weights (.pt binary)."""
    async with model_lock:
        weights = global_model.get_weights()
    buf = io.BytesIO()
    torch.save(weights, buf)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=global_model.pt"},
    )


@app.post("/update")
async def receive_update(
    background_tasks: BackgroundTasks,
    client_id: str      = Form(...),
    file:      UploadFile = File(...),
):
    """
    Receives a client's locally-trained .pt weight file.

    Flow:
        1. Deserialise weights.
        2. Byzantine detection against current global model.
        3. REJECT → log + return.
        4. ACCEPT → enqueue; if queue >= MIN_AGGREGATE_SIZE, aggregate.
    """
    global pending_updates

    try:
        raw = await file.read()
        client_weights: Dict[str, torch.Tensor] = torch.load(
            io.BytesIO(raw), weights_only=True, map_location="cpu"
        )
    except Exception as exc:
        raise HTTPException(400, f"Invalid weight file: {exc}")

    async with model_lock:
        global_weights = global_model.get_weights()
        curr_vid = current_version_id

    status, reason, norm_val, dist_val = detect_update(
        client_weights, global_weights, cfg=runtime_cfg
    )

    # DB log (best-effort — db errors must not crash the endpoint)
    if storage:
        try:
            db_uuid = storage.register_client(client_id)
            storage.log_client_update(curr_vid, db_uuid, status, norm_val, dist_val, reason)
        except Exception as db_err:
            logger.warning(f"DB log failed for {client_id}: {db_err}")

    if status == "REJECT":
        logger.warning(f"[Update] REJECT [{client_id}]: {reason}")
        return {"status": "REJECT", "reason": reason}

    async with model_lock:
        pending_updates.append((client_id, client_weights))
        q = len(pending_updates)
        logger.info(f"[Update] ACCEPT [{client_id}] — queue {q}/{runtime_cfg.MIN_AGGREGATE_SIZE}")

        if q >= runtime_cfg.MIN_AGGREGATE_SIZE:
            batch = list(pending_updates)
            pending_updates.clear()
            background_tasks.add_task(_run_aggregation_sync, batch)

    return {"status": "ACCEPT", "message": "Update queued for aggregation."}


# ---------------------------------------------------------------------------
# Dashboard read endpoints
# ---------------------------------------------------------------------------

@app.get("/metrics")
async def get_metrics():
    """Evaluation + aggregation history for the dashboard."""
    if not storage:
        raise HTTPException(500, "Storage uninitialized.")
    evals = (
        storage.supabase.table("evaluation_metrics")
        .select("*").order("version_id", desc=True).limit(20).execute()
    )
    aggs = (
        storage.supabase.table("aggregation_logs")
        .select("*").order("version_id", desc=True).limit(20).execute()
    )
    return {
        "current_version":    current_version_num,
        "evaluations":        evals.data,
        "aggregations":       aggs.data,
        "pending_queue_size": len(pending_updates),
    }


@app.get("/clients")
async def get_clients():
    """Recent client update attempts for the dashboard."""
    if not storage:
        raise HTTPException(500, "Storage uninitialized.")
    try:
        rows = (
            storage.supabase.table("client_updates")
            .select("*").order("created_at", desc=True).limit(50).execute()
        )
        return {"data": rows.data}
    except Exception as exc:
        logger.error(f"Failed to fetch clients: {exc}")
        return {"data": []}


@app.get("/versions")
async def get_versions():
    """List all saved global model checkpoints."""
    if not storage:
        raise HTTPException(500, "Storage uninitialized.")
    try:
        rows = (
            storage.supabase.table("model_versions")
            .select("*").order("version_num", desc=True).limit(50).execute()
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
                .select("file_path").eq("id", version_id).single().execute()
            )
            file_path = row.data.get("file_path") if row.data else None
        else:
            file_path = storage.get_latest_version().get("file_path")

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


# ---------------------------------------------------------------------------
# Simulation  (demo / testing  — fires a real training loop)
# ---------------------------------------------------------------------------

class SimulationRequest(BaseModel):
    client_name:          str
    is_malicious:         bool  = False
    malicious_multiplier: float = cfg.NORM_THRESHOLD * 3


@app.post("/simulate")
async def run_simulation(background_tasks: BackgroundTasks, req: SimulationRequest):
    """
    Fires a real local PyTorch training run in a background thread.
    Uses the same global model + FLSettings as real clients.
    Malicious clients inject large Gaussian noise before submission.
    """
    def _worker(name: str, malicious: bool, mult: float, vid):
        import asyncio as _as
        from clients.local_training import get_updated_weights

        logger.info(f"[Sim] Starting {name}")
        db_client_id = storage.register_client(f"MOCK-{name}") if storage else name

        gw = _as.run(_async_get_weights())

        # Build a local copy of the global model
        local_m = build_model(
            input_shape=global_model.input_shape,
            num_classes=global_model.num_classes,
            hidden_dims=cfg.hidden_dims(),
        )
        local_m.set_weights(gw)
        local_m.train()

        optimizer = torch.optim.SGD(local_m.parameters(),
                                    lr=cfg.LEARNING_RATE, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()

        # Synthetic mini-batches shaped to match the model's input_shape
        c, *spatial = global_model.input_shape if len(global_model.input_shape) == 3 else (global_model.input_shape[0],)
        batch_shape = (16, *global_model.input_shape)

        for _ in range(5):
            inputs  = torch.randn(*batch_shape)
            targets = torch.randint(0, global_model.num_classes, (16,))
            optimizer.zero_grad()
            loss = criterion(local_m(inputs), targets)
            loss.backward()
            optimizer.step()

        weights = get_updated_weights(local_m)

        if malicious:
            for key in weights:
                weights[key] += torch.randn_like(weights[key]) * mult

        status, reason, norm_val, dist_val = detect_update(weights, gw, cfg=runtime_cfg)
        if storage:
            try:
                storage.log_client_update(vid, db_client_id, status, norm_val, dist_val, reason)
            except Exception:
                pass

        if status == "ACCEPT":
            _as.run(_enqueue_and_aggregate(db_client_id, weights))
        logger.info(f"[Sim] {name} → {status}")

    background_tasks.add_task(
        _worker, req.client_name, req.is_malicious,
        req.malicious_multiplier, current_version_id,
    )
    return {"status": "Simulation started in background."}


async def _async_get_weights() -> Dict[str, torch.Tensor]:
    async with model_lock:
        return global_model.get_weights()


async def _enqueue_and_aggregate(client_id: str, weights: Dict[str, torch.Tensor]):
    global pending_updates
    async with model_lock:
        pending_updates.append((client_id, weights))
        if len(pending_updates) >= runtime_cfg.MIN_AGGREGATE_SIZE:
            batch = list(pending_updates)
            pending_updates.clear()
            await _aggregate_and_update(batch)


# ---------------------------------------------------------------------------
# Admin config  (hot-update DP/queue settings without restart)
# ---------------------------------------------------------------------------

class ConfigUpdate(BaseModel):
    dp_enabled:       bool  = False
    dp_clip_norm:     float = 10.0
    dp_noise_mult:    float = 1.0
    min_update_queue: int   = 1


@app.get("/admin/config")
async def get_config():
    return {
        "dp_enabled":       runtime_cfg.DP_ENABLED,
        "dp_clip_norm":     runtime_cfg.DP_CLIP_NORM,
        "dp_noise_mult":    runtime_cfg.DP_NOISE_MULT,
        "min_update_queue": runtime_cfg.MIN_AGGREGATE_SIZE,
    }


@app.post("/admin/config")
async def set_config(update: ConfigUpdate):
    """Hot-updates runtime DP & queue settings in memory (no restart needed)."""
    global runtime_cfg
    # Create a patched copy using model_copy (Pydantic v2)
    runtime_cfg = runtime_cfg.model_copy(update={
        "DP_ENABLED":       update.dp_enabled,
        "DP_CLIP_NORM":     update.dp_clip_norm,
        "DP_NOISE_MULT":    update.dp_noise_mult,
        "MIN_AGGREGATE_SIZE": update.min_update_queue,
    })
    logger.info(
        f"[Config] Updated: DP={runtime_cfg.DP_ENABLED}, "
        f"clip={runtime_cfg.DP_CLIP_NORM}, σ={runtime_cfg.DP_NOISE_MULT}, "
        f"queue={runtime_cfg.MIN_AGGREGATE_SIZE}"
    )
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Dataset upload + per-client weight serving
# ---------------------------------------------------------------------------

def _train_client_background(client_id: str, file_path: str):
    """
    Background task: trains on the uploaded CSV using the universal csv_trainer.
    Uses FLSettings for epochs — no magic numbers.
    """
    import asyncio as _as
    logger.info(f"[BG Train] Starting for {client_id} on {file_path}")
    try:
        from clients.csv_trainer import train_on_csv
        gw = _as.run(_async_get_weights())
        saved = train_on_csv(
            client_id     = client_id,
            csv_path      = file_path,
            global_weights= gw,
            epochs        = cfg.BG_TRAIN_EPOCHS,
            device        = cfg.DEVICE,
        )
        if saved:
            logger.info(f"[BG Train] ✅ {client_id} → {saved}")
        else:
            logger.warning(f"[BG Train] ⚠️ No weights produced for {client_id}")
    except Exception as exc:
        logger.error(f"[BG Train] ❌ {client_id}: {exc}", exc_info=True)


@app.post("/api/dataset/upload")
async def upload_dataset(
    background_tasks: BackgroundTasks,
    client_id:   str        = Form(...),
    file:        UploadFile = File(None),
    dataset_url: str        = Form(None),
):
    """Upload a dataset (file or URL) and trigger background local training."""
    if not file and not dataset_url:
        raise HTTPException(400, "Provide file or dataset_url.")

    client_dir = os.path.join(
        os.path.dirname(__file__), "..", "data", f"client_{client_id}"
    )
    os.makedirs(client_dir, exist_ok=True)
    file_path = ""

    if file:
        file_path = os.path.join(client_dir, file.filename or "dataset.csv")
        # Stream-write in 1 MB chunks — avoids OOM for large files
        CHUNK = 1024 * 1024
        with open(file_path, "wb") as out:
            while True:
                chunk = await file.read(CHUNK)
                if not chunk:
                    break
                out.write(chunk)
        logger.info(f"Dataset '{file.filename}' saved for {client_id} (streaming).")

    elif dataset_url:
        parsed    = urlparse(dataset_url)
        filename  = os.path.basename(parsed.path) or "dataset.csv"
        file_path = os.path.join(client_dir, filename)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode    = ssl.CERT_NONE
        req = urllib.request.Request(dataset_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx) as resp, \
             open(file_path, "wb") as out:
            shutil.copyfileobj(resp, out)
        logger.info(f"Dataset downloaded from {dataset_url} for {client_id}.")

    background_tasks.add_task(_train_client_background, client_id, file_path)
    return {"message": "Dataset received. Background training started."}


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
