import os
import io
import asyncio
import logging
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from pydantic import BaseModel
from dotenv import load_dotenv
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Load custom FL modules
from model import CNNModel
from detection import detect_update
from aggregator import dp_trimmed_mean
from storage import SupabaseManager
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.evaluation import evaluate_model

# -------------------------------------------------------------
# Configuration & Setup
# -------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Production Asynchronous Federated Learning")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global State
global_model = CNNModel()
model_lock = asyncio.Lock()
storage: SupabaseManager = None

# In-memory queue of valid updates waiting to be aggregated
# List of Tuples (client_id, weight_dict)
pending_updates: List[tuple] = []
MIN_UPDATE_QUEUE = int(os.getenv("MIN_AGGREGATE_SIZE", "3"))

# Differential Privacy Configuration
# DP_ENABLED       – set to "false" to disable DP (ablation mode)
# DP_CLIP_NORM     – max L2 norm per client update (sensitivity bound, C)
# DP_NOISE_MULT    – noise multiplier (sigma = noise_mult * clip_norm / n)
#                    Higher = more privacy, more noise. Typical range: 0.5 – 2.0
DP_ENABLED = os.getenv("DP_ENABLED", "true").lower() == "true"
DP_CLIP_NORM = float(os.getenv("DP_CLIP_NORM", "10.0"))
DP_NOISE_MULT = float(os.getenv("DP_NOISE_MULT", "1.0"))

# Validation Dataset Tracker (Loaded on startup to save memory)
val_loader: DataLoader = None

# Active DB Record Trackers
current_version_id: int = None
current_version_num: int = 0


@app.on_event("startup")
async def startup_event():
    global storage, val_loader, current_version_id, current_version_num
    try:
        storage = SupabaseManager()
        logger.info("SupabaseManager Initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize SupabaseManager: {e}")
        # Not raising, allowing server to start for visibility, but it won't work well

    # Attempt to load latest version
    if storage:
        latest = storage.get_latest_version()
        if (
            latest["version_num"] > 0
            and latest["file_path"]
            and os.path.exists(latest["file_path"])
        ):
            global_model.load_state_dict(torch.load(latest["file_path"]))
            current_version_num = latest["version_num"]
            current_version_id = latest["id"]
            logger.info(f"Loaded existing global model: Version {current_version_num}")
        else:
            # First time startup
            current_version_num = 1
            current_version_id = storage.save_model_version(
                global_model.get_weights(), current_version_num
            )
            logger.info("Initialized Brand New Global Model (Version 1).")

    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context

    # Load MNIST Validation Set
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    val_dataset = datasets.MNIST(
        "../data", train=False, download=True, transform=transform
    )
    val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)
    logger.info("MNIST Evaluation Dataset Loaded.")


# -------------------------------------------------------------
# Core Execution Handlers
# -------------------------------------------------------------


async def trigger_aggregation():
    """
    Background Task: Empties the queue safely, aggregates weights strictly on CPU,
    updates the model, evaluates it, and stores everything in Supabase.
    """
    global pending_updates, current_version_num, current_version_id

    async with model_lock:
        if len(pending_updates) < MIN_UPDATE_QUEUE:
            return  # Should not realistically trigger

        logger.info(f"Triggering Aggregation on {len(pending_updates)} clients...")

        # 1. Extract valid weights
        updates_to_process = [u[1] for u in pending_updates]
        accepted_clients_count = len(updates_to_process)

        # 2. Aggregation Math (Trimmed Mean + Gaussian DP noise)
        try:
            aggregated_weights = dp_trimmed_mean(
                updates_to_process,
                trim_ratio=0.2,
                clip_norm=DP_CLIP_NORM,
                noise_multiplier=DP_NOISE_MULT,
                enabled=DP_ENABLED,
            )
        except Exception as e:
            logger.error(f"Aggregation Math Failed: {e}")
            return

        # Clear queue since we captured them
        pending_updates.clear()

        # 3. Apply the Weights
        global_model.set_weights(aggregated_weights)
        current_version_num += 1

        # 4. Evaluate the new iteration (Takes time, but we are in background)
        metrics = evaluate_model(global_model, val_loader)

        # 5. Db logging
        new_vid = storage.save_model_version(aggregated_weights, current_version_num)

        storage.log_aggregation(
            new_vid,
            accepted=accepted_clients_count,
            rejected=0,
            method="DP-TrimmedMean" if DP_ENABLED else "TrimmedMean",
        )
        storage.log_evaluation(new_vid, metrics["loss"], metrics["accuracy"])

        current_version_id = new_vid
        logger.info(
            f"Aggregation Round {current_version_num} Complete! Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.4f}"
        )


# -------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------


@app.get("/metrics")
async def get_metrics():
    """Endpoint for Dashboard Frontend."""
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
        "evaluations": evals.data,
        "aggregations": aggs.data,
        "pending_queue_size": len(pending_updates),
    }


@app.get("/clients")
async def get_clients():
    """Endpoint for Dashboard Edge Clients Page."""
    if not storage:
        raise HTTPException(500, "Storage uninitialized.")

    try:
        # Get the latest 50 client update attempts
        client_logs = (
            storage.supabase.table("client_updates")
            .select("*")
            .order("created_at", desc=True)
            .limit(50)
            .execute()
        )
        return {"data": client_logs.data}
    except Exception as e:
        logger.error(f"Failed to fetch clients: {e}")
        return {"data": []}


@app.post("/update")
async def receive_update(
    background_tasks: BackgroundTasks,
    client_id: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Endpoint for edge clients to push their weights back asynchronously.
    Reads a PyTorch .pt byte stream.
    """
    global pending_updates

    try:
        content = await file.read()
        buffer = io.BytesIO(content)
        client_weights = torch.load(buffer, weights_only=True)
    except Exception as e:
        raise HTTPException(400, f"Invalid PyTorch weight file: {e}")

    # Enter lock just to grab the canonical weights to distance check against
    async with model_lock:
        global_weights = global_model.get_weights()
        curr_vid = current_version_id

    # Validation step
    status, reason, norm_val, dist_val = detect_update(client_weights, global_weights)

    # Store attempt to DB
    storage.log_client_update(curr_vid, client_id, status, norm_val, dist_val, reason)

    if status == "REJECT":
        logger.warning(f"Rejected Client {client_id}: {reason}")
        return {"status": "REJECT", "reason": reason}

    # Queue updates securely
    async with model_lock:
        pending_updates.append((client_id, client_weights))
        q_len = len(pending_updates)
        logger.info(
            f"Accepted Client {client_id}. Queue size: {q_len}/{MIN_UPDATE_QUEUE}"
        )

        # Fire background aggregation if threshold criteria hit
        if q_len >= MIN_UPDATE_QUEUE:
            background_tasks.add_task(trigger_aggregation)

    return {"status": "ACCEPT", "message": "Update queued."}


# -------------------------------------------------------------
# Dataset Upload and Background Training
# -------------------------------------------------------------
import shutil
from fastapi.responses import FileResponse

@app.post("/api/dataset/upload")
async def upload_dataset(
    background_tasks: BackgroundTasks,
    client_id: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Uploads a dataset for a specific client and triggers background training.
    """
    client_dir = os.path.join(os.path.dirname(__file__), "..", "data", f"client_{client_id}")
    os.makedirs(client_dir, exist_ok=True)
    
    file_path = os.path.join(client_dir, file.filename)
    
    # Save the file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
        
    logger.info(f"Dataset {file.filename} saved for client {client_id}")
    
    # Trigger background training
    from clients.trainer import train_client_background
    background_tasks.add_task(train_client_background, client_id, file_path)
    
    return {"message": "Dataset uploaded successfully. Background training started."}


@app.get("/api/model/weights/{client_id}")
async def get_client_weights(client_id: str):
    """
    Retrieves the trained .pt weights for a specific client if training has finished.
    """
    client_dir = os.path.join(os.path.dirname(__file__), "..", "data", f"client_{client_id}")
    model_path = os.path.join(client_dir, "model.pt")
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Weights not found. Training may still be in progress.")
        
    return FileResponse(
        path=model_path, 
        filename=f"{client_id}_model.pt", 
        media_type="application/octet-stream"
    )

# -------------------------------------------------------------
# Local Simulation Helper
# -------------------------------------------------------------
class SimulationRequest(BaseModel):
    client_name: str
    is_malicious: bool = False
    malicious_multiplier: float = 50.0


@app.post("/simulate")
async def run_local_simulation(
    background_tasks: BackgroundTasks, req: SimulationRequest
):
    """
    Kicks off an authentic PyTorch local training routine on a thread in the background,
    then fires it to the primary /update endpoint locally.
    """

    def local_train_worker(name: str, malicious: bool, mult: float, vid: int):
        logger.info(f"Starting simulated local training for MOCK-{name}")

        db_client_id = storage.register_client(f"MOCK-{name}")

        # Emulate loading from API
        local_model = CNNModel()

        async def fetch_w():
            async with model_lock:
                return global_model.get_weights()

        # Simple local training loop on 1 batch of pure noise (representing decoupled data)
        # Using real PyTorch grad backprop to drift the model authentically!
        weights_to_train = asyncio.run(fetch_w())
        local_model.set_weights(weights_to_train)
        local_model.train()

        optimizer = torch.optim.SGD(local_model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        # Synthesize real training drift (non-iid simulation)
        for _ in range(5):
            inputs = torch.randn(16, 1, 28, 28)
            targets = torch.randint(0, 10, (16,))

            optimizer.zero_grad()
            outputs = local_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        client_weights = local_model.get_weights()

        if malicious:
            for key in client_weights.keys():
                client_weights[key] += torch.randn_like(client_weights[key]) * mult

        # Manually invoke the pipeline via memory bypassing REST (to keep file parsing simple in simulation)
        status, reason, norm_val, dist_val = detect_update(
            client_weights, asyncio.run(fetch_w())
        )
        storage.log_client_update(vid, db_client_id, status, norm_val, dist_val, reason)

        async def handle_valid():
            global pending_updates
            async with model_lock:
                pending_updates.append((db_client_id, client_weights))
                if len(pending_updates) >= MIN_UPDATE_QUEUE:
                    await trigger_aggregation()

        if status == "ACCEPT":
            asyncio.run(handle_valid())

        logger.info(f"Simulation MOCK-{name} finished. Result: {status}")

    background_tasks.add_task(
        local_train_worker,
        req.client_name,
        req.is_malicious,
        req.malicious_multiplier,
        current_version_id,
    )
    return {"status": "Simulating in background"}
