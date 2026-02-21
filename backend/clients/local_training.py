"""
client/local_training.py

Implements the client-side local training pipeline for Federated Learning.
All logic uses real PyTorch operations — no mocking, no fake gradients.

Responsibilities:
  - Load global weights from the server into a local model.
  - Run a real, full training loop on local data.
  - Return updated weights for submission to the server.
"""

import os
import sys
import copy
import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Resolve path so we can import CNNModel from server/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "server")))
from model import CNNModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Weight Loading
# ---------------------------------------------------------------------------

def load_global_weights(model: nn.Module, weights_dict: Dict[str, torch.Tensor]) -> None:
    """
    Safely loads the server's global weights into the local model.

    Validates that layer names match exactly to detect architecture drift
    between server and client versions.

    Args:
        model:       The local PyTorch model instance to update.
        weights_dict: State dict received from the server (CPU tensors).

    Raises:
        KeyError:   If any layer key present in the server dict is missing
                    from the local model (or vice-versa).
        RuntimeError: If tensor shapes are incompatible.
    """
    local_keys  = set(model.state_dict().keys())
    server_keys = set(weights_dict.keys())

    missing_in_local  = server_keys - local_keys
    extra_in_local    = local_keys  - server_keys

    if missing_in_local:
        raise KeyError(
            f"Server weights contain layers not found in local model: {missing_in_local}"
        )
    if extra_in_local:
        raise KeyError(
            f"Local model contains layers not present in server weights: {extra_in_local}"
        )

    # strict=True ensures shape mismatches raise RuntimeError
    model.load_state_dict(weights_dict, strict=True)
    logger.info("Global weights loaded into local model successfully.")


# ---------------------------------------------------------------------------
# 2. Local Training Loop
# ---------------------------------------------------------------------------

def train_local_model(
    model: nn.Module,
    dataloader: DataLoader,
    epochs: int = 1,
    lr: float = float(os.getenv("CLIENT_LR", "0.01")),
    device: str = "cpu",
    optimizer_type: str = os.getenv("CLIENT_OPTIMIZER", "sgd"),
) -> Dict[str, torch.Tensor]:
    """
    Runs a real local training loop on the client's private data.

    Uses:
      - CrossEntropyLoss (standard for classification).
      - Adam or SGD depending on CLIENT_OPTIMIZER env var.
      - Genuine forward pass, backward pass, and optimizer step each batch.

    Args:
        model:          The local model (already loaded with global weights).
        dataloader:     DataLoader over the client's local dataset.
        epochs:         Number of local epochs to run.
        lr:             Learning rate (from env CLIENT_LR, default 0.01).
        device:         Torch device string, e.g. "cpu" or "cuda".
        optimizer_type: "sgd" or "adam" (from env CLIENT_OPTIMIZER).

    Returns:
        Updated state_dict (Dict[str, Tensor]) on CPU after training.

    Raises:
        ValueError: If dataloader is empty.
    """
    if len(dataloader.dataset) == 0:
        raise ValueError("Local dataloader is empty — cannot train.")

    model = model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()

    if optimizer_type.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        # Default to SGD with momentum
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    total_batches_processed = 0

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        correct    = 0
        total      = 0

        for batch_idx, (data, targets) in enumerate(dataloader):
            data    = data.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # Real forward pass
            outputs = model(data)

            # Real loss computation
            loss = criterion(outputs, targets)

            # Real backward pass
            loss.backward()

            # Real parameter update
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            correct      += (predicted == targets).sum().item()
            total        += targets.size(0)
            total_batches_processed += 1

        avg_loss = epoch_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0.0

        logger.info(
            f"[Local Training] Epoch {epoch}/{epochs} — "
            f"Loss: {avg_loss:.4f} | Accuracy: {accuracy * 100:.2f}%"
        )

    logger.info(
        f"[Local Training] Complete. {epochs} epoch(s), "
        f"{total_batches_processed} batches processed."
    )

    # Return updated weights isolated on CPU
    return get_updated_weights(model)


# ---------------------------------------------------------------------------
# 3. Weight Extraction
# ---------------------------------------------------------------------------

def get_updated_weights(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Returns a CPU copy of the model's current state dict.

    Detaches all tensors from the computational graph to ensure safe
    serialization and transmission.

    Args:
        model: A trained (or partially trained) PyTorch model.

    Returns:
        A deep-copied state dict with all tensors on CPU, detached.
    """
    return {
        key: tensor.cpu().detach().clone()
        for key, tensor in model.state_dict().items()
    }
