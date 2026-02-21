"""
clients/csv_trainer.py

Real CSV dataset training for the FL pipeline.
Parses an uploaded CSV file and trains the CNN on its actual pixel values.

Expected CSV format (MNIST-style):
  label, px0, px1, ..., px783
Each row = one 28x28 grayscale image flattened to 784 pixels (0-255 range).

This module is called by the server's background task after a CSV upload.
Keeps ML logic completely separate from DB/HTTP logic.
"""

import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSV Dataset
# ---------------------------------------------------------------------------

class MNISTCSVDataset(Dataset):
    """
    Parses a CSV file in MNIST format:
      - Column 0 : label (integer 0-9)
      - Columns 1-784 : pixel values (0-255), will be normalised to [0,1]

    Falls back gracefully if the CSV doesn't match this layout.
    """

    def __init__(self, csv_path: str, max_rows: int = 20_000):
        super().__init__()
        self.samples: list[tuple[torch.Tensor, int]] = []
        self._parse(csv_path, max_rows)

    def _parse(self, path: str, max_rows: int):
        import csv

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)          # skip header row if present

            skipped = 0
            for i, row in enumerate(reader):
                if i >= max_rows:
                    break
                try:
                    values = [float(v) for v in row]
                    if len(values) < 2:
                        skipped += 1
                        continue

                    label  = int(values[0]) % 10          # keep in 0-9
                    pixels = torch.tensor(
                        values[1:785], dtype=torch.float32
                    )

                    # Pad or truncate to exactly 784 values
                    if pixels.numel() < 784:
                        pixels = torch.nn.functional.pad(pixels, (0, 784 - pixels.numel()))
                    else:
                        pixels = pixels[:784]

                    # Normalise from [0,255] → [0,1], then standardise
                    pixels = pixels / 255.0
                    pixels = (pixels - 0.1307) / 0.3081

                    # Reshape to (1, 28, 28) for the CNN
                    img = pixels.view(1, 28, 28)
                    self.samples.append((img, label))

                except (ValueError, IndexError):
                    skipped += 1

        logger.info(
            f"MNISTCSVDataset: loaded {len(self.samples)} samples "
            f"(skipped {skipped} malformed rows) from {os.path.basename(path)}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Training entry-point called by server background task
# ---------------------------------------------------------------------------

def train_on_csv(
    client_id: str,
    csv_path: str,
    global_weights: dict,
    epochs: int = 3,
    batch_size: int = 64,
    lr: float = 0.01,
    device: str = "cpu",
) -> Optional[str]:
    """
    Loads a CSV dataset, initialises the CNN with current global weights,
    runs a real local training loop, and saves the result to disk.

    Args:
        client_id:      String identifier for this client.
        csv_path:       Absolute path to the uploaded CSV file.
        global_weights: Current global model state_dict (CPU tensors).
        epochs:         Number of local epochs.
        batch_size:     Mini-batch size.
        lr:             SGD learning rate.
        device:         "cpu" or "cuda".

    Returns:
        Absolute path to the saved weights file, or None on failure.
    """
    # Import CNNModel inline to avoid circular imports when called from server
    server_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "server"))
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)
    from model import CNNModel

    from clients.local_training import load_global_weights, train_local_model

    client_dir   = os.path.join(os.path.dirname(csv_path), )  # same dir as CSV
    weights_path = os.path.join(client_dir, f"weights_{client_id}.pt")

    logger.info(f"[CSV Train] Starting for client {client_id}, file: {csv_path}")

    try:
        dataset = MNISTCSVDataset(csv_path)

        if len(dataset) == 0:
            logger.warning(
                f"[CSV Train] No valid rows found in CSV for client {client_id}. "
                "Make sure the CSV has columns: label, px0, px1, ..., px783"
            )
            return None

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        local_model = CNNModel()
        load_global_weights(local_model, global_weights)

        updated_weights = train_local_model(
            local_model, loader, epochs=epochs, lr=lr, device=device
        )

        torch.save(updated_weights, weights_path)
        logger.info(f"[CSV Train] Complete for client {client_id}. Weights → {weights_path}")
        return weights_path

    except Exception as exc:
        logger.error(f"[CSV Train] Failed for client {client_id}: {exc}", exc_info=True)
        return None
