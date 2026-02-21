import os
import json
import torch
from typing import Dict, Any

class StorageManager:
    def __init__(self, base_dir: str = "checkpoints"):
        self.base_dir = base_dir
        self.models_dir = os.path.join(self.base_dir, "models")
        self.metrics_dir = os.path.join(self.base_dir, "metrics")
        self._create_directories()

    def _create_directories(self):
        """Creates necessary directories for storing checkpoints and metrics."""
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

    def save_model(self, model_state_dict: Dict[str, torch.Tensor], round_num: int):
        """
        Saves the PyTorch model state dictionary to a .pt file.
        
        Args:
            model_state_dict: Dictionary of PyTorch tensors representing the model weights.
            round_num: The current federated learning round number.
        """
        file_path = os.path.join(self.models_dir, f"global_model_round_{round_num}.pt")
        torch.save(model_state_dict, file_path)
        print(f"[*] Model checkpoint saved to {file_path}")

    def save_metrics(self, metrics: Dict[str, Any], round_num: int):
        """
        Saves round metrics to a JSON file.
        
        Args:
            metrics: Dictionary containing metrics like loss, accuracy, accepted_clients, etc.
            round_num: The current federated learning round number.
        """
        file_path = os.path.join(self.metrics_dir, f"metrics_round_{round_num}.json")
        with open(file_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"[*] Metrics saved to {file_path}")
