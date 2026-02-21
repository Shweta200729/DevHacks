import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import sys
import os

# Add parent dir to path so we can import server modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from server.detection import detect_update, cosine_similarity_check
from server.aggregator import fedavg, trimmed_mean
from experiments.evaluation import evaluate_model

def run_checks():
    print("--- Running detection.py sanity checks ---")
    global_weights = {
        "layer1": torch.tensor([1.0, 2.0]),
        "layer2": torch.tensor([[0.5, 0.5], [1.0, 1.0]])
    }
    
    # 1. Normal update
    normal_update = {
        "layer1": torch.tensor([1.1, 1.9]),
        "layer2": torch.tensor([[0.6, 0.4], [1.1, 0.9]])
    }
    status, reason = detect_update(normal_update, global_weights)
    print(f"Normal update: {status} ({reason})")
    assert status == "ACCEPT"
    
    print(f"Cosine similarity: {cosine_similarity_check(normal_update, global_weights):.4f}")
    
    # 2. Huge norm
    huge_update = {
        "layer1": torch.tensor([100.0, 200.0]),
        "layer2": torch.tensor([[0.5, 0.5], [1.0, 1.0]])
    }
    status, reason = detect_update(huge_update, global_weights)
    print(f"Huge update: {status} ({reason})")
    assert status == "REJECT"
    
    # 3. Far distance
    far_update = {
        "layer1": torch.tensor([-1.0, -2.0]),
        "layer2": torch.tensor([[-0.5, -0.5], [-1.0, -1.0]])
    }
    status, reason = detect_update(far_update, global_weights)
    print(f"Far update: {status} ({reason})")
    assert status == "REJECT"

    print("\n--- Running aggregator.py sanity checks ---")
    update1 = {"layer1": torch.tensor([1.0, 2.0])}
    update2 = {"layer1": torch.tensor([3.0, 4.0])}
    update3 = {"layer1": torch.tensor([5.0, 6.0])}
    updates = [update1, update2, update3]
    
    avg_weights = fedavg(updates)
    print(f"FedAvg: {avg_weights['layer1']}")
    assert torch.allclose(avg_weights['layer1'], torch.tensor([3.0, 4.0]))
    
    trim_weights = trimmed_mean(updates, trim_ratio=0.3)
    # 3 clients, 0.3 * 3 = 0.9 = 0 trimmed from each side. Should just be mean.
    print(f"Trimmed Mean (ratio 0.3 on 3 clients): {trim_weights['layer1']}")
    
    # 5 clients
    updates5 = [
        {"layer1": torch.tensor([1.0])},
        {"layer1": torch.tensor([100.0])}, # To be trimmed
        {"layer1": torch.tensor([2.0])},
        {"layer1": torch.tensor([3.0])},
        {"layer1": torch.tensor([-100.0])}, # To be trimmed
    ]
    trim_weights5 = trimmed_mean(updates5, trim_ratio=0.2) # 5 * 0.2 = 1. Removes top 1 and bottom 1.
    print(f"Trimmed Mean (ratio 0.2 on 5 clients): {trim_weights5['layer1']}")
    assert torch.allclose(trim_weights5['layer1'], torch.tensor([2.0]))

    print("\n--- Running evaluation.py sanity checks ---")
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2)
            
        def forward(self, x):
            return self.linear(x)
            
    model = SimpleModel()
    # Dummy data
    X = torch.randn(10, 2)
    y = torch.randint(0, 2, (10,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=2)
    
    metrics = evaluate_model(model, dataloader)
    print(f"Metrics: {metrics}")
    assert "loss" in metrics and "accuracy" in metrics
    
    print("\nAll sanity checks passed successfully!")

if __name__ == "__main__":
    run_checks()
