import torch
from torch.utils.data import TensorDataset, DataLoader
import copy

from aggregator import trimmed_mean
from detection import detect_update
from storage import StorageManager
from model import SimpleModel
import sys
import os

# Ensure experiments module can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.evaluation import evaluate_model

def simulate_fl_round(round_num: int, global_model: SimpleModel, storage: StorageManager, dataloader: DataLoader):
    print(f"\n{'='*40}")
    print(f"Starting Federated Learning Round: {round_num}")
    print(f"{'='*40}")

    global_weights = global_model.state_dict()
    client_updates = []
    
    # 1. Simulate 5 normal client updates
    for i in range(1, 6):
        # We simulate a client training by adding small random noise to the global model
        simulated_update = {}
        for key, tensor in global_weights.items():
            noise = torch.randn_like(tensor) * 0.05  # Small noise representing normal learning
            simulated_update[key] = tensor + noise
        client_updates.append((f"Client_{i}", simulated_update))

    # 2. Inject 1 malicious update (huge norm)
    malicious_update = {}
    for key, tensor in global_weights.items():
        malicious_noise = torch.randn_like(tensor) * 50.0  # Very large noise
        malicious_update[key] = tensor + malicious_noise
    client_updates.append(("Malicious_Client", malicious_update))

    print(f"\n[Validation Phase]")
    valid_updates = []
    accepted_count = 0
    rejected_count = 0

    # 3. Use validate_update (detect_update)
    for client_id, update in client_updates:
        status, reason = detect_update(update, global_weights, norm_threshold=40.0, distance_threshold=20.0)
        
        if status == "ACCEPT":
            print(f"  [+] {client_id}: ACCEPTED")
            valid_updates.append(update)
            accepted_count += 1
        else:
            print(f"  [-] {client_id}: REJECTED (Reason: {reason})")
            rejected_count += 1

    # 4. Aggregate valid updates
    print(f"\n[Aggregation Phase]")
    if valid_updates:
        print(f"  Aggregating {len(valid_updates)} valid updates using Trimmed Mean...")
        aggregated_weights = trimmed_mean(valid_updates, trim_ratio=0.2)
        
        # 5. Update global model
        global_model.load_state_dict(aggregated_weights)
        print("  Global model updated successfully.")
    else:
        print("  No valid updates to aggregate. Skipping model update.")

    # 6. Evaluate model
    print(f"\n[Evaluation Phase]")
    eval_metrics = evaluate_model(global_model, dataloader)
    print(f"  Loss: {eval_metrics['loss']:.4f}")
    print(f"  Accuracy: {eval_metrics['accuracy']:.4f}")

    # 7. Use storage.py to save model + metrics
    print(f"\n[Storage Phase]")
    storage.save_model(global_model.state_dict(), round_num)
    
    round_metrics = {
        "round": round_num,
        "accepted_clients": accepted_count,
        "rejected_clients": rejected_count,
        "loss": eval_metrics['loss'],
        "accuracy": eval_metrics['accuracy']
    }
    storage.save_metrics(round_metrics, round_num)


def main():
    # Setup
    global_model = SimpleModel()
    storage = StorageManager(base_dir="fl_simulation_data")
    
    # Create a dummy validation dataset
    X_val = torch.randn(100, 10)
    y_val = torch.randint(0, 2, (100,))
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Run Simulation
    simulate_fl_round(round_num=1, global_model=global_model, storage=storage, dataloader=val_loader)

if __name__ == "__main__":
    main()
