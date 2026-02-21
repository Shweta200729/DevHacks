import os
import torch
from typing import Tuple, Dict

def flattened_l2_norm(weights_dict: Dict[str, torch.Tensor]) -> float:
    """
    Computes the total L2 norm of a dictionary of model weights.
    All tensors are flattened and summed before taking the square root.
    """
    squared_sum = 0.0
    with torch.no_grad():
        for tensor in weights_dict.values():
            squared_sum += torch.sum(tensor ** 2).item()
    return squared_sum ** 0.5

def detect_update(
    update_weights: Dict[str, torch.Tensor],
    global_weights: Dict[str, torch.Tensor]
) -> Tuple[str, str, float, float]:
    """
    Detects if a client's weight update is potentially malicious by checking its L2 norm
    and its L2 distance from the global model using mathematically valid environments limits.

    Returns:
        A tuple (status, reason, calculated_norm, calculated_distance), 
        where status is either "ACCEPT" or "REJECT".
    """
    # Environment boundaries setup
    norm_threshold = float(os.getenv("DETECTION_NORM_THRESHOLD", "50.0"))
    distance_threshold = float(os.getenv("DETECTION_DISTANCE_THRESHOLD", "25.0"))

    # 1. Compute total L2 norm of update weights.
    update_norm = flattened_l2_norm(update_weights)
    
    # 2. Compute L2 Euclidean distance between update_weights and global_weights.
    distance_squared_sum = 0.0
    with torch.no_grad():
        for key in update_weights.keys():
            if key in global_weights:
                diff = update_weights[key].cpu() - global_weights[key].cpu()
                distance_squared_sum += torch.sum(diff ** 2).item()
            else:
                # Handle key mismatch gracefully by adding the update weight's norm squared
                distance_squared_sum += torch.sum(update_weights[key].cpu() ** 2).item()
            
    distance = distance_squared_sum ** 0.5

    # 3. Decision bounds checking
    if update_norm > norm_threshold:
        return ("REJECT", f"Update norm too large ({update_norm:.2f} > {norm_threshold})", update_norm, distance)
    
    if distance > distance_threshold:
        return ("REJECT", f"Update too far from global model ({distance:.2f} > {distance_threshold})", update_norm, distance)
        
    return ("ACCEPT", "Update valid", update_norm, distance)
