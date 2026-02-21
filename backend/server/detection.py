import torch
from typing import Tuple, Dict

def flattened_l2_norm(weights_dict: Dict[str, torch.Tensor]) -> float:
    """
    Computes the total L2 norm of a dictionary of model weights.
    All tensors are flattened and summed before taking the square root.
    """
    squared_sum = 0.0
    for tensor in weights_dict.values():
        squared_sum += torch.sum(tensor ** 2).item()
    return squared_sum ** 0.5

def detect_update(
    update_weights: Dict[str, torch.Tensor],
    global_weights: Dict[str, torch.Tensor],
    norm_threshold: float = 10.0,
    distance_threshold: float = 5.0
) -> Tuple[str, str]:
    """
    Detects if a client's weight update is potentially malicious by checking its L2 norm
    and its L2 distance from the global model.

    Args:
        update_weights: Dictionary of weight tensors from the client.
        global_weights: Dictionary of current global model weight tensors.
        norm_threshold: Maximum allowable L2 norm of the update.
        distance_threshold: Maximum allowable L2 distance between update and global model.

    Returns:
        A tuple (status, reason), where status is either "ACCEPT" or "REJECT".
    """
    # 1. Compute total L2 norm of update weights.
    update_norm = flattened_l2_norm(update_weights)
    if update_norm > norm_threshold:
        return ("REJECT", "Update norm too large")
    
    # 2. Compute L2 distance between update_weights and global_weights.
    distance_squared_sum = 0.0
    for key in update_weights.keys():
        if key in global_weights:
            diff = update_weights[key] - global_weights[key]
            distance_squared_sum += torch.sum(diff ** 2).item()
        else:
            # Handle key mismatch gracefully by adding the update weight's norm squared
            distance_squared_sum += torch.sum(update_weights[key] ** 2).item()
            
    distance = distance_squared_sum ** 0.5
    
    if distance > distance_threshold:
        return ("REJECT", "Update too far from global model")
        
    # 3. Otherwise Accept
    return ("ACCEPT", "Update valid")

def cosine_similarity_check(
    update_weights: Dict[str, torch.Tensor],
    global_weights: Dict[str, torch.Tensor]
) -> float:
    """
    Computes the cosine similarity between the update weights and global weights.
    Returns a float between -1.0 and 1.0.
    """
    dot_product = 0.0
    update_norm_sq = 0.0
    global_norm_sq = 0.0
    
    for key in update_weights.keys():
        if key in global_weights:
            dot_product += torch.sum(update_weights[key] * global_weights[key]).item()
            update_norm_sq += torch.sum(update_weights[key] ** 2).item()
            global_norm_sq += torch.sum(global_weights[key] ** 2).item()
            
    return dot_product / ((update_norm_sq ** 0.5) * (global_norm_sq ** 0.5) + 1e-10)
