import torch
from typing import List, Dict

def fedavg(updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Performs Federated Averaging (FedAvg) aggregation over a list of client updates.
    
    Args:
        updates: A list containing weight dictionaries from all clients.
        
    Returns:
        A dictionary of the aggregated weights.
        
    Raises:
        ValueError: If the updates list is empty.
    """
    if not updates:
        raise ValueError("Cannot aggregate an empty list of updates.")
        
    aggregated_weights = {}
    
    # Get all keys from the first update
    keys = updates[0].keys()
    
    for key in keys:
        tensors = []
        for update in updates:
            if key in update:
                tensors.append(update[key])
                
        if tensors:
            # Stack along a new client dimension (dim=0)
            stacked_tensors = torch.stack(tensors)
            # Compute element-wise mean across all clients
            aggregated_weights[key] = torch.mean(stacked_tensors, dim=0)
            
    return aggregated_weights

def trimmed_mean(
    updates: List[Dict[str, torch.Tensor]],
    trim_ratio: float = 0.2
) -> Dict[str, torch.Tensor]:
    """
    Performs Trimmed Mean aggregation over a list of client updates.
    At each position in the tensor, it sorts the values from all clients,
    removes the highest and lowest `trim_ratio` fraction of values, and
    takes the mean of the remaining values.
    
    Args:
        updates: A list containing weight dictionaries from all clients.
        trim_ratio: The fraction of extreme values to remove from each end (0.0 to <0.5).
        
    Returns:
        A dictionary of the aggregated weights.
        
    Raises:
        ValueError: If the updates list is empty or invalid trim_ratio.
    """
    if not updates:
        raise ValueError("Cannot aggregate an empty list of updates.")
    if not (0.0 <= trim_ratio < 0.5):
        raise ValueError("trim_ratio must be between 0.0 and <0.5")
        
    num_clients = len(updates)
    trim_count = int(num_clients * trim_ratio)
    
    # Check if trimming removes all clients (or too many, leaving 0)
    if 2 * trim_count >= num_clients:
        trim_count = 0  # Fallback to simple average if too few clients
        
    aggregated_weights = {}
    keys = updates[0].keys()
    
    for key in keys:
        tensors = []
        for update in updates:
            if key in update:
                tensors.append(update[key])
                
        if tensors:
            stacked_tensors = torch.stack(tensors) # Shape: (num_clients, ...)
            
            if trim_count == 0:
                # If nothing to trim (e.g., small num_clients), just do simple mean
                aggregated_weights[key] = torch.mean(stacked_tensors, dim=0)
                continue
                
            # Sort along the client dimension (dim=0)
            sorted_tensors, _ = torch.sort(stacked_tensors, dim=0)
            
            # Remove top and bottom trim_count slices
            trimmed_tensors = sorted_tensors[trim_count:-trim_count]
            
            # Compute mean of the remaining
            aggregated_weights[key] = torch.mean(trimmed_tensors, dim=0)
            
    return aggregated_weights
