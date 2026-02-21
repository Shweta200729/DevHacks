import torch
from typing import List, Dict

def fedavg(updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Performs Federated Averaging (FedAvg) aggregation over a list of client updates.
    """
    if not updates:
        raise ValueError("Cannot aggregate an empty list of updates.")
        
    aggregated_weights = {}
    keys = updates[0].keys()
    
    for key in keys:
        tensors = []
        for update in updates:
            if key in update:
                tensors.append(update[key].cpu())
                
        if tensors:
            stacked_tensors = torch.stack(tensors)
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
    """
    if not updates:
        raise ValueError("Cannot aggregate an empty list of updates.")
    if not (0.0 <= trim_ratio < 0.5):
        raise ValueError("trim_ratio must be between 0.0 and <0.5")
        
    num_clients = len(updates)
    trim_count = int(num_clients * trim_ratio)
    
    if 2 * trim_count >= num_clients:
        trim_count = 0  
        
    aggregated_weights = {}
    keys = updates[0].keys()
    
    for key in keys:
        tensors = []
        for update in updates:
            if key in update:
                tensors.append(update[key].cpu())
                
        if tensors:
            stacked_tensors = torch.stack(tensors) # Shape: (num_clients, ...)
            
            if trim_count == 0:
                aggregated_weights[key] = torch.mean(stacked_tensors, dim=0)
                continue
                
            # Real tensor sort over client dimension 0
            sorted_tensors, _ = torch.sort(stacked_tensors, dim=0)
            
            # Trim extreme values
            trimmed_tensors = sorted_tensors[trim_count:-trim_count]
            
            # Mean the valid bounded values
            aggregated_weights[key] = torch.mean(trimmed_tensors, dim=0)
            
    return aggregated_weights
