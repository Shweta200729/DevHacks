import torch
import torch.nn as nn
from typing import Dict, Any

def evaluate_model(model: nn.Module, dataloader: Any, device: str = "cpu") -> Dict[str, float]:
    """
    Evaluates a PyTorch model on a given dataloader.
    
    Args:
        model: The PyTorch model to evaluate.
        dataloader: A PyTorch DataLoader containing the validation dataset.
        device: The device to run evaluation on, e.g., "cpu" or "cuda".
        
    Returns:
        A dictionary containing the "loss" and "accuracy".
    """
    # Set model to evaluation mode
    model.eval()
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # Disable gradients for validation to save memory and computations
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute cross entropy loss
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            
            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)
            
    average_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    
    return {
        "loss": average_loss,
        "accuracy": accuracy
    }
