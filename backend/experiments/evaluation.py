import torch
import torch.nn as nn
from typing import Dict, Any

def evaluate_model(model: nn.Module, dataloader: Any, device: str = "cpu") -> Dict[str, float]:
    """
    Evaluates a PyTorch model on a given dataloader.
    Returns real cross-entropy loss and accuracy metrics.
    """
    model.eval()
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)
            
    average_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    
    return {
        "loss": average_loss,
        "accuracy": accuracy
    }
