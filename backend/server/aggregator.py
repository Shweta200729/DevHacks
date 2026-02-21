import os
import math
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
    updates: List[Dict[str, torch.Tensor]], trim_ratio: float = 0.2
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
            stacked_tensors = torch.stack(tensors)  # Shape: (num_clients, ...)

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


# ===========================================================
# Differential Privacy Utilities
# ===========================================================


def clip_weights(
    weights: Dict[str, torch.Tensor], clip_norm: float
) -> Dict[str, torch.Tensor]:
    """
    Clips a client's weight dictionary so that its total L2 norm
    does not exceed `clip_norm`. This is the standard per-sample
    gradient clipping step used in DP-SGD.
    """
    # Compute total L2 norm across all tensors
    total_norm_sq = 0.0
    with torch.no_grad():
        for v in weights.values():
            total_norm_sq += torch.sum(v.cpu() ** 2).item()
    total_norm = math.sqrt(total_norm_sq)

    # Scale factor (no-op if norm is already within bound)
    scale = min(1.0, clip_norm / (total_norm + 1e-9))

    return {k: v.cpu() * scale for k, v in weights.items()}


def add_gaussian_dp_noise(
    aggregated: Dict[str, torch.Tensor],
    clip_norm: float,
    noise_multiplier: float,
    num_clients: int,
) -> Dict[str, torch.Tensor]:
    """
    Adds calibrated Gaussian noise to an already-aggregated weight dict.

    The noise standard deviation per coordinate is:
        sigma = noise_multiplier * clip_norm / num_clients

    This is the standard DP-FL formulation (Geyer et al., 2017).
    A higher noise_multiplier = stronger privacy guarantee (larger epsilon cost).

    Args:
        aggregated:       Aggregated weight dict (output of trimmed_mean / fedavg).
        clip_norm:        The clipping threshold used on individual updates.
        noise_multiplier: Noise scale relative to clip_norm (sigma = nm * C / n).
        num_clients:      Number of clients that contributed to this round.

    Returns:
        Noisy aggregated weight dict.
    """
    sigma = noise_multiplier * clip_norm / max(num_clients, 1)
    noisy_weights = {}
    with torch.no_grad():
        for key, tensor in aggregated.items():
            noise = torch.randn_like(tensor) * sigma
            noisy_weights[key] = tensor + noise
    return noisy_weights


def dp_trimmed_mean(
    updates: List[Dict[str, torch.Tensor]],
    trim_ratio: float = 0.2,
    clip_norm: float = 10.0,
    noise_multiplier: float = 1.0,
    enabled: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Full DP-aware aggregation pipeline:
    1. Clip each client's update to `clip_norm`.
    2. Run Trimmed Mean aggregation.
    3. Add calibrated Gaussian noise to the result.

    If `enabled=False`, falls back to plain trimmed_mean (for ablations).
    """
    if not enabled:
        return trimmed_mean(updates, trim_ratio=trim_ratio)

    # Step 1 — per-client clipping
    clipped = [clip_weights(u, clip_norm) for u in updates]

    # Step 2 — robust aggregation
    aggregated = trimmed_mean(clipped, trim_ratio=trim_ratio)

    # Step 3 — Gaussian noise injection
    noisy = add_gaussian_dp_noise(
        aggregated,
        clip_norm=clip_norm,
        noise_multiplier=noise_multiplier,
        num_clients=len(updates),
    )
    return noisy
