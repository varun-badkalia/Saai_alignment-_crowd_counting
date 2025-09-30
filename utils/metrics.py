import torch
import numpy as np

def calculate_mae(pred_density, gt_count):
    """Calculate Mean Absolute Error"""
    pred_count = pred_density.sum(dim=(2, 3)).squeeze()
    if isinstance(gt_count, list):
        gt_count = torch.tensor(gt_count, dtype=torch.float32).to(pred_density.device)
    return torch.mean(torch.abs(pred_count - gt_count)).item()

def calculate_mse(pred_density, gt_count):
    """Calculate Mean Squared Error"""
    pred_count = pred_density.sum(dim=(2, 3)).squeeze()
    if isinstance(gt_count, list):
        gt_count = torch.tensor(gt_count, dtype=torch.float32).to(pred_density.device)
    return torch.mean((pred_count - gt_count) ** 2).item()

def calculate_rmse(pred_density, gt_count):
    """Calculate Root Mean Squared Error"""
    return np.sqrt(calculate_mse(pred_density, gt_count))
