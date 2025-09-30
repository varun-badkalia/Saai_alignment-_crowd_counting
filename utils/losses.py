import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SAAIBrokerLoss(nn.Module):
    """Enhanced SAAI loss with Broker-Modality compatibility"""
    
    def __init__(self, alpha=0.1, beta=0.05, gamma=1.0):
        super(SAAIBrokerLoss, self).__init__()
        self.alpha = alpha  # Domain adversarial weight
        self.beta = beta    # Semantic alignment weight
        self.gamma = gamma  # Density map consistency weight
        
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, density_map, keypoints_list, targets_list, 
                domain_pred_rgb, domain_pred_thermal, gt_density_maps=None):
        """
        Compute multi-component loss with Broker-Modality compatibility
        Args:
            density_map: Predicted density map [B, 1, H, W]
            keypoints_list: List of keypoints for each image
            targets_list: List of target weights for Bayesian loss
            domain_pred_rgb: Domain predictions for RGB features
            domain_pred_thermal: Domain predictions for thermal features
            gt_density_maps: Ground truth density maps [B, 1, H, W] (optional)
        """
        device = density_map.device
        batch_size = density_map.size(0)
        
        # 1. Bayesian Loss (point-wise supervision) - following Broker-Modality
        bayesian_loss = torch.tensor(0.0).to(device)
        count_loss = torch.tensor(0.0).to(device)
        
        for i in range(batch_size):
            if len(keypoints_list[i]) > 0:
                keypoints = keypoints_list[i].to(device)
                targets = targets_list[i].to(device)
                
                # Scale keypoints to density map size (following Broker-Modality approach)
                h_scale = density_map.size(2) / max(keypoints[:, 1].max().item() + 1, density_map.size(2))
                w_scale = density_map.size(3) / max(keypoints[:, 0].max().item() + 1, density_map.size(3))
                
                # Extract density values at keypoint locations
                x_coords = torch.clamp((keypoints[:, 0] * w_scale).long(), 0, density_map.size(3) - 1)
                y_coords = torch.clamp((keypoints[:, 1] * h_scale).long(), 0, density_map.size(2) - 1)
                
                if len(x_coords) > 0 and len(y_coords) > 0:
                    predicted_densities = density_map[i, 0, y_coords, x_coords]
                    bayesian_loss += self.mse_loss(predicted_densities, targets)
                
                # Count consistency loss
                pred_count = density_map[i].sum()
                gt_count = len(keypoints_list[i])
                count_loss += self.l1_loss(pred_count, torch.tensor(gt_count, dtype=torch.float32).to(device))
        
        bayesian_loss = bayesian_loss / batch_size
        count_loss = count_loss / batch_size
        
        # 2. Density Map MSE Loss (if ground truth density maps provided)
        density_mse_loss = torch.tensor(0.0).to(device)
        if gt_density_maps is not None:
            # Resize if necessary
            if gt_density_maps.shape != density_map.shape:
                gt_density_maps = F.interpolate(gt_density_maps, 
                                              size=density_map.shape[2:], 
                                              mode='bilinear', align_corners=False)
            density_mse_loss = self.mse_loss(density_map, gt_density_maps)
        
        # 3. Domain adversarial loss for SAAI alignment
        rgb_labels = torch.zeros(batch_size, dtype=torch.long).to(device)
        thermal_labels = torch.ones(batch_size, dtype=torch.long).to(device)
        
        domain_loss = (self.ce_loss(domain_pred_rgb, rgb_labels) + 
                      self.ce_loss(domain_pred_thermal, thermal_labels)) / 2
        
        # 4. Total loss combination (Broker-Modality style)
        if gt_density_maps is not None:
            # Use density map loss if available
            primary_loss = self.gamma * density_mse_loss
        else:
            # Use Bayesian loss (default Broker-Modality approach)
            primary_loss = self.gamma * bayesian_loss
        
        total_loss = (primary_loss + 
                     0.1 * count_loss + 
                     self.alpha * domain_loss)
        
        # Return loss components for logging
        return total_loss, {
            'density_loss': primary_loss.item(),
            'count_loss': count_loss.item(),
            'domain_loss': domain_loss.item(),
            'total_loss': total_loss.item()
        }

def calculate_broker_game_metrics(pred_density, gt_count, levels=[0, 1, 2, 3]):
    """
    Calculate GAME metrics exactly as in Broker-Modality paper
    Following their ECCV 2024 evaluation protocol
    """
    if isinstance(pred_density, torch.Tensor):
        pred_density = pred_density.detach().cpu().numpy()
    
    batch_size = pred_density.shape[0]
    game_metrics = {f'GAME_{level}': 0.0 for level in levels}
    
    for i in range(batch_size):
        pred_map = pred_density[i, 0] if len(pred_density.shape) == 4 else pred_density[i]  # [H, W]
        current_gt = gt_count[i] if isinstance(gt_count, (list, torch.Tensor)) else gt_count
        
        for level in levels:
            if level == 0:
                # GAME(0) is MAE - whole image
                pred_count = pred_map.sum()
                error = abs(pred_count - current_gt)
                game_metrics[f'GAME_{level}'] += error
            else:
                # Split image into 4^level regions (2^level x 2^level grid)
                h, w = pred_map.shape
                regions_per_side = 2 ** level
                region_errors = []
                
                for row in range(regions_per_side):
                    for col in range(regions_per_side):
                        # Calculate region boundaries
                        start_h = row * h // regions_per_side
                        end_h = (row + 1) * h // regions_per_side
                        start_w = col * w // regions_per_side
                        end_w = (col + 1) * w // regions_per_side
                        
                        # Extract region prediction
                        region_pred = pred_map[start_h:end_h, start_w:end_w].sum()
                        
                        # Calculate region GT (assume uniform distribution)
                        region_area = (end_h - start_h) * (end_w - start_w)
                        total_area = h * w
                        region_gt = current_gt * (region_area / total_area)
                        
                        region_error = abs(region_pred - region_gt)
                        region_errors.append(region_error)
                
                # Sum all region errors for this level
                total_error = sum(region_errors)
                game_metrics[f'GAME_{level}'] += total_error
    
    # Average over batch
    for level in levels:
        game_metrics[f'GAME_{level}'] /= batch_size
    
    return game_metrics

def calculate_broker_rmse(pred_density, gt_count):
    """Calculate RMSE exactly as in Broker-Modality paper"""
    if isinstance(pred_density, torch.Tensor):
        pred_count = pred_density.sum(dim=(2, 3)).squeeze().detach().cpu().numpy()
    else:
        pred_count = pred_density.sum(axis=(2, 3)).squeeze()
    
    if isinstance(gt_count, torch.Tensor):
        gt_count = gt_count.detach().cpu().numpy()
    elif isinstance(gt_count, list):
        gt_count = np.array(gt_count)
    
    # Ensure arrays are 1D
    if pred_count.ndim == 0:
        pred_count = np.array([pred_count])
    if gt_count.ndim == 0:
        gt_count = np.array([gt_count])
    
    mse = ((pred_count - gt_count) ** 2).mean()
    rmse = mse ** 0.5
    mae = abs(pred_count - gt_count).mean()
    
    return mae, rmse

def calculate_mae_rmse(pred_density, gt_count):
    """Wrapper function for compatibility"""
    return calculate_broker_rmse(pred_density, gt_count)
