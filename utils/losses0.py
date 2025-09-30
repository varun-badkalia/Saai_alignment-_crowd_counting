import torch
import torch.nn as nn
import torch.nn.functional as F

class SAAIBrokerLoss(nn.Module):
    """SAAI loss for Broker-Modality style training"""
    
    def __init__(self, alpha=0.1, beta=0.05):
        super(SAAIBrokerLoss, self).__init__()
        self.alpha = alpha  # Domain adversarial weight
        self.beta = beta    # Semantic alignment weight
        
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, density_map, keypoints_list, targets_list, 
                domain_pred_rgb, domain_pred_thermal):
        """
        Compute loss using Broker-Modality Bayesian Loss style
        """
        device = density_map.device
        batch_size = density_map.size(0)
        
        # Bayesian Loss calculation (similar to original Broker-Modality)
        density_loss = torch.tensor(0.0).to(device)
        
        for i in range(batch_size):
            if len(keypoints_list[i]) > 0:
                keypoints = keypoints_list[i].to(device)
                targets = targets_list[i].to(device)
                
                # Scale keypoints to density map size
                scale_h = density_map.size(2) / keypoints[:, 1].max().item() if keypoints[:, 1].max().item() > 0 else 1.0
                scale_w = density_map.size(3) / keypoints[:, 0].max().item() if keypoints[:, 0].max().item() > 0 else 1.0
                
                # Extract density values at keypoint locations
                x_coords = torch.clamp((keypoints[:, 0] * scale_w).long(), 0, density_map.size(3) - 1)
                y_coords = torch.clamp((keypoints[:, 1] * scale_h).long(), 0, density_map.size(2) - 1)
                
                predicted_densities = density_map[i, 0, y_coords, x_coords]
                density_loss += self.mse_loss(predicted_densities, targets)
        
        density_loss = density_loss / batch_size
        
        # Domain adversarial loss
        rgb_labels = torch.zeros(batch_size, dtype=torch.long).to(device)
        thermal_labels = torch.ones(batch_size, dtype=torch.long).to(device)
        
        domain_loss = (self.ce_loss(domain_pred_rgb, rgb_labels) + 
                      self.ce_loss(domain_pred_thermal, thermal_labels)) / 2
        
        # Total loss
        total_loss = density_loss + self.alpha * domain_loss
        
        return total_loss, {
            'density_loss': density_loss.item(),
            'domain_loss': domain_loss.item(),
            'total_loss': total_loss.item()
        }
