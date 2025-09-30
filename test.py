import os
import torch
from torch.utils.data import DataLoader
from datasets.saai_crowd_dataset import SAAICrowdDataset, saai_crowd_collate
from models.crowd_counter import SAAICrowdCounter
from utils.metrics import calculate_mae, calculate_rmse
from config import get_config

def test_saai_model():
    """Test function for SAAI crowd counter"""
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test dataset
    test_dataset = SAAICrowdDataset(
        root_path=os.path.join(config.data_path, 'test'),
        crop_size=config.crop_size,
        downsample_ratio=config.downsample_ratio,
        method='test',
        enable_domain_labels=False,
        enable_density_maps=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=saai_crowd_collate,
        num_workers=4
    )
    
    # Load model
    model = SAAICrowdCounter(
        backbone_name=config.backbone,
        pretrained=False,
        feature_dim=config.feature_dim
    ).to(device)
    
    checkpoint = torch.load(os.path.join(config.save_dir, 'best_saai_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    total_mae = 0
    total_rmse = 0
    results = []
    
    print("Starting testing...")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            rgb, thermal, target, gt_count, name = batch_data[:5]
            
            rgb, thermal = rgb.to(device), thermal.to(device)
            
            pred_density, _, _ = model(rgb, thermal)
            pred_count = pred_density.sum().item()
            
            mae = abs(pred_count - gt_count)
            rmse = (pred_count - gt_count) ** 2
            
            total_mae += mae
            total_rmse += rmse
            
            results.append({
                'name': name[0],
                'gt_count': gt_count,
                'pred_count': pred_count,
                'mae': mae
            })
            
            if batch_idx % 100 == 0:
                print(f'Processed {batch_idx}/{len(test_loader)} images')
    
    final_mae = total_mae / len(test_loader)
    final_rmse = (total_rmse / len(test_loader)) ** 0.5
    
    print(f'Test Results:')
    print(f'MAE: {final_mae:.2f}')
    print(f'RMSE: {final_rmse:.2f}')
    
    return results

if __name__ == '__main__':
    test_saai_model()
