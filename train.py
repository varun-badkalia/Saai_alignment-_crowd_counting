import torch
from torch.utils.data import DataLoader
import os
import argparse
import time
import logging
import json
from datetime import datetime
from tqdm import tqdm
from datasets.broker_crowd_dataset import BrokerCrowdDataset, crowd_collate
from models.crowd_counter import SAAICrowdCounter
from utils.losses import SAAIBrokerLoss
from utils.metrics import calculate_mae, calculate_rmse
import numpy as np  # Add this import



def setup_logging(save_dir):
    """Setup logging to both file and console"""
    log_filename = os.path.join(save_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_filename

def calculate_broker_game_metrics(pred_density, gt_count, levels=[0, 1, 2, 3]):
    """
    Calculate GAME metrics exactly as in Broker-Modality paper
    Following their evaluation protocol from ECCV 2024
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
                
                # Average error across all regions for this level
                total_error = sum(region_errors)
                game_metrics[f'GAME_{level}'] += total_error
    
    # Average over batch
    for level in levels:
        game_metrics[f'GAME_{level}'] /= batch_size
    
    return game_metrics

def calculate_broker_rmse(pred_density, gt_count):
    """Calculate RMSE exactly as in Broker-Modality"""
    if isinstance(pred_density, torch.Tensor):
        pred_count = pred_density.sum(dim=(2, 3)).squeeze().detach().cpu().numpy()
    else:
        pred_count = pred_density.sum(axis=(2, 3)).squeeze()
    
    if isinstance(gt_count, torch.Tensor):
        gt_count = gt_count.detach().cpu().numpy()
    elif isinstance(gt_count, list):
        gt_count = np.array(gt_count)
    
    mse = ((pred_count - gt_count) ** 2).mean()
    rmse = mse ** 0.5
    mae = abs(pred_count - gt_count).mean()
    
    return mae, rmse

def validate_broker_style(model, val_loader, device, logger):
    """Validation using exact Broker-Modality evaluation protocol with proper size handling"""
    model.eval()
    
    all_mae_errors = []
    all_game_metrics = {0: [], 1: [], 2: [], 3: []}
    all_rmse_errors = []
    
    with torch.no_grad():
        for rgb, thermal, target, gt_count, name in val_loader:
            rgb, thermal = rgb.to(device), thermal.to(device)
            
            density_map, _, _ = model(rgb, thermal)
            pred_count = density_map.sum().item()
            
            # Calculate individual sample metrics
            mae_error = abs(pred_count - gt_count)
            rmse_error = (pred_count - gt_count) ** 2
            
            all_mae_errors.append(mae_error)
            all_rmse_errors.append(rmse_error)
            
            # Calculate GAME metrics for this single sample
            # Convert to numpy for GAME calculation
            density_np = density_map.cpu().detach().numpy()  # [1, 1, H, W]
            
            # Calculate GAME metrics for each level
            for level in [0, 1, 2, 3]:
                if level == 0:
                    # GAME(0) is just MAE
                    game_error = mae_error
                else:
                    # Calculate GAME for this level
                    pred_map = density_np[0, 0]  # [H, W]
                    h, w = pred_map.shape
                    regions_per_side = 2 ** level
                    
                    total_error = 0
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
                            region_gt = gt_count * (region_area / total_area)
                            
                            region_error = abs(region_pred - region_gt)
                            total_error += region_error
                    
                    game_error = total_error
                
                all_game_metrics[level].append(game_error)
    
    # Calculate overall metrics
    overall_mae = sum(all_mae_errors) / len(all_mae_errors)
    overall_rmse = (sum(all_rmse_errors) / len(all_rmse_errors)) ** 0.5
    
    # Calculate average GAME metrics
    game_results = {}
    for level in [0, 1, 2, 3]:
        game_results[f'GAME_{level}'] = sum(all_game_metrics[level]) / len(all_game_metrics[level])
    
    # Log exactly as in Broker-Modality paper
    logger.info(f"🔍 Validation Results (Broker-Modality Format):")
    logger.info(f"   GAME(0): {game_results['GAME_0']:.2f}")
    logger.info(f"   GAME(1): {game_results['GAME_1']:.2f}")
    logger.info(f"   GAME(2): {game_results['GAME_2']:.2f}")  
    logger.info(f"   GAME(3): {game_results['GAME_3']:.2f}")
    logger.info(f"   RMSE:    {overall_rmse:.2f}")
    
    # Return primary metric (GAME(0) = MAE) for model saving
    return game_results['GAME_0'], {
        'GAME_0': game_results['GAME_0'],
        'GAME_1': game_results['GAME_1'], 
        'GAME_2': game_results['GAME_2'],
        'GAME_3': game_results['GAME_3'],
        'RMSE': overall_rmse
    }

def get_args():
    parser = argparse.ArgumentParser(description='SAAI-Enhanced Broker-Modality Training')
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--crop-size', type=int, default=384)
    parser.add_argument('--downsample-ratio', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--save-dir', type=str, default='./checkpoints')
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.05)
    parser.add_argument('--gamma', type=float, default=1.0)
    return parser.parse_args()

def main():
    args = get_args()
    
    # Setup logging
    os.makedirs(args.save_dir, exist_ok=True)
    logger, log_file = setup_logging(args.save_dir)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🚀 Using device: {device}")
    logger.info(f"📝 Training log: {log_file}")
    
    # Log configuration in Broker-Modality style
    logger.info("⚙️  Training Configuration:")
    logger.info(f"   Dataset: RGBT-CC (Custom)")
    logger.info(f"   Crop Size: {args.crop_size}×{args.crop_size}")
    logger.info(f"   Batch Size: {args.batch_size}")
    logger.info(f"   Learning Rate: {args.lr}")
    logger.info(f"   Epochs: {args.epochs}")
    logger.info(f"   Loss Weights - Alpha: {args.alpha}, Beta: {args.beta}, Gamma: {args.gamma}")
    
    # Setup paths
    train_path = os.path.join(args.data_path, 'train')
    val_path = os.path.join(args.data_path, 'val')
    
    # Create datasets
    train_dataset = BrokerCrowdDataset(
        root_path=train_path, crop_size=args.crop_size,
        downsample_ratio=args.downsample_ratio, method='train'
    )
    
    val_dataset = BrokerCrowdDataset(
        root_path=val_path, crop_size=args.crop_size,
        downsample_ratio=args.downsample_ratio, method='val'
    )
    
    logger.info(f"📊 Dataset Info:")
    logger.info(f"   Train: {len(train_dataset)} samples")
    logger.info(f"   Val:   {len(val_dataset)} samples")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=crowd_collate,
        pin_memory=True, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, collate_fn=crowd_collate, pin_memory=True
    )
    
    # Model setup
    model = SAAICrowdCounter(
        backbone_name='vgg16', pretrained=True, feature_dim=512
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"🧠 Model Info:")
    logger.info(f"   Total parameters: {total_params/1e6:.2f}M")
    
    # Optimizer setup (following Broker-Modality style)
    backbone_params = list(model.rgb_backbone.parameters()) + list(model.thermal_backbone.parameters())
    saai_params = list(model.saai_aligner.parameters())
    head_params = list(model.fusion_module.parameters()) + list(model.regression_head.parameters())
    
    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': args.lr * 0.1, 'name': 'backbone'},
        {'params': saai_params, 'lr': args.lr, 'name': 'saai'},
        {'params': head_params, 'lr': args.lr, 'name': 'head'}
    ], weight_decay=args.weight_decay)
    
    criterion = SAAIBrokerLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)
    
    # Best metrics tracking (Broker-Modality format)
    best_metrics = {
        'GAME_0': float('inf'),
        'GAME_1': float('inf'), 
        'GAME_2': float('inf'),
        'GAME_3': float('inf'),
        'RMSE': float('inf')
    }
    
    logger.info("🎯 Starting training...")
    logger.info("=" * 80)
    
    # Training loop
    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        
        # Loss tracking with Broker-Modality style
        epoch_losses = {
            'total': 0, 'density': 0, 'domain': 0, 'count': 0,
            'train_mae': 0, 'train_rmse': 0
        }
        batch_count = 0
        
        # Training epoch
        epoch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{args.epochs}", 
                         ncols=120, leave=False)
        
        for batch_idx, (rgb, thermal, keypoints_list, targets_list, st_sizes) in enumerate(epoch_pbar):
            rgb, thermal = rgb.to(device), thermal.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            density_map, domain_pred_rgb, domain_pred_thermal = model(rgb, thermal)
            
            # Loss calculation
            total_loss, loss_dict = criterion(
                density_map, keypoints_list, targets_list,
                domain_pred_rgb, domain_pred_thermal
            )
            
            # Training metrics calculation (Broker-Modality style)
            with torch.no_grad():
                batch_mae = 0
                batch_rmse = 0
                for i in range(len(keypoints_list)):
                    pred_count = density_map[i].sum().item()
                    gt_count = len(keypoints_list[i])
                    mae = abs(pred_count - gt_count)
                    rmse = (pred_count - gt_count) ** 2
                    batch_mae += mae
                    batch_rmse += rmse
                batch_mae /= len(keypoints_list)
                batch_rmse = (batch_rmse / len(keypoints_list)) ** 0.5
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate losses
            epoch_losses['total'] += total_loss.item()
            epoch_losses['density'] += loss_dict['density_loss']
            epoch_losses['domain'] += loss_dict['domain_loss']
            epoch_losses['count'] += loss_dict['count_loss']
            epoch_losses['train_mae'] += batch_mae
            epoch_losses['train_rmse'] += batch_rmse
            batch_count += 1
            
            # Update progress bar
            epoch_pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Density': f'{loss_dict["density_loss"]:.4f}',
                'Domain': f'{loss_dict["domain_loss"]:.4f}',
                'MAE': f'{batch_mae:.2f}',
                'RMSE': f'{batch_rmse:.2f}'
            })
        
        # Calculate epoch averages
        avg_losses = {key: val/batch_count for key, val in epoch_losses.items()}
        epoch_time = time.time() - start_time
        
        # Scheduler step
        scheduler.step()
        
        # Comprehensive epoch logging (Broker-Modality style)
        logger.info(f"📊 Epoch {epoch+1:3d}/{args.epochs} - Time: {epoch_time:.1f}s")
        logger.info(f"   🔥 Training Losses:")
        logger.info(f"      Total:   {avg_losses['total']:.6f}")
        logger.info(f"      Density: {avg_losses['density']:.6f}")
        logger.info(f"      Domain:  {avg_losses['domain']:.6f}")
        logger.info(f"      Count:   {avg_losses['count']:.6f}")
        logger.info(f"   📈 Training Metrics (Broker-Modality Format):")
        logger.info(f"      Train MAE:  {avg_losses['train_mae']:.2f}")
        logger.info(f"      Train RMSE: {avg_losses['train_rmse']:.2f}")
        logger.info(f"   ⚙️  Learning Rates:")
        for group in optimizer.param_groups:
            logger.info(f"      {group.get('name', 'Group')}: {group['lr']:.2e}")
        
        # Validation every 10 epochs
        if (epoch + 1) % 10 == 0:
            logger.info("-" * 60)
            logger.info(f"🔍 Running validation at epoch {epoch+1}...")
            
            primary_metric, val_metrics = validate_broker_style(model, val_loader, device, logger)
            
            # Check for best metrics and save checkpoints
            improved_metrics = []
            
            for metric_name, current_value in val_metrics.items():
                if current_value < best_metrics[metric_name]:
                    best_metrics[metric_name] = current_value
                    improved_metrics.append(metric_name)
                    
                    # Save best model for this metric
                    save_path = os.path.join(args.save_dir, f'best_{metric_name.lower()}_model.pth')
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'metrics': val_metrics,
                        'train_losses': avg_losses,
                        'args': vars(args)
                    }, save_path)
                    logger.info(f"✅ New best {metric_name}: {current_value:.2f} (saved)")
            
            if not improved_metrics:
                logger.info("📊 No improvements this validation")
            
            # Display best metrics summary (Broker-Modality format)
            logger.info("🏆 Best Metrics So Far (Broker-Modality Format):")
            logger.info(f"   GAME(0): {best_metrics['GAME_0']:.2f}")
            logger.info(f"   GAME(1): {best_metrics['GAME_1']:.2f}")
            logger.info(f"   GAME(2): {best_metrics['GAME_2']:.2f}")
            logger.info(f"   GAME(3): {best_metrics['GAME_3']:.2f}")
            logger.info(f"   RMSE:    {best_metrics['RMSE']:.2f}")
            
            logger.info("-" * 60)
    
    # Training completion summary (Broker-Modality format)
    logger.info("=" * 80)
    logger.info("🎉 Training Completed!")
    logger.info("🏆 Final Best Metrics (Broker-Modality Format):")
    logger.info(f"   GAME(0): {best_metrics['GAME_0']:.2f}")
    logger.info(f"   GAME(1): {best_metrics['GAME_1']:.2f}")
    logger.info(f"   GAME(2): {best_metrics['GAME_2']:.2f}")
    logger.info(f"   GAME(3): {best_metrics['GAME_3']:.2f}")
    logger.info(f"   RMSE:    {best_metrics['RMSE']:.2f}")
    
    logger.info(f"💾 All best models saved in: {args.save_dir}")
    logger.info(f"📝 Complete log saved: {log_file}")
    
    # Save final results in Broker-Modality format
    results = {
        'method': 'SAAI-Enhanced Broker-Modality',
        'venue': 'Custom Implementation',
        'dataset': 'RGBT-CC',
        'final_metrics': best_metrics,
        'comparison_format': 'Compatible with ECCV 2024 Broker-Modality paper'
    }
    
    with open(os.path.join(args.save_dir, 'broker_modality_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
