import torch
from torch.utils.data import DataLoader
import os
import argparse
import time
import logging
from datetime import datetime
from tqdm import tqdm
from datasets.broker_crowd_dataset import BrokerCrowdDataset, crowd_collate
from models.crowd_counter import SAAICrowdCounter
from utils.losses import SAAIBrokerLoss
from utils.metrics import calculate_mae, calculate_rmse

def setup_logging(save_dir):
    """Setup logging to both file and console"""
    log_filename = os.path.join(save_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_filename

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
    return parser.parse_args()

def validate(model, val_loader, device, logger):
    model.eval()
    total_mae = 0
    num_samples = 0
    
    with torch.no_grad():
        for rgb, thermal, target, gt_count, name in val_loader:
            rgb, thermal = rgb.to(device), thermal.to(device)
            
            density_map, _, _ = model(rgb, thermal)
            pred_count = density_map.sum().item()
            mae = abs(pred_count - gt_count)
            
            total_mae += mae
            num_samples += 1
    
    avg_mae = total_mae / num_samples
    logger.info(f"Validation completed: MAE = {avg_mae:.2f}")
    return avg_mae

def main():
    args = get_args()
    
    # Setup logging
    os.makedirs(args.save_dir, exist_ok=True)
    logger, log_file = setup_logging(args.save_dir)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Training log will be saved to: {log_file}")
    
    # Log training configuration
    logger.info("Training Configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Check directory structure
    train_path = os.path.join(args.data_path, 'train')
    val_path = os.path.join(args.data_path, 'val')
    
    logger.info(f"Train path: {train_path}")
    logger.info(f"Val path: {val_path}")
    
    # Create datasets
    train_dataset = BrokerCrowdDataset(
        root_path=train_path,
        crop_size=args.crop_size,
        downsample_ratio=args.downsample_ratio,
        method='train'
    )
    
    val_dataset = BrokerCrowdDataset(
        root_path=val_path,
        crop_size=args.crop_size,
        downsample_ratio=args.downsample_ratio,
        method='val'
    )
    
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=crowd_collate,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=crowd_collate,
        pin_memory=True
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # Model
    model = SAAICrowdCounter(
        backbone_name='vgg16',
        pretrained=True,
        feature_dim=512
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    backbone_params = list(model.rgb_backbone.parameters()) + list(model.thermal_backbone.parameters())
    saai_params = list(model.saai_aligner.parameters())
    head_params = list(model.fusion_module.parameters()) + list(model.regression_head.parameters())
    
    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': saai_params, 'lr': args.lr},
        {'params': head_params, 'lr': args.lr}
    ], weight_decay=args.weight_decay)
    
    criterion = SAAIBrokerLoss(alpha=args.alpha, beta=args.beta)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)
    
    best_mae = float('inf')
    
    logger.info("Starting training...")
    logger.info("=" * 80)
    
    # Training progress tracking
    epoch_pbar = tqdm(range(args.epochs), desc="Training Progress", ncols=100)
    
    for epoch in epoch_pbar:
        start_time = time.time()
        model.train()
        
        # Loss tracking for the epoch
        epoch_losses = {
            'total': 0,
            'density': 0,
            'domain': 0,
        }
        batch_count = 0
        
        # Simple progress for batches within epoch (no detailed logging)
        for batch_idx, (rgb, thermal, keypoints_list, targets_list, st_sizes) in enumerate(train_loader):
            rgb = rgb.to(device)
            thermal = thermal.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            density_map, domain_pred_rgb, domain_pred_thermal = model(rgb, thermal)
            
            # Loss calculation
            total_loss, loss_dict = criterion(
                density_map, keypoints_list, targets_list,
                domain_pred_rgb, domain_pred_thermal
            )
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate losses
            epoch_losses['total'] += total_loss.item()
            epoch_losses['density'] += loss_dict['density_loss']
            epoch_losses['domain'] += loss_dict['domain_loss']
            batch_count += 1
        
        # Calculate epoch averages
        avg_losses = {key: val/batch_count for key, val in epoch_losses.items()}
        
        # Scheduler step
        scheduler.step()
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'Loss': f'{avg_losses["total"]:.4f}',
            'Density': f'{avg_losses["density"]:.4f}',
            'Domain': f'{avg_losses["domain"]:.4f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.1e}'
        })
        
        # Log epoch results
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Time: {epoch_time:.1f}s")
        logger.info(f"  Losses - Total: {avg_losses['total']:.6f}, "
                   f"Density: {avg_losses['density']:.6f}, "
                   f"Domain: {avg_losses['domain']:.6f}")
        logger.info(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Validation every 10 epochs
        if (epoch + 1) % 10 == 0:
            logger.info("-" * 50)
            logger.info(f"Running validation at epoch {epoch+1}...")
            val_mae = validate(model, val_loader, device, logger)
            
            # Save best model
            if val_mae < best_mae:
                best_mae = val_mae
                model_save_path = os.path.join(args.save_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_mae': best_mae,
                    'train_losses': avg_losses,
                    'args': args,
                }, model_save_path)
                logger.info(f"✅ New best model saved! MAE: {best_mae:.2f} -> {model_save_path}")
            
            logger.info(f"Validation MAE: {val_mae:.2f} (Best: {best_mae:.2f})")
            logger.info("-" * 50)
        
        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': avg_losses,
                'args': args,
            }, checkpoint_path)
            logger.info(f"💾 Checkpoint saved: checkpoint_epoch_{epoch+1}.pth")

    logger.info("=" * 80)
    logger.info("🎉 Training completed!")
    logger.info(f"🏆 Best validation MAE: {best_mae:.2f}")
    logger.info(f"💾 Best model and logs saved in: {args.save_dir}")
    logger.info(f"📝 Training log: {log_file}")

if __name__ == '__main__':
    main()
