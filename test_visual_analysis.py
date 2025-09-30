import torch
from torch.utils.data import DataLoader
import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import cv2
from PIL import Image
import seaborn as sns
from datasets.broker_crowd_dataset import BrokerCrowdDataset, crowd_collate
from models.crowd_counter import SAAICrowdCounter

# Set matplotlib backend for headless servers
plt.switch_backend('Agg')
plt.style.use('seaborn-v0_8')

class VisualAnalyzer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'aligned_images'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'density_maps'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'comparison_plots'), exist_ok=True)
        
    def create_broker_style_aligned_images(self, rgb_tensor, thermal_tensor, density_map, 
                                         gt_count, pred_count, image_name, idx):
        """Create Broker-Modality style aligned image visualization"""
        
        # Convert tensors to numpy arrays
        rgb_np = rgb_tensor.cpu().squeeze().numpy().transpose(1, 2, 0)
        thermal_np = thermal_tensor.cpu().squeeze().numpy().transpose(1, 2, 0)
        density_np = density_map.cpu().squeeze().numpy()
        
        # Denormalize RGB and Thermal
        rgb_mean = np.array([0.485, 0.456, 0.406])
        rgb_std = np.array([0.229, 0.224, 0.225])
        rgb_np = rgb_np * rgb_std + rgb_mean
        rgb_np = np.clip(rgb_np, 0, 1)
        
        thermal_np = thermal_np * rgb_std + rgb_mean  # Same normalization was used
        thermal_np = np.clip(thermal_np, 0, 1)
        
        # Create Broker-Modality style visualization
        fig = plt.figure(figsize=(16, 4))
        gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 1])
        
        # RGB Image
        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(rgb_np)
        ax1.set_title(f'RGB Image\nGT: {gt_count}', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Thermal Image  
        ax2 = fig.add_subplot(gs[1])
        ax2.imshow(thermal_np)
        ax2.set_title(f'Thermal Image\nPred: {pred_count:.1f}', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # Density Map
        ax3 = fig.add_subplot(gs[2])
        density_colored = ax3.imshow(density_np, cmap='jet', alpha=0.8)
        ax3.set_title(f'Predicted Density\nMAE: {abs(pred_count - gt_count):.1f}', 
                     fontsize=12, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(density_colored, ax=ax3, fraction=0.046, pad=0.04)
        
        # Overlay: RGB + Density
        ax4 = fig.add_subplot(gs[3])
        ax4.imshow(rgb_np)
        density_overlay = ax4.imshow(density_np, cmap='hot', alpha=0.6)
        ax4.set_title(f'RGB + Density Overlay\nError: {((pred_count-gt_count)/gt_count*100):+.1f}%', 
                     fontsize=12, fontweight='bold')
        ax4.axis('off')
        
        # Main title
        error_status = "✅ Good" if abs(pred_count - gt_count) < 5 else "⚠️ Fair" if abs(pred_count - gt_count) < 15 else "❌ Poor"
        fig.suptitle(f'{image_name} - SAAI Broker-Modality Analysis | {error_status}', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, 'aligned_images', f'{idx:03d}_{image_name}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_comparison_plot(self, results_list):
        """Create comprehensive prediction vs ground truth comparison"""
        
        gt_counts = [r['gt_count'] for r in results_list]
        pred_counts = [r['pred_count'] for r in results_list]
        mae_errors = [r['mae'] for r in results_list]
        
        # Create comparison plots
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Prediction vs Ground Truth Scatter
        ax1 = fig.add_subplot(gs[0, 0])
        scatter = ax1.scatter(gt_counts, pred_counts, c=mae_errors, cmap='viridis', alpha=0.7, s=50)
        ax1.plot([min(gt_counts), max(gt_counts)], [min(gt_counts), max(gt_counts)], 'r--', alpha=0.8)
        ax1.set_xlabel('Ground Truth Count')
        ax1.set_ylabel('Predicted Count')
        ax1.set_title('Prediction vs Ground Truth')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='MAE')
        
        # 2. Error Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        errors = np.array(pred_counts) - np.array(gt_counts)
        ax2.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(0, color='red', linestyle='--', alpha=0.8)
        ax2.set_xlabel('Prediction Error (Pred - GT)')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Error Distribution\nMean Bias: {np.mean(errors):.2f}')
        ax2.grid(True, alpha=0.3)
        
        # 3. MAE vs Ground Truth Count
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.scatter(gt_counts, mae_errors, alpha=0.6, color='orange')
        ax3.set_xlabel('Ground Truth Count')
        ax3.set_ylabel('Mean Absolute Error')
        ax3.set_title('MAE vs Crowd Density')
        ax3.grid(True, alpha=0.3)
        
        # 4. Cumulative Error
        ax4 = fig.add_subplot(gs[1, 0])
        sorted_mae = np.sort(mae_errors)
        cumulative_pct = np.arange(1, len(sorted_mae) + 1) / len(sorted_mae) * 100
        ax4.plot(sorted_mae, cumulative_pct, linewidth=2, color='green')
        ax4.axvline(np.median(sorted_mae), color='red', linestyle='--', label=f'Median: {np.median(sorted_mae):.1f}')
        ax4.set_xlabel('MAE')
        ax4.set_ylabel('Cumulative Percentage')
        ax4.set_title('Cumulative MAE Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Performance by Crowd Density Ranges
        ax5 = fig.add_subplot(gs[1, 1])
        density_ranges = ['Low (0-50)', 'Medium (50-100)', 'High (100-200)', 'Very High (200+)']
        range_maes = []
        
        for i, (low, high) in enumerate([(0, 50), (50, 100), (100, 200), (200, 1000)]):
            mask = (np.array(gt_counts) >= low) & (np.array(gt_counts) < high)
            if np.any(mask):
                range_maes.append(np.mean(np.array(mae_errors)[mask]))
            else:
                range_maes.append(0)
        
        bars = ax5.bar(density_ranges, range_maes, color=['lightgreen', 'yellow', 'orange', 'red'], alpha=0.7)
        ax5.set_ylabel('Average MAE')
        ax5.set_title('Performance by Crowd Density')
        ax5.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, range_maes):
            if value > 0:
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{value:.1f}', ha='center', va='bottom')
        
        # 6. Sample Images Performance (Top 10)
        ax6 = fig.add_subplot(gs[1, 2])
        sample_indices = range(min(10, len(results_list)))
        sample_gt = [gt_counts[i] for i in sample_indices]
        sample_pred = [pred_counts[i] for i in sample_indices]
        
        x_pos = np.arange(len(sample_indices))
        width = 0.35
        
        ax6.bar(x_pos - width/2, sample_gt, width, label='Ground Truth', color='skyblue', alpha=0.8)
        ax6.bar(x_pos + width/2, sample_pred, width, label='Predicted', color='orange', alpha=0.8)
        ax6.set_xlabel('Sample Index')
        ax6.set_ylabel('Count')
        ax6.set_title('Sample Predictions (First 10)')
        ax6.set_xticks(x_pos)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Accuracy Statistics Table
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        # Calculate statistics
        stats = {
            'Total Samples': len(results_list),
            'Mean MAE': f'{np.mean(mae_errors):.2f}',
            'Median MAE': f'{np.median(mae_errors):.2f}',
            'RMSE': f'{np.sqrt(np.mean(errors**2)):.2f}',
            'Mean Bias': f'{np.mean(errors):+.2f}',
            'Std Error': f'{np.std(errors):.2f}',
            'Accuracy ±5': f'{(np.array(mae_errors) <= 5).mean()*100:.1f}%',
            'Accuracy ±10': f'{(np.array(mae_errors) <= 10).mean()*100:.1f}%',
            'Accuracy ±15': f'{(np.array(mae_errors) <= 15).mean()*100:.1f}%',
            'Max Error': f'{np.max(mae_errors):.1f}',
            'Min Error': f'{np.min(mae_errors):.1f}'
        }
        
        # Create statistics table
        table_data = []
        for i, (key, value) in enumerate(stats.items()):
            if i % 2 == 0:
                table_data.append([key, value, '', ''])
            else:
                table_data[-1][2] = key
                table_data[-1][3] = value
        
        table = ax7.table(cellText=table_data,
                         colLabels=['Metric', 'Value', 'Metric', 'Value'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(table_data) + 1):
            for j in range(4):
                cell = table[(i, j)]
                if i == 0:  # Header row
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        plt.suptitle('SAAI Broker-Modality: Comprehensive Performance Analysis (First 100 Test Images)', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        save_path = os.path.join(self.save_dir, 'comparison_plots', 'comprehensive_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path

def get_args():
    parser = argparse.ArgumentParser(description='SAAI Visual Analysis - Broker-Modality Style')
    parser.add_argument('--data-path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--save-dir', type=str, default='./visual_analysis', help='Directory to save results')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of test samples to analyze')
    parser.add_argument('--crop-size', type=int, default=384, help='Crop size used during training')
    return parser.parse_args()

def main():
    args = get_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎨 SAAI Visual Analysis - Broker-Modality Style")
    print(f"📱 Device: {device}")
    print(f"📁 Model: {args.model_path}")
    print(f"📊 Analyzing first {args.num_samples} test images")
    print(f"💾 Saving to: {args.save_dir}")
    
    # Initialize visual analyzer
    analyzer = VisualAnalyzer(args.save_dir)
    
    # Create test dataset
    test_dataset = BrokerCrowdDataset(
        root_path=os.path.join(args.data_path, 'test'),
        crop_size=args.crop_size,
        downsample_ratio=8,
        method='val'
    )
    
    if len(test_dataset) == 0:
        print("⚠️  Test set not found, using validation set...")
        test_dataset = BrokerCrowdDataset(
            root_path=os.path.join(args.data_path, 'val'),
            crop_size=args.crop_size,
            downsample_ratio=8,
            method='val'
        )
    
    print(f"📊 Dataset: {len(test_dataset)} total samples")
    
    # Limit to specified number of samples
    if len(test_dataset) > args.num_samples:
        test_dataset.rgbt_list = test_dataset.rgbt_list[:args.num_samples]
    
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        collate_fn=crowd_collate, num_workers=0
    )
    
    # Load model
    print("🧠 Loading model...")
    model = SAAICrowdCounter(
        backbone_name='vgg16', pretrained=False, feature_dim=512
    ).to(device)
    
    try:
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    model.eval()
    
    # Process test images
    print(f"\n🔍 Processing {len(test_loader)} test images...")
    results_list = []
    
    with torch.no_grad():
        for idx, (rgb, thermal, target, gt_count, name) in enumerate(test_loader):
            rgb, thermal = rgb.to(device), thermal.to(device)
            
            # Model inference
            density_map, domain_pred_rgb, domain_pred_thermal = model(rgb, thermal)
            pred_count = density_map.sum().item()
            
            # Store results
            result = {
                'idx': idx,
                'image_name': name,
                'gt_count': gt_count,
                'pred_count': pred_count,
                'mae': abs(pred_count - gt_count),
                'error_percentage': (pred_count - gt_count) / gt_count * 100 if gt_count > 0 else 0
            }
            results_list.append(result)
            
            # Create Broker-Modality style aligned images
            if idx < 20:  # Create detailed visualizations for first 20 images
                image_path = analyzer.create_broker_style_aligned_images(
                    rgb, thermal, density_map, gt_count, pred_count, name, idx
                )
                print(f"   📸 Created aligned image {idx+1}/20: {os.path.basename(image_path)}")
            
            # Progress update
            if (idx + 1) % 25 == 0:
                print(f"   Processed {idx + 1}/{len(test_loader)} images...")
    
    # Create comprehensive comparison plot
    print(f"\n📊 Creating comprehensive analysis plots...")
    comparison_path = analyzer.create_comparison_plot(results_list)
    print(f"✅ Comparison plot saved: {comparison_path}")
    
    # Save detailed results to JSON
    detailed_results = {
        'model_info': {
            'model_path': args.model_path,
            'test_samples': len(results_list),
            'analysis_date': str(np.datetime64('now'))
        },
        'summary_statistics': {
            'mean_mae': float(np.mean([r['mae'] for r in results_list])),
            'median_mae': float(np.median([r['mae'] for r in results_list])),
            'rmse': float(np.sqrt(np.mean([r['mae']**2 for r in results_list]))),
            'total_gt': sum([r['gt_count'] for r in results_list]),
            'total_pred': sum([r['pred_count'] for r in results_list]),
            'accuracy_within_5': float((np.array([r['mae'] for r in results_list]) <= 5).mean() * 100),
            'accuracy_within_10': float((np.array([r['mae'] for r in results_list]) <= 10).mean() * 100),
            'accuracy_within_15': float((np.array([r['mae'] for r in results_list]) <= 15).mean() * 100)
        },
        'detailed_predictions': results_list
    }
    
    results_path = os.path.join(args.save_dir, 'detailed_results.json')
    with open(results_path, 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    
    print(f"📋 Detailed results saved: {results_path}")
    
    # Print summary table
    print(f"\n🏆 VISUAL ANALYSIS SUMMARY")
    print("="*60)
    stats = detailed_results['summary_statistics']
    print(f"📊 Samples Analyzed:        {len(results_list)}")
    print(f"📈 Mean MAE:               {stats['mean_mae']:.2f}")
    print(f"📉 Median MAE:             {stats['median_mae']:.2f}")
    print(f"📐 RMSE:                   {stats['rmse']:.2f}")
    print(f"🎯 Total GT Count:         {stats['total_gt']:,}")
    print(f"🤖 Total Pred Count:       {stats['total_pred']:,.0f}")
    print(f"✅ Accuracy (±5):          {stats['accuracy_within_5']:.1f}%")
    print(f"✅ Accuracy (±10):         {stats['accuracy_within_10']:.1f}%")
    print(f"✅ Accuracy (±15):         {stats['accuracy_within_15']:.1f}%")
    
    print(f"\n📁 Files Generated:")
    print(f"   🖼️  Aligned Images:        {args.save_dir}/aligned_images/ (First 20)")
    print(f"   📊 Comparison Plots:      {args.save_dir}/comparison_plots/")
    print(f"   📋 Detailed Results:      {results_path}")
    
    print(f"\n🎉 Visual analysis completed successfully!")

if __name__ == '__main__':
    main()
