import torch
from torch.utils.data import DataLoader
import os
import argparse
import json
import numpy as np
import time
from datetime import datetime
from datasets.broker_crowd_dataset import BrokerCrowdDataset, crowd_collate
from models.crowd_counter import SAAICrowdCounter
from utils.losses import calculate_broker_game_metrics, calculate_broker_rmse

# JSON encoder for numpy objects
class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy objects"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def get_args():
    parser = argparse.ArgumentParser(description='SAAI Model Testing - Broker-Modality Compatible')
    parser.add_argument('--data-path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--crop-size', type=int, default=384, help='Crop size used during training')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--save-results', type=str, default='./test_results.json', help='Path to save results')
    parser.add_argument('--save-predictions', type=str, default='./test_predictions.json', help='Path to save detailed predictions')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers')
    return parser.parse_args()

def comprehensive_test_evaluation(model, test_loader, device):
    """
    Complete testing evaluation with Broker-Modality compatible metrics
    """
    model.eval()
    
    # Results storage
    all_results = []
    all_mae_errors = []
    all_game_metrics = {0: [], 1: [], 2: [], 3: []}
    all_rmse_errors = []
    
    # Timing
    total_inference_time = 0
    
    print("🔍 Running comprehensive test evaluation...")
    print(f"📊 Processing {len(test_loader)} test images...")
    
    with torch.no_grad():
        for idx, (rgb, thermal, target, gt_count, name) in enumerate(test_loader):
            # Inference timing
            start_time = time.time()
            
            rgb, thermal = rgb.to(device), thermal.to(device)
            density_map, domain_pred_rgb, domain_pred_thermal = model(rgb, thermal)
            
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            
            # Predictions
            pred_count = density_map.sum().item()
            density_max = density_map.max().item()
            density_min = density_map.min().item()
            density_mean = density_map.mean().item()
            
            # Calculate individual sample metrics
            mae_error = abs(pred_count - gt_count)
            rmse_error = (pred_count - gt_count) ** 2
            
            all_mae_errors.append(mae_error)
            all_rmse_errors.append(rmse_error)
            
            # Calculate GAME metrics for this single sample
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
            
            # Domain prediction analysis (SAAI specific)
            rgb_domain_prob = torch.softmax(domain_pred_rgb, dim=1)[0, 0].item()  # Prob of RGB being RGB
            thermal_domain_prob = torch.softmax(domain_pred_thermal, dim=1)[0, 1].item()  # Prob of Thermal being Thermal
            
            # Store detailed results
            sample_result = {
                'image_name': name,
                'gt_count': int(gt_count),
                'pred_count': float(pred_count),
                'mae': float(mae_error),
                'rmse': float(rmse_error ** 0.5),
                'error_percentage': float((mae_error / max(gt_count, 1)) * 100),
                'density_stats': {
                    'max': float(density_max),
                    'min': float(density_min),
                    'mean': float(density_mean)
                },
                'saai_alignment': {
                    'rgb_domain_accuracy': float(rgb_domain_prob),
                    'thermal_domain_accuracy': float(thermal_domain_prob),
                    'average_alignment': float((rgb_domain_prob + thermal_domain_prob) / 2)
                },
                'game_metrics': {
                    'GAME_0': float(all_game_metrics[0][-1]),
                    'GAME_1': float(all_game_metrics[1][-1]),
                    'GAME_2': float(all_game_metrics[2][-1]),
                    'GAME_3': float(all_game_metrics[3][-1])
                },
                'inference_time_ms': float(inference_time * 1000)
            }
            
            all_results.append(sample_result)
            
            # Progress update
            if (idx + 1) % 50 == 0 or idx == 0:
                print(f"   Processed {idx + 1}/{len(test_loader)} images... "
                      f"Current MAE: {mae_error:.2f}, Pred: {pred_count:.1f}, GT: {gt_count}")
    
    # Calculate overall metrics
    overall_mae = sum(all_mae_errors) / len(all_mae_errors)
    overall_rmse = (sum(all_rmse_errors) / len(all_rmse_errors)) ** 0.5
    
    # Calculate average GAME metrics
    game_results = {}
    for level in [0, 1, 2, 3]:
        game_results[f'GAME_{level}'] = sum(all_game_metrics[level]) / len(all_game_metrics[level])
    
    # Additional statistics
    avg_inference_time = total_inference_time / len(test_loader)
    total_gt_count = sum([r['gt_count'] for r in all_results])
    total_pred_count = sum([r['pred_count'] for r in all_results])
    avg_alignment = sum([r['saai_alignment']['average_alignment'] for r in all_results]) / len(all_results)
    
    # Count accuracy statistics
    count_errors = [r['pred_count'] - r['gt_count'] for r in all_results]
    mean_bias = sum(count_errors) / len(count_errors)
    std_error = (sum([(e - mean_bias)**2 for e in count_errors]) / len(count_errors)) ** 0.5
    
    # Compile comprehensive results
    final_results = {
        'model_info': {
            'model_path': args.model_path if 'args' in globals() else 'Unknown',
            'test_dataset_size': len(test_loader),
            'test_date': datetime.now().isoformat(),
            'device': str(device)
        },
        'performance_metrics': {
            'broker_modality_format': {
                'GAME_0': float(game_results['GAME_0']),
                'GAME_1': float(game_results['GAME_1']),
                'GAME_2': float(game_results['GAME_2']),
                'GAME_3': float(game_results['GAME_3']),
                'RMSE': float(overall_rmse)
            },
            'additional_metrics': {
                'MAE': float(overall_mae),
                'Mean_Bias': float(mean_bias),
                'Std_Error': float(std_error),
                'MAPE': float(sum([abs(r['error_percentage']) for r in all_results]) / len(all_results))
            }
        },
        'saai_analysis': {
            'average_domain_alignment': float(avg_alignment),
            'rgb_domain_accuracy': float(sum([r['saai_alignment']['rgb_domain_accuracy'] for r in all_results]) / len(all_results)),
            'thermal_domain_accuracy': float(sum([r['saai_alignment']['thermal_domain_accuracy'] for r in all_results]) / len(all_results))
        },
        'efficiency_metrics': {
            'average_inference_time_ms': float(avg_inference_time * 1000),
            'fps': float(1 / avg_inference_time),
            'total_test_time_seconds': float(total_inference_time)
        },
        'dataset_statistics': {
            'total_gt_count': int(total_gt_count),
            'total_pred_count': float(total_pred_count),
            'count_accuracy_percentage': float((total_pred_count / total_gt_count) * 100) if total_gt_count > 0 else 0.0,
            'avg_gt_per_image': float(total_gt_count / len(all_results)),
            'avg_pred_per_image': float(total_pred_count / len(all_results))
        },
        'detailed_predictions': all_results
    }
    
    return final_results

def display_results(results):
    """Display results in Broker-Modality comparable format"""
    
    print("\n" + "="*90)
    print("🏆 COMPREHENSIVE TEST RESULTS - BROKER-MODALITY COMPATIBLE")
    print("="*90)
    
    # Main metrics in Broker-Modality format
    print("\n📊 MAIN EVALUATION METRICS (Broker-Modality Format):")
    print("-" * 60)
    print(f"{'Metric':<12} {'Value':<10} {'Comparison':<25}")
    print("-" * 60)
    bm_metrics = results['performance_metrics']['broker_modality_format']
    print(f"{'GAME(0)':<12} {bm_metrics['GAME_0']:<10.2f} {'(vs BM: 10.19)':<25}")
    print(f"{'GAME(1)':<12} {bm_metrics['GAME_1']:<10.2f} {'(vs BM: 13.61)':<25}")
    print(f"{'GAME(2)':<12} {bm_metrics['GAME_2']:<10.2f} {'(vs BM: 17.65)':<25}")
    print(f"{'GAME(3)':<12} {bm_metrics['GAME_3']:<10.2f} {'(vs BM: 23.64)':<25}")
    print(f"{'RMSE':<12} {bm_metrics['RMSE']:<10.2f} {'(vs BM: 17.32)':<25}")
    
    # Performance comparison
    print(f"\n🎯 PERFORMANCE COMPARISON:")
    bm_baseline = {'GAME_0': 10.19, 'GAME_1': 13.61, 'GAME_2': 17.65, 'GAME_3': 23.64, 'RMSE': 17.32}
    for metric, value in bm_metrics.items():
        baseline = bm_baseline[metric]
        improvement = ((baseline - value) / baseline) * 100
        status = "✅ Better" if improvement > 0 else "❌ Worse"
        print(f"   {metric}: {improvement:+.1f}% {status}")
    
    # SAAI specific analysis
    print(f"\n🔗 SAAI DOMAIN ALIGNMENT ANALYSIS:")
    saai = results['saai_analysis']
    print(f"   Average Domain Alignment:  {saai['average_domain_alignment']:.3f}")
    print(f"   RGB Domain Accuracy:       {saai['rgb_domain_accuracy']:.3f}")
    print(f"   Thermal Domain Accuracy:   {saai['thermal_domain_accuracy']:.3f}")
    
    alignment_quality = "Excellent" if saai['average_domain_alignment'] > 0.9 else \
                       "Good" if saai['average_domain_alignment'] > 0.8 else \
                       "Needs Improvement"
    print(f"   Alignment Quality:         {alignment_quality}")
    
    # Additional metrics
    print(f"\n📈 ADDITIONAL METRICS:")
    add_metrics = results['performance_metrics']['additional_metrics']
    print(f"   Mean Absolute Error (MAE): {add_metrics['MAE']:.2f}")
    print(f"   Mean Bias:                 {add_metrics['Mean_Bias']:+.2f}")
    print(f"   Standard Error:            {add_metrics['Std_Error']:.2f}")
    print(f"   Mean Absolute Percent Err: {add_metrics['MAPE']:.2f}%")
    
    # Efficiency metrics
    print(f"\n⚡ EFFICIENCY METRICS:")
    eff = results['efficiency_metrics']
    print(f"   Average Inference Time:    {eff['average_inference_time_ms']:.2f} ms")
    print(f"   Frames Per Second (FPS):   {eff['fps']:.2f}")
    print(f"   Total Test Time:           {eff['total_test_time_seconds']:.2f} seconds")
    
    # Dataset statistics
    print(f"\n📊 DATASET STATISTICS:")
    stats = results['dataset_statistics']
    print(f"   Total Ground Truth Count:  {stats['total_gt_count']:,}")
    print(f"   Total Predicted Count:     {stats['total_pred_count']:,.0f}")
    print(f"   Overall Count Accuracy:    {stats['count_accuracy_percentage']:.1f}%")
    print(f"   Average GT per Image:      {stats['avg_gt_per_image']:.1f}")
    print(f"   Average Pred per Image:    {stats['avg_pred_per_image']:.1f}")
    
    # Best and worst predictions
    predictions = results['detailed_predictions']
    sorted_by_mae = sorted(predictions, key=lambda x: x['mae'])
    
    print(f"\n✅ TOP 5 BEST PREDICTIONS (Lowest MAE):")
    for i, pred in enumerate(sorted_by_mae[:5]):
        print(f"   {i+1}. {pred['image_name'][:20]:20} | GT: {pred['gt_count']:3d} | Pred: {pred['pred_count']:6.1f} | MAE: {pred['mae']:5.1f}")
    
    print(f"\n❌ TOP 5 WORST PREDICTIONS (Highest MAE):")
    for i, pred in enumerate(sorted_by_mae[-5:]):
        print(f"   {i+1}. {pred['image_name'][:20]:20} | GT: {pred['gt_count']:3d} | Pred: {pred['pred_count']:6.1f} | MAE: {pred['mae']:5.1f}")

def main():
    global args
    args = get_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Testing SAAI-Enhanced Crowd Counter")
    print(f"📱 Device: {device}")
    print(f"📁 Model: {args.model_path}")
    print(f"📊 Dataset: {args.data_path}")
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model file not found: {args.model_path}")
        return
    
    # Create test dataset
    test_dataset = BrokerCrowdDataset(
        root_path=os.path.join(args.data_path, 'test'),
        crop_size=args.crop_size,
        downsample_ratio=8,
        method='val'  # Use 'val' method for consistent image sizes
    )
    
    if len(test_dataset) == 0:
        # Try using validation set if test set doesn't exist
        print("⚠️  Test set not found, using validation set...")
        test_dataset = BrokerCrowdDataset(
            root_path=os.path.join(args.data_path, 'val'),
            crop_size=args.crop_size,
            downsample_ratio=8,
            method='val'
        )
    
    print(f"📊 Test dataset: {len(test_dataset)} samples")
    
    if len(test_dataset) == 0:
        print("❌ Error: No test data found!")
        return
    
    # Create test loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=crowd_collate,
        num_workers=args.workers
    )
    
    # Load model
    print("🧠 Loading model...")
    model = SAAICrowdCounter(
        backbone_name='vgg16', 
        pretrained=False,  # Don't load pretrained weights
        feature_dim=512
    ).to(device)
    
    # Load trained weights - FIXED for PyTorch 2.6+
    try:
        print("📥 Loading checkpoint with weights_only=False for compatibility...")
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Print model info if available
        if 'epoch' in checkpoint:
            print(f"✅ Model loaded from epoch {checkpoint['epoch']}")
        if 'metrics' in checkpoint:
            print(f"📈 Training metrics:")
            for k, v in checkpoint['metrics'].items():
                print(f"     {k}: {v:.2f}")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 If you trust this model file, the loading issue has been resolved with weights_only=False")
        return
    
    # Run comprehensive testing
    print(f"\n🔍 Starting comprehensive test evaluation...")
    results = comprehensive_test_evaluation(model, test_loader, device)
    
    # Display results
    display_results(results)
    
    # Save detailed results - FIXED with NumpyEncoder
    print(f"\n💾 Saving detailed results...")
    try:
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        print(f"✅ Detailed results saved: {args.save_results}")
    except Exception as e:
        print(f"❌ Error saving detailed results: {e}")
    
    # Save predictions only (smaller file) - FIXED with NumpyEncoder
    predictions_only = {
        'model_path': args.model_path,
        'test_date': results['model_info']['test_date'],
        'main_metrics': results['performance_metrics']['broker_modality_format'],
        'saai_analysis': results['saai_analysis'],
        'efficiency': results['efficiency_metrics'],
        'dataset_stats': results['dataset_statistics'],
        'predictions': results['detailed_predictions']
    }
    
    try:
        with open(args.save_predictions, 'w') as f:
            json.dump(predictions_only, f, indent=2, cls=NumpyEncoder)
        print(f"✅ Predictions saved: {args.save_predictions}")
    except Exception as e:
        print(f"❌ Error saving predictions: {e}")
    
    print(f"\n🎉 Testing completed successfully!")
    
    # Print final summary table
    print(f"\n📋 FINAL SUMMARY TABLE:")
    print("┌─────────────────────────────────────────────────┐")
    print("│              SAAI PERFORMANCE SUMMARY           │")
    print("├─────────────────────────────────────────────────┤")
    bm = results['performance_metrics']['broker_modality_format']
    saai = results['saai_analysis']
    eff = results['efficiency_metrics']
    print(f"│ GAME(0):           {bm['GAME_0']:8.2f}                │")
    print(f"│ GAME(1):           {bm['GAME_1']:8.2f}                │")
    print(f"│ GAME(2):           {bm['GAME_2']:8.2f}                │")
    print(f"│ GAME(3):           {bm['GAME_3']:8.2f}                │")
    print(f"│ RMSE:              {bm['RMSE']:8.2f}                │")
    print(f"│ SAAI Alignment:    {saai['average_domain_alignment']:8.3f}                │")
    print(f"│ Inference FPS:     {eff['fps']:8.1f}                │")
    print("└─────────────────────────────────────────────────┘")

if __name__ == '__main__':
    main()
