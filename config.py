import argparse

def get_config():
    parser = argparse.ArgumentParser(description='SAAI Enhanced Crowd Counting')
    
    # Dataset settings
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to RGBT-CC dataset')
    parser.add_argument('--crop_size', type=int, default=384,
                        help='Random crop size for training')
    parser.add_argument('--downsample_ratio', type=int, default=8,
                        help='Downsample ratio for density maps')
    
    # Model settings
    parser.add_argument('--backbone', type=str, default='vgg16',
                        choices=['vgg16', 'resnet50'],
                        help='Backbone architecture')
    parser.add_argument('--feature_dim', type=int, default=512,
                        help='Feature dimension for SAAI alignment')
    parser.add_argument('--num_prototypes', type=int, default=64,
                        help='Number of semantic prototypes')
    
    # Training settings
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    
    # Loss weights
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Domain adversarial loss weight')
    parser.add_argument('--beta', type=float, default=0.05,
                        help='Semantic alignment loss weight')
    
    # Logging and saving
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='Logging interval')
    
    return parser.parse_args()
