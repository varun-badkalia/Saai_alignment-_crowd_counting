import os
import cv2
from glob import glob
from PIL import Image

def check_dataset_structure(data_path):
    """Check dataset structure and image sizes"""
    print(f"Checking dataset at: {data_path}")
    
    train_path = os.path.join(data_path, 'train')
    if not os.path.exists(train_path):
        print(f"Train directory not found at: {train_path}")
        print("Available directories:")
        for item in os.listdir(data_path):
            if os.path.isdir(os.path.join(data_path, item)):
                print(f"  - {item}")
        return
    
    # Check for RGBT files
    rgbt_files = sorted(glob(os.path.join(train_path, '*.png')))
    print(f"Found {len(rgbt_files)} RGBT files")
    
    if len(rgbt_files) == 0:
        print("No RGBT.png files found. Checking all files:")
        all_files = os.listdir(train_path)
        for f in all_files[:10]:  # Show first 10 files
            print(f"  - {f}")
        return
    
    # Check first few files
    for i, rgbt_path in enumerate(rgbt_files[:3]):
        print(f"\nChecking file {i+1}: {os.path.basename(rgbt_path)}")
        
        # Check corresponding files
        rgb_path = rgbt_path.replace('RGBT.png', 'RGB.jpg')
        t_path = rgbt_path.replace('RGBT.png', 'T.jpg')
        gt_path = rgbt_path.replace('RGBT.png', 'GT.npy')
        
        files_to_check = [
            ('RGB', rgb_path),
            ('Thermal', t_path), 
            ('Ground Truth', gt_path)
        ]
        
        for name, path in files_to_check:
            if os.path.exists(path):
                if name != 'Ground Truth':
                    try:
                        img = Image.open(path)
                        print(f"  {name}: {img.size} (W x H)")
                    except Exception as e:
                        print(f"  {name}: Error loading - {e}")
                else:
                    try:
                        import numpy as np
                        gt = np.load(path)
                        print(f"  {name}: {gt.shape} - {len(gt)} people")
                    except Exception as e:
                        print(f"  {name}: Error loading - {e}")
            else:
                print(f"  {name}: NOT FOUND")

if __name__ == '__main__':
    data_path = "/home/varun/1_MY WORK/Pipeline/VarunPipeline/RGBT-CC-And-Initial-Broker"
    check_dataset_structure(data_path)
