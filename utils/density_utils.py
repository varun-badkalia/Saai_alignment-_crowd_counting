import torch
import numpy as np

def generate_density_map_tensor(keypoints, image_shape, sigma=3.0):
    """
    Generate density map tensor from keypoints
    Args:
        keypoints: Tensor [N, 2] with (x, y) coordinates
        image_shape: (H, W) target shape
        sigma: Gaussian sigma for density map
    Returns:
        density_map: Tensor [H, W]
    """
    H, W = image_shape
    density_map = torch.zeros((H, W), dtype=torch.float32)

    if keypoints is None or len(keypoints) == 0:
        return density_map

    if isinstance(keypoints, torch.Tensor):
        pts = keypoints.detach().cpu().numpy()
    else:
        pts = keypoints

    # Precompute Gaussian kernel
    size = int(max(3, 6 * sigma))
    if size % 2 == 0:
        size += 1
        
    ax = torch.arange(-(size // 2), (size // 2) + 1, dtype=torch.float32)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))

    for p in pts:
        x, y = int(p[0]), int(p[1])
        if not (0 <= x < W and 0 <= y < H):
            continue

        # Apply kernel with boundary checking
        y1 = max(0, y - size // 2)
        y2 = min(H, y + size // 2 + 1)
        x1 = max(0, x - size // 2)
        x2 = min(W, x + size // 2 + 1)

        ky1 = max(0, (size // 2) - y)
        ky2 = ky1 + (y2 - y1)
        kx1 = max(0, (size // 2) - x)
        kx2 = kx1 + (x2 - x1)

        density_map[y1:y2, x1:x2] += kernel[ky1:ky2, kx1:kx2]

    return density_map

def generate_density_batch(keypoints_list, target_hw, sigma=3.0):
    """
    Generate batch of density maps
    Args:
        keypoints_list: List of keypoint tensors [Ni, 2]
        target_hw: (H, W) target shape
        sigma: Gaussian sigma
    Returns:
        density_batch: Tensor [B, 1, H, W]
    """
    H, W = target_hw
    B = len(keypoints_list)
    out = torch.zeros((B, 1, H, W), dtype=torch.float32)
    
    for i, kps in enumerate(keypoints_list):
        dm = generate_density_map_tensor(kps, (H, W), sigma=sigma)
        out[i, 0] = dm
    
    return out
