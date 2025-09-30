import cv2
from PIL import Image
import torch.utils.data as data
import os
from glob import glob
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import random
import numpy as np

def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w

def cal_innner_area(c_left, c_up, c_right, c_down, bbox):
    inner_left = np.maximum(c_left, bbox[:, 0])
    inner_up = np.maximum(c_up, bbox[:, 1])
    inner_right = np.minimum(c_right, bbox[:, 2])
    inner_down = np.minimum(c_down, bbox[:, 3])
    inner_area = np.maximum(inner_right - inner_left, 0.0) * np.maximum(inner_down - inner_up, 0.0)
    return inner_area

class BrokerCrowdDataset(data.Dataset):
    """Broker-Modality style dataset for SAAI enhancement"""
    
    def __init__(self, root_path, crop_size, downsample_ratio, method):
        self.root_path = root_path
        self.rgbt_list = sorted(glob(os.path.join(self.root_path, '*.png')))
        
        if method not in ['train', 'val', 'test']:
            raise Exception("Method not implemented")
        
        self.method = method
        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio

        # Standard ImageNet normalization
        self.rgb_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.t_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.rgbt_list)

    def __getitem__(self, item):
        rgbt_path = self.rgbt_list[item]
        rgb_path = rgbt_path.replace('RGBT.png', 'RGB.jpg')
        t_path = rgbt_path.replace('RGBT.png', 'T.jpg')
        gd_path = rgbt_path.replace('RGBT.png', 'GT.npy')
        
        rgb = Image.open(rgb_path).convert('RGB')
        t = Image.open(t_path).convert('RGB')

        if self.method == 'train':
            keypoints = np.load(gd_path)
            return self.train_transform(rgb, t, keypoints)
        else:
            keypoints = np.load(gd_path)
            shape = cv2.imread(t_path)[..., ::-1].copy().shape
            gt = keypoints
            k = np.zeros((shape[0], shape[1]))
            for i in range(0, len(gt)):
                if int(gt[i][1]) < shape[0] and int(gt[i][0]) < shape[1]:
                    k[int(gt[i][1]), int(gt[i][0])] = 1
                    
            rgb = self.rgb_trans(rgb)
            t = self.t_trans(t)
            target = k
            name = os.path.basename(rgbt_path).split('.')[0]
            return rgb, t, target, len(keypoints), name

    def train_transform(self, rgb, t, keypoints):
        """Training transform with Bayesian Loss format"""
        wd, ht = rgb.size
        st_size = min(wd, ht)
        
        # Handle different image sizes
        effective_crop_size = min(self.c_size, st_size)
        if self.c_size > st_size:
            effective_crop_size = st_size
        
        assert len(keypoints) > 0
        
        # Random crop
        i, j, h, w = random_crop(ht, wd, effective_crop_size, effective_crop_size)
        rgb = F.crop(rgb, i, j, h, w)
        t = F.crop(t, i, j, h, w)

        # Process keypoints using Bayesian Loss format
        nearest_dis = np.clip(keypoints[:, 2], 4.0, 128.0)
        points_left_up = keypoints[:, :2] - nearest_dis[:, None] / 2.0
        points_right_down = keypoints[:, :2] + nearest_dis[:, None] / 2.0
        bbox = np.concatenate((points_left_up, points_right_down), axis=1)
        inner_area = cal_innner_area(j, i, j + w, i + h, bbox)
        origin_area = nearest_dis * nearest_dis
        ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)
        mask = (ratio >= 0.3)

        target = ratio[mask]
        keypoints = keypoints[mask]
        
        # Handle empty keypoints
        if len(keypoints) == 0:
            keypoints = np.array([[w/2, h/2]], dtype=np.float32)
            target = np.array([0.5], dtype=np.float32)
        else:
            keypoints = keypoints[:, :2] - [j, i]
        
        # Horizontal flip augmentation
        if len(keypoints) > 0:
            if random.random() > 0.5:
                rgb = F.hflip(rgb)
                t = F.hflip(t)
                keypoints[:, 0] = w - keypoints[:, 0]
        else:
            if random.random() > 0.5:
                rgb = F.hflip(rgb)
                t = F.hflip(t)

        # Apply normalization
        rgb = self.rgb_trans(rgb)
        t = self.t_trans(t)
        
        return rgb, t, torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(target.copy()).float(), st_size

def crowd_collate(batch):
    """Simple collate function"""
    rgb_list, t_list, keypoints_list, targets_list, st_sizes = [], [], [], [], []
    
    for rgb, t, keypoints, target, st_size in batch:
        rgb_list.append(rgb)
        t_list.append(t)
        keypoints_list.append(keypoints)
        targets_list.append(target)
        st_sizes.append(st_size)
    
    rgb = torch.stack(rgb_list, 0)
    t = torch.stack(t_list, 0)
    st_sizes = torch.tensor(st_sizes, dtype=torch.float)
    
    return rgb, t, keypoints_list, targets_list, st_sizes
