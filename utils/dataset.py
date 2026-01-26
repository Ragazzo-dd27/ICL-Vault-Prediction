import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import cv2

class KeratitisDataset(Dataset):
    def __init__(self, img_dir='data/public_datasets/keratitis_oct/images',
                 mask_dir='data/public_datasets/keratitis_oct/masks',
                 resize=(256, 256)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.resize = resize
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith('.bmp')]
        self.img_names.sort()  # 保证顺序一致

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # ------- 加载图片 -------
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('L')  # 转灰度
        # 图片 resize & 归一化在后面做，为了 mask 坐标对齐，先不 resize

        img_np = np.array(img).astype(np.float32)
        orig_h, orig_w = img_np.shape[:2]

        # ------- 打开并解析Mask JSON -------
        mask_name = os.path.splitext(img_name)[0] + '.json'
        mask_path = os.path.join(self.mask_dir, mask_name)
        with open(mask_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # LabelMe 标准格式：shapes为多边形标注列表
        shapes = data.get('shapes', [])
        # 默认mask黑底
        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        for shape in shapes:
            # points字段是List[[x1,y1],...], 可能是浮点数
            points = shape.get('points', [])
            if not points:
                continue
            pts = np.array(points, dtype=np.float32)
            pts = np.round(pts).astype(np.int32)  # 强制转整
            pts = pts.reshape(-1, 1, 2)
            # 填充到mask，填充值为 1
            cv2.fillPoly(mask, [pts], color=1)

        # ------ Resize img & mask 到统一尺寸 ------
        img_resized = cv2.resize(img_np, self.resize, interpolation=cv2.INTER_LINEAR)
        img_resized = img_resized.astype(np.float32) / 255.0  # 归一化
        img_resized = np.expand_dims(img_resized, axis=0)  # [1, H, W]

        mask_resized = cv2.resize(mask, self.resize, interpolation=cv2.INTER_NEAREST)
        mask_resized = (mask_resized > 0).astype(np.float32)
        mask_resized = np.expand_dims(mask_resized, axis=0)  # [1, H, W]

        # 转为Tensor
        img_tensor = torch.from_numpy(img_resized)
        mask_tensor = torch.from_numpy(mask_resized)

        return img_tensor, mask_tensor

if __name__ == '__main__':
    ds = KeratitisDataset()
    img, mask = ds[0]
    print('image shape:', img.shape)   # [1, 256, 256]
    print('mask shape:', mask.shape)
    print('mask unique values:', torch.unique(mask))
    has_one = (mask == 1).sum().item() > 0
    print('Mask contains 1:', has_one)