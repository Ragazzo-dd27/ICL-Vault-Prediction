import torch
import torch
import random
import matplotlib.pyplot as plt
import numpy as np

from models.unet import LightweightUNet
from utils.dataset import KeratitisDataset

# === 修复中文显示乱码 ===
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像是负号'-'显示为方块的问题

# 1. 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. 加载模型结构与权重
model = LightweightUNet()
model.load_state_dict(torch.load('unet_keratitis.pth', map_location=device))
model.to(device)
model.eval()

# 3. 实例化数据集
dataset = KeratitisDataset(resize=(256, 256))     # 设定和训练时一致
dataset_len = len(dataset)

# 4. 随机选取3张图片
indices = random.sample(range(dataset_len), 3)
samples = [dataset[i] for i in indices]

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
titles = ["原始图像", "真实标签(Ground Truth)", "模型预测(Prediction)"]

for row, sample in enumerate(samples):
    img, mask = sample  # img: [1, H, W], mask: [1, H, W]
    
    # 数据准备
    input_img = img.unsqueeze(0).to(device)          # [1, 1, H, W]
    with torch.no_grad():
        logits = model(input_img)                   # [1, 1, H, W]
        pred_mask = torch.sigmoid(logits)
        pred_mask = (pred_mask > 0.5).float()
    
    # 转回CPU并 squeeze，准备画图
    img_np = img.squeeze().cpu().numpy()
    mask_np = mask.squeeze().cpu().numpy()
    pred_np = pred_mask.squeeze().cpu().numpy()
    
    # 画: 原图
    axes[row,0].imshow(img_np, cmap='gray')
    axes[row,0].set_title(titles[0])
    axes[row,0].axis('off')
    # 画: GT
    axes[row,1].imshow(mask_np, cmap='gray')
    axes[row,1].set_title(titles[1])
    axes[row,1].axis('off')
    # 画: 预测
    axes[row,2].imshow(pred_np, cmap='gray')
    axes[row,2].set_title(titles[2])
    axes[row,2].axis('off')

plt.tight_layout()
plt.show()