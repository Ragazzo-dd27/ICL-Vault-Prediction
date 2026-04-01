import torch
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

from utils.dataset import KeratitisDataset             # 导入自定义数据集
from models.unet import LightweightUNet                # 导入自定义U-Net模型

# -------------------- 1. 选择设备 --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# -------------------- 2. 加载与划分数据集 --------------------
full_dataset = KeratitisDataset(resize=(256, 256))
data_len = len(full_dataset)
train_size = int(0.8 * data_len)
val_size = data_len - train_size
# 随机划分训练集和验证集
train_set, val_set = random_split(full_dataset, [train_size, val_size])

# -------------------- 3. 创建DataLoader --------------------
batch_size = 4  # 可以适当增大看显存
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# -------------------- 4. 实例化模型、损失函数、优化器 --------------------
model = LightweightUNet().to(device)
criterion = nn.BCEWithLogitsLoss()                 # 适合单通道二分类segmentation
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# -------------------- 5. 训练循环 --------------------
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for imgs, masks in train_loader:
        imgs = imgs.to(device)                    # [B, 1, H, W]
        masks = masks.to(device)                  # [B, 1, H, W]
        
        optimizer.zero_grad()
        outputs = model(imgs)                     # [B, 1, H, W]
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * imgs.size(0)
    
    avg_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# -------------------- 6. 保存模型权重 --------------------
torch.save(model.state_dict(), "unet_keratitis.pth")
print("Training completed. Model weights saved as 'unet_keratitis.pth'.")