import os
import os
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam

from models.multimodal_net import VaultPredictor
from utils.multimodal_dataset import SimulatedMultimodalDataset

# 配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 8
lr = 1e-4
epochs = 5

# 数据准备
dataset = SimulatedMultimodalDataset(length=1000)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 模型准备
model = VaultPredictor().to(device)
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=lr)

# 训练循环
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in loader:
        oct_img = batch['oct_img'].to(device)
        ubm_img = batch['ubm_img'].to(device)
        clinical_feats = batch['clinical_feats'].to(device)
        label = batch['label'].to(device)

        pred = model(oct_img, ubm_img, clinical_feats)
        loss = criterion(pred.squeeze(), label.squeeze())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * oct_img.size(0)

    avg_loss = total_loss / len(dataset)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# 日志与保存
ckpt_dir = "checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(ckpt_dir, "multimodal_simulated.pth"))