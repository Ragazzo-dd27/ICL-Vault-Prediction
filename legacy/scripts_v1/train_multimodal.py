import os
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam

from models.multimodal_net import VaultPredictor
from utils.multimodal_dataset import SimulatedMultimodalDataset

# ================= 配置参数 =================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 8
lr = 1e-4
epochs = 5

# MCOA 预训练权重路径 (昨晚跑出来的那个文件)
pretrained_ckpt = 'checkpoints/resnet18_mcoa_pretrained.pth'

# ================= 数据准备 =================
print("正在生成模拟多模态数据...")
dataset = SimulatedMultimodalDataset(length=1000)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ================= 模型准备 =================
# 检查预训练权重是否存在
if os.path.exists(pretrained_ckpt):
    print(f"🚀 正在加载 MCOA 域适应权重: {pretrained_ckpt}")
    model = VaultPredictor(pretrained_path=pretrained_ckpt).to(device)
else:
    print(f"⚠️ 未找到预训练权重 {pretrained_ckpt}，将使用默认 ImageNet 权重。")
    model = VaultPredictor().to(device)

criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=lr)

# ================= 训练循环 =================
print("开始全链路训练...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for batch in loader:
        # 解包数据并移至 GPU
        oct_img = batch['oct_img'].to(device)
        ubm_img = batch['ubm_img'].to(device)
        clinical_feats = batch['clinical_feats'].to(device)
        label = batch['label'].to(device)

        # 前向传播
        pred = model(oct_img, ubm_img, clinical_feats)
        
        # 计算 Loss (注意 squeeze 确保维度一致)
        loss = criterion(pred.squeeze(), label.squeeze())
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * oct_img.size(0)

    avg_loss = total_loss / len(dataset)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# ================= 保存结果 =================
ckpt_dir = "checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)
save_name = "multimodal_simulated_finetuned.pth"
torch.save(model.state_dict(), os.path.join(ckpt_dir, save_name))
print(f"✅ 训练完成，最终模型已保存至: {os.path.join(ckpt_dir, save_name)}")