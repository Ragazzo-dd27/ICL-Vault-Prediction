import os
import sys

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from legacy.models_v1.multimodal_net import VaultPredictor
from legacy.utils_v1.multimodal_dataset import SimulatedMultimodalDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 8
lr = 1e-4
epochs = 5

pretrained_ckpt = os.path.join(
    PROJECT_ROOT,
    "legacy",
    "artifacts_v1",
    "checkpoints",
    "resnet18_mcoa_pretrained.pth",
)

print("正在生成模拟多模态数据...")
dataset = SimulatedMultimodalDataset(length=1000)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

if os.path.exists(pretrained_ckpt):
    print(f"正在加载 MCOA 预训练权重: {pretrained_ckpt}")
    model = VaultPredictor(pretrained_path=pretrained_ckpt).to(device)
else:
    print(f"未找到预训练权重 {pretrained_ckpt}，将使用默认 ImageNet 权重。")
    model = VaultPredictor().to(device)

criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=lr)

print("开始全链路训练...")
for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for batch in loader:
        oct_img = batch["oct_img"].to(device)
        ubm_img = batch["ubm_img"].to(device)
        clinical_feats = batch["clinical_feats"].to(device)
        label = batch["label"].to(device)

        pred = model(oct_img, ubm_img, clinical_feats)
        loss = criterion(pred.squeeze(), label.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * oct_img.size(0)

    avg_loss = total_loss / len(dataset)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

ckpt_dir = os.path.join(PROJECT_ROOT, "legacy", "artifacts_v1", "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)
save_name = "multimodal_simulated_finetuned.pth"
save_path = os.path.join(ckpt_dir, save_name)
torch.save(model.state_dict(), save_path)
print(f"训练完成，最终模型已保存至: {save_path}")
