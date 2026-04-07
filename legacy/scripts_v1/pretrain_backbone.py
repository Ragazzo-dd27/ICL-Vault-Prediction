import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from legacy.utils_v1.mcoa_dataset import MCOADataset


data_dir = r"D:\Code\ICL_Vault_Project\data\public_datasets\mcoa_oct"
save_path = os.path.join(PROJECT_ROOT, "legacy", "artifacts_v1", "checkpoints", "resnet18_mcoa_pretrained.pth")
batch_size = 16
lr = 1e-4
epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(data_dir):
    print(f"错误：找不到数据集路径 {data_dir}")
    raise SystemExit(1)

print(f"正在加载数据集 {data_dir}")
dataset = MCOADataset(data_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

print("正在初始化 ResNet18...")
model = models.resnet18(pretrained=True)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print(f"开始训练，设备: {device}")
for epoch in range(epochs):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    print(f"Epoch {epoch + 1}/{epochs} 正在运行...")

    for i, (imgs, labels) in enumerate(dataloader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += imgs.size(0)

        if (i + 1) % 100 == 0:
            print(f"  > Step {i + 1}, Current Loss: {loss.item():.4f}")

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    print(f"Epoch [{epoch + 1}/{epochs}] Avg Loss: {avg_loss:.4f} Acc: {acc:.4f}")

os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.state_dict(), save_path)
print(f"训练完成，ResNet18 主干模型权重已保存至: {save_path}")
