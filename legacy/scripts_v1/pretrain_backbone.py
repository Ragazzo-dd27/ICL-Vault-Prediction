import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import os
import sys

# 确保能找到 utils 模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.mcoa_dataset import MCOADataset

# ================= 预训练主干网络 (ResNet18) =================
# 本脚本用于在MCOA分类任务上预训练主干网络
# ============================================================

# 1. 超参数与配置
# --- 修改点：指向你的真实解压路径 ---
data_dir = r"D:\Code\ICL_Vault_Project\data\public_datasets\mcoa_oct" 
# ----------------------------------

save_path = 'checkpoints/resnet18_mcoa_pretrained.pth'
batch_size = 16
lr = 1e-4
epochs = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 检查路径是否存在，避免瞎跑
if not os.path.exists(data_dir):
    print(f"❌ 错误：找不到数据集路径: {data_dir}")
    print("请检查文件夹名是否正确，或者是否解压到了子文件夹里。")
    exit()

print(f"🚀 正在加载数据集: {data_dir}")

# 2. 准备数据集与加载器
# Windows下如果报错，请将 num_workers 改为 0
dataset = MCOADataset(data_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0) 

# 3. 初始化模型
print("正在初始化 ResNet18...")
model = models.resnet18(pretrained=True)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 2)  # 修改为2分类
model = model.to(device)

# 4. 损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 5. 训练循环
print(f"开始训练，设备: {device}")
for epoch in range(epochs):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    
    # 进度提示
    print(f"Epoch {epoch+1}/{epochs} 正在运行...")
    
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
        
        # 每100个batch打印一次，防止你也睡着了不知道进度
        if (i + 1) % 100 == 0:
            print(f"  > Step {i+1}, Current Loss: {loss.item():.4f}")

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    print(f"✅ Epoch [{epoch+1}/{epochs}]  Avg Loss: {avg_loss:.4f}  Acc: {acc:.4f}")

# 6. 保存权重
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.state_dict(), save_path)
print(f"🎉 训练完成！ResNet18主干模型权重已保存至: {save_path}")