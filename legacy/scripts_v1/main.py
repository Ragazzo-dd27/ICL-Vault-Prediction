import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from legacy.models_v1.multimodal_net import VaultPredictor


ckpt_path = os.path.join(PROJECT_ROOT, "legacy", "artifacts_v1", "checkpoints", "resnet18_mcoa_pretrained.pth")

print(f"正在尝试加载预训练权重 {ckpt_path}")
model = VaultPredictor(pretrained_path=ckpt_path)

batch_size = 4
oct_imgs = torch.randn(batch_size, 3, 224, 224)
ubm_imgs = torch.randn(batch_size, 3, 224, 224)
clinical_data = torch.randn(batch_size, 10)

output = model(oct_imgs, ubm_imgs, clinical_data)
print("模型输出形状:", output.shape)

target = torch.randn(batch_size, 1)
loss_fn = torch.nn.MSELoss()
loss = loss_fn(output, target)
print("损失值:", loss.item())

loss.backward()
print("反向传播成功，梯度已计算。")
