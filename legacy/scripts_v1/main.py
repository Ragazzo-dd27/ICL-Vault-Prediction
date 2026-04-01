import torch
import torch
from models.multimodal_net import VaultPredictor

# 定义权重路径 (指向你昨晚跑出来的文件)
ckpt_path = 'checkpoints/resnet18_mcoa_pretrained.pth'

# 实例化模型时传入路径
print(f"正在尝试加载预训练权重: {ckpt_path}")
model = VaultPredictor(pretrained_path=ckpt_path)

# 生成假数据（模拟输入）
batch_size = 4

# 模拟 BatchSize=4 的 OCT 图像，[4, 3, 224, 224]
oct_imgs = torch.randn(batch_size, 3, 224, 224)

# 模拟 BatchSize=4 的 UBM 图像，[4, 3, 224, 224]
ubm_imgs = torch.randn(batch_size, 3, 224, 224)

# 模拟 10 项临床数值参数，[4, 10]
clinical_data = torch.randn(batch_size, 10)

# === 前向传播测试 ===
# 将三组模拟数据输入模型，得到输出
output = model(oct_imgs, ubm_imgs, clinical_data)

# 打印输出张量的形状，应该为 [4, 1]
print("模型输出形状:", output.shape)

# === 损失函数与反向传播示例 ===
# 构造一个模拟的真实标签（回归目标），与输出形状相同
target = torch.randn(batch_size, 1)

# 使用 MSELoss 作为回归损失
loss_fn = torch.nn.MSELoss()

# 计算损失
loss = loss_fn(output, target)
print("损失值:", loss.item())

# 反向传播
loss.backward()
print("反向传播成功，梯度已计算。")