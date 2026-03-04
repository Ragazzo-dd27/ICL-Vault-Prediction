import matplotlib.pyplot as plt
import numpy as np

# === 修复中文显示乱码 ===
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False    

# === 从日志中提取的真实数据 ===
epochs = np.arange(1, 6)
# 训练集 Loss
losses =[0.0577, 0.0153, 0.0070, 0.0139, 0.0140]
# 训练集 Accuracy
accuracies =[0.9803, 0.9949, 0.9974, 0.9952, 0.9956]

# === 开始绘图 ===
fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)

# 绘制 Loss 曲线 (红色, 左轴)
color1 = 'tab:red'
ax1.set_xlabel('Epochs', fontsize=12)
ax1.set_ylabel('Training Loss', color=color1, fontsize=12)
line1 = ax1.plot(epochs, losses, color=color1, marker='o', linewidth=2, label='Loss')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xticks(epochs)
ax1.grid(True, linestyle='--', alpha=0.6)

# 实例化共享x轴的第二个y轴
ax2 = ax1.twinx()  

# 绘制 Accuracy 曲线 (蓝色, 右轴)
color2 = 'tab:blue'
ax2.set_ylabel('Accuracy', color=color2, fontsize=12)
line2 = ax2.plot(epochs, accuracies, color=color2, marker='s', linewidth=2, label='Accuracy')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0.97, 1.0) # 设置y轴范围让曲线更好看

# 合并图例
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right', fontsize=11)

plt.title('MCOA Dataset Pre-training: Loss & Accuracy', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()

# 保存高清图片
plt.savefig('mcoa_pretrain_metrics.png')
print("✅ 图表已保存为 mcoa_pretrain_metrics.png")
plt.show()