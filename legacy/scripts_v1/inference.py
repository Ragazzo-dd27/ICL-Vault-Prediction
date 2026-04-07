import os
import random
import sys

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from legacy.models_v1.unet import LightweightUNet
from legacy.utils_v1.dataset import KeratitisDataset


plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = LightweightUNet()
ckpt_path = os.path.join(PROJECT_ROOT, "legacy", "artifacts_v1", "checkpoints", "unet_keratitis.pth")
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.to(device)
model.eval()

dataset = KeratitisDataset(resize=(256, 256))
indices = random.sample(range(len(dataset)), 3)
samples = [dataset[i] for i in indices]

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
titles = ["原始图像", "真实标签(Ground Truth)", "模型预测(Prediction)"]

for row, sample in enumerate(samples):
    img, mask = sample
    input_img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_img)
        pred_mask = torch.sigmoid(logits)
        pred_mask = (pred_mask > 0.5).float()

    img_np = img.squeeze().cpu().numpy()
    mask_np = mask.squeeze().cpu().numpy()
    pred_np = pred_mask.squeeze().cpu().numpy()

    axes[row, 0].imshow(img_np, cmap="gray")
    axes[row, 0].set_title(titles[0])
    axes[row, 0].axis("off")

    axes[row, 1].imshow(mask_np, cmap="gray")
    axes[row, 1].set_title(titles[1])
    axes[row, 1].axis("off")

    axes[row, 2].imshow(pred_np, cmap="gray")
    axes[row, 2].set_title(titles[2])
    axes[row, 2].axis("off")

plt.tight_layout()
plt.show()
