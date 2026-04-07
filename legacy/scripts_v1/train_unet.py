import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from legacy.models_v1.unet import LightweightUNet
from legacy.utils_v1.dataset import KeratitisDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

full_dataset = KeratitisDataset(resize=(256, 256))
data_len = len(full_dataset)
train_size = int(0.8 * data_len)
val_size = data_len - train_size
train_set, val_set = random_split(full_dataset, [train_size, val_size])

batch_size = 4
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

model = LightweightUNet().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for imgs, masks in train_loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    avg_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

save_path = os.path.join(PROJECT_ROOT, "legacy", "artifacts_v1", "checkpoints", "unet_keratitis.pth")
torch.save(model.state_dict(), save_path)
print(f"Training completed. Model weights saved as '{save_path}'.")
