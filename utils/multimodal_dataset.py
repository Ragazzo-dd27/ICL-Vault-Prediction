import torch
from torch.utils.data import Dataset, DataLoader

class SimulatedMultimodalDataset(Dataset):
    def __init__(self, length=1000):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        oct_img = torch.randn(3, 224, 224)
        ubm_img = torch.randn(3, 224, 224)
        clinical_feats = torch.randn(10)
        label = torch.randint(0, 1001, (1,), dtype=torch.float32)
        return {
            'oct_img': oct_img,
            'ubm_img': ubm_img,
            'clinical_feats': clinical_feats,
            'label': label
        }

if __name__ == '__main__':
    dataset = SimulatedMultimodalDataset()
    loader = DataLoader(dataset, batch_size=4)
    batch = next(iter(loader))
    for k, v in batch.items():
        print(f"{k}: shape {v.shape}")