import os
from torch.utils.data import Dataset
import glob
from PIL import Image
from torchvision import transforms

class MCOADataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        # 支持常见图片格式且忽略大小写
        exts = ['jpg', 'jpeg', 'png']
        img_paths = []
        for ext in exts:
            img_paths.extend(glob.glob(os.path.join(root_dir, '**', f'*.{ext}'), recursive=True))
            img_paths.extend(glob.glob(os.path.join(root_dir, '**', f'*.{ext.upper()}'), recursive=True))
        self.img_paths = img_paths

        # 调试信息
        n_imgs = len(self.img_paths)
        print("============================================")
        print(f"在路径 [{root_dir}] 下共找到 [{n_imgs}] 张图片")
        print("============================================")

        if n_imgs == 0:
            raise RuntimeError(f"未在路径 [{root_dir}] 下找到任何图片文件，请检查路径是否正确，以及文件是否存在.")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        if 'Normal' in img_path:
            label = 0
        elif 'Opaque' in img_path:
            label = 1
        else:
            raise ValueError(f"Image path does not contain 'Normal' or 'Opaque': {img_path}")
        return img_tensor, label

if __name__ == "__main__":
    # 这里替换为你的数据集根目录
    root = '/path/to/mcoa'
    dataset = MCOADataset(root)
    print(f"Found {len(dataset)} images.")
    if len(dataset) > 0:
        img, label = dataset[0]
        print(f"Loaded image shape: {img.shape}, label: {label}")