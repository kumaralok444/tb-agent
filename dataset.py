from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch


TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

CLASS_NAMES = ["normal", "tb"]  # 0 = normal, 1 = tb


class TBDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def _load_samples(data_dir):
    samples = []
    data_dir = Path(data_dir)
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing folder: {class_dir}")
        for img_path in class_dir.glob("*"):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                samples.append((img_path, label))
    return samples


def get_dataloaders(data_dir, batch_size=32, val_split=0.2):
    all_samples = _load_samples(data_dir)
    n = len(all_samples)
    n_val = int(n * val_split)
    n_train = n - n_val

    # shuffle before splitting so val set is representative
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(n, generator=generator).tolist()
    train_samples = [all_samples[i] for i in indices[:n_train]]
    val_samples   = [all_samples[i] for i in indices[n_train:]]

    train_dataset = TBDataset(train_samples, transform=TRAIN_TRANSFORMS)
    val_dataset   = TBDataset(val_samples,   transform=EVAL_TRANSFORMS)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Dataset: {n} total | {n_train} train | {n_val} val")
    return train_loader, val_loader
