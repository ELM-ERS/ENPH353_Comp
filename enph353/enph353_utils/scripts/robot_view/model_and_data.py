# model_and_data.py
import os
import csv
from typing import Tuple, List

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import torchvision.transforms as T


class CNNRegressor(nn.Module):
    """
    Lightweight CNN that maps an RGB image -> 2D vector.

    Uses global average pooling so the number of parameters does NOT depend
    on the input image resolution. The `input_size` arg is kept for API
    compatibility but is not used internally.
    """

    def __init__(self, num_outputs: int = 2, input_size: int = 680):
        super().__init__()

        # Feature extractor
        self.features = nn.Sequential(
            # Block 1: downsample early with stride=2
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # further /2
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Global average pooling → always [B, 128, 1, 1] regardless of HxW
        self.gap = nn.AdaptiveAvgPool2d((16, 16))

        # Work out flatten_dim programmatically so you can change blocks/input_size later
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            feat = self.features(dummy)
            feat = self.gap(feat)
            self.flatten_dim = feat.view(1, -1).size(1)  # ~= 128 * 8 * 8 = 8192

        # Dense head: big but not insane
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)  # [B, 128, H', W']
        x = self.gap(x)  # [B, 128, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 128]
        out = self.fc(x)  # [B, num_outputs]
        return out


class ImageToVectorDataset(Dataset):
    """
    Dataset that expects a layout like:

        root_dir/
          images/
            000001.png
            000002.png
            ...
          labels.csv

    labels.csv must contain: filename,x,y
    """

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "images")
        self.transform = transform

        labels_path = os.path.join(root_dir, "labels.csv")
        self.samples = []

        with open(labels_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row["filename"]
                x = float(row["x"])
                y = float(row["y"])
                img_path = os.path.join(self.img_dir, filename)
                self.samples.append(
                    (img_path, torch.tensor([x, y], dtype=torch.float32))
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, target = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, target


def build_transforms(image_size: int = 680):
    """
    Build train/val transforms that resize to the fixed model size.
    You can tune augmentations here.
    """
    train_transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            # Add augmentation if you want (only if valid for your driving task)
            # T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            # T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    val_transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    return train_transform, val_transform


def build_loaders(
    train_dirs: List[str],
    val_dirs: List[str],
    image_size: int = 680,
    batch_size: int = 32,
    num_workers: int = 4,
):
    """
    Given lists of train/val dataset directories, build PyTorch DataLoaders.

    Each directory in train_dirs / val_dirs should have the structure
    expected by ImageToVectorDataset.

    This lets you have multiple datasets with overlapping filenames,
    since each dataset lives in its own folder.
    """
    train_transform, val_transform = build_transforms(image_size=image_size)

    # Build one dataset per directory, then concatenate
    train_datasets = [
        ImageToVectorDataset(d, transform=train_transform) for d in train_dirs
    ]
    val_datasets = [ImageToVectorDataset(d, transform=val_transform) for d in val_dirs]

    train_dataset = (
        ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    )
    val_dataset = (
        ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader
