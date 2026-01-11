# src/train.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---- Robust imports for both:
# 1) python -m src.train
# 2) python src/train.py
try:
    from .engine import train_one_epoch, validate, save_checkpoint
    from .dataset import CSVDataset
    from .transforms import get_transforms
    from .utils.seed import seed_everything
except Exception:
    # fallback when running as script
    from engine import train_one_epoch, validate, save_checkpoint
    from dataset import CSVDataset
    from transforms import get_transforms
    from utils.seed import seed_everything


def parse_args():
    p = argparse.ArgumentParser(description="Flower Classification Training")

    # Repo bạn đang dùng --model -> giữ nguyên
    p.add_argument("--model", type=str, default="cnn_baseline",
                   choices=["cnn_baseline", "cnn_transfer", "vit"],
                   help="Which model pipeline to train")

    # Alias để bạn chạy kiểu --model_name (không bắt buộc)
    p.add_argument("--model_name", type=str, default=None,
                   help="Alias of --model (optional). If provided, override --model")

    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)

    # ViT options (repo bạn có vit_name/freeeze_backbone)
    p.add_argument("--vit_name", type=str, default="vit_base_patch16_224",
                   help="timm ViT model name (when --model vit)")
    p.add_argument("--freeze_backbone", action="store_true",
                   help="Freeze backbone for transfer learning (when --model cnn_transfer)")

    # IMPORTANT for your pipeline: splits folder (train.csv/val.csv/test.csv)
    p.add_argument("--splits_dir", type=str, default="splits",
                   help="Folder containing train.csv/val.csv/test.csv")

    # Output dirs (sheet yêu cầu output theo model name)
    p.add_argument("--model_dir", type=str, default="models")
    p.add_argument("--output_dir", type=str, default="outputs")

    # speed
    p.add_argument("--num_workers", type=int, default=2)

    return p.parse_args()


class SimpleCNN(nn.Module):
    """CNN baseline ổn định, không hardcode flatten."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def build_model(args, num_classes: int) -> nn.Module:
    """
    Build model based on args.model
    - cnn_baseline: SimpleCNN
    - cnn_transfer: torchvision resnet18 pretrained
    - vit: timm ViT
    """
    if args.model == "cnn_baseline":
        return SimpleCNN(num_classes)

    if args.model == "cnn_transfer":
        from torchvision import models
        # weights API may differ across torchvision versions; this is safe-ish
        try:
            m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception:
            m = models.resnet18(pretrained=True)

        if args.freeze_backbone:
            for p in m.parameters():
                p.requires_grad = False

        # replace classifier
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)

        # ensure classifier trainable
        for p in m.fc.parameters():
            p.requires_grad = True

        return m

    if args.model ==
