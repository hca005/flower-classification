# src/train.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Make src importable regardless of where you run
import sys
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from engine import train_one_epoch, validate, save_checkpoint
from dataset import CSVDataset
from transforms import get_transforms

# (Keep ONLY ONE seed function in your repo)
from utils.seed import seed_everything


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="cnn_baseline")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--splits_dir", type=str, default="splits")   # where train.csv/val.csv/test.csv
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--model_dir", type=str, default="models")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class SimpleCNN(nn.Module):
    """Baseline CNN that works with any img_size (no hard-coded flatten size)."""
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
        x = self.fc2(x)
        return x


def main():
    args = parse_args()
    repo_root = Path.cwd()  # assume you run from repo root in Colab/terminal

    # Output folders per model_name (REQUIRED)
    model_path = (repo_root / args.model_dir / args.model_name)
    output_path = (repo_root / args.output_dir / args.model_name)
    model_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    # Seed + device
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Dataset + Dataloader (from CSV)
    train_tf, val_tf = get_transforms(img_size=args.img_size)
    splits_dir = repo_root / args.splits_dir

    train_dataset = CSVDataset(str(splits_dir / "train.csv"), transform=train_tf)
    val_dataset = CSVDataset(
        str(splits_dir / "val.csv"),
        transform=val_tf,
        label_to_idx=train_dataset.label_to_idx
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    num_classes = len(train_dataset.label_to_idx)
    print("Num classes:", num_classes)

    # Model (team can swap their own model here)
    model = SimpleCNN(num_classes=num_classes).to(device)

    # Optimizer + loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Train loop + log
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save best checkpoint (REQUIRED)
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, optimizer, epoch, str(model_path / "best.pt"))

    # Save history.csv (REQUIRED)
    pd.DataFrame(history).to_csv(output_path / "history.csv", index=False)

    # Save curves.png (REQUIRED)
    plt.figure(figsize=(8, 4))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"], label="val_acc")
    plt.legend()
    plt.title(f"{args.model_name} Training Curves")
    plt.savefig(output_path / "curves.png", dpi=150)
    plt.close()

    print("\nDONE ")
    print("Best Val Acc:", best_acc)
    print("Saved:", model_path / "best.pt")
    print("Saved:", output_path / "history.csv")
    print("Saved:", output_path / "curves.png")


if __name__ == "__main__":
    main()
