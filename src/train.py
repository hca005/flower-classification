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

# Ensure src/ is importable when running from repo root
import sys
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from engine import train_one_epoch, validate, save_checkpoint
from dataset import CSVDataset
from transforms import get_transforms
from utils.seed import seed_everything


def parse_args():
    p = argparse.ArgumentParser(description="Flower Classification - Training Script")

    # Required by sheet: output per model name
    p.add_argument("--model_name", type=str, default="cnn_baseline")

    # Training hyperparams
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--img_size", type=int, default=224)

    # Data / splits
    p.add_argument("--splits_dir", type=str, default="splits")  # train.csv/val.csv/test.csv

    # Output directories
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--model_dir", type=str, default="models")

    # Reproducibility
    p.add_argument("--seed", type=int, default=42)

    # Speed / stability
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--use_amp", action="store_true", help="Use mixed precision on CUDA (faster).")

    return p.parse_args()


class SimpleCNN(nn.Module):
    """
    Baseline CNN robust to different img_size (no hard-coded flatten size).
    Members can replace this model with ViT/CNN/etc but keep the same training pipeline.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # stable output regardless of input size
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.gap(x)                 # (B, 64, 1, 1)
        x = x.view(x.size(0), -1)       # (B, 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    args = parse_args()

    # Assume you run from repo root (Colab: /content/flower-classification)
    repo_root = Path.cwd()

    # Output paths per model name (REQUIRED by sheet)
    model_path = (repo_root / args.model_dir / args.model_name).resolve()
    output_path = (repo_root / args.output_dir / args.model_name).resolve()
    model_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save run args for reproducibility (nice for demo)
    (output_path / "run_args.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # Seed + device
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Dataset + Dataloader (from CSV splits)
    train_tf, val_tf = get_transforms(img_size=args.img_size)

    splits_dir = (repo_root / args.splits_dir).resolve()
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"

    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(
            f"Missing splits CSV. Expected:\n- {train_csv}\n- {val_csv}\n"
            "Please generate splits first (src/split_data.py or src/split_data module)."
        )

    train_dataset = CSVDataset(str(train_csv), transform=train_tf)
    val_dataset = CSVDataset(
        str(val_csv),
        transform=val_tf,
        label_to_idx=train_dataset.label_to_idx
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    num_classes = len(train_dataset.label_to_idx)
    print("Num classes:", num_classes)

    # Model (swap here if needed)
    model = SimpleCNN(num_classes=num_classes).to(device)

    # Optimizer + loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Train loop + log
    best_acc = 0.0
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, use_amp=args.use_amp
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        print(
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save best checkpoint (REQUIRED)
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_acc=best_acc,
                path=str(model_path / "best.pt"),
                extra={
                    "model_name": args.model_name,
                    "num_classes": num_classes,
                    "label_to_idx": getattr(train_dataset, "label_to_idx", None),
                },
            )

    # Save history.csv (REQUIRED)
    df = pd.DataFrame(history)
    df.to_csv(output_path / "history.csv", index=False)

    # Save curves.png (REQUIRED)
    plt.figure(figsize=(8, 4))
    plt.plot(history["epoch"], history["train_loss"], label="train_loss")
    plt.plot(history["epoch"], history["val_loss"], label="val_loss")
    plt.plot(history["epoch"], history["train_acc"], label="train_acc")
    plt.plot(history["epoch"], history["val_acc"], label="val_acc")
    plt.legend()
    plt.title(f"{args.model_name} Training Curves")
    plt.savefig(output_path / "curves.png", dpi=150)
    plt.close()

    print("\nDONE")
    print("Best Val Acc:", best_acc)
    print("Saved best checkpoint:", model_path / "best.pt")
    print("Saved history:", output_path / "history.csv")
    print("Saved curves:", output_path / "curves.png")
    print("Saved run args:", output_path / "run_args.json")


if __name__ == "__main__":
    main()
