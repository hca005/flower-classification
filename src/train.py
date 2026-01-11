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

    if args.model == "vit":
        import timm
        m = timm.create_model(args.vit_name, pretrained=True, num_classes=num_classes)
        return m

    raise ValueError(f"Unknown model: {args.model}")


def main():
    args = parse_args()

    # Alias: if --model_name provided, override --model
    if args.model_name:
        args.model = args.model_name

    # Seed
    seed_everything(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    repo_root = Path.cwd()

    # Output paths per model name (SHEET REQUIREMENT)
    model_out_dir = (repo_root / args.model_dir / args.model).resolve()
    run_out_dir = (repo_root / args.output_dir / args.model).resolve()
    model_out_dir.mkdir(parents=True, exist_ok=True)
    run_out_dir.mkdir(parents=True, exist_ok=True)

    # Save args for demo/repro
    (run_out_dir / "run_args.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # Data
    splits_dir = (repo_root / args.splits_dir).resolve()
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"

    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(
            f"Missing splits CSVs.\nExpected:\n- {train_csv}\n- {val_csv}\n"
            f"Hint: generate with split_data.py into folder '{args.splits_dir}'."
        )

    train_tf, val_tf = get_transforms(img_size=args.img_size)

    train_ds = CSVDataset(str(train_csv), transform=train_tf)
    val_ds = CSVDataset(str(val_csv), transform=val_tf, label_to_idx=train_ds.label_to_idx)

    num_classes = len(train_ds.label_to_idx)
    print("Num classes:", num_classes)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Model
    model = build_model(args, num_classes=num_classes).to(device)

    # Optimizer: only trainable params (important for freeze_backbone)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Train loop
    best_acc = 0.0
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save best checkpoint (SHEET REQUIREMENT)
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                path=str(model_out_dir / "best.pt"),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_acc=best_acc,
                extra={
                    "model": args.model,
                    "num_classes": num_classes,
                    "label_to_idx": getattr(train_ds, "label_to_idx", None),
                }
            )

    # Save history.csv (SHEET REQUIREMENT)
    pd.DataFrame(history).to_csv(run_out_dir / "history.csv", index=False)

    # Save curves.png (SHEET REQUIREMENT)
    plt.figure(figsize=(8, 4))
    plt.plot(history["epoch"], history["train_loss"], label="train_loss")
    plt.plot(history["epoch"], history["val_loss"], label="val_loss")
    plt.plot(history["epoch"], history["train_acc"], label="train_acc")
    plt.plot(history["epoch"], history["val_acc"], label="val_acc")
    plt.legend()
    plt.title(f"{args.model} Training Curves")
    plt.savefig(run_out_dir / "curves.png", dpi=150)
    plt.close()

    print("\nDONE ✅ (Train engine)")
    print("Best Val Acc:", best_acc)
    print("Saved best checkpoint:", model_out_dir / "best.pt")
    print("Saved history:", run_out_dir / "history.csv")
    print("Saved curves:", run_out_dir / "curves.png")
    print("Saved run args:", run_out_dir / "run_args.json")


if __name__ == "__main__":
    main()
