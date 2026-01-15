import argparse
from pathlib import Path
import time
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import CSVDataset
from src.transforms import get_transforms
from src.utils.seed import seed_everything

def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

def get_model(model_key: str, num_classes: int, freeze_backbone: bool, vit_name: str):
    if model_key == "cnn_baseline":
        from src.models.cnn_baseline import build
        return build(num_classes)
    if model_key == "cnn_transfer":
        from src.models.cnn_transfer import build
        return build(num_classes, freeze_backbone=freeze_backbone)
    if model_key == "vit":
        from src.models.vit import build
        return build(num_classes, model_name=vit_name, pretrained=True)
    raise ValueError(f"Unknown model: {model_key}")

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, y) * bs
        n += bs
    return total_loss / n, total_acc / n

def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, y) * bs
        n += bs
    return total_loss / n, total_acc / n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="cnn_baseline",
                    choices=["cnn_baseline", "cnn_transfer", "vit"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--vit_name", type=str, default="vit_base_patch16_224")
    args = ap.parse_args()

    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_tf, eval_tf = get_transforms(img_size=args.img_size)
    train_ds = CSVDataset("splits/train.csv", transform=train_tf)
    val_ds = CSVDataset("splits/val.csv", transform=eval_tf, label_to_idx=train_ds.label_to_idx)

    num_classes = len(train_ds.label_to_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = get_model(args.model, num_classes, args.freeze_backbone, args.vit_name).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # output dirs
    model_dir = Path("models") / args.model
    out_dir = Path("outputs") / args.model
    model_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    history_path = out_dir / "history.csv"
    best_path = model_dir / "best.pt"
    best_val_acc = -1.0

    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "seconds"])

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()

            tr_loss, tr_acc = train_one_epoch(model, train_loader, device, optimizer, criterion)
            va_loss, va_acc = evaluate(model, val_loader, device, criterion)

            sec = time.time() - t0
            writer.writerow([epoch, tr_loss, tr_acc, va_loss, va_acc, sec])
            f.flush()

            print(f"[{args.model}] epoch {epoch}/{args.epochs} | "
                  f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"val loss {va_loss:.4f} acc {va_acc:.4f} | {sec:.1f}s")

            if va_acc > best_val_acc:
                best_val_acc = va_acc
                torch.save(
                    {"model": args.model, "state_dict": model.state_dict(),
                     "label_to_idx": train_ds.label_to_idx, "img_size": args.img_size},
                    best_path
                )
                print(f"✅ Saved best to {best_path} (val_acc={best_val_acc:.4f})")

    print("Done. Best val acc:", best_val_acc)

if __name__ == "__main__":
    main()
