import argparse, time, csv
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import CSVDataset
from src.transforms import get_transforms
from src.utils.seed import seed_everything

def get_model(model_key, num_classes, freeze_backbone, vit_name):
    if model_key == "cnn_baseline":
        from models.cnn_baseline.model import build
        return build(num_classes, freeze_backbone)

    if model_key == "cnn_transfer":
        from models.cnn_transfer.model import build
        return build(num_classes, freeze_backbone)

    if model_key == "vit":
        from models.vit_timm.model import build
        return build(num_classes, vit_name, True)

    raise ValueError(model_key)

def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--vit_name", default="vit_base_patch16_224")
    args = ap.parse_args()

    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_tf, eval_tf = get_transforms(224)
    train_ds = CSVDataset("splits/train.csv", train_tf)
    val_ds = CSVDataset("splits/val.csv", eval_tf, train_ds.label_to_idx)

    model = get_model(
        args.model,
        len(train_ds.label_to_idx),
        args.freeze_backbone,
        args.vit_name
    ).to(device)

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, args.batch_size)

    model_dir = Path("models") / args.model
    model_dir.mkdir(parents=True, exist_ok=True)

    best = 0.0
    for e in range(args.epochs):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()

        model.eval()
        acc = sum(accuracy(model(x.to(device)), y.to(device)) for x, y in val_dl) / len(val_dl)

        if acc > best:
            best = acc
            torch.save({"state_dict": model.state_dict()}, model_dir / "best.pt")

        print(f"[{args.model}] epoch {e+1} acc={acc:.4f}")

if __name__ == "__main__":
    main()
