import os
import csv
import json
import time
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

# torchvision / timm
from torchvision import models as tv_models
from torchvision import transforms as T
import timm

# sklearn metrics (for report + confusion matrix)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# -----------------------------
# Config
# -----------------------------
@dataclass
class TrainConfig:
    model: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    num_workers: int
    img_size: int
    device: str
    seed: int
    amp: bool
    grad_clip: float
    early_stop_patience: int
    scheduler: str  # none | cosine | step
    mixup: float    # 0.0 = off, else alpha for Beta distribution


# -----------------------------
# Utils
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_split_csv(csv_path: str) -> List[Tuple[str, str]]:
    """
    CSV format:
    path,label
    data\\raw\\...\\img.jpg,classname
    """
    items = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row["path"].strip()
            y = row["label"].strip()
            items.append((p, y))
    return items


def make_label_map(train_items: List[Tuple[str, str]]) -> Dict[str, int]:
    labels = sorted(list({y for _, y in train_items}))
    return {lab: i for i, lab in enumerate(labels)}


def invert_map(d: Dict[str, int]) -> Dict[int, str]:
    return {v: k for k, v in d.items()}


def normalize_path(p: str) -> str:
    return p.replace("/", os.sep).replace("\\", os.sep)


# -----------------------------
# Dataset
# -----------------------------
class CSVDataset(Dataset):
    def __init__(self, items: List[Tuple[str, str]], label2idx: Dict[str, int], root_dir: str, transform=None):
        self.items = items
        self.label2idx = label2idx
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rel_path, lab = self.items[idx]
        img_path = os.path.join(self.root_dir, normalize_path(rel_path))
        img = Image.open(img_path).convert("RGB")
        y = self.label2idx[lab]
        if self.transform is not None:
            img = self.transform(img)
        return img, y


# -----------------------------
# Transforms
# -----------------------------
def build_transforms(img_size: int, train: bool):
    # ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if train:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])


# -----------------------------
# Models (3 models + improve ViT)
# -----------------------------
def build_model(model_name: str, num_classes: int, img_size: int) -> nn.Module:
    name = model_name.lower().strip()

    if name == "cnn_baseline":
        # MobileNetV3 Small (pretrained)
        m = tv_models.mobilenet_v3_small(weights="DEFAULT")
        in_features = m.classifier[3].in_features
        m.classifier[3] = nn.Linear(in_features, num_classes)
        return m

    if name == "cnn_transfer":
        # EfficientNet-B0 (pretrained)
        m = tv_models.efficientnet_b0(weights="DEFAULT")
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, num_classes)
        return m

    if name == "vit_timm":
        # Improve ViT: use a better head (dropout) + optionally use pretrained
        # vit_tiny runs fast and fits CPU/GPU easier
        base = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=0)  # num_classes=0 => no head
        feat_dim = base.num_features

        head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(p=0.2),
            nn.Linear(feat_dim, num_classes),
        )
        base.reset_classifier(0)
        # timm model forward_features exists
        class ViTWrap(nn.Module):
            def __init__(self, backbone, head):
                super().__init__()
                self.backbone = backbone
                self.head = head

            def forward(self, x):
                feats = self.backbone.forward_features(x)
                # timm ViT outputs [B, tokens, C] or [B, C] depending; handle both
                if feats.dim() == 3:
                    # take cls token at index 0
                    feats = feats[:, 0]
                return self.head(feats)

        return ViTWrap(base, head)

    raise ValueError("Unknown model. Use: cnn_baseline | cnn_transfer | vit_timm")


# -----------------------------
# Mixup (optional)
# -----------------------------
def mixup_data(x, y, alpha: float):
    if alpha <= 0:
        return x, y, None, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# -----------------------------
# Train / Eval Engine
# -----------------------------
def train_one_epoch(model, loader, optimizer, device, scaler, cfg: TrainConfig):
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    y_true = []
    y_pred = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # mixup
        if cfg.mixup > 0:
            x, y_a, y_b, lam = mixup_data(x, y, cfg.mixup)

        with torch.cuda.amp.autocast(enabled=cfg.amp):
            logits = model(x)
            if cfg.mixup > 0:
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            else:
                loss = criterion(logits, y)

        if cfg.amp:
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        total_loss += loss.item() * x.size(0)

        # accuracy logging (if mixup, use original y for metric)
        pred = torch.argmax(logits, dim=1)
        y_true.append(y.detach().cpu().numpy())
        y_pred.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    acc = accuracy_score(y_true, y_pred)
    avg_loss = total_loss / max(len(loader.dataset), 1)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    y_true = []
    y_pred = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        pred = torch.argmax(logits, dim=1)

        y_true.append(y.detach().cpu().numpy())
        y_pred.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    avg_loss = total_loss / max(len(loader.dataset), 1)
    acc = accuracy_score(y_true, y_pred)
    return avg_loss, acc, y_true, y_pred


# -----------------------------
# Scheduler
# -----------------------------
def build_scheduler(scheduler_name: str, optimizer, epochs: int):
    name = scheduler_name.lower().strip()
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    raise ValueError("scheduler must be none | cosine | step")


# -----------------------------
# Plots / Reports
# -----------------------------
def save_history_csv(history: List[dict], out_path: str):
    ensure_dir(os.path.dirname(out_path))
    keys = list(history[0].keys()) if history else []
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in history:
            w.writerow(row)


def plot_curves(history: List[dict], out_loss_path: str, out_acc_path: str):
    epochs = [h["epoch"] for h in history]
    tr_loss = [h["train_loss"] for h in history]
    va_loss = [h["val_loss"] for h in history]
    tr_acc = [h["train_acc"] for h in history]
    va_acc = [h["val_acc"] for h in history]

    plt.figure()
    plt.plot(epochs, tr_loss, label="train_loss")
    plt.plot(epochs, va_loss, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.tight_layout()
    plt.savefig(out_loss_path)
    plt.close()

    plt.figure()
    plt.plot(epochs, tr_acc, label="train_acc")
    plt.plot(epochs, va_acc, label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.tight_layout()
    plt.savefig(out_acc_path)
    plt.close()


def save_confusion_matrix(cm: np.ndarray, labels: List[str], out_path: str, title: str = "Confusion Matrix"):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.savefig(out_path)
    plt.close()


# -----------------------------
# Checkpoint
# -----------------------------
def save_checkpoint(path: str, model: nn.Module, model_name: str, label2idx: Dict[str, int]):
    ckpt = {
        "model_name": model_name,
        "label2idx": label2idx,
        "state_dict": model.state_dict(),
    }
    ensure_dir(os.path.dirname(path))
    torch.save(ckpt, path)


def load_checkpoint(path: str, device: str):
    ckpt = torch.load(path, map_location=device)
    return ckpt


# -----------------------------
# Inference
# -----------------------------
@torch.no_grad()
def infer_one_image(model: nn.Module, img_path: str, tfm, device: str, idx2label: Dict[int, str], topk: int = 3):
    model.eval()
    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    logits = model(x)
    prob = torch.softmax(logits, dim=1).squeeze(0)
    topk = min(topk, prob.numel())
    vals, inds = torch.topk(prob, k=topk)

    result = []
    for v, i in zip(vals.detach().cpu().numpy(), inds.detach().cpu().numpy()):
        result.append((idx2label[int(i)], float(v)))
    return result


def list_images_in_dir(folder: str):
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    files = []
    for root, _, fnames in os.walk(folder):
        for fn in fnames:
            if fn.lower().endswith(exts):
                files.append(os.path.join(root, fn))
    return sorted(files)


# -----------------------------
# Commands
# -----------------------------
def cmd_train(args):
    cfg = TrainConfig(
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        img_size=args.img_size,
        device=args.device,
        seed=args.seed,
        amp=args.amp,
        grad_clip=args.grad_clip,
        early_stop_patience=args.early_stop,
        scheduler=args.scheduler,
        mixup=args.mixup,
    )

    set_seed(cfg.seed)

    project_root = os.getcwd()
    splits_dir = os.path.join(project_root, "splits")
    train_csv = os.path.join(splits_dir, "train.csv")
    val_csv = os.path.join(splits_dir, "val.csv")
    test_csv = os.path.join(splits_dir, "test.csv")

    train_items = read_split_csv(train_csv)
    val_items = read_split_csv(val_csv)
    test_items = read_split_csv(test_csv)

    label2idx = make_label_map(train_items)
    idx2label = invert_map(label2idx)
    num_classes = len(label2idx)

    print(f"[INFO] num_classes = {num_classes}")
    print(f"[INFO] device = {cfg.device}")
    print(f"[INFO] model = {cfg.model}")
    print(f"[INFO] amp = {cfg.amp} | scheduler = {cfg.scheduler} | mixup = {cfg.mixup}")

    tf_train = build_transforms(cfg.img_size, train=True)
    tf_eval = build_transforms(cfg.img_size, train=False)

    ds_train = CSVDataset(train_items, label2idx, root_dir=project_root, transform=tf_train)
    ds_val = CSVDataset(val_items, label2idx, root_dir=project_root, transform=tf_eval)
    ds_test = CSVDataset(test_items, label2idx, root_dir=project_root, transform=tf_eval)

    pin = (cfg.device.startswith("cuda"))
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=pin)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=pin)
    dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=pin)

    model = build_model(cfg.model, num_classes=num_classes, img_size=cfg.img_size).to(cfg.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = build_scheduler(cfg.scheduler, optimizer, cfg.epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    out_dir = os.path.join(project_root, "outputs", cfg.model)
    ckpt_dir = os.path.join(project_root, "models", cfg.model)
    ensure_dir(out_dir)
    ensure_dir(ckpt_dir)

    best_path = os.path.join(ckpt_dir, "best.pt")
    history_path = os.path.join(out_dir, "history.csv")
    loss_curve_path = os.path.join(out_dir, "loss_curve.png")
    acc_curve_path = os.path.join(out_dir, "acc_curve.png")
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    report_path = os.path.join(out_dir, "classification_report.txt")
    meta_path = os.path.join(out_dir, "meta.json")

    history: List[dict] = []
    best_val_acc = -1.0
    best_epoch = -1
    bad_epochs = 0

    start_time = time.time()

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, dl_train, optimizer, cfg.device, scaler, cfg)
        va_loss, va_acc, _, _ = evaluate(model, dl_val, cfg.device)

        lr_now = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_acc": va_acc,
        }
        history.append(row)

        dt = time.time() - t0
        print(
            f"[E{epoch:02d}/{cfg.epochs}] "
            f"lr={lr_now:.2e} | "
            f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
            f"va_loss={va_loss:.4f} va_acc={va_acc:.4f} | "
            f"time={dt:.1f}s"
        )

        # scheduler step
        if scheduler is not None:
            scheduler.step()

        # save best
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(best_path, model, cfg.model, label2idx)
        else:
            bad_epochs += 1

        # early stop
        if cfg.early_stop_patience > 0 and bad_epochs >= cfg.early_stop_patience:
            print(f"[EARLY STOP] No improvement for {cfg.early_stop_patience} epochs.")
            break

    # Save history + curves
    save_history_csv(history, history_path)
    plot_curves(history, loss_curve_path, acc_curve_path)

    # Save meta
    meta = {
        "model": cfg.model,
        "num_classes": num_classes,
        "img_size": cfg.img_size,
        "best_val_acc": float(best_val_acc),
        "best_epoch": int(best_epoch),
        "best_ckpt": os.path.relpath(best_path, project_root),
        "outputs_dir": os.path.relpath(out_dir, project_root),
        "train_time_sec": float(time.time() - start_time),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[SAVED] {os.path.relpath(history_path, project_root)}")
    print(f"[SAVED] {os.path.relpath(loss_curve_path, project_root)}")
    print(f"[SAVED] {os.path.relpath(acc_curve_path, project_root)}")
    print(f"[SAVED] {os.path.relpath(best_path, project_root)}")

    # Final test evaluation with best
    if os.path.exists(best_path):
        ckpt = load_checkpoint(best_path, cfg.device)
        label2idx = ckpt["label2idx"]
        idx2label = invert_map(label2idx)
        model = build_model(cfg.model, num_classes=len(label2idx), img_size=cfg.img_size).to(cfg.device)
        model.load_state_dict(ckpt["state_dict"])

        te_loss, te_acc, y_true, y_pred = evaluate(model, dl_test, cfg.device)
        print(f"[TEST] loss={te_loss:.4f} acc={te_acc:.4f}")

        labels_sorted = [idx2label[i] for i in range(len(idx2label))]
        rep = classification_report(y_true, y_pred, target_names=labels_sorted, digits=4)
        cm = confusion_matrix(y_true, y_pred)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(rep)

        save_confusion_matrix(cm, labels_sorted, cm_path, title=f"Confusion Matrix ({cfg.model})")

        print(f"[SAVED] {os.path.relpath(report_path, project_root)}")
        print(f"[SAVED] {os.path.relpath(cm_path, project_root)}")


def cmd_eval(args):
    project_root = os.getcwd()
    ckpt_path = args.ckpt
    device = args.device

    ckpt = load_checkpoint(ckpt_path, device)
    label2idx = ckpt["label2idx"]
    idx2label = invert_map(label2idx)

    splits_dir = os.path.join(project_root, "splits")
    test_csv = os.path.join(splits_dir, "test.csv")
    test_items = read_split_csv(test_csv)

    tf_eval = build_transforms(args.img_size, train=False)
    ds_test = CSVDataset(test_items, label2idx, root_dir=project_root, transform=tf_eval)

    pin = device.startswith("cuda")
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    model_name = ckpt.get("model_name", args.model)
    model = build_model(model_name, num_classes=len(label2idx), img_size=args.img_size).to(device)
    model.load_state_dict(ckpt["state_dict"])

    te_loss, te_acc, y_true, y_pred = evaluate(model, dl_test, device)
    print(f"[EVAL] model={model_name} test_loss={te_loss:.4f} test_acc={te_acc:.4f}")

    labels_sorted = [idx2label[i] for i in range(len(idx2label))]
    rep = classification_report(y_true, y_pred, target_names=labels_sorted, digits=4)
    print(rep)

    if args.save_dir:
        out_dir = os.path.join(project_root, args.save_dir)
        ensure_dir(out_dir)

        report_path = os.path.join(out_dir, "classification_report.txt")
        cm_path = os.path.join(out_dir, "confusion_matrix.png")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(rep)

        cm = confusion_matrix(y_true, y_pred)
        save_confusion_matrix(cm, labels_sorted, cm_path, title=f"Confusion Matrix ({model_name})")

        print(f"[SAVED] {os.path.relpath(report_path, project_root)}")
        print(f"[SAVED] {os.path.relpath(cm_path, project_root)}")


def cmd_infer(args):
    project_root = os.getcwd()
    device = args.device

    ckpt = load_checkpoint(args.ckpt, device)
    label2idx = ckpt["label2idx"]
    idx2label = invert_map(label2idx)
    model_name = ckpt.get("model_name", args.model)

    model = build_model(model_name, num_classes=len(label2idx), img_size=args.img_size).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tfm = build_transforms(args.img_size, train=False)

    # single image
    if args.image:
        preds = infer_one_image(model, args.image, tfm, device, idx2label, topk=args.topk)
        print(f"[IMAGE] {args.image}")
        for lab, p in preds:
            print(f"  - {lab}: {p:.4f}")
        return

    # folder
    if args.folder:
        imgs = list_images_in_dir(args.folder)
        if not imgs:
            print("[INFER] No images found in folder.")
            return

        out_json = []
        for p in imgs:
            preds = infer_one_image(model, p, tfm, device, idx2label, topk=args.topk)
            out_json.append({
                "image": p,
                "preds": [{"label": lab, "prob": prob} for lab, prob in preds]
            })
            # print short
            top1 = preds[0]
            print(f"{os.path.basename(p)} -> {top1[0]} ({top1[1]:.4f})")

        if args.out:
            ensure_dir(os.path.dirname(args.out) if os.path.dirname(args.out) else ".")
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(out_json, f, indent=2, ensure_ascii=False)
            print(f"[SAVED] {args.out}")
        return

    print("Provide --image <path> or --folder <path>")


# -----------------------------
# CLI
# -----------------------------
def build_argparser():
    p = argparse.ArgumentParser("Flower Classification - All-in-one train/eval/infer")
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    p_train = sub.add_parser("train")
    p_train.add_argument("--model", type=str, default="cnn_baseline", choices=["cnn_baseline", "cnn_transfer", "vit_timm"])
    p_train.add_argument("--epochs", type=int, default=5)
    p_train.add_argument("--batch_size", type=int, default=16)
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--weight_decay", type=float, default=1e-4)
    p_train.add_argument("--num_workers", type=int, default=2)
    p_train.add_argument("--img_size", type=int, default=224)
    p_train.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--amp", action="store_true")
    p_train.add_argument("--grad_clip", type=float, default=1.0)
    p_train.add_argument("--early_stop", type=int, default=3)  # patience
    p_train.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine", "step"])
    p_train.add_argument("--mixup", type=float, default=0.0)

    # eval
    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--ckpt", type=str, required=True)
    p_eval.add_argument("--model", type=str, default="cnn_baseline")
    p_eval.add_argument("--batch_size", type=int, default=32)
    p_eval.add_argument("--num_workers", type=int, default=2)
    p_eval.add_argument("--img_size", type=int, default=224)
    p_eval.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p_eval.add_argument("--save_dir", type=str, default="")

    # infer
    p_inf = sub.add_parser("infer")
    p_inf.add_argument("--ckpt", type=str, required=True)
    p_inf.add_argument("--model", type=str, default="cnn_baseline")
    p_inf.add_argument("--img_size", type=int, default=224)
    p_inf.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p_inf.add_argument("--topk", type=int, default=3)
    p_inf.add_argument("--image", type=str, default="")
    p_inf.add_argument("--folder", type=str, default="")
    p_inf.add_argument("--out", type=str, default="")

    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()

    if args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    elif args.cmd == "infer":
        cmd_infer(args)
    else:
        raise ValueError("Unknown command")


if __name__ == "__main__":
    main()
