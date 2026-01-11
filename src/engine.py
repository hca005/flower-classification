# src/engine.py
from __future__ import annotations
from typing import Tuple, Dict
import torch
from tqdm import tqdm


def _to_device(batch, device):
    x, y = batch
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def train_one_epoch(model, loader, optimizer, criterion, device, use_amp: bool = True) -> Tuple[float, float]:
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        x, y = _to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += bs

        pbar.set_postfix(loss=loss.item())

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def validate(model, loader, criterion, device) -> Tuple[float, float]:
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    pbar = tqdm(loader, desc="Val", leave=False)
    for batch in pbar:
        x, y = _to_device(batch, device)
        logits = model(x)
        loss = criterion(logits, y)

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += bs

        pbar.set_postfix(loss=loss.item())

    return total_loss / total_samples, total_correct / total_samples


def save_checkpoint(model, optimizer, epoch: int, best_acc: float, path: str, extra: Dict | None = None):
    ckpt = {
        "epoch": epoch,
        "best_acc": best_acc,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)
