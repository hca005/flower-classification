# src/train.py
# -----------------------
# Main training script
# -----------------------

import sys
import os

# -----------------------
# Fix ModuleNotFoundError: thêm folder src vào sys.path
# Khi chạy "python src/train.py", Python tìm module từ root folder
# sys.path.insert(0, ...) đảm bảo engine.py, dataset.py, transforms.py được tìm thấy
# -----------------------
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import argparse
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

# -----------------------
# Import modules trong src/
# -----------------------
from engine import train_one_epoch, validate, save_checkpoint
from dataset import CSVDataset
from transforms import get_transforms
from utils.seed_everything import seed_everything

# -----------------------
# ARGUMENTS
# -----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="cnn_baseline")
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--data_dir", type=str, default="../data")
parser.add_argument("--output_dir", type=str, default="../outputs")
parser.add_argument("--model_dir", type=str, default="../models")
args = parser.parse_args()

# -----------------------
# Set seed
# -----------------------
seed_everything(42)

# -----------------------
# Device
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------
# Dataset + Dataloader
# -----------------------
train_tf, val_tf = get_transforms(img_size=args.img_size)

train_dataset = CSVDataset(os.path.join(args.data_dir, "train.csv"), transform=train_tf)
val_dataset = CSVDataset(os.path.join(args.data_dir, "val.csv"), transform=val_tf,
                         label_to_idx=train_dataset.label_to_idx)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

num_classes = len(train_dataset.label_to_idx)
print(f"Num classes: {num_classes}")

# -----------------------
# Simple CNN model
# -----------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64*56*56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN(num_classes=num_classes).to(device)

# -----------------------
# Optimizer + Loss
# -----------------------
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

# -----------------------
# Output paths
# -----------------------
model_path = os.path.join(args.model_dir, args.model_name)
os.makedirs(model_path, exist_ok=True)
output_path = os.path.join(args.output_dir, args.model_name)
os.makedirs(output_path, exist_ok=True)

# -----------------------
# Training loop
# -----------------------
best_acc = 0.0
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

for epoch in range(1, args.epochs+1):
    print(f"\nEpoch {epoch}/{args.epochs}")
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    # Save best checkpoint
    if val_acc > best_acc:
        best_acc = val_acc
        save_checkpoint(model, optimizer, epoch, os.path.join(model_path, "best.pt"))

# -----------------------
# Save history + plot curves
# -----------------------
df = pd.DataFrame(history)
df.to_csv(os.path.join(output_path, "history.csv"), index=False)

plt.figure(figsize=(8,4))
plt.plot(history["train_loss"], label="train_loss")
plt.plot(history["val_loss"], label="val_loss")
plt.plot(history["train_acc"], label="train_acc")
plt.plot(history["val_acc"], label="val_acc")
plt.legend()
plt.title(f"{args.model_name} Training Curves")
plt.savefig(os.path.join(output_path, "curves.png"))
plt.close()

print("\nTraining done. Best Val Acc:", best_acc)
