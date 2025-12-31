from torch.utils.data import DataLoader
import torch

from src.dataset import CSVDataset
from src.transforms import get_transforms

def main():
    train_tf, eval_tf = get_transforms(img_size=224)

    train_ds = CSVDataset("splits/train.csv", transform=train_tf)
    val_ds   = CSVDataset("splits/val.csv", transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    x, y = next(iter(train_loader))

    print("Batch OK")
    print("x shape:", tuple(x.shape))  # (B, 3, 224, 224)
    print("y shape:", tuple(y.shape))  # (B,)
    print("x dtype:", x.dtype, "y dtype:", y.dtype)
    print("num classes:", len(train_ds.label_to_idx))

    assert x.shape[1:] == (3, 224, 224)

if __name__ == "__main__":
    main()
