import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    hist = pd.read_csv(args.history)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # loss
    plt.figure()
    plt.plot(hist["epoch"], hist["train_loss"], label="train_loss")
    plt.plot(hist["epoch"], hist["val_loss"], label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
    plt.savefig(out_dir / "loss_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    # acc
    plt.figure()
    plt.plot(hist["epoch"], hist["train_acc"], label="train_acc")
    plt.plot(hist["epoch"], hist["val_acc"], label="val_acc")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend()
    plt.savefig(out_dir / "acc_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved curves to:", out_dir)

if __name__ == "__main__":
    main()
