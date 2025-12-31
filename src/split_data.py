import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def scan_dir(split_dir: Path) -> pd.DataFrame:
    rows = []
    for class_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
        label = class_dir.name
        for p in class_dir.rglob("*"):
            if p.suffix.lower() in IMG_EXTS:
                rows.append({"path": str(p), "label": label})
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, required=True,
                    help="Folder containing train/ and val/ subfolders")
    ap.add_argument("--out_dir", type=str, default="splits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_from_train", type=float, default=0.1,
                    help="Take this fraction from TRAIN to create validation CSV")
    ap.add_argument("--classes", type=str, default="",
                    help="Optional: comma-separated subset of classes")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    train_dir = root / "train"
    test_dir = root / "val"   # dùng val sẵn làm TEST

    df_train_all = scan_dir(train_dir)
    df_test = scan_dir(test_dir)

    if df_train_all.empty or df_test.empty:
        raise ValueError("No images found. Check dataset_root path.")

    if args.classes.strip():
        keep = set([c.strip() for c in args.classes.split(",") if c.strip()])
        df_train_all = df_train_all[df_train_all["label"].isin(keep)].reset_index(drop=True)
        df_test = df_test[df_test["label"].isin(keep)].reset_index(drop=True)

    train_df, val_df = train_test_split(
        df_train_all,
        test_size=args.val_from_train,
        random_state=args.seed,
        shuffle=True,
        stratify=df_train_all["label"],
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    df_test.to_csv(out_dir / "test.csv", index=False)

    print("✅ Split CSV written:")
    print("train:", len(train_df))
    print("val:", len(val_df))
    print("test:", len(df_test))
    print("num_classes:", train_df["label"].nunique())

if __name__ == "__main__":
    main()
