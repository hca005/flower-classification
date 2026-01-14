import os
import csv
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from src.test_loader import get_test_loader
from src.utils.visualization import plot_confusion_matrix
from src.utils.metrics import compute_metrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# MODEL CONFIG
# =====================================================
MODELS = {
    "cnn_baseline": {
        "ckpt": "models/cnn_baseline/best.pt"
    },
    "cnn_transfer": {
        "ckpt": "models/cnn_transfer/best.pt"
    },
    "vit": {
        "ckpt": "models/vit_timm/best.pt"
    }
}

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================
# BUILD MODEL
# =====================================================
def get_model(model_name, num_classes):
    if model_name == "cnn_baseline":
        from models.cnn_baseline.model import build
        return build(num_classes)

    if model_name == "cnn_transfer":
        from models.cnn_transfer.model import build
        return build(num_classes)

    if model_name == "vit":
        from models.vit_timm.model import build
        return build(num_classes)

    raise ValueError(f"Unknown model: {model_name}")


# =====================================================
# EVALUATION LOOP
# =====================================================
@torch.no_grad()
def evaluate_model(model_name, model, loader):
    model.eval()
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc=f"Evaluating {model_name}"):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    return metrics, cm


# =====================================================
# MAIN
# =====================================================
def main():
    # 🔹 READ CLASS NAMES
    class_file = "data/raw/flower_classification/classname.txt"
    with open(class_file, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines()]

    num_classes = len(class_names)

    # 🔹 TEST LOADER
    test_loader = get_test_loader(
        csv_path="splits/test.csv",
        batch_size=32,
        num_workers=0
    )

    metrics_rows = []

    # =================================================
    # LOOP MODELS (SAFE)
    # =================================================
    for model_name, cfg in MODELS.items():
        print(f"\n=== Evaluating {model_name} ===")

        try:
            model = get_model(model_name, num_classes).to(DEVICE)

            state_dict = torch.load(cfg["ckpt"], map_location=DEVICE)
            model.load_state_dict(state_dict)
            model.eval()

            metrics, cm = evaluate_model(model_name, model, test_loader)

            plot_confusion_matrix(
                cm,
                class_names,
                os.path.join(OUTPUT_DIR, f"confusion_matrix_{model_name}.png")
            )

            metrics["model"] = model_name
            metrics_rows.append(metrics)

        except Exception as e:
            print(f"[ERROR] {model_name}: {e}")
            continue

    # =================================================
    # SAVE METRICS CSV
    # =================================================
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "accuracy", "f1_macro", "f1_weighted"]
        )
        writer.writeheader()
        writer.writerows(metrics_rows)

    print("\n✅ DONE Evaluation")


if __name__ == "__main__":
    main()
