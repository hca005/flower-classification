from torch.utils.data import DataLoader
from src.dataset import CSVDataset
from src.transforms import get_transforms
import pandas as pd


def get_test_loader(
    csv_path="splits/test.csv",
    batch_size=16,
    num_workers=0
):
    # Transform cho evaluation
    _, eval_tf = get_transforms(img_size=224)

    # Đọc CSV để tạo label mapping nhất quán
    df = pd.read_csv(csv_path)
    labels = sorted(df["label"].unique())
    label_to_idx = {l: i for i, l in enumerate(labels)}

    # Dataset
    test_ds = CSVDataset(
        csv_path=csv_path,
        transform=eval_tf,
        label_to_idx=label_to_idx
    )

    # DataLoader
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return test_loader
