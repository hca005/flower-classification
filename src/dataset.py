import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    def __init__(self, csv_path, transform=None, label_to_idx=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        labels = sorted(self.df["label"].unique())
        if label_to_idx is None:
            self.label_to_idx = {l: i for i, l in enumerate(labels)}
        else:
            self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["path"]
        label_str = row["label"]
        y = self.label_to_idx[label_str]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, y
