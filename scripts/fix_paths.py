import pandas as pd

for name in ["train", "val", "test"]:
    p = f"splits/{name}.csv"
    df = pd.read_csv(p)
    df["path"] = df["path"].astype(str).str.replace("\\", "/")
    df.to_csv(p, index=False)
    print("fixed", p, "sample:", df["path"].iloc[0])
