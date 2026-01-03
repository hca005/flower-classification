# Flower Classification (CNN + ViT)

Repo này xây pipeline tối thiểu cho bài Flower Classification:
- Chuẩn hoá cấu trúc data (train/val)
- Tạo file split CSV (train/val/test)
- Dataset + transforms + DataLoader (sanity check batch)
- (Tuỳ chọn) train model + vẽ loss/acc curves


## 0) Project Structure
flower-classification/
├─ configs/ # (tuỳ chọn) config yaml/json nếu muốn
├─ data/
│ └─ raw/
│ └─ flower_classification/
│ ├─ train/
│ │ ├─ class_1/
│ │ ├─ class_2/
│ │ └─ ...
│ └─ val/
│ ├─ class_1/
│ ├─ class_2/
│ └─ ...
├─ splits/ # CSV danh sách ảnh + label
│ ├─ train.csv
│ ├─ val.csv
│ └─ test.csv
├─ src/
│ ├─ init.py
│ ├─ sanity.py # test seed + root path
│ ├─ split_data.py # tạo splits/*.csv
│ ├─ transforms.py # resize/normalize/augment
│ ├─ dataset.py # Dataset đọc từ CSV
│ ├─ test_loader.py # sanity check dataloader batch
│ ├─ train.py # (tuỳ chọn) train
│ ├─ plot_curves.py # (tuỳ chọn) vẽ curve từ history.csv
│ └─ utils/
│ ├─ seed.py # seed_everything()
│ └─ paths.py # PROJECT_ROOT
├─ models/ # checkpoint (local)
├─ outputs/ # log/plot (local)
├─ requirements.txt
└─ .gitignore
## 1) Setup môi trường (Windows PowerShell)

Tại root repo:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

