import torch
import torch.nn as nn
from torchvision import models

def build(num_classes):
    # EfficientNet-B0 là bản cân bằng nhất cho bài toán vừa và nhỏ
    model = models.efficientnet_b0(weights='DEFAULT')
    
    # Thay đổi head cuối cùng
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model

if __name__ == "__main__":
    model = build(num_classes=10)
    print("CNN Transfer (EfficientNet-B0) initialized.")