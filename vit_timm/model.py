import torch
import torch.nn as nn
import timm

def build(num_classes):
    # Sử dụng bản vit_tiny để train nhanh và tránh tràn RAM/GPU
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    
    # Thay đổi head cho khớp với số lớp đầu ra
    model.head = nn.Linear(model.head.in_features, num_classes)
    
    return model

if __name__ == "__main__":
    model = build(num_classes=10)
    print("ViT (Vision Transformer) initialized via timm.")