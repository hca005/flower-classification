import timm
import torch.nn as nn

def build(num_classes: int, model_name: str = "vit_base_patch16_224", pretrained: bool = True):
    m = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return m
