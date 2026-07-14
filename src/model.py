import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from src.config import NUM_CLASSES


def build_model(num_classes: int = NUM_CLASSES, freeze_backbone: bool = True) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def unfreeze_backbone(model: nn.Module) -> nn.Module:
    for param in model.parameters():
        param.requires_grad = True
    return model
