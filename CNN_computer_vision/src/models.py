"""CNN and Vision Transformer backbones for 40-way face identification.
Các backbone CNN và Vision Transformer cho nhận dạng 40 người.
"""

from __future__ import annotations

import timm
import torch
from torch import nn
from torchvision import models as tvm

from src.config import N_PERSONS


class SimpleCNN(nn.Module):
    """Three conv blocks trained from scratch on 64x64 grayscale faces.
    Ba khối conv huấn luyện từ đầu trên ảnh mặt xám 64x64.
    """

    def __init__(self, num_classes: int = N_PERSONS) -> None:
        super().__init__()

        def block(c_in: int, c_out: int) -> nn.Sequential:
            # Conv-BN-ReLU twice, then halve resolution / Conv-BN-ReLU hai lần rồi giảm nửa độ phân giải
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.features(x)).flatten(1)
        return self.fc(self.dropout(x))


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    """Create a backbone and replace its head with a 40-way classifier.
    Tạo backbone và thay head bằng bộ phân loại 40 lớp.
    """
    if name == "SimpleCNN":
        return SimpleCNN()
    if name == "ResNet-18":
        model = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, N_PERSONS)
        return model
    if name == "MobileNetV3-S":
        weights = tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, N_PERSONS)
        return model
    if name == "DeiT-Tiny":
        return timm.create_model("deit_tiny_patch16_224", pretrained=pretrained, num_classes=N_PERSONS)
    raise ValueError(f"Unknown model: {name}")


def classifier_head(model: nn.Module, name: str) -> nn.Module:
    """Return the final linear layer; its input is the embedding.
    Trả về lớp tuyến tính cuối; input của nó chính là embedding.
    """
    if name in ("SimpleCNN", "ResNet-18"):
        return model.fc
    if name == "MobileNetV3-S":
        return model.classifier[3]
    return model.head


def gradcam_layer(model: nn.Module, name: str) -> nn.Module | None:
    """Last convolutional stage used for Grad-CAM (None for ViT).
    Tầng tích chập cuối dùng cho Grad-CAM (None với ViT).
    """
    if name == "SimpleCNN":
        return model.features[-1]
    if name == "ResNet-18":
        return model.layer4[-1]
    if name == "MobileNetV3-S":
        return model.features[-1]
    return None


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters / Số tham số huấn luyện được."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
