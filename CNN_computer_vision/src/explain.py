"""Feature maps, Grad-CAM, and ViT attention rollout.
Feature map, Grad-CAM và attention rollout của ViT.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@torch.no_grad()
def resnet_feature_maps(model: nn.Module, image: torch.Tensor) -> dict[str, np.ndarray]:
    """Activations after the stem and each ResNet stage for one image (1,C,H,W).
    Kích hoạt sau stem và từng stage của ResNet cho một ảnh (1,C,H,W).
    """
    model.eval()
    x = model.maxpool(model.relu(model.bn1(model.conv1(image))))
    maps = {"stem": x}
    for stage in ("layer1", "layer2", "layer3", "layer4"):
        x = getattr(model, stage)(x)
        maps[stage] = x
    return {k: v[0].cpu().numpy() for k, v in maps.items()}


def grad_cam(model: nn.Module, layer: nn.Module, image: torch.Tensor, class_idx: int) -> np.ndarray:
    """Grad-CAM heatmap in [0, 1] at the input resolution.
    Bản đồ nhiệt Grad-CAM trong [0, 1] ở độ phân giải input.
    """
    model.eval()
    store: dict[str, torch.Tensor] = {}

    def capture(_m: nn.Module, _i: tuple, out: torch.Tensor) -> None:
        # Tensor hook avoids conflicts with in-place activations /
        # Hook trên tensor tránh xung đột với activation in-place
        store["act"] = out
        out.register_hook(lambda grad: store.__setitem__("grad", grad))

    fwd = layer.register_forward_hook(capture)
    image = image.clone().requires_grad_(True)
    logits = model(image)
    model.zero_grad(set_to_none=True)
    logits[0, class_idx].backward()
    fwd.remove()
    # Channel weights = spatially averaged gradients / Trọng số kênh = gradient lấy trung bình không gian
    weights = store["grad"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * store["act"]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


@torch.no_grad()
def attention_rollout(model: nn.Module, image: torch.Tensor) -> np.ndarray:
    """Attention rollout (Abnar & Zuidema, 2020) from the CLS token to patches.
    Attention rollout (Abnar & Zuidema, 2020) từ token CLS tới các patch.
    """
    model.eval()
    attentions: list[torch.Tensor] = []
    hooks = []
    for block in model.blocks:
        # Disable fused SDPA so attention probabilities pass through attn_drop /
        # Tắt SDPA gộp để xác suất attention đi qua attn_drop và bắt được bằng hook
        block.attn.fused_attn = False
        hooks.append(block.attn.attn_drop.register_forward_hook(
            lambda _m, _i, out: attentions.append(out.detach().cpu())
        ))
    model(image)
    for hook in hooks:
        hook.remove()
    for block in model.blocks:
        block.attn.fused_attn = True

    n_tokens = attentions[0].shape[-1]
    rollout = torch.eye(n_tokens)
    for attn in attentions:
        # Average heads, add residual identity, renormalize rows /
        # Trung bình các head, cộng identity của residual, chuẩn hoá lại theo hàng
        a = attn[0].mean(dim=0) + torch.eye(n_tokens)
        a = a / a.sum(dim=-1, keepdim=True)
        rollout = a @ rollout
    prefix = getattr(model, "num_prefix_tokens", 1)
    cls_to_patch = rollout[0, prefix:]
    side = int(cls_to_patch.numel() ** 0.5)
    heat = cls_to_patch.reshape(1, 1, side, side)
    heat = F.interpolate(heat, size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0].numpy()
    return (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
