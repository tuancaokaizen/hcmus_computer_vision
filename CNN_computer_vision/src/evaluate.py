"""Prediction and metric helpers.
Hàm dự đoán và tính chỉ số đánh giá.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader

from src.config import N_PERSONS
from src.models import classifier_head


@torch.no_grad()
def predict(
    model: nn.Module, name: str, loader: DataLoader, device: torch.device
) -> dict[str, np.ndarray | float]:
    """Return predictions, probabilities, embeddings, and ms per image.
    Trả về nhãn dự đoán, xác suất, embedding và thời gian ms mỗi ảnh.
    """
    model.eval()
    captured: list[torch.Tensor] = []
    # Embedding = input of the final linear layer / Embedding = input của lớp tuyến tính cuối
    hook = classifier_head(model, name).register_forward_pre_hook(
        lambda _m, inputs: captured.append(inputs[0].detach().cpu())
    )
    probs, labels = [], []
    elapsed = 0.0
    # Warm-up pass so kernel compilation is not timed / Chạy khởi động để không tính thời gian biên dịch kernel
    model(next(iter(loader))[0].to(device))
    captured.clear()
    for images, targets in loader:
        images = images.to(device)
        start = time.perf_counter()
        logits = model(images)
        if device.type == "mps":
            torch.mps.synchronize()
        elapsed += time.perf_counter() - start
        probs.append(torch.softmax(logits, dim=1).cpu())
        labels.append(targets)
    hook.remove()
    prob = torch.cat(probs).numpy()
    return {
        "y_true": torch.cat(labels).numpy(),
        "y_pred": prob.argmax(axis=1),
        "prob": prob,
        "embeddings": torch.cat(captured).numpy(),
        "ms_per_image": 1000.0 * elapsed / len(prob),
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy, macro P/R/F1, per-class arrays, confusion matrix, text report.
    Accuracy, P/R/F1 trung bình macro, mảng theo lớp, ma trận nhầm lẫn, báo cáo text.
    """
    labels = np.arange(N_PERSONS)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "f1": float(f1.mean()),
        "per_class": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "support": support.tolist(),
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        "report": classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0),
    }
