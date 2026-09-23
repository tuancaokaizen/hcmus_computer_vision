"""Evaluation helpers / Hàm hỗ trợ đánh giá."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> dict[str, Any]:
    """Compute accuracy, report, and confusion matrix.
    Tính accuracy, báo cáo lớp, và ma trận nhầm lẫn.
    """
    return {
        "name": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "report": classification_report(
            y_true,
            y_pred,
            digits=4,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }
