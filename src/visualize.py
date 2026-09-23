"""Figures for the traditional face-recognition report.
Hình minh họa cho báo cáo nhận dạng khuôn mặt truyền thống.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from src.config import FIGURES_DIR, IMAGE_SHAPE, N_PERSONS


def _save(fig: plt.Figure, name: str) -> None:
    """Save figure with enough padding to avoid title overlap.
    Lưu hình với đệm đủ để tránh chồng tiêu đề.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def plot_sample_gallery(images: np.ndarray, labels: np.ndarray, n_persons: int = 10) -> None:
    """Show one face per identity / Hiện một khuôn mặt cho mỗi người."""
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.8))
    for person_id, ax in enumerate(axes.ravel()):
        sample = images[labels == person_id][0]
        ax.imshow(sample, cmap="gray")
        ax.set_title(f"ID {person_id}", fontsize=10, pad=6)
        ax.axis("off")
    fig.subplots_adjust(hspace=0.45, wspace=0.15)
    _save(fig, "sample_gallery.png")


def plot_eigenfaces(mean_face: np.ndarray, eigenfaces: np.ndarray, n_show: int = 12) -> None:
    """Visualize mean face and top eigenfaces.
    Vẽ mặt trung bình và các eigenfaces hàng đầu.
    """
    n_cols = 5
    n_total = n_show + 1
    n_rows = int(np.ceil(n_total / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 2.4 * n_rows))
    axes = np.atleast_2d(axes).ravel()

    axes[0].imshow(mean_face, cmap="gray")
    axes[0].set_title("Mean face", fontsize=10, pad=6)
    axes[0].axis("off")

    for i in range(n_show):
        ax = axes[i + 1]
        ax.imshow(eigenfaces[i], cmap="gray")
        ax.set_title(f"PC {i + 1}", fontsize=10, pad=6)
        ax.axis("off")

    for ax in axes[n_total:]:
        ax.axis("off")

    fig.subplots_adjust(hspace=0.55, wspace=0.2, top=0.92, bottom=0.05)
    _save(fig, "eigenfaces.png")


def plot_reconstructions(
    original: np.ndarray,
    pca_images: Sequence[tuple[int, np.ndarray]],
) -> None:
    """Compare reconstructions by number of components.
    So sánh tái tạo theo số thành phần chính.
    """
    n_cols = len(pca_images) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3.2))
    axes[0].imshow(original.reshape(IMAGE_SHAPE), cmap="gray")
    axes[0].set_title("Original", fontsize=11, pad=8)
    axes[0].axis("off")
    for ax, (n_comp, image) in zip(axes[1:], pca_images):
        ax.imshow(image.reshape(IMAGE_SHAPE), cmap="gray")
        ax.set_title(f"{n_comp} PCs", fontsize=11, pad=8)
        ax.axis("off")
    fig.subplots_adjust(wspace=0.25, top=0.82)
    _save(fig, "reconstructions.png")


def plot_accuracy_bars(names: Sequence[str], scores: Sequence[float]) -> None:
    """Horizontal accuracy bars (readable in a narrow column).
    Biểu đồ cột ngang (dễ đọc trong cột hẹp).
    """
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(names))
    bars = ax.barh(y, scores, color="#2563eb", height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(list(names), fontsize=10)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Accuracy")
    ax.set_title("Traditional descriptors + classifiers", pad=10)
    ax.invert_yaxis()
    for bar, score in zip(bars, scores):
        ax.text(
            score + 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            va="center",
            ha="left",
            fontsize=9,
        )
    fig.subplots_adjust(left=0.28, right=0.95, top=0.88, bottom=0.15)
    _save(fig, "accuracy_comparison.png")


def plot_confusion_matrix(matrix: np.ndarray, title: str, filename: str) -> None:
    """Save a 40-class confusion matrix / Lưu ma trận nhầm lẫn 40 lớp."""
    fig, ax = plt.subplots(figsize=(10, 10))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=list(range(N_PERSONS)),
    )
    display.plot(include_values=False, cmap="Blues", ax=ax, colorbar=True)
    ax.set_title(title, pad=12)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)
    _save(fig, filename)


def plot_prediction_samples(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Show correct and incorrect predictions without empty axes.
    Hiện mẫu đúng/sai, không để ô trống.
    """
    correct = np.where(y_true == y_pred)[0]
    wrong = np.where(y_true != y_pred)[0]
    chosen = list(correct[:4]) + list(wrong[:4])
    if not chosen:
        return

    n = len(chosen)
    n_cols = min(4, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 3.0 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, idx in zip(axes, chosen):
        ax.imshow(images[idx], cmap="gray")
        ok = y_true[idx] == y_pred[idx]
        ax.set_title(
            f"true {y_true[idx]} / pred {y_pred[idx]}",
            color="green" if ok else "red",
            fontsize=10,
            pad=8,
        )
        ax.axis("off")

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.subplots_adjust(hspace=0.55, wspace=0.25, top=0.90, bottom=0.05)
    _save(fig, "prediction_samples.png")
