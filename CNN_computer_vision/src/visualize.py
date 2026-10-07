"""Figures for the CNN / ViT report.
Hình minh hoạ cho báo cáo CNN / ViT.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch import nn
from torchvision.transforms import v2

from src.config import MODEL_SPECS, N_PERSONS, RANDOM_STATE
from src.dataset import Splits, build_transform
from src.explain import attention_rollout, grad_cam, resnet_feature_maps
from src.models import gradcam_layer

MODEL_COLORS = {
    "SimpleCNN": "#7f7f7f",
    "ResNet-18": "#1f4e78",
    "MobileNetV3-S": "#2e8b57",
    "DeiT-Tiny": "#c0504d",
}


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_sample_gallery(images: np.ndarray, labels: np.ndarray, path: Path) -> None:
    """One face per identity (4x10 grid) / Mỗi người một ảnh (lưới 4x10)."""
    fig, axes = plt.subplots(4, 10, figsize=(10, 4.4))
    for person, ax in enumerate(axes.flat):
        ax.imshow(images[np.flatnonzero(labels == person)[0]], cmap="gray")
        ax.set_title(f"ID {person}", fontsize=7)
        ax.axis("off")
    fig.suptitle("Olivetti Faces: one sample per identity (40 persons)", fontsize=10)
    fig.tight_layout()
    _save(fig, path)


def plot_augmentations(image: np.ndarray, path: Path) -> None:
    """Original face plus random training augmentations / Ảnh gốc và các biến thể tăng cường."""
    torch.manual_seed(RANDOM_STATE)
    augment = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomAffine(degrees=10, translate=(0.06, 0.06), scale=(0.93, 1.07)),
        v2.ColorJitter(brightness=0.25, contrast=0.25),
    ])
    tensor = torch.from_numpy(image).unsqueeze(0)
    fig, axes = plt.subplots(1, 8, figsize=(10, 1.6))
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("original", fontsize=8)
    for ax in axes[1:]:
        ax.imshow(augment(tensor)[0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title("augmented", fontsize=8)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    _save(fig, path)


def plot_training_curves(histories: dict[str, dict], path: Path) -> None:
    """Train loss, validation loss, and validation accuracy per epoch.
    Loss train, loss validation và accuracy validation theo epoch.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for name, hist in histories.items():
        epochs = np.arange(1, len(hist["train_loss"]) + 1)
        color = MODEL_COLORS[name]
        axes[0].plot(epochs, hist["train_loss"], color=color, label=name)
        axes[1].plot(epochs, hist["val_loss"], color=color, label=name)
        axes[2].plot(epochs, hist["val_acc"], color=color, label=name)
    for ax, title in zip(axes, ("Train loss (label smoothing 0.1)", "Validation loss", "Validation accuracy")):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[2].set_ylim(0, 1.02)
    axes[2].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    _save(fig, path)


def plot_metrics_comparison(rows: list[dict], reference: dict, path: Path) -> None:
    """Grouped bars of accuracy / macro P / R / F1 vs the assignment-1 baseline.
    Cột nhóm accuracy / P / R / F1 macro so với baseline bài 1.
    """
    metrics = ["accuracy", "precision", "recall", "f1"]
    entries = [(r["name"], [r[m] for m in metrics], MODEL_COLORS[r["name"]]) for r in rows]
    entries.append((reference["name"], [reference[m] for m in metrics], "#d9a441"))
    width = 0.8 / len(entries)
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i, (name, values, color) in enumerate(entries):
        bars = ax.bar(x + (i - (len(entries) - 1) / 2) * width, values, width, label=name, color=color,
                      hatch="//" if "A1" in name else None, edgecolor="white")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.004, f"{value:.3f}",
                    ha="center", va="bottom", fontsize=6, rotation=90)
    ax.set_xticks(x, ["Accuracy", "Macro precision", "Macro recall", "Macro F1"])
    low = min(min(v) for _, v, _ in entries)
    ax.set_ylim(max(0.0, low - 0.08), 1.06)
    ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Test-set metrics (80 images, 40 identities)", fontsize=10)
    fig.tight_layout()
    _save(fig, path)


def plot_confusion_matrix(cm: np.ndarray, name: str, path: Path) -> None:
    """40x40 confusion matrix with off-diagonal errors annotated.
    Ma trận nhầm lẫn 40x40, chú thích các lỗi ngoài đường chéo.
    """
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    for i, j in zip(*np.nonzero(cm)):
        if i != j:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", lw=1.5))
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="red", fontsize=7)
    ticks = np.arange(0, N_PERSONS, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("Predicted identity")
    ax.set_ylabel("True identity")
    errors = int(cm.sum() - np.trace(cm))
    ax.set_title(f"Confusion matrix: {name} ({errors} errors / {int(cm.sum())})", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, path)


def plot_confusion_grid(cms: dict[str, np.ndarray], accs: dict[str, float], path: Path) -> None:
    """Confusion matrices of all models side by side / Ma trận nhầm lẫn của mọi model."""
    fig, axes = plt.subplots(1, len(cms), figsize=(3.2 * len(cms), 3.4))
    for ax, (name, cm) in zip(axes, cms.items()):
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
        for i, j in zip(*np.nonzero(cm)):
            if i != j:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", lw=1.2))
        ax.set_title(f"{name}\nacc = {accs[name]:.3f}", fontsize=9)
        ax.set_xticks([0, 20, 39])
        ax.set_yticks([0, 20, 39])
        ax.set_xlabel("predicted", fontsize=8)
    axes[0].set_ylabel("true", fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def plot_per_class_prf(per_class: dict, name: str, path: Path) -> None:
    """Per-identity precision, recall, F1 bars for one model.
    Cột precision, recall, F1 theo từng người cho một model.
    """
    ids = np.arange(N_PERSONS)
    fig, axes = plt.subplots(3, 1, figsize=(10, 5.4), sharex=True)
    for ax, key, color in zip(axes, ("precision", "recall", "f1"), ("#1f4e78", "#2e8b57", "#c0504d")):
        values = np.array(per_class[key])
        colors = [color if v >= 0.999 else "#e46c0a" for v in values]
        ax.bar(ids, values, color=colors)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(key.capitalize() if key != "f1" else "F1")
        ax.axhline(values.mean(), color="black", ls="--", lw=0.8)
        ax.text(N_PERSONS - 0.5, values.mean() + 0.02, f"macro = {values.mean():.3f}", ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.3)
    axes[-1].set_xticks(ids)
    axes[-1].tick_params(axis="x", labelsize=6)
    axes[-1].set_xlabel("Identity (orange = below 1.0)")
    axes[0].set_title(f"Per-identity precision / recall / F1: {name}", fontsize=10)
    fig.tight_layout()
    _save(fig, path)


def plot_per_class_heatmap(per_class: dict[str, dict], path: Path) -> None:
    """Heatmap of per-identity precision and recall for every model.
    Heatmap precision và recall theo từng người cho mọi model.
    """
    rows, labels = [], []
    for name, values in per_class.items():
        for key in ("precision", "recall"):
            rows.append(values[key])
            labels.append(f"{name} {key[0].upper()}")
    matrix = np.array(rows)
    fig, ax = plt.subplots(figsize=(10, 0.36 * len(rows) + 1.2))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=7)
    ax.set_xticks(np.arange(N_PERSONS))
    ax.tick_params(axis="x", labelsize=6)
    ax.set_xlabel("Identity")
    for i, j in zip(*np.nonzero(matrix < 0.999)):
        ax.text(j, i, f"{matrix[i, j]:.2f}".lstrip("0"), ha="center", va="center", fontsize=5)
    ax.set_title("Per-identity precision (P) and recall (R), all models", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    fig.tight_layout()
    _save(fig, path)


def plot_efficiency(rows: list[dict], path: Path) -> None:
    """Accuracy vs parameter count; marker area ~ inference time.
    Accuracy theo số tham số; kích thước điểm ~ thời gian suy luận.
    """
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for r in rows:
        ax.scatter(r["params"] / 1e6, r["accuracy"], s=70, color=MODEL_COLORS[r["name"]],
                   edgecolor="black", zorder=3)
        ax.annotate(f"{r['name']}\n{r['ms_per_image']:.2f} ms/img", (r["params"] / 1e6, r["accuracy"]),
                    textcoords="offset points", xytext=(0, -24), ha="center", fontsize=7)
    accs = [r["accuracy"] for r in rows]
    ax.set_ylim(min(accs) - 0.02, 1.008)
    ax.set_xlim(0.12, 40)
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (millions, log scale)")
    ax.set_ylabel("Test accuracy")
    ax.grid(alpha=0.3)
    ax.set_title("Accuracy vs model size (label: inference time per image)", fontsize=9)
    fig.tight_layout()
    _save(fig, path)


def _input_tensor(image: np.ndarray, name: str) -> torch.Tensor:
    spec = MODEL_SPECS[name]
    transform = build_transform(spec["input_size"], spec["channels"], train=False)
    return transform(torch.from_numpy(image).unsqueeze(0)).unsqueeze(0)


def plot_feature_maps(model: nn.Module, image: np.ndarray, path: Path) -> None:
    """Strongest 8 channels after each ResNet-18 stage / 8 kênh mạnh nhất sau mỗi stage ResNet-18."""
    maps = resnet_feature_maps(model, _input_tensor(image, "ResNet-18"))
    fig, axes = plt.subplots(len(maps), 9, figsize=(10, 1.25 * len(maps) + 0.4))
    for row, (stage, activation) in enumerate(maps.items()):
        axes[row, 0].imshow(image, cmap="gray")
        axes[row, 0].set_ylabel(f"{stage}\n{activation.shape[0]}x{activation.shape[1]}x{activation.shape[2]}",
                                fontsize=7)
        top = np.argsort(activation.mean(axis=(1, 2)))[::-1][:8]
        for col, channel in enumerate(top, start=1):
            axes[row, col].imshow(activation[channel], cmap="viridis")
            axes[row, col].set_title(f"ch {channel}", fontsize=6)
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("ResNet-18 feature maps (input → deeper stages, 8 most active channels)", fontsize=10)
    fig.tight_layout()
    _save(fig, path)


def _overlay(ax: plt.Axes, image: np.ndarray, heat: np.ndarray, title: str) -> None:
    ax.imshow(image, cmap="gray", extent=(0, 1, 0, 1))
    ax.imshow(heat, cmap="jet", alpha=0.45, extent=(0, 1, 0, 1))
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def plot_explanations(models: dict[str, nn.Module], splits: Splits, outputs: dict, path: Path) -> None:
    """Grad-CAM for CNNs and attention rollout for DeiT on six test faces.
    Grad-CAM cho CNN và attention rollout cho DeiT trên sáu ảnh test.
    """
    # Six test faces from distinct identities / Sáu ảnh test thuộc sáu người khác nhau
    chosen, seen = [], set()
    for idx, label in enumerate(splits.y_test):
        if label not in seen:
            chosen.append(idx)
            seen.add(label)
        if len(chosen) == 6:
            break
    names = list(models)
    fig, axes = plt.subplots(len(chosen), len(names) + 1, figsize=(1.75 * (len(names) + 1), 1.9 * len(chosen)))
    for row, idx in enumerate(chosen):
        image, label = splits.x_test[idx], int(splits.y_test[idx])
        axes[row, 0].imshow(image, cmap="gray")
        axes[row, 0].set_title(f"input (ID {label})", fontsize=7)
        axes[row, 0].axis("off")
        for col, name in enumerate(names, start=1):
            pred = int(outputs[name]["test"]["y_pred"][idx])
            tensor = _input_tensor(image, name)
            layer = gradcam_layer(models[name], name)
            heat = grad_cam(models[name], layer, tensor, pred) if layer is not None \
                else attention_rollout(models[name], tensor)
            method = "Grad-CAM" if layer is not None else "attn rollout"
            mark = "" if pred == label else " ✗"
            _overlay(axes[row, col], image, heat, f"{name}\n{method}, pred {pred}{mark}")
    fig.tight_layout()
    _save(fig, path)


def plot_tsne(embeddings: dict[str, np.ndarray], labels: np.ndarray, path: Path) -> None:
    """2-D t-SNE of the penultimate embeddings for all 400 faces.
    t-SNE 2 chiều của embedding lớp áp chót cho cả 400 ảnh.
    """
    cmap = plt.get_cmap("tab20")
    markers = ["o", "s"]
    fig, axes = plt.subplots(1, len(embeddings), figsize=(3.3 * len(embeddings), 3.4))
    for ax, (name, emb) in zip(axes, embeddings.items()):
        points = TSNE(n_components=2, perplexity=20, init="pca", random_state=RANDOM_STATE).fit_transform(emb)
        for person in range(N_PERSONS):
            mask = labels == person
            ax.scatter(points[mask, 0], points[mask, 1], s=8, color=cmap(person % 20),
                       marker=markers[person // 20], linewidths=0)
        ax.set_title(f"{name} ({emb.shape[1]}-D)", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("t-SNE of learned embeddings (colour/marker = identity)", fontsize=10)
    fig.tight_layout()
    _save(fig, path)


def plot_errors_all(splits: Splits, outputs: dict, path: Path) -> None:
    """Every test error of every model next to a training face of the predicted identity.
    Mọi lỗi test của mọi model, đặt cạnh một ảnh train của người bị dự đoán nhầm.
    """
    errors = []
    for name, out in outputs.items():
        y_true, y_pred = out["test"]["y_true"], out["test"]["y_pred"]
        errors += [(name, i, int(y_true[i]), int(y_pred[i])) for i in np.flatnonzero(y_true != y_pred)]
    if not errors:
        return
    fig, axes = plt.subplots(2, len(errors), figsize=(1.6 * len(errors) + 0.4, 3.6), squeeze=False)
    for col, (name, idx, true_id, pred_id) in enumerate(errors):
        look_alike = splits.x_train_full[np.flatnonzero(splits.y_train_full == pred_id)[0]]
        axes[0, col].imshow(splits.x_test[idx], cmap="gray")
        axes[0, col].set_title(f"{name}\ntest face, ID {true_id}", fontsize=7)
        axes[1, col].imshow(look_alike, cmap="gray")
        axes[1, col].set_title(f"predicted ID {pred_id}\n(train sample)", fontsize=7, color="red")
        for ax in axes[:, col]:
            ax.axis("off")
    fig.suptitle("All test errors: misclassified face (top) vs predicted identity (bottom)", fontsize=10)
    fig.tight_layout()
    _save(fig, path)


def plot_predictions(x_test: np.ndarray, out: dict, name: str, path: Path) -> None:
    """Test predictions with confidence; errors first, in red.
    Dự đoán trên tập test kèm độ tin cậy; lỗi hiển thị trước, màu đỏ.
    """
    y_true, y_pred, prob = out["y_true"], out["y_pred"], out["prob"]
    wrong = np.flatnonzero(y_true != y_pred)
    right = np.flatnonzero(y_true == y_pred)
    order = np.concatenate([wrong, right])[:16]
    fig, axes = plt.subplots(2, 8, figsize=(10, 3.1))
    for ax, idx in zip(axes.flat, order):
        ok = y_true[idx] == y_pred[idx]
        ax.imshow(x_test[idx], cmap="gray")
        ax.set_title(f"true {y_true[idx]} / pred {y_pred[idx]}\np = {prob[idx].max():.2f}", fontsize=7,
                     color="black" if ok else "red")
        ax.axis("off")
    fig.suptitle(f"{name}: test predictions ({len(wrong)} errors shown first)", fontsize=10)
    fig.tight_layout()
    _save(fig, path)
