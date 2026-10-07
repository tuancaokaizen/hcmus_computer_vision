"""Fine-tune every backbone, evaluate on the held-out test set, and save artifacts.
Fine-tune từng backbone, đánh giá trên tập test và lưu kết quả.
"""

from __future__ import annotations

import json
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import (
    BATCH_SIZE,
    FIGURES_DIR,
    LABEL_SMOOTHING,
    MODEL_SPECS,
    MODELS_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    TRADITIONAL_REFERENCE,
    WEIGHT_DECAY,
)
from src.dataset import FaceDataset, Splits, build_transform, load_olivetti_faces, make_splits
from src.evaluate import compute_metrics, predict
from src.models import build_model, count_parameters
from src import visualize


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch / Đặt seed cho Python, NumPy và PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device() -> torch.device:
    """Prefer Apple MPS, then CUDA, then CPU / Ưu tiên MPS, rồi CUDA, rồi CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _loader(images: np.ndarray, labels: np.ndarray, spec: dict, train: bool) -> DataLoader:
    transform = build_transform(spec["input_size"], spec["channels"], train=train)
    generator = torch.Generator().manual_seed(RANDOM_STATE)
    return DataLoader(
        FaceDataset(images, labels, transform),
        batch_size=BATCH_SIZE,
        shuffle=train,
        generator=generator,
    )


def _evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Mean cross-entropy and accuracy on a loader / Cross-entropy trung bình và accuracy."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            total_loss += criterion(logits, targets).item() * len(targets)
            correct += (logits.argmax(1) == targets).sum().item()
            count += len(targets)
    return total_loss / count, correct / count


def train_model(
    name: str,
    spec: dict,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    epochs: int,
    device: torch.device,
) -> tuple[nn.Module, dict, int]:
    """Train with AdamW + cosine LR; keep the best-validation weights if val is given.
    Huấn luyện bằng AdamW + LR cosine; giữ trọng số tốt nhất trên validation nếu có.
    """
    seed_everything(RANDOM_STATE)
    model = build_model(name, pretrained=spec["pretrained"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_key, best_state, best_epoch = None, None, epochs

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, count = 0.0, 0, 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)
            correct += (logits.argmax(1) == targets).sum().item()
            count += len(targets)
        scheduler.step()
        history["train_loss"].append(total_loss / count)
        history["train_acc"].append(correct / count)

        if val_loader is not None:
            val_loss, val_acc = _evaluate_loss(model, val_loader, device)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            # Highest val accuracy, ties broken by lower val loss /
            # Chọn val accuracy cao nhất, hoà thì chọn val loss thấp hơn
            key = (val_acc, -val_loss)
            if best_key is None or key > best_key:
                best_key, best_epoch = key, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"  [{name}] epoch {epoch:02d}/{epochs} loss={total_loss / count:.3f} "
                  f"val_loss={val_loss:.3f} val_acc={val_acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_epoch


def _summary_row(name: str, metrics: dict, extra: dict) -> dict:
    return {
        "name": name,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        **extra,
        "per_class": metrics["per_class"],
        "report": metrics["report"],
    }


def run_experiment(quick: bool = False) -> None:
    """Full pipeline: data -> train/refit each model -> metrics -> figures.
    Toàn bộ pipeline: dữ liệu -> train/huấn luyện lại từng model -> chỉ số -> hình.
    """
    seed_everything(RANDOM_STATE)
    device = pick_device()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    images, labels = load_olivetti_faces()
    splits: Splits = make_splits(images, labels)
    print(f"Train {len(splits.y_train)} / Val {len(splits.y_val)} / Test {len(splits.y_test)} "
          f"(refit on {len(splits.y_train_full)})")
    visualize.plot_sample_gallery(images, labels, FIGURES_DIR / "sample_gallery.png")
    visualize.plot_augmentations(splits.x_train[0], FIGURES_DIR / "augmentation_examples.png")

    trained: dict[str, nn.Module] = {}
    outputs: dict[str, dict] = {}
    histories: dict[str, dict] = {}
    rows: list[dict] = []

    for name, spec in MODEL_SPECS.items():
        epochs = 2 if quick else spec["epochs"]
        print(f"\n=== {name} ===")
        start = time.perf_counter()
        # Phase 1: select the epoch count on the validation split /
        # Giai đoạn 1: chọn số epoch trên tập validation
        _, history, best_epoch = train_model(
            name, spec,
            _loader(splits.x_train, splits.y_train, spec, train=True),
            _loader(splits.x_val, splits.y_val, spec, train=False),
            epochs, device,
        )
        # Phase 2: refit on all 320 train images for best_epoch epochs (same data as A1) /
        # Giai đoạn 2: huấn luyện lại trên toàn bộ 320 ảnh train với best_epoch epoch (cùng dữ liệu bài 1)
        model, _, _ = train_model(
            name, spec,
            _loader(splits.x_train_full, splits.y_train_full, spec, train=True),
            None, best_epoch, device,
        )
        train_seconds = time.perf_counter() - start

        test_out = predict(model, name, _loader(splits.x_test, splits.y_test, spec, False), device)
        all_out = predict(model, name, _loader(images, labels, spec, False), device)
        metrics = compute_metrics(test_out["y_true"], test_out["y_pred"])
        print(f"  test acc={metrics['accuracy']:.4f} macro-F1={metrics['f1']:.4f} "
              f"best_epoch={best_epoch} time={train_seconds:.0f}s")

        torch.save(model.state_dict(), MODELS_DIR / f"{name}.pt")
        trained[name] = model.cpu().eval()
        histories[name] = history
        outputs[name] = {"test": test_out, "all": all_out, "metrics": metrics}
        rows.append(_summary_row(name, metrics, {
            "params": count_parameters(model),
            "best_epoch": best_epoch,
            "train_seconds": round(train_seconds, 1),
            "ms_per_image": round(test_out["ms_per_image"], 3),
            "pretrained": spec["pretrained"],
            "input_size": spec["input_size"],
            "misclassified": int((test_out["y_true"] != test_out["y_pred"]).sum()),
        }))

    # Rank by test accuracy, then macro-F1, then fewer parameters /
    # Xếp hạng theo accuracy test, rồi macro-F1, rồi ít tham số hơn
    best = max(rows, key=lambda r: (r["accuracy"], r["f1"], -r["params"]))["name"]
    print(f"\nBest model: {best}")

    visualize.plot_training_curves(histories, FIGURES_DIR / "training_curves.png")
    visualize.plot_metrics_comparison(rows, TRADITIONAL_REFERENCE, FIGURES_DIR / "metrics_comparison.png")
    visualize.plot_confusion_matrix(
        outputs[best]["metrics"]["confusion_matrix"], best, FIGURES_DIR / "confusion_matrix_best.png"
    )
    visualize.plot_confusion_grid(
        {n: o["metrics"]["confusion_matrix"] for n, o in outputs.items()},
        {r["name"]: r["accuracy"] for r in rows},
        FIGURES_DIR / "confusion_matrices_all.png",
    )
    visualize.plot_per_class_prf(
        outputs[best]["metrics"]["per_class"], best, FIGURES_DIR / "per_class_prf_best.png"
    )
    visualize.plot_per_class_heatmap(
        {n: o["metrics"]["per_class"] for n, o in outputs.items()},
        FIGURES_DIR / "per_class_recall_precision_all.png",
    )
    visualize.plot_efficiency(rows, FIGURES_DIR / "efficiency.png")
    visualize.plot_feature_maps(trained["ResNet-18"], splits.x_test[0], FIGURES_DIR / "feature_maps_resnet18.png")
    visualize.plot_explanations(trained, splits, outputs, FIGURES_DIR / "explanations_gradcam_attention.png")
    visualize.plot_tsne({n: o["all"]["embeddings"] for n, o in outputs.items()}, labels,
                        FIGURES_DIR / "tsne_embeddings.png")
    visualize.plot_predictions(splits.x_test, outputs[best]["test"], best, FIGURES_DIR / "prediction_samples.png")
    visualize.plot_errors_all(splits, outputs, FIGURES_DIR / "errors_all_models.png")

    summary = {
        "dataset": "Olivetti Faces (AT&T ORL)",
        "device": str(device),
        "quick_mode": quick,
        "n_train": int(len(splits.y_train)),
        "n_val": int(len(splits.y_val)),
        "n_train_refit": int(len(splits.y_train_full)),
        "n_test": int(len(splits.y_test)),
        "best_model": best,
        "best_accuracy": next(r["accuracy"] for r in rows if r["name"] == best),
        "traditional_reference": TRADITIONAL_REFERENCE,
        "models": rows,
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved metrics -> {RESULTS_DIR / 'metrics.json'}")
