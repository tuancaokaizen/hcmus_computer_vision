"""Train traditional face-recognition models and export results.
Huấn luyện mô hình nhận dạng mặt truyền thống và xuất kết quả.
"""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
from sklearn.decomposition import PCA

from src.classifiers import make_knn, make_svm
from src.config import FIGURES_DIR, MODELS_DIR, N_COMPONENTS_PCA, RESULTS_DIR
from src.dataset import equalize_faces, flatten_images, load_olivetti_faces, split_by_person
from src.evaluate import evaluate_predictions
from src.features import EigenfacesExtractor, FisherfacesExtractor, extract_hog, extract_lbp
from src.visualize import (
    plot_accuracy_bars,
    plot_confusion_matrix,
    plot_eigenfaces,
    plot_prediction_samples,
    plot_reconstructions,
    plot_sample_gallery,
)


def _fit_predict(model, X_train, y_train, X_test) -> np.ndarray:
    model.fit(X_train, y_train)
    return model.predict(X_test)


def run_experiment() -> dict[str, Any]:
    """Run the full traditional pipeline / Chạy toàn bộ pipeline truyền thống."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    images, labels = load_olivetti_faces()
    images = equalize_faces(images)
    X_train_img, X_test_img, y_train, y_test = split_by_person(images, labels)
    X_train_flat = flatten_images(X_train_img)
    X_test_flat = flatten_images(X_test_img)

    # Eigenfaces / Eigenfaces
    eigen = EigenfacesExtractor(n_components=N_COMPONENTS_PCA)
    Z_train = eigen.fit_transform(X_train_flat)
    Z_test = eigen.transform(X_test_flat)

    # Fisherfaces / Fisherfaces
    fisher = FisherfacesExtractor(n_pca=N_COMPONENTS_PCA)
    F_train = fisher.fit_transform(X_train_flat, y_train)
    F_test = fisher.transform(X_test_flat)

    # Local texture and gradient descriptors / Đặc trưng texture và gradient cục bộ
    lbp_train = extract_lbp(X_train_img)
    lbp_test = extract_lbp(X_test_img)
    hog_train = extract_hog(X_train_img)
    hog_test = extract_hog(X_test_img)

    experiments = [
        ("pixels+SVM", X_train_flat, X_test_flat, make_svm("rbf")),
        ("eigenfaces+kNN", Z_train, Z_test, make_knn(3)),
        ("eigenfaces+SVM", Z_train, Z_test, make_svm("rbf")),
        ("fisherfaces+kNN", F_train, F_test, make_knn(3)),
        ("lbp+SVM", lbp_train, lbp_test, make_svm("linear")),
        ("hog+SVM", hog_train, hog_test, make_svm("rbf")),
    ]

    results = []
    fitted_models = {}
    predictions = {}
    for name, X_tr, X_te, model in experiments:
        y_pred = _fit_predict(model, X_tr, y_train, X_te)
        metrics = evaluate_predictions(y_test, y_pred, name)
        results.append(metrics)
        fitted_models[name] = model
        predictions[name] = y_pred
        joblib.dump(model, MODELS_DIR / f"{name.replace('+', '_')}.joblib")

    joblib.dump(eigen, MODELS_DIR / "eigenfaces_pca.joblib")
    joblib.dump(fisher, MODELS_DIR / "fisherfaces.joblib")

    # Reconstruction examples for the report / Ví dụ tái tạo cho báo cáo
    sample = X_test_flat[0]
    reconstructions = []
    for n_comp in (10, 50, 100):
        pca_tmp = PCA(n_components=n_comp, svd_solver="full", random_state=42)
        pca_tmp.fit(X_train_flat)
        restored = pca_tmp.inverse_transform(pca_tmp.transform(sample.reshape(1, -1)))[0]
        reconstructions.append((n_comp, restored))

    plot_sample_gallery(images, labels)
    plot_eigenfaces(eigen.mean_face, eigen.eigenfaces)
    plot_reconstructions(sample, reconstructions)

    names = [item["name"] for item in results]
    scores = [item["accuracy"] for item in results]
    plot_accuracy_bars(names, scores)

    best = max(results, key=lambda item: item["accuracy"])
    plot_confusion_matrix(
        np.asarray(best["confusion_matrix"]),
        title=f"Confusion matrix — {best['name']}",
        filename="confusion_matrix_best.png",
    )
    plot_prediction_samples(X_test_img, y_test, np.asarray(predictions[best["name"]]))

    summary = {
        "dataset": "Olivetti Faces",
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "best_model": best["name"],
        "best_accuracy": best["accuracy"],
        "models": [
            {
                "name": item["name"],
                "accuracy": item["accuracy"],
                "report": item["report"],
            }
            for item in results
        ],
    }
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("=== Traditional face recognition / Nhận dạng khuôn mặt truyền thống ===")
    for item in results:
        print(f"{item['name']:18s}  accuracy={item['accuracy']:.4f}")
    print(f"Best: {best['name']} ({best['accuracy']:.4f})")
    print(f"Figures: {FIGURES_DIR}")
    print(f"Metrics: {RESULTS_DIR / 'metrics.json'}")
    return summary
