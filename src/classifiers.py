"""Traditional classifiers used with visual features.
Bộ phân lớp truyền thống dùng kèm đặc trưng thị giác.
"""

from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def make_knn(n_neighbors: int = 3) -> KNeighborsClassifier:
    """kNN in descriptor space / kNN trong không gian đặc trưng."""
    return KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")


def make_svm(kernel: str = "rbf", C: float = 10.0) -> Pipeline:
    """SVM with feature scaling / SVM kèm chuẩn hóa đặc trưng."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svc",
                SVC(
                    kernel=kernel,
                    C=C,
                    gamma="scale",
                    class_weight="balanced",
                ),
            ),
        ]
    )
