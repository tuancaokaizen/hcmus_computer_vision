"""Traditional visual descriptors for faces.
Đặc trưng thị giác truyền thống cho ảnh khuôn mặt.
"""

from __future__ import annotations

import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src.config import (
    HOG_CELLS_PER_BLOCK,
    HOG_ORIENTATIONS,
    HOG_PIXELS_PER_CELL,
    IMAGE_SHAPE,
    LBP_GRID,
    LBP_POINTS,
    LBP_RADIUS,
    N_COMPONENTS_PCA,
    RANDOM_STATE,
)


def extract_hog(images: np.ndarray) -> np.ndarray:
    """Extract HOG vectors / Trích vector HOG."""
    features = [
        hog(
            image,
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=HOG_PIXELS_PER_CELL,
            cells_per_block=HOG_CELLS_PER_BLOCK,
            block_norm="L2-Hys",
            feature_vector=True,
        )
        for image in images
    ]
    return np.asarray(features, dtype=np.float32)


def _lbp_histogram(image: np.ndarray) -> np.ndarray:
    """Spatial histogram of uniform LBP codes.
    Histogram không gian của mã LBP đều.
    """
    # LBP expects integer pixels / LBP cần pixel kiểu nguyên
    image_u8 = np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8)
    lbp = local_binary_pattern(image_u8, LBP_POINTS, LBP_RADIUS, method="uniform")
    n_bins = int(LBP_POINTS + 2)
    grid_y, grid_x = LBP_GRID
    height, width = image.shape
    cell_h = height // grid_y
    cell_w = width // grid_x
    histograms = []

    for row in range(grid_y):
        for col in range(grid_x):
            cell = lbp[
                row * cell_h : (row + 1) * cell_h,
                col * cell_w : (col + 1) * cell_w,
            ]
            hist, _ = np.histogram(
                cell,
                bins=n_bins,
                range=(0, n_bins),
                density=True,
            )
            histograms.append(hist.astype(np.float32))

    return np.concatenate(histograms)


def extract_lbp(images: np.ndarray) -> np.ndarray:
    """Extract concatenated spatial LBP histograms.
    Trích histogram LBP không gian đã nối.
    """
    return np.vstack([_lbp_histogram(image) for image in images])


class EigenfacesExtractor:
    """PCA subspace used as Eigenfaces.
    Không gian con PCA dùng làm Eigenfaces.
    """

    def __init__(self, n_components: int = N_COMPONENTS_PCA) -> None:
        self.pca = PCA(
            n_components=n_components,
            whiten=True,
            svd_solver="full",
            random_state=RANDOM_STATE,
        )

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.fit_transform(X).astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(X).astype(np.float32)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return self.pca.inverse_transform(Z)

    @property
    def eigenfaces(self) -> np.ndarray:
        """Principal components reshaped as faces / Thành phần chính reshape thành mặt."""
        return self.pca.components_.reshape(-1, *IMAGE_SHAPE)

    @property
    def mean_face(self) -> np.ndarray:
        return self.pca.mean_.reshape(IMAGE_SHAPE)


class FisherfacesExtractor:
    """PCA followed by LDA (Fisherfaces).
    PCA rồi LDA (Fisherfaces).
    """

    def __init__(self, n_pca: int = N_COMPONENTS_PCA) -> None:
        self.pca = PCA(
            n_components=n_pca,
            svd_solver="full",
            random_state=RANDOM_STATE,
        )
        self.lda = LinearDiscriminantAnalysis()

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        # LDA needs n_samples > n_classes; PCA reduces dim first
        # LDA cần số mẫu > số lớp; PCA giảm chiều trước
        X_pca = self.pca.fit_transform(X)
        return self.lda.fit_transform(X_pca, y).astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_pca = self.pca.transform(X)
        return self.lda.transform(X_pca).astype(np.float32)
