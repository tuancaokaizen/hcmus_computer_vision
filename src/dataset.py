"""Load and preprocess Olivetti faces.
Tải và tiền xử lý bộ Olivetti Faces.
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from scipy.io import loadmat
from skimage import exposure
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_DIR,
    IMAGE_SHAPE,
    OLIVETTI_MAT_PATH,
    OLIVETTI_MAT_URL,
    RANDOM_STATE,
    TEST_SIZE,
)


def _download_olivetti_mat(dest: Path) -> None:
    """Download the MATLAB archive if it is missing.
    Tải file MATLAB nếu chưa có sẵn.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Olivetti Faces -> {dest}")
    urlretrieve(OLIVETTI_MAT_URL, dest)


def _faces_from_mat(mat_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse ORL faces from the Roweis .mat file.
    Đọc ảnh ORL từ file .mat của Roweis.
    """
    payload = loadmat(mat_path)
    # sklearn layout: (4096, 400) then transpose, rescale, swap axes
    # Bố cục sklearn: (4096, 400) rồi transpose, scale, đổi trục
    faces = payload["faces"].T.copy()
    faces = np.float32(faces)
    faces -= faces.min()
    faces /= faces.max()
    images = faces.reshape((400, 64, 64)).transpose(0, 2, 1)
    labels = np.array([i // 10 for i in range(400)], dtype=np.int32)
    return images, labels


def load_olivetti_faces() -> tuple[np.ndarray, np.ndarray]:
    """Load local Olivetti Faces, downloading only if needed.
    Tải Olivetti Faces local, chỉ download khi thiếu file.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not OLIVETTI_MAT_PATH.exists():
        _download_olivetti_mat(OLIVETTI_MAT_PATH)
    return _faces_from_mat(OLIVETTI_MAT_PATH)


def equalize_faces(images: np.ndarray) -> np.ndarray:
    """Apply histogram equalization per image.
    Cân bằng histogram cho từng ảnh.
    """
    equalized = np.empty_like(images, dtype=np.float32)
    for i, image in enumerate(images):
        equalized[i] = exposure.equalize_hist(image).astype(np.float32)
    return equalized


def split_by_person(
    images: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split so every person appears in train and test.
    Tách có tầng để mọi người đều xuất hiện ở tập train và test.
    """
    return train_test_split(
        images,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )


def flatten_images(images: np.ndarray) -> np.ndarray:
    """Flatten HxW faces into vectors / Duỗi ảnh HxW thành vector."""
    n_samples = images.shape[0]
    return images.reshape(n_samples, IMAGE_SHAPE[0] * IMAGE_SHAPE[1])
