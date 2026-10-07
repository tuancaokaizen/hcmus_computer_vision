"""Load Olivetti faces, split them, and wrap them as PyTorch datasets.
Tải Olivetti Faces, chia tập và đóng gói thành dataset PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms import v2

from src.config import (
    DATA_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    N_PERSONS,
    OLIVETTI_MAT_PATH,
    OLIVETTI_MAT_URL,
    RANDOM_STATE,
    TEST_SIZE,
    VAL_PER_PERSON,
)


@dataclass
class Splits:
    """Train / validation / test arrays plus the full train set for refitting.
    Mảng train / validation / test và tập train đầy đủ để huấn luyện lại.
    """

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    x_train_full: np.ndarray
    y_train_full: np.ndarray


def _download_olivetti_mat(dest: Path) -> None:
    """Download the MATLAB archive if it is missing.
    Tải file MATLAB nếu chưa có sẵn.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Olivetti Faces -> {dest}")
    urlretrieve(OLIVETTI_MAT_URL, dest)


def load_olivetti_faces() -> tuple[np.ndarray, np.ndarray]:
    """Return 400 faces in [0, 1] (64x64) and identity labels 0..39.
    Trả về 400 ảnh mặt trong [0, 1] (64x64) và nhãn danh tính 0..39.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not OLIVETTI_MAT_PATH.exists():
        _download_olivetti_mat(OLIVETTI_MAT_PATH)
    faces = np.float32(loadmat(OLIVETTI_MAT_PATH)["faces"].T.copy())
    faces -= faces.min()
    faces /= faces.max()
    images = faces.reshape((400, 64, 64)).transpose(0, 2, 1)
    labels = np.array([i // 10 for i in range(400)], dtype=np.int64)
    return images, labels


def make_splits(images: np.ndarray, labels: np.ndarray) -> Splits:
    """Reproduce the assignment-1 test split, then carve a validation set from train.
    Tái tạo tập test của bài 1, rồi tách validation từ tập train.
    """
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        images, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=VAL_PER_PERSON * N_PERSONS,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )
    return Splits(x_train, y_train, x_val, y_val, x_test, y_test, x_train_full, y_train_full)


def build_transform(input_size: int, channels: int, train: bool) -> v2.Compose:
    """Augment (train only), resize, replicate channels, and normalize.
    Tăng cường dữ liệu (chỉ khi train), resize, nhân kênh và chuẩn hoá.
    """
    steps: list = []
    if train:
        # Mild geometric + photometric jitter keeps faces recognizable /
        # Biến đổi hình học + độ sáng nhẹ để mặt vẫn nhận ra được
        steps += [
            v2.RandomHorizontalFlip(),
            v2.RandomAffine(degrees=10, translate=(0.06, 0.06), scale=(0.93, 1.07)),
            v2.ColorJitter(brightness=0.25, contrast=0.25),
        ]
    if input_size != 64:
        steps.append(v2.Resize((input_size, input_size), antialias=True))
    if channels == 3:
        # Pretrained ImageNet models expect RGB statistics /
        # Model pretrained ImageNet cần 3 kênh và thống kê RGB
        steps += [v2.Lambda(lambda t: t.repeat(3, 1, 1)), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    else:
        steps.append(v2.Normalize((0.5,), (0.5,)))
    return v2.Compose(steps)


class FaceDataset(Dataset):
    """Grayscale face tensors with an on-the-fly transform.
    Tensor ảnh mặt xám với phép biến đổi thực hiện khi lấy mẫu.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform: v2.Compose) -> None:
        self.images = torch.from_numpy(images).unsqueeze(1)
        self.labels = torch.from_numpy(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.transform(self.images[index]), self.labels[index]
