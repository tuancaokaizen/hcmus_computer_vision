"""Experiment paths and hyperparameters.
Đường dẫn thí nghiệm và siêu tham số.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"
CACHE_DIR = ROOT_DIR / ".cache"

# Sam Roweis MATLAB copy of the AT&T ORL faces / Bản MATLAB ORL của Sam Roweis
OLIVETTI_MAT_PATH = DATA_DIR / "olivettifaces.mat"
OLIVETTI_MAT_URL = "https://ndownloader.figshare.com/files/5976027"

IMAGE_SHAPE = (64, 64)
N_PERSONS = 40
RANDOM_STATE = 42
# Same 8/2 split as assignment 1 / Cùng cách chia 8/2 như bài 1
TEST_SIZE = 0.2
# One train image per person held out for early stopping /
# Giữ 1 ảnh train mỗi người làm validation để dừng sớm
VAL_PER_PERSON = 1

BATCH_SIZE = 32
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Per-model settings: input size, channels, learning rate, max epochs /
# Cấu hình từng model: kích thước input, số kênh, learning rate, số epoch tối đa
MODEL_SPECS = {
    "SimpleCNN": {"input_size": 64, "channels": 1, "lr": 3e-3, "epochs": 60, "pretrained": False},
    "ResNet-18": {"input_size": 224, "channels": 3, "lr": 3e-4, "epochs": 25, "pretrained": True},
    "MobileNetV3-S": {"input_size": 224, "channels": 3, "lr": 1e-3, "epochs": 30, "pretrained": True},
    "DeiT-Tiny": {"input_size": 224, "channels": 3, "lr": 2e-4, "epochs": 25, "pretrained": True},
}

# Best traditional result from assignment 1 (eigenfaces+SVM) /
# Kết quả truyền thống tốt nhất của bài 1 (eigenfaces+SVM)
TRADITIONAL_REFERENCE = {
    "name": "Eigenfaces+SVM (A1)",
    "accuracy": 0.975,
    "precision": 0.9833,
    "recall": 0.975,
    "f1": 0.9733,
}
