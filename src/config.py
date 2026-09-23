"""Experiment paths and hyperparameters.
Đường dẫn thí nghiệm và siêu tham số.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"

# Sam Roweis MATLAB copy of the AT&T ORL faces / Bản MATLAB ORL của Sam Roweis
OLIVETTI_MAT_PATH = DATA_DIR / "olivettifaces.mat"
OLIVETTI_MAT_URL = "https://ndownloader.figshare.com/files/5976027"

IMAGE_SHAPE = (64, 64)
N_PERSONS = 40
IMAGES_PER_PERSON = 10
RANDOM_STATE = 42
# 2 test images per person / 2 ảnh kiểm tra cho mỗi người
TEST_SIZE = 0.2

# Eigenfaces keep 100 principal components / Eigenfaces giữ 100 thành phần chính
N_COMPONENTS_PCA = 100

# Uniform LBP on an 8x8 spatial grid / LBP đều trên lưới không gian 8x8
LBP_POINTS = 8
LBP_RADIUS = 2
LBP_GRID = (8, 8)

# HOG on 64x64 faces / HOG trên ảnh mặt 64x64
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
