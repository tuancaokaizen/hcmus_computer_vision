# Nhận dạng khuôn mặt truyền thống (Olivetti Faces)

Bài tập thị giác máy tính cá nhân: **trích đặc trưng thị giác truyền thống + bộ phân lớp**. Không dùng CNN.

**Sinh viên:** Cao Anh Tuan — MSHV **25C01025**

## Bài toán

Nhận dạng danh tính (closed-set) trên **Olivetti Faces** (AT&T ORL): 400 ảnh, 40 người × 10 ảnh, 64×64 grayscale. Dataset local: `data/olivettifaces.mat` (bản MATLAB Sam Roweis; tự tải lại nếu thiếu).

## Phương pháp (chỉ truyền thống)

| Bước | Kỹ thuật |
|---|---|
| Tiền xử lý | Histogram equalization (từng ảnh) |
| Đặc trưng | Pixel thô (baseline), **Eigenfaces (PCA, k=100, whitening)**, **Fisherfaces (PCA+LDA)**, **LBP** (uniform P=8, R=2, lưới 8×8), **HOG** (9 bins, cell 8×8, block 2×2) |
| Phân lớp | **kNN** (k=3) hoặc **SVM** (RBF / linear cho LBP) |
| Đánh giá | Stratified 8 train / 2 test mỗi người (`random_state=42`), accuracy + confusion matrix |

Không CNN, không embedding học sâu, không ArcFace.

## Cài đặt & chạy

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Kết quả:

- `results/metrics.json` — accuracy từng mô hình
- `results/figures/` — gallery, eigenfaces, reconstructions, accuracy (bar ngang), confusion matrix, prediction samples

## Kết quả (80 ảnh test)

| Mô hình | Accuracy | Đúng / 80 |
|---|---|---|
| pixels+SVM (baseline) | 95.0% | 76 |
| eigenfaces+kNN | 87.5% | 70 |
| **eigenfaces+SVM** | **97.5%** | **78** |
| fisherfaces+kNN | 97.5% | 78 |
| lbp+SVM | 97.5% | 78 |
| hog+SVM | 97.5% | 78 |

Mô hình đề xuất: **Eigenfaces + SVM**. Cùng descriptor Eigenfaces, SVM hơn kNN +10 điểm.

## Cấu trúc repo

```text
.
├── main.py                 # entry point
├── requirements.txt
├── data/olivettifaces.mat  # ORL local
├── src/
│   ├── config.py           # đường dẫn + hyperparams
│   ├── dataset.py          # load .mat, equalize, split
│   ├── features.py         # Eigenfaces, Fisherfaces, LBP, HOG
│   ├── classifiers.py      # SVM, kNN
│   ├── train.py            # thí nghiệm so sánh
│   ├── evaluate.py
│   └── visualize.py
├── results/                # metrics + figures
└── reports/report.md       # báo cáo Markdown (VI)
```

## Báo cáo

- Markdown: `reports/report.md`

## Nộp bài

Copy toàn bộ project vào thư mục `<MSSV>` (ví dụ `25C01025`), kèm báo cáo PDF (nếu giảng viên yêu cầu), rồi nén `.zip` / `.rar`.
