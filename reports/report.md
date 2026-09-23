# Báo cáo — Nhận dạng khuôn mặt bằng đặc trưng thị giác truyền thống

**Sinh viên:** Cao Anh Tuan  
**MSHV:** 25C01025  
**Môn:** Computer Vision — bài tập cá nhân

## 1. Giới thiệu

Bài toán: nhận dạng danh tính người (closed-set) từ ảnh khuôn mặt thẳng, nền tối, chiếu sáng/biểu cảm thay đổi nhẹ. Bộ **Olivetti Faces** (AT&T ORL): 40 người × 10 ảnh 64×64. Yêu cầu: **phương pháp truyền thống** — trích đặc trưng thị giác + bộ phân lớp, không dùng CNN.

Không chọn MNIST vì đề gợi ý ảnh thật (mặt, lá, quả, cảm xúc); Olivetti trùng ví dụ “nhận dạng khuôn mặt” và có dataset sẵn.

## 2. Phương pháp

### 2.1. Tiền xử lý

Mỗi ảnh cân bằng histogram độc lập (`skimage.exposure.equalize_hist`). Split stratified 80/20 → **8 train / 2 test mỗi người**, `random_state=42` (320 / 80).

### 2.2. Đặc trưng (`src/features.py`)

| Đặc trưng | Chi tiết |
|---|---|
| Pixel thô (baseline) | Vector 4096 chiều |
| Eigenfaces | PCA k=100, whitening (Turk & Pentland, 1991) |
| Fisherfaces | PCA → LDA, tối đa C−1=39 chiều (Belhumeur et al., 1997) |
| LBP | Uniform P=8, R=2 trên `uint8`; lưới 8×8 → histogram nối (Ahonen) |
| HOG | 9 hướng, cell 8×8, block 2×2, L2-Hys (Dalal & Triggs) |

### 2.3. Bộ phân lớp (`src/classifiers.py`)

| Descriptor | Classifier |
|---|---|
| pixels | SVM RBF (C=10) |
| Eigenfaces | kNN-3 **và** SVM RBF |
| Fisherfaces | kNN-3 |
| LBP | SVM linear |
| HOG | SVM RBF |

Không CNN, không Siamese/triplet, không ArcFace.

### 2.4. Pipeline code

`main.py` → `src/train.py`: load `.mat` → equalize → extract features → fit 6 mô hình → ghi `results/metrics.json` + figures.

## 3. Thí nghiệm

```bash
pip install -r requirements.txt
python main.py
```

Dataset: `data/olivettifaces.mat`. Metric: top-1 accuracy trên 80 ảnh test + confusion matrix 40 lớp.

## 4. Kết quả

Số liệu từ `results/metrics.json` (seed 42):

| Mô hình | Accuracy | Đúng / 80 |
|---|---|---|
| pixels+SVM (baseline) | 95.00% | 76 |
| eigenfaces+kNN | 87.50% | 70 |
| **eigenfaces+SVM** | **97.50%** | **78** |
| fisherfaces+kNN | 97.50% | 78 |
| lbp+SVM | 97.50% | 78 |
| hog+SVM | 97.50% | 78 |

Hình trong `results/figures/`:

- `sample_gallery.png` — một ảnh / người (ID 0–9)
- `eigenfaces.png` — mean face + PC 1…12
- `reconstructions.png` — tái tạo 10 / 50 / 100 PCs
- `accuracy_comparison.png` — bar **ngang** (dễ đọc trong cột hẹp)
- `confusion_matrix_best.png` — Eigenfaces+SVM
- `prediction_samples.png` — đúng (xanh) / sai (đỏ); không còn ô trống chồng nhãn

### Nhận xét

- Baseline pixel+SVM đã 95% vì ORL căn chỉnh, gallery nhỏ — không thay thế yêu cầu “có đặc trưng thị giác”.
- Cùng Eigenfaces: SVM (97.5%) hơn kNN (87.5%) **+10 điểm** → lựa chọn classifier quan trọng không kém descriptor.
- Bốn mô hình truyền thống cùng 78/80: lưới accuracy thô (mỗi lỗi = 1.25%); coi là hòa, không xếp hạng cứng.
- Hai lỗi Eigenfaces+SVM: nhầm danh tính (true 9→pred 7, true 4→pred 39), không phải lệch pose mạnh.

**Mô hình đề xuất:** Eigenfaces + SVM.

## 5. Hạn chế & hướng mở rộng

Olivetti là gallery kiểm soát — 97.5% không suy ra LFW/IJB. Split 8/2 có thể chia sẻ cue phiên (kính, biểu cảm). Mở rộng vẫn truyền thống: LBP đa tỉ lệ, Gabor, leave-one-out, grid search k/C. Deep learning nằm ngoài phạm vi bài tập.

## 6. Kết luận

Pipeline truyền thống (histogram equalization → Eigenfaces / Fisherfaces / LBP / HOG → SVM / kNN) đạt yêu cầu đề. Eigenfaces+SVM (và ba cặp tương đương) **97.5%**; baseline pixel 95.0%.

## Tài liệu tham khảo

1. Turk, M. & Pentland, A. (1991). Eigenfaces for Recognition.
2. Belhumeur, P. N., Hespanha, J. P. & Kriegman, D. J. (1997). Eigenfaces vs. Fisherfaces.
3. Ahonen, T., Hadid, A. & Pietikäinen, M. (2004/2006). Face Description with Local Binary Patterns.
4. Dalal, N. & Triggs, B. (2005). Histograms of Oriented Gradients for Human Detection.
5. Samaria, F. S. & Harter, A. C. (1994). ORL / Olivetti face database.
6. Zhao, W. et al. (2003). Face Recognition: A Literature Survey.
