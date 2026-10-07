# Nhận dạng khuôn mặt bằng CNN / Vision Transformer (Olivetti Faces)

Bài tập lập trình 2 môn Thị giác máy tính: **cùng bài toán với bài 1** (nhận dạng 40 người trên Olivetti Faces), nhưng giải bằng **CNN và Vision Transformer** thay cho đặc trưng truyền thống.

**Sinh viên:** Cao Anh Tuan — MSHV **25C01025**

## Bài toán và dữ liệu

- **Olivetti Faces** (AT&T ORL): 400 ảnh xám 64×64, 40 người × 10 ảnh. File local: `data/olivettifaces.mat`.
- **Chia tập giống hệt bài 1**: stratified 8 train / 2 test mỗi người, `random_state=42` → 320 / 80.
- Trong 320 ảnh train, tách 1 ảnh/người (40 ảnh) làm **validation** để chọn số epoch, sau đó **huấn luyện lại trên đủ 320 ảnh**. 80 ảnh test chỉ dùng để đánh giá cuối.

## Kỹ thuật áp dụng

| Model | Loại | Tham số | Ghi chú |
|---|---|---|---|
| SimpleCNN | CNN train từ đầu | 0.29M | 3 khối [conv3×3-BN-ReLU]×2 + max-pool, input 64×64×1 |
| ResNet-18 | CNN kinh điển (residual) | 11.2M | pretrained ImageNet, fine-tune toàn bộ |
| MobileNetV3-Small | CNN nhẹ (depthwise, SE) | 1.56M | pretrained ImageNet, fine-tune toàn bộ |
| DeiT-Tiny | Vision Transformer | 5.53M | pretrained ImageNet (timm), patch 16×16 |

Huấn luyện: augmentation (lật ngang, xoay ±10°, dịch/scale, độ sáng/tương phản), AdamW (weight decay 0.05), cosine LR, label smoothing 0.1, batch 32. Input 224×224×3 chuẩn hoá ImageNet cho model pretrained.

Minh hoạ đặc trưng: feature map ResNet-18, Grad-CAM (CNN), attention rollout (ViT), t-SNE embedding.

## Cài đặt & chạy

```bash
git clone https://github.com/tuancaokaizen/hcmus_computer_vision.git
cd hcmus_computer_vision/CNN_computer_vision

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 main.py            # chạy đầy đủ, ~4 phút trên Apple Silicon (MPS)
python3 main.py --quick    # 2 epoch mỗi model, chỉ để kiểm tra pipeline
```

- Cần khoảng **1 GB** cho `.venv` (PyTorch) và ~80 MB pretrained weights (tự tải vào `.cache/` ở lần chạy đầu).
- Tự chọn thiết bị: MPS (Mac) → CUDA → CPU. Trên CPU vẫn chạy được nhưng chậm hơn.
- Nếu `python3 -m venv .venv` treo ở bước `ensurepip`, dùng `python3 -m venv --without-pip .venv` rồi cài pip bằng `get-pip.py` (xem README bài 1).

Kết quả:

- `results/metrics.json` — accuracy, macro precision/recall/F1, per-class, thời gian
- `results/figures/` — toàn bộ hình minh hoạ (xem bảng dưới)

## Kết quả (80 ảnh test)

| Model | Accuracy | Precision | Recall | F1 | Sai / 80 |
|---|---|---|---|---|---|
| SimpleCNN (từ đầu) | 95.00% | 0.967 | 0.950 | 0.947 | 4 |
| ResNet-18 | **100%** | **1.000** | **1.000** | **1.000** | 0 |
| **MobileNetV3-Small** | **100%** | **1.000** | **1.000** | **1.000** | 0 |
| DeiT-Tiny (ViT) | 98.75% | 0.992 | 0.988 | 0.987 | 1 |
| *Eigenfaces+SVM (bài 1)* | *97.50%* | *0.983* | *0.975* | *0.973* | *2* |

Precision / Recall / F1 là trung bình macro trên 40 người. Model đề xuất: **MobileNetV3-Small** (bằng ResNet-18 về độ chính xác nhưng ít tham số hơn 7 lần).

## Hình minh hoạ (`results/figures/`)

| File | Nội dung |
|---|---|
| `sample_gallery.png`, `augmentation_examples.png` | Dữ liệu và augmentation |
| `training_curves.png` | Loss / accuracy theo epoch |
| `metrics_comparison.png`, `efficiency.png` | So sánh Accuracy / Precision / Recall / F1, kích thước model |
| `confusion_matrix_best.png`, `confusion_matrices_all.png` | Confusion matrix |
| `per_class_prf_best.png`, `per_class_recall_precision_all.png` | Precision / Recall theo từng người |
| `feature_maps_resnet18.png` | Feature map qua các stage của ResNet-18 |
| `explanations_gradcam_attention.png` | Grad-CAM (CNN) và attention rollout (ViT) |
| `tsne_embeddings.png` | t-SNE của embedding |
| `errors_all_models.png`, `prediction_samples.png` | Phân tích lỗi, ví dụ dự đoán |

## Cấu trúc

```text
CNN_computer_vision/
├── main.py                 # entry point (--quick để chạy thử)
├── requirements.txt
├── data/olivettifaces.mat
├── src/
│   ├── config.py           # đường dẫn + siêu tham số từng model
│   ├── dataset.py          # load, chia train/val/test, augmentation
│   ├── models.py           # SimpleCNN, ResNet-18, MobileNetV3-S, DeiT-Tiny
│   ├── train.py            # huấn luyện 2 giai đoạn + xuất kết quả
│   ├── evaluate.py         # accuracy, precision, recall, F1, confusion matrix
│   ├── explain.py          # feature map, Grad-CAM, attention rollout
│   └── visualize.py        # vẽ hình
├── results/                # metrics.json + figures/
└── reports/report.md       # báo cáo Markdown (VI)
```

## Báo cáo

- Markdown: `reports/report.md`
