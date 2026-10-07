"""Entry point for the CNN / Vision Transformer Olivetti face-recognition assignment.
Điểm chạy bài nhận dạng khuôn mặt Olivetti bằng CNN / Vision Transformer.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Keep matplotlib config and pretrained weights inside the project /
# Giữ cấu hình matplotlib và trọng số pretrained trong thư mục dự án
ROOT = Path(__file__).resolve().parent
_cache = ROOT / ".cache"
(_cache / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache / "matplotlib"))
os.environ.setdefault("TORCH_HOME", str(_cache / "torch"))
os.environ.setdefault("HF_HOME", str(_cache / "huggingface"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# python.org macOS builds lack system CA certs; use certifi for weight downloads /
# Python cài từ python.org trên macOS thiếu chứng chỉ CA; dùng certifi để tải trọng số
try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass

import matplotlib

matplotlib.use("Agg")

from src.train import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="CNN / ViT face recognition on Olivetti Faces")
    parser.add_argument("--quick", action="store_true", help="2 epochs per model (smoke test)")
    run_experiment(quick=parser.parse_args().quick)


if __name__ == "__main__":
    main()
