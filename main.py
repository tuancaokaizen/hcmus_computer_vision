"""Entry point for the traditional Olivetti face-recognition assignment.
Điểm chạy bài nhận dạng khuôn mặt Olivetti bằng phương pháp truyền thống.
"""

from __future__ import annotations

import os
from pathlib import Path

# Writable caches for sandbox / macOS matplotlib + joblib
# Cache ghi được cho matplotlib + joblib trên macOS
ROOT = Path(__file__).resolve().parent
_mpl_dir = ROOT / "results" / ".mplconfig"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib

matplotlib.use("Agg")

from src.train import run_experiment


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    main()
