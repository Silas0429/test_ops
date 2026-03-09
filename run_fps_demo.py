"""
Minimal demo for FPS (pytorch3d) and stats reporting.

Run:
  python -m test_ops.run_fps_demo
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Ensure repo root is on sys.path when running directly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import point_ops
# Demo config (edit as needed)
DEVICE = "gpu"  # "gpu" | "cpu"
B = 4
N = 1024
NUM_SAMPLES = 16
DETERMINISTIC = True

def main() -> None:
    point_ops.config.set_mode("reference")
    point_ops.config.set_stats_enabled(True)
    point_ops.config.set_device("cuda" if DEVICE == "gpu" else "cpu")

    use_cuda = DEVICE == "gpu" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # B=4, N=32, D=3
    xyz = torch.arange(B * N * 3, dtype=torch.float32, device=device).reshape(B, N, 3)
    print("device:", device)
    print(xyz)

    op = point_ops.FPS(num_samples=NUM_SAMPLES, deterministic=DETERMINISTIC)
    try:
        idx = op(xyz)
    except ImportError as exc:
        print("pytorch3d not available:", exc)
        return

    print("idx:\n", idx)
    print("\n=== stats report ===")
    print(point_ops.stats_report())


if __name__ == "__main__":
    main()
