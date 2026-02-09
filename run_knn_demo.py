"""
Minimal demo for KNN (pytorch3d) and stats reporting.

Run:
  python -m test_ops.run_knn_demo
"""

import sys
from pathlib import Path

import torch

# Ensure point_ops is on sys.path when running directly.
ROOT = Path(__file__).resolve().parent.parent
POINT_OPS = ROOT / "point_ops"
if str(POINT_OPS) not in sys.path:
    sys.path.insert(0, str(POINT_OPS))

import config
from stats import report as stats_report
# Demo config (edit as needed)
DEVICE = "gpu"  # "gpu" | "cpu"
B = 4
N = 1024
K = 16
INCLUDE_SELF = True

from ops.knn import KNN


def main() -> None:
    config.set_mode("reference")
    config.set_stats_enabled(True)
    config.set_device("cuda" if DEVICE == "gpu" else "cpu")

    use_cuda = DEVICE == "gpu" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # B=1, N=32, D=3
    x = torch.arange(N, dtype=torch.float32, device=device)
    xyz = torch.stack([x, torch.zeros_like(x), torch.zeros_like(x)], dim=1).unsqueeze(0)

    print("device:", device)
    print("--- include_self=True ---")
    op1 = KNN(k=K, include_self=INCLUDE_SELF)
    idx1, dist1, mask1 = op1(xyz)
    print("idx:", idx1)
    print("dist2:", dist1)
    print("mask:", mask1)

    print("--- include_self=False ---")
    op2 = KNN(k=K, include_self=False)
    idx2, dist2, mask2 = op2(xyz)
    print("idx:", idx2)
    print("dist2:", dist2)
    print("mask:", mask2)

    print("=== stats report ===")
    print(stats_report())


if __name__ == "__main__":
    main()
