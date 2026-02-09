"""
Minimal demo for Group (feature gather) and stats reporting.

Run:
  python -m test_ops.run_group_demo
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
B = 2
N = 8
C = 4
M = 3
K = 5

from ops.group import Group


def _build_feat() -> torch.Tensor:
    # Build an easy-to-check feature tensor: value encodes (b, n, c)
    # feat[b, n, c] = 1000*b + 10*n + c
    feat = torch.zeros(B, N, C, dtype=torch.float32)
    for b in range(B):
        for n in range(N):
            for c in range(C):
                feat[b, n, c] = 1000 * b + 10 * n + c
    return feat


def main() -> None:
    config.set_mode("reference")
    config.set_stats_enabled(True)
    config.set_device("cuda" if DEVICE == "gpu" else "cpu")

    use_cuda = DEVICE == "gpu" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    feat = _build_feat().to(device)

    # Build idx with a mix of valid indices and -1 padding
    idx = torch.tensor(
        [
            [[0, 2, 4, -1, 7], [1, 3, -1, 5, 6], [7, 6, 5, 4, -1]],
            [[2, 1, -1, -1, 0], [6, 7, 5, 4, 3], [0, -1, 1, 2, 3]],
        ],
        dtype=torch.int64,
        device=device,
    )

    op = Group()
    grouped = op(feat, idx)

    # Shape checks
    assert grouped.shape == (B, M, K, C), f"Unexpected shape: {grouped.shape}"

    # Value checks: compare against manual gather (CPU) with -1 -> 0
    feat_cpu = feat.cpu()
    idx_cpu = idx.cpu()
    grouped_cpu = grouped.cpu()

    for b in range(B):
        for m in range(M):
            for k in range(K):
                j = int(idx_cpu[b, m, k])
                if j < 0:
                    assert torch.all(grouped_cpu[b, m, k] == 0), "Invalid idx not zeroed"
                else:
                    expected = feat_cpu[b, j]
                    got = grouped_cpu[b, m, k]
                    assert torch.allclose(got, expected), f"Mismatch at b={b},m={m},k={k}"

    print("device:", device)
    print("feat shape:", tuple(feat.shape))
    print("idx shape:", tuple(idx.shape))
    print("grouped shape:", tuple(grouped.shape))
    print("\n=== stats report ===")
    print(stats_report())


if __name__ == "__main__":
    main()
