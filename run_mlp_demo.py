"""
Minimal demo for point_ops MLP.

Run:
  python -m test_ops.run_mlp_demo

Edit demo parameters at the top of this file.
"""

import sys
from pathlib import Path

# Ensure point_ops is on sys.path when running directly.
ROOT = Path(__file__).resolve().parent.parent
POINT_OPS = ROOT / "point_ops"
if str(POINT_OPS) not in sys.path:
    sys.path.insert(0, str(POINT_OPS))

import config
from stats import report as stats_report
from ops.mlp.linear import Linear
from ops.mlp.normalization import Normalization
from ops.mlp.activation import Activation

# Demo config (edit as needed)
DEVICE = "gpu"  # "cpu" | "gpu"
B = 2
N = 5
C_IN = 3
HIDDEN = 64
C_OUT = 128
NORM_TYPE = "ln"  # "ln" | "bn" | "none"
ACT_TYPE = "relu"

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise ImportError("torch is required for this demo") from exc


def main() -> None:
    config.set_mode("reference")
    config.set_stats_enabled(True)
    config.set_device("cuda" if DEVICE == "gpu" else "cpu")

    use_cuda = DEVICE == "gpu" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # [B, N, C] input
    x = torch.randn(B, N, C_IN, device=device)

    # Simulate a tiny MLP: Linear -> Norm -> Act -> Linear
    fc1 = Linear(C_IN, HIDDEN)
    norm1 = Normalization(HIDDEN, norm_type=NORM_TYPE)
    act1 = Activation(ACT_TYPE)
    fc2 = Linear(HIDDEN, C_OUT)

    y = fc2(act1(norm1(fc1(x))))

    print("device:", device)
    print("input shape:", tuple(x.shape))
    print("output shape:", tuple(y.shape))
    print("\n=== stats report ===")
    print(stats_report())


if __name__ == "__main__":
    main()
