"""
Minimal demo for SVD transform solver and stats reporting.

Run:
  python -m test_ops.run_svd_demo
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
from ops.svd import SVD

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise ImportError("torch is required for this demo") from exc


# Demo config (edit as needed)
DEVICE = "gpu"  # "gpu" | "cpu"
SEED = 7
B = 2
P = 64
NOISE_STD = 0.0
RIGID_ERR_TOL = 5e-4
SIM_ERR_TOL = 5e-4
SCALE_TOL = 5e-4


def _random_rotation(batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    a = torch.randn(batch, 3, 3, device=device, dtype=dtype)
    q, _ = torch.linalg.qr(a)
    det = torch.linalg.det(q)
    q[det < 0, :, 2] *= -1.0
    return q


def _transform_points(
    src: torch.Tensor, r: torch.Tensor, t: torch.Tensor, s: torch.Tensor
) -> torch.Tensor:
    # Row-vector form: x' = s * x * R^T + t
    return s[:, None, None] * (src @ r.transpose(1, 2)) + t[:, None, :]


def _apply_t(src: torch.Tensor, tmat: torch.Tensor) -> torch.Tensor:
    rs = tmat[:, :3, :3]
    tt = tmat[:, :3, 3]
    return src @ rs.transpose(1, 2) + tt[:, None, :]


def _extract_scale(rs: torch.Tensor) -> torch.Tensor:
    # For rs = sR, each column norm is |s|. Take mean column norm.
    return torch.linalg.norm(rs, dim=1).mean(dim=1)


def main() -> None:
    config.set_mode("reference")
    config.set_stats_enabled(True)
    config.set_device("cuda" if DEVICE == "gpu" else "cpu")

    use_cuda = DEVICE == "gpu" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(SEED)

    src = torch.randn(B, P, 3, device=device, dtype=torch.float32)
    if NOISE_STD > 0:
        src = src + NOISE_STD * torch.randn_like(src)

    # Case A: rigid (no scaling)
    r_gt = _random_rotation(B, device, src.dtype)
    t_gt = torch.randn(B, 3, device=device, dtype=src.dtype)
    s_gt = torch.ones(B, device=device, dtype=src.dtype)
    tgt_rigid = _transform_points(src, r_gt, t_gt, s_gt)

    rigid_op = SVD(with_scaling=False, name="svd_rigid_demo")
    t_rigid = rigid_op(src, tgt_rigid)
    pred_rigid = _apply_t(src, t_rigid)
    err_rigid = torch.linalg.norm(pred_rigid - tgt_rigid, dim=-1)
    det_rigid = torch.linalg.det(t_rigid[:, :3, :3])

    assert t_rigid.shape == (B, 4, 4), f"Unexpected rigid T shape: {t_rigid.shape}"
    assert torch.all(det_rigid > 0.0), "Rigid solve returned reflection (det <= 0)."
    assert float(err_rigid.max()) < RIGID_ERR_TOL, f"Rigid reprojection error too high: {float(err_rigid.max())}"

    print("device:", device)
    print("[Case A] rigid")
    print("  T shape:", tuple(t_rigid.shape))
    print("  det(R):", det_rigid.tolist())
    print("  reproj mean/max:", float(err_rigid.mean()), float(err_rigid.max()))

    # Case B: similarity (with scaling)
    r_gt2 = _random_rotation(B, device, src.dtype)
    t_gt2 = torch.randn(B, 3, device=device, dtype=src.dtype)
    s_gt2 = 0.5 + 1.5 * torch.rand(B, device=device, dtype=src.dtype)
    tgt_sim = _transform_points(src, r_gt2, t_gt2, s_gt2)

    sim_op = SVD(with_scaling=True, name="svd_sim_demo")
    t_sim = sim_op(src, tgt_sim)
    pred_sim = _apply_t(src, t_sim)
    err_sim = torch.linalg.norm(pred_sim - tgt_sim, dim=-1)

    rs_sim = t_sim[:, :3, :3]
    s_hat = _extract_scale(rs_sim)
    r_hat = rs_sim / s_hat[:, None, None].clamp_min(torch.finfo(rs_sim.dtype).eps)
    det_sim = torch.linalg.det(r_hat)

    assert t_sim.shape == (B, 4, 4), f"Unexpected similarity T shape: {t_sim.shape}"
    assert torch.all(det_sim > 0.0), "Similarity solve returned reflection (det <= 0)."
    assert float(err_sim.max()) < SIM_ERR_TOL, f"Similarity reprojection error too high: {float(err_sim.max())}"
    assert float(torch.max(torch.abs(s_hat - s_gt2))) < SCALE_TOL, "Estimated scale mismatch."

    print("[Case B] similarity")
    print("  det(R):", det_sim.tolist())
    print("  scale gt:", s_gt2.tolist())
    print("  scale pred:", s_hat.tolist())
    print("  reproj mean/max:", float(err_sim.mean()), float(err_sim.max()))

    # Case C: reflection target (solver must still output proper rotation det > 0)
    refl = torch.tensor(
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=src.dtype,
    ).unsqueeze(0).repeat(B, 1, 1)
    t_ref = torch.randn(B, 3, device=device, dtype=src.dtype)
    tgt_reflect = src @ refl.transpose(1, 2) + t_ref[:, None, :]

    reflect_op = SVD(with_scaling=False, name="svd_reflect_demo")
    t_reflect = reflect_op(src, tgt_reflect)
    det_reflect = torch.linalg.det(t_reflect[:, :3, :3])
    err_reflect = torch.linalg.norm(_apply_t(src, t_reflect) - tgt_reflect, dim=-1)

    assert torch.all(det_reflect > 0.0), "Reflection case returned improper rotation."
    print("[Case C] reflection correction")
    print("  det(R):", det_reflect.tolist())
    print("  reproj mean/max:", float(err_reflect.mean()), float(err_reflect.max()))

    print("\n=== stats report ===")
    print(stats_report())


if __name__ == "__main__":
    main()
