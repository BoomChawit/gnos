from __future__ import annotations

import torch

Tensor = torch.Tensor


def relative_l2(pred: Tensor, ref: Tensor, x_nodes: Tensor | None = None, eps: float = 1e-12) -> Tensor:
    p = pred.reshape(-1).to(dtype=torch.float64)
    r = ref.reshape(-1).to(device=p.device, dtype=p.dtype)
    if x_nodes is None:
        return torch.linalg.norm(p - r) / torch.linalg.norm(r).clamp_min(eps)
    x = x_nodes.reshape(-1).to(device=p.device, dtype=p.dtype)
    diff2 = (p - r).square()
    ref2 = r.square()
    num = torch.trapezoid(diff2, x).clamp_min(0.0).sqrt()
    den = torch.trapezoid(ref2, x).clamp_min(eps).sqrt()
    return num / den


def to_float_dict(metrics: dict[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in metrics.items():
        if isinstance(value, Tensor):
            out[key] = float(value.detach().cpu())
        else:
            out[key] = value
    return out

