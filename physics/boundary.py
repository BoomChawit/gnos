from __future__ import annotations

import torch

Tensor = torch.Tensor


def make_dirichlet_fixed_fixed(
    n_node: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    idx = torch.tensor([0, n_node - 1], device=device, dtype=torch.long)
    val = torch.zeros(2, device=device, dtype=dtype)
    return idx, val


def adf_1d_fixed_fixed(x_normalized: Tensor) -> Tensor:
    """Approximate distance function for fixed-fixed 1D boundaries."""
    x = x_normalized.reshape(-1)
    return x * (1.0 - x)


def apply_dirichlet_envelope(raw: Tensor, x_normalized: Tensor, left: float = 0.0, right: float = 0.0) -> Tensor:
    x = x_normalized.reshape(-1).to(device=raw.device, dtype=raw.dtype)
    phi = adf_1d_fixed_fixed(x).to(device=raw.device, dtype=raw.dtype)
    lift = (1.0 - x) * left + x * right
    if raw.ndim == 1:
        return lift + phi * raw
    return lift.unsqueeze(0) + phi.unsqueeze(0) * raw


def bc_error(u: Tensor, dirichlet_idx: Tensor, dirichlet_val: Tensor) -> Tensor:
    if u.ndim == 1:
        u = u.unsqueeze(0)
    vals = dirichlet_val.to(device=u.device, dtype=u.dtype).reshape(1, -1)
    return torch.linalg.norm(u[:, dirichlet_idx] - vals)

