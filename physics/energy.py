from __future__ import annotations

from collections.abc import Callable

import torch

from physics.derivatives import element_gradient_1d

Tensor = torch.Tensor


def _as_batch(u: Tensor) -> Tensor:
    if u.ndim == 1:
        return u.unsqueeze(0)
    if u.ndim == 2:
        return u
    if u.ndim == 3 and u.shape[-1] == 1:
        return u[..., 0]
    raise ValueError(f"Expected scalar 1D field, got {tuple(u.shape)}")


def linear_elastic_energy(u: Tensor, x_nodes: Tensor, f_ext: Tensor, *, area: float, young: float) -> Tensor:
    u_b = _as_batch(u)
    x = x_nodes.reshape(-1).to(device=u_b.device, dtype=u_b.dtype)
    f = f_ext.reshape(-1).to(device=u_b.device, dtype=u_b.dtype)
    h = torch.diff(x)
    eps = element_gradient_1d(u_b, x)
    strain_energy = 0.5 * area * young * torch.sum(eps.square() * h.unsqueeze(0), dim=1)
    external_work = torch.sum(u_b * f.unsqueeze(0), dim=1)
    return torch.mean(strain_energy - external_work)


def nonlinear_elastic_energy(
    u: Tensor,
    x_nodes: Tensor,
    f_ext: Tensor,
    *,
    area: float,
    energy_density: Callable[[Tensor], Tensor],
) -> Tensor:
    u_b = _as_batch(u)
    x = x_nodes.reshape(-1).to(device=u_b.device, dtype=u_b.dtype)
    f = f_ext.reshape(-1).to(device=u_b.device, dtype=u_b.dtype)
    h = torch.diff(x)
    eps = element_gradient_1d(u_b, x)
    strain_energy = area * torch.sum(energy_density(eps) * h.unsqueeze(0), dim=1)
    external_work = torch.sum(u_b * f.unsqueeze(0), dim=1)
    return torch.mean(strain_energy - external_work)


def residual_norm_from_energy(
    energy_fn: Callable[[Tensor], Tensor],
    u: Tensor,
    free_idx: Tensor,
    f_ext: Tensor | None = None,
    eps: float = 1e-12,
) -> Tensor:
    u_var = _as_batch(u).detach().clone().requires_grad_(True)
    energy = energy_fn(u_var)
    residual = torch.autograd.grad(energy, u_var, create_graph=False)[0][:, free_idx]
    denom = torch.ones((), device=u_var.device, dtype=u_var.dtype)
    if f_ext is not None:
        denom = torch.linalg.norm(f_ext.reshape(-1).to(u_var)[free_idx]).clamp_min(eps)
    return torch.linalg.norm(residual) / denom

