from __future__ import annotations

from collections.abc import Callable

import torch

from physics.derivatives import element_gradient_1d, q4_element_strain

Tensor = torch.Tensor


def _as_batch(u: Tensor) -> Tensor:
    if u.ndim == 1:
        return u.unsqueeze(0)
    if u.ndim == 2:
        return u
    if u.ndim == 3 and u.shape[-1] == 1:
        return u[..., 0]
    raise ValueError(f"Expected scalar 1D field, got {tuple(u.shape)}")


def _as_batch_vector(u: Tensor) -> Tensor:
    if u.ndim == 2:
        return u.unsqueeze(0)
    if u.ndim == 3:
        return u
    raise ValueError(f"Expected vector nodal field, got {tuple(u.shape)}")


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


def thermo_nonlinear_elastic_energy_1d(
    u: Tensor,
    x_nodes: Tensor,
    f_ext: Tensor,
    *,
    area: float,
    thermal_strain_elem: Tensor,
    energy_density: Callable[[Tensor], Tensor],
) -> Tensor:
    u_b = _as_batch(u)
    x = x_nodes.reshape(-1).to(device=u_b.device, dtype=u_b.dtype)
    f = f_ext.reshape(-1).to(device=u_b.device, dtype=u_b.dtype)
    h = torch.diff(x)
    eps = element_gradient_1d(u_b, x)
    eps_th = thermal_strain_elem.reshape(1, -1).to(device=u_b.device, dtype=u_b.dtype)
    mech = eps - eps_th
    strain_energy = area * torch.sum(energy_density(mech) * h.unsqueeze(0), dim=1)
    external_work = torch.sum(u_b * f.unsqueeze(0), dim=1)
    return torch.mean(strain_energy - external_work)


def diffusion_source_energy_1d(
    field: Tensor,
    x_nodes: Tensor,
    f_ext: Tensor,
    *,
    diffusion_area_fn: Callable[[Tensor], Tensor],
) -> Tensor:
    """Energy for -d/dx(a(x) du/dx) = q with Dirichlet boundaries."""
    u_b = _as_batch(field)
    x = x_nodes.reshape(-1).to(device=u_b.device, dtype=u_b.dtype)
    f = f_ext.reshape(-1).to(device=u_b.device, dtype=u_b.dtype)
    h = torch.diff(x)
    x_mid = 0.5 * (x[:-1] + x[1:])
    grad = element_gradient_1d(u_b, x)
    a_mid = diffusion_area_fn(x_mid).to(device=u_b.device, dtype=u_b.dtype)
    stored = 0.5 * torch.sum(a_mid.unsqueeze(0) * grad.square() * h.unsqueeze(0), dim=1)
    external = torch.sum(u_b * f.unsqueeze(0), dim=1)
    return torch.mean(stored - external)


def linear_elastic_q4_energy_2d(
    u: Tensor,
    elements: Tensor,
    b_mats: Tensor,
    det_j: Tensor,
    weights: Tensor,
    d_matrix: Tensor,
    *,
    thickness: float = 1.0,
) -> Tensor:
    u_b = _as_batch_vector(u)
    elems = elements.to(device=u_b.device, dtype=torch.long)
    b = b_mats.to(device=u_b.device, dtype=u_b.dtype)
    det = det_j.to(device=u_b.device, dtype=u_b.dtype)
    w = weights.to(device=u_b.device, dtype=u_b.dtype)
    d = d_matrix.to(device=u_b.device, dtype=u_b.dtype)
    strain = q4_element_strain(u_b, elems, b)
    stress = torch.einsum("ij,begj->begi", d, strain)
    density = 0.5 * torch.sum(strain * stress, dim=-1)
    energy = torch.sum(density * det.unsqueeze(0) * w.reshape(1, 1, -1), dim=(1, 2))
    return torch.mean(energy * thickness)


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
