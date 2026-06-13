from __future__ import annotations

from collections.abc import Callable

import torch

from physics.derivatives import element_gradient_1d, q4_element_strain, q4_scalar_gradient

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


def _as_batch_scalar_nodes(u: Tensor) -> Tensor:
    if u.ndim == 1:
        return u.unsqueeze(0)
    if u.ndim == 2 and u.shape[-1] == 1:
        return u[:, 0].unsqueeze(0)
    if u.ndim == 2:
        return u
    if u.ndim == 3 and u.shape[-1] == 1:
        return u[..., 0]
    raise ValueError(f"Expected scalar nodal field, got {tuple(u.shape)}")


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


def nonlinear_hardening_q4_energy_2d(
    u: Tensor,
    elements: Tensor,
    b_mats: Tensor,
    det_j: Tensor,
    weights: Tensor,
    d_matrix: Tensor,
    *,
    alpha: float,
    p: float,
    thickness: float = 1.0,
    eps_reg: float = 1e-12,
) -> Tensor:
    u_b = _as_batch_vector(u)
    elems = elements.to(device=u_b.device, dtype=torch.long)
    b = b_mats.to(device=u_b.device, dtype=u_b.dtype)
    det = det_j.to(device=u_b.device, dtype=u_b.dtype)
    w = weights.to(device=u_b.device, dtype=u_b.dtype)
    d = d_matrix.to(device=u_b.device, dtype=u_b.dtype)
    strain = q4_element_strain(u_b, elems, b)
    d_eps = torch.einsum("ij,begj->begi", d, strain)
    q = torch.sum(strain * d_eps, dim=-1).clamp_min(0.0)
    kappa = torch.sqrt(q + eps_reg**2)
    density = 0.5 * q + alpha / (p + 2.0) * (kappa ** (p + 2.0) - eps_reg ** (p + 2.0))
    energy = torch.sum(density * det.unsqueeze(0) * w.reshape(1, 1, -1), dim=(1, 2))
    return torch.mean(energy * thickness)


def heat_q4_energy_2d(
    field: Tensor,
    elements: Tensor,
    shape_vals: Tensor,
    grad_mats: Tensor,
    det_j: Tensor,
    weights: Tensor,
    *,
    k: float | Tensor,
    source: float | Tensor = 0.0,
    thickness: float = 1.0,
) -> Tensor:
    t_b = _as_batch_scalar_nodes(field)
    elems = elements.to(device=t_b.device, dtype=torch.long)
    n = shape_vals.to(device=t_b.device, dtype=t_b.dtype)
    grad_m = grad_mats.to(device=t_b.device, dtype=t_b.dtype)
    det = det_j.to(device=t_b.device, dtype=t_b.dtype)
    w = weights.to(device=t_b.device, dtype=t_b.dtype)
    k_t = torch.as_tensor(k, device=t_b.device, dtype=t_b.dtype)

    t_e = t_b[:, elems]
    grad = q4_scalar_gradient(t_b, elems, grad_m)
    t_gp = torch.einsum("gj,bej->beg", n, t_e)

    if torch.is_tensor(source):
        src = source.to(device=t_b.device, dtype=t_b.dtype)
        if src.ndim == 0:
            source_gp = src
        else:
            source_b = _as_batch_scalar_nodes(src)
            source_e = source_b[:, elems]
            source_gp = torch.einsum("gj,bej->beg", n, source_e)
    else:
        source_gp = torch.as_tensor(float(source), device=t_b.device, dtype=t_b.dtype)

    density = 0.5 * k_t * torch.sum(grad.square(), dim=-1) - source_gp * t_gp
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
