from __future__ import annotations

import time

import numpy as np
import torch

from physics.derivatives import q4_scalar_grad_matrices, q4_scalar_gradient
from src.validation.fem.linear_elasticity_2d import make_rect_mesh

Tensor = torch.Tensor

K_HT = 1.0
SOURCE_HT = 0.0
THICK_HT = 1.0
CASE_HT = "case1"


def boundary_profiles(ys: np.ndarray, case: str = CASE_HT) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(ys, dtype=np.float64)
    case = str(case).lower()
    if case in ["case1", "bc1", "1"]:
        t_left = 1.00 + 0.20 * np.exp(-((y - 0.15) / 0.18) ** 2) - 0.10 * np.exp(-((y - 0.75) / 0.22) ** 2)
        t_right = -0.10 + 0.35 * np.exp(-((y - 0.30) / 0.12) ** 2) - 0.20 * np.exp(-((y - 0.08) / 0.08) ** 2)
    elif case in ["case2", "bc2", "2"]:
        t_left = 0.85 + 0.25 * np.sin(np.pi * y) + 0.10 * np.sin(3.0 * np.pi * y)
        t_right = 0.05 - 0.25 * np.sin(np.pi * y) + 0.15 * np.cos(2.0 * np.pi * y)
    else:
        raise ValueError("case must be 'case1' or 'case2'")
    return t_left.astype(np.float64), t_right.astype(np.float64)


def make_dirichlet_nodes(xs: np.ndarray, ys: np.ndarray, t_left: np.ndarray, t_right: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx = len(xs)
    ny = len(ys)
    bc: dict[int, float] = {}
    for j in range(ny):
        bc[j * nx] = float(t_left[j])
        bc[j * nx + nx - 1] = float(t_right[j])

    fixed = np.fromiter(sorted(bc), dtype=np.int64)
    fixed_vals = np.asarray([bc[int(n)] for n in fixed], dtype=np.float64)
    all_nodes = np.arange(nx * ny, dtype=np.int64)
    free = np.setdiff1d(all_nodes, fixed, assume_unique=True)
    return fixed, fixed_vals, free


def bc_mask_values(xs: np.ndarray, ys: np.ndarray, t_left: np.ndarray, t_right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nx = len(xs)
    ny = len(ys)
    mask = np.zeros(nx * ny, dtype=bool)
    values = np.zeros(nx * ny, dtype=np.float64)
    fixed, fixed_vals, _ = make_dirichlet_nodes(xs, ys, t_left, t_right)
    mask[fixed] = True
    values[fixed] = fixed_vals
    return mask, values


def assemble_heat_system(
    coords: np.ndarray,
    elements: np.ndarray,
    *,
    k: float = K_HT,
    source: float = SOURCE_HT,
    thickness: float = THICK_HT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coords_t = torch.tensor(coords, dtype=torch.float64)
    elems_t = torch.tensor(elements, dtype=torch.long)
    shape_t, grad_t, det_t, weights_t = q4_scalar_grad_matrices(coords_t, elems_t)
    shape = shape_t.cpu().numpy()
    grad = grad_t.cpu().numpy()
    det_j = det_t.cpu().numpy()
    weights = weights_t.cpu().numpy()

    n_nodes = coords.shape[0]
    k_global = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    f_global = np.zeros(n_nodes, dtype=np.float64)

    for e, conn in enumerate(elements):
        ke = np.zeros((4, 4), dtype=np.float64)
        fe = np.zeros(4, dtype=np.float64)
        for g in range(weights.shape[0]):
            b = grad[e, g]
            n = shape[g]
            dvol = det_j[e, g] * weights[g] * thickness
            ke += k * (b.T @ b) * dvol
            fe += source * n * dvol
        k_global[np.ix_(conn, conn)] += ke
        f_global[conn] += fe

    return k_global, f_global, shape, grad, det_j, weights


def solve_reference(
    nx: int = 21,
    ny: int = 21,
    *,
    case: str = CASE_HT,
    k: float = K_HT,
    source: float = SOURCE_HT,
    thickness: float = THICK_HT,
) -> dict[str, object]:
    t0 = time.perf_counter()
    mesh = make_rect_mesh(nx, ny)
    xs = mesh["xs"]
    ys = mesh["ys"]
    coords = mesh["coords"]
    elems = mesh["elements"]
    t_left, t_right = boundary_profiles(ys, case)
    stiffness, force, shape, grad_mats, det_j, weights = assemble_heat_system(
        coords,
        elems,
        k=k,
        source=source,
        thickness=thickness,
    )
    fixed, fixed_vals, free = make_dirichlet_nodes(xs, ys, t_left, t_right)

    temp = np.zeros(coords.shape[0], dtype=np.float64)
    temp[fixed] = fixed_vals
    rhs = force[free] - stiffness[np.ix_(free, fixed)] @ fixed_vals
    temp[free] = np.linalg.solve(stiffness[np.ix_(free, free)], rhs)

    fields = element_gauss_fields_np(temp, elems, shape, grad_mats, k=k, source=source)
    bc_mask, bc_values = bc_mask_values(xs, ys, t_left, t_right)
    energy = 0.5 * float(temp @ (stiffness @ temp)) - float(force @ temp)
    return {
        **mesh,
        "case": str(case).lower(),
        "k": float(k),
        "source": float(source),
        "thickness": float(thickness),
        "K": stiffness,
        "F": force,
        "shape_vals": shape,
        "grad_mats": grad_mats,
        "det_j": det_j,
        "weights": weights,
        "fixed_nodes": fixed,
        "fixed_vals": fixed_vals,
        "free_nodes": free,
        "bc_mask": bc_mask,
        "bc_values": bc_values,
        "T_left": t_left,
        "T_right": t_right,
        "T": temp,
        "T_grid": temp.reshape(ny, nx),
        "grad_gp": fields["grad_gp"],
        "flux_gp": fields["flux_gp"],
        "qmag_gp": fields["qmag_gp"],
        "energy_density_gp": fields["energy_density_gp"],
        "internal_energy": energy,
        "wall": time.perf_counter() - t0,
    }


def element_gauss_fields_np(
    temp: np.ndarray,
    elements: np.ndarray,
    shape_vals: np.ndarray,
    grad_mats: np.ndarray,
    *,
    k: float = K_HT,
    source: float = SOURCE_HT,
) -> dict[str, np.ndarray]:
    t_nodes = np.asarray(temp, dtype=np.float64).reshape(-1)
    t_e = t_nodes[elements]
    t_gp = np.einsum("gj,ej->eg", shape_vals, t_e)
    grad_gp = np.einsum("egij,ej->egi", grad_mats, t_e)
    flux_gp = -float(k) * grad_gp
    qmag_gp = np.linalg.norm(flux_gp, axis=-1)
    energy_density = 0.5 * float(k) * np.sum(grad_gp**2, axis=-1) - float(source) * t_gp
    return {
        "T_gp": t_gp,
        "grad_gp": grad_gp,
        "flux_gp": flux_gp,
        "qmag_gp": qmag_gp,
        "energy_density_gp": energy_density,
    }


def element_gauss_fields_torch(
    temp: Tensor,
    elements: Tensor,
    shape_vals: Tensor,
    grad_mats: Tensor,
    *,
    k: float = K_HT,
    source: float = SOURCE_HT,
) -> dict[str, Tensor]:
    elems = elements.to(device=temp.device, dtype=torch.long)
    n = shape_vals.to(device=temp.device, dtype=temp.dtype)
    grad_m = grad_mats.to(device=temp.device, dtype=temp.dtype)
    if temp.ndim == 1:
        t_b = temp.unsqueeze(0)
    elif temp.ndim == 2 and temp.shape[-1] == 1:
        t_b = temp[:, 0].unsqueeze(0)
    elif temp.ndim == 2:
        t_b = temp
    elif temp.ndim == 3 and temp.shape[-1] == 1:
        t_b = temp[..., 0]
    else:
        raise ValueError(f"Expected scalar temperature field, got {tuple(temp.shape)}")

    t_e = t_b[:, elems]
    t_gp = torch.einsum("gj,bej->beg", n, t_e)
    grad_gp = q4_scalar_gradient(t_b, elems, grad_m)
    flux_gp = -float(k) * grad_gp
    qmag_gp = torch.linalg.norm(flux_gp, dim=-1)
    energy_density = 0.5 * float(k) * torch.sum(grad_gp.square(), dim=-1) - float(source) * t_gp
    return {
        "T_gp": t_gp,
        "grad_gp": grad_gp,
        "flux_gp": flux_gp,
        "qmag_gp": qmag_gp,
        "energy_density_gp": energy_density,
    }


def particular_linear_bc_np(xs: np.ndarray, ys: np.ndarray, t_left: np.ndarray, t_right: np.ndarray) -> np.ndarray:
    xnorm = (np.asarray(xs, dtype=np.float64) - xs[0]) / (xs[-1] - xs[0])
    return (1.0 - xnorm[None, :]) * t_left[:, None] + xnorm[None, :] * t_right[:, None]


def envelope_phi_np(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    xnorm = (np.asarray(xs, dtype=np.float64) - xs[0]) / (xs[-1] - xs[0])
    phi_x = xnorm * (1.0 - xnorm)
    phi_x = phi_x / max(float(np.max(np.abs(phi_x))), 1e-12)
    return np.tile(phi_x[None, :], (len(ys), 1))


def make_node_features(
    coords: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    t_left: np.ndarray,
    t_right: np.ndarray,
    *,
    case: str = CASE_HT,
    k: float = K_HT,
    source: float = SOURCE_HT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    nx = len(xs)
    ny = len(ys)
    g = particular_linear_bc_np(xs, ys, t_left, t_right)
    phi = envelope_phi_np(xs, ys)
    left_mask = np.zeros((ny, nx), dtype=np.float64)
    right_mask = np.zeros((ny, nx), dtype=np.float64)
    left_mask[:, 0] = 1.0
    right_mask[:, -1] = 1.0
    temp_scale = max(float(np.max(np.abs(t_left))), float(np.max(np.abs(t_right))), 1.0)
    case_id = 0.0 if str(case).lower() in ["case1", "bc1", "1"] else 1.0
    xnorm = (coords[:, 0] - xs[0]) / (xs[-1] - xs[0])
    ynorm = (coords[:, 1] - ys[0]) / (ys[-1] - ys[0])
    features = np.stack(
        [
            xnorm,
            ynorm,
            left_mask.reshape(-1),
            right_mask.reshape(-1),
            g.reshape(-1) / temp_scale,
            phi.reshape(-1),
            float(k) * np.ones(nx * ny, dtype=np.float64),
            float(source) * np.ones(nx * ny, dtype=np.float64),
            case_id * np.ones(nx * ny, dtype=np.float64),
        ],
        axis=-1,
    )
    return features.astype(np.float64), g.astype(np.float64), phi.astype(np.float64), temp_scale


def make_envelope_nodes(
    xs: np.ndarray,
    ys: np.ndarray,
    t_left: np.ndarray,
    t_right: np.ndarray,
    *,
    device=None,
    dtype=torch.float32,
) -> tuple[Tensor, Tensor]:
    g = particular_linear_bc_np(xs, ys, t_left, t_right)
    phi = envelope_phi_np(xs, ys)
    g_nodes = torch.tensor(g.reshape(-1, 1), device=device, dtype=dtype)
    phi_nodes = torch.tensor(phi.reshape(-1, 1), device=device, dtype=dtype)
    return g_nodes, phi_nodes


def apply_envelope(raw: Tensor, g_nodes: Tensor, phi_nodes: Tensor) -> Tensor:
    if raw.ndim == 2:
        return g_nodes + phi_nodes * raw
    return g_nodes.unsqueeze(0) + phi_nodes.unsqueeze(0) * raw


def bc_error(temp: Tensor, fixed_nodes: Tensor, fixed_vals: Tensor) -> Tensor:
    t = temp.reshape(-1)
    nodes = fixed_nodes.to(device=t.device, dtype=torch.long)
    vals = fixed_vals.to(device=t.device, dtype=t.dtype).reshape(-1)
    return torch.max(torch.abs(t[nodes] - vals))


def bc_error_rect_panel_np(T_grid: np.ndarray, ys: np.ndarray, t_left: np.ndarray, t_right: np.ndarray) -> dict[str, float]:
    t = np.asarray(T_grid, dtype=np.float64)
    dy = float(ys[1] - ys[0])
    left_err = t[:, 0] - t_left
    right_err = t[:, -1] - t_right
    bottom_neu = (t[1, :] - t[0, :]) / dy
    top_neu = (t[-1, :] - t[-2, :]) / dy
    return {
        "dirichlet_max_abs": float(np.max(np.abs(np.concatenate([left_err, right_err])))),
        "neumann_max_abs": float(np.max(np.abs(np.concatenate([bottom_neu, top_neu])))),
        "left_max_abs": float(np.max(np.abs(left_err))),
        "right_max_abs": float(np.max(np.abs(right_err))),
    }


def rel_l2(pred: Tensor, ref: Tensor, eps: float = 1e-12) -> Tensor:
    p = pred.reshape(-1).to(dtype=torch.float64)
    r = ref.reshape(-1).to(device=p.device, dtype=p.dtype)
    return torch.linalg.norm(p - r) / torch.linalg.norm(r).clamp_min(eps)


def residual_norm_from_energy(energy_fn, temp: Tensor, free_nodes: Tensor, eps: float = 1e-12) -> Tensor:
    t_var = temp.detach().clone().requires_grad_(True)
    energy = energy_fn(t_var)
    grad = torch.autograd.grad(energy, t_var, create_graph=False)[0].reshape(-1)
    free = free_nodes.to(device=grad.device, dtype=torch.long)
    denom = torch.linalg.norm(grad).clamp_min(eps)
    return torch.linalg.norm(grad[free]) / denom
