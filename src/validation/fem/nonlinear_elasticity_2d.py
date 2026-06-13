from __future__ import annotations

import time

import numpy as np
import torch

from src.validation.fem import linear_elasticity_2d as le

Tensor = torch.Tensor

E_NE = 10.0
NU_NE = 0.2
THICK_NE = 1.0
TOP_V_NE = 0.1
PLANE_NE = "stress"
HARD_ALPHA_NE = 50.0
HARD_P_NE = 2.0


def hardening_response_np(
    eps_vec: np.ndarray,
    *,
    E: float = E_NE,
    nu: float = NU_NE,
    plane: str = PLANE_NE,
    alpha: float = HARD_ALPHA_NE,
    p: float = HARD_P_NE,
    eps_reg: float = 1e-12,
) -> dict[str, np.ndarray]:
    eps = np.asarray(eps_vec, dtype=np.float64)
    d = le.d_matrix_np(E, nu, plane)
    d_eps = np.einsum("ij,...j->...i", d, eps)
    q = np.maximum(np.einsum("...i,...i->...", eps, d_eps), 0.0)
    kappa = np.sqrt(q + eps_reg**2)
    psi = 0.5 * q + alpha / (p + 2.0) * (kappa ** (p + 2.0) - eps_reg ** (p + 2.0))
    scale = 1.0 + alpha * kappa**p
    sig = scale[..., None] * d_eps
    return {"psi": psi, "sig_vec": sig, "q": q, "kappa": kappa, "scale": scale}


def hardening_response_tangent_np(
    eps_vec: np.ndarray,
    *,
    E: float = E_NE,
    nu: float = NU_NE,
    plane: str = PLANE_NE,
    alpha: float = HARD_ALPHA_NE,
    p: float = HARD_P_NE,
    eps_reg: float = 1e-12,
) -> tuple[float, np.ndarray, np.ndarray]:
    eps = np.asarray(eps_vec, dtype=np.float64)
    d = le.d_matrix_np(E, nu, plane)
    d_eps = d @ eps
    q = max(float(eps @ d_eps), 0.0)
    kappa = np.sqrt(q + eps_reg**2)
    psi = 0.5 * q + alpha / (p + 2.0) * (kappa ** (p + 2.0) - eps_reg ** (p + 2.0))
    scale = 1.0 + alpha * kappa**p
    sig = scale * d_eps
    tangent = scale * d + alpha * p * (kappa ** (p - 2.0)) * np.outer(d_eps, d_eps)
    return float(psi), sig, tangent


def hardening_response_torch(
    eps_vec: Tensor,
    d_matrix: Tensor,
    *,
    alpha: float = HARD_ALPHA_NE,
    p: float = HARD_P_NE,
    eps_reg: float = 1e-12,
) -> dict[str, Tensor]:
    d = d_matrix.to(device=eps_vec.device, dtype=eps_vec.dtype)
    d_eps = torch.einsum("ij,...j->...i", d, eps_vec)
    q = torch.sum(eps_vec * d_eps, dim=-1).clamp_min(0.0)
    kappa = torch.sqrt(q + eps_reg**2)
    psi = 0.5 * q + alpha / (p + 2.0) * (kappa ** (p + 2.0) - eps_reg ** (p + 2.0))
    scale = 1.0 + alpha * kappa**p
    sig = scale.unsqueeze(-1) * d_eps
    return {"psi": psi, "sig_vec": sig, "q": q, "kappa": kappa, "scale": scale}


def assemble_internal_tangent(
    u_vec: np.ndarray,
    elements: np.ndarray,
    b_mats: np.ndarray,
    det_j: np.ndarray,
    weights: np.ndarray,
    *,
    thickness: float = THICK_NE,
) -> tuple[np.ndarray, np.ndarray, float]:
    ndof = u_vec.shape[0]
    fint = np.zeros(ndof, dtype=np.float64)
    ktan = np.zeros((ndof, ndof), dtype=np.float64)
    energy = 0.0
    for e, conn in enumerate(elements):
        dofs = np.empty(8, dtype=np.int64)
        dofs[0::2] = 2 * conn
        dofs[1::2] = 2 * conn + 1
        ue = u_vec[dofs]
        fe = np.zeros(8, dtype=np.float64)
        ke = np.zeros((8, 8), dtype=np.float64)
        for g in range(4):
            b = b_mats[e, g]
            eps = b @ ue
            psi, sig, tangent = hardening_response_tangent_np(eps)
            dvol = thickness * det_j[e, g] * weights[g]
            fe += b.T @ sig * dvol
            ke += b.T @ tangent @ b * dvol
            energy += psi * dvol
        fint[dofs] += fe
        ktan[np.ix_(dofs, dofs)] += ke
    return fint, ktan, float(energy)


def solve_reference(
    nx: int = 21,
    ny: int = 21,
    *,
    n_steps: int = 8,
    newton_tol: float = 1e-8,
    newton_max_iter: int = 25,
    damping: float = 1.0,
) -> dict[str, object]:
    mesh = le.make_rect_mesh(nx, ny)
    coords = mesh["coords"]
    elems = mesh["elements"]
    _, b_mats, det_j, weights = le.assemble_stiffness(coords, elems)
    u = np.zeros(2 * coords.shape[0], dtype=np.float64)
    hist = {"step": [], "load_factor": [], "iter": [], "res_norm": [], "res_rel": [], "energy": []}
    converged_all = True
    t0 = time.perf_counter()

    for step in range(1, n_steps + 1):
        lf = step / n_steps
        fixed_dofs, fixed_vals, free_dofs = le.make_dirichlet_dofs(mesh["xs"], mesh["ys"], top_v=TOP_V_NE * lf)
        u[fixed_dofs] = fixed_vals
        step_res0 = None
        step_ok = False
        for it in range(newton_max_iter + 1):
            u[fixed_dofs] = fixed_vals
            fint, ktan, energy = assemble_internal_tangent(u, elems, b_mats, det_j, weights)
            r_free = fint[free_dofs]
            res_norm = float(np.linalg.norm(r_free))
            if step_res0 is None:
                step_res0 = max(res_norm, 1.0)
            res_rel = res_norm / step_res0
            hist["step"].append(step)
            hist["load_factor"].append(lf)
            hist["iter"].append(it)
            hist["res_norm"].append(res_norm)
            hist["res_rel"].append(res_rel)
            hist["energy"].append(float(energy))
            if res_rel < newton_tol or res_norm < newton_tol:
                step_ok = True
                break
            du = np.linalg.solve(ktan[np.ix_(free_dofs, free_dofs)], -r_free)
            u[free_dofs] += damping * du
        if not step_ok:
            converged_all = False

    disp = u.reshape(-1, 2)
    strain_gp, stress_gp, psi_gp = element_gauss_fields_np(disp, elems, b_mats)
    mask, values = le.bc_mask_values(coords, mesh["xs"], mesh["ys"], TOP_V_NE)
    free_full = le.make_dirichlet_dofs(mesh["xs"], mesh["ys"], TOP_V_NE)[2]
    return {
        **mesh,
        "b_mats": b_mats,
        "det_j": det_j,
        "weights": weights,
        "free_dofs": free_full,
        "bc_mask": mask,
        "bc_values": values,
        "disp": disp,
        "U": disp[:, 0].reshape(ny, nx),
        "V": disp[:, 1].reshape(ny, nx),
        "Umag": np.linalg.norm(disp, axis=1).reshape(ny, nx),
        "strain_gp": strain_gp,
        "stress_gp": stress_gp,
        "psi_gp": psi_gp,
        "internal_energy": float(np.sum(psi_gp * det_j * weights.reshape(1, -1)) * THICK_NE),
        "hist": hist,
        "wall": time.perf_counter() - t0,
        "converged_all": converged_all,
    }


def element_gauss_fields_np(disp: np.ndarray, elements: np.ndarray, b_mats: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u_e = disp[elements].reshape(elements.shape[0], 8)
    strain = np.einsum("egij,ej->egi", b_mats, u_e)
    response = hardening_response_np(strain)
    return strain, response["sig_vec"], response["psi"]


def make_node_features(coords: np.ndarray, bc_values: np.ndarray) -> np.ndarray:
    base = le.make_node_features(coords, np.zeros_like(bc_values, dtype=bool), bc_values, TOP_V_NE)
    n = coords.shape[0]
    return np.concatenate(
        [
            base,
            np.ones((n, 1), dtype=np.float64),
            NU_NE * np.ones((n, 1), dtype=np.float64),
            np.ones((n, 1), dtype=np.float64),
            np.ones((n, 1), dtype=np.float64),
        ],
        axis=1,
    )


def residual_norm_from_energy(energy_fn, u: Tensor, free_dofs: Tensor, eps: float = 1e-12) -> Tensor:
    u_var = u.detach().clone().requires_grad_(True)
    energy = energy_fn(u_var)
    grad = torch.autograd.grad(energy, u_var, create_graph=False)[0].reshape(-1)
    denom = torch.linalg.norm(grad).clamp_min(eps)
    return torch.linalg.norm(grad[free_dofs]) / denom
