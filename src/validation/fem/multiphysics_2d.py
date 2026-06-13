from __future__ import annotations

import time

import numpy as np
import torch

from physics.derivatives import q4_b_matrices, q4_element_strain, q4_scalar_grad_matrices
from src.validation.fem import heat_transfer_2d as ht
from src.validation.fem import linear_elasticity_2d as le

Tensor = torch.Tensor

E_MP = 10.0
NU_MP = 0.30
THICK_MP = 1.0
TOP_V_MP = 0.1
PLANE_MP = "stress"
HARD_ALPHA_MP = 50.0
HARD_P_MP = 2.0
ALPHA_T_MP = 1.0e-2
TEMP_SCALE_MP = 1.0
T0_MP = 0.0


def build_temperature(
    nx: int,
    ny: int,
    *,
    heat_case: str = ht.CASE_HT,
    temp_scale: float = TEMP_SCALE_MP,
    recenter: bool = True,
) -> dict[str, object]:
    heat_ref = ht.solve_reference(nx, ny, case=heat_case, k=ht.K_HT, source=ht.SOURCE_HT, thickness=ht.THICK_HT)
    raw = np.asarray(heat_ref["T_grid"], dtype=np.float64)
    raw_mean = float(np.mean(raw))
    temp = temp_scale * (raw - raw_mean) if recenter else temp_scale * raw
    return {
        "xs": heat_ref["xs"],
        "ys": heat_ref["ys"],
        "T_raw": raw,
        "T": temp,
        "T0": T0_MP,
        "raw_mean": raw_mean,
        "source": "heat_transfer_2d",
        "temp_scale": float(temp_scale),
        "recenter": bool(recenter),
        "heat_ref": heat_ref,
    }


def thermo_hardening_response_tangent_np(
    eps_total: np.ndarray,
    temp_gp: float,
    *,
    E: float = E_MP,
    nu: float = NU_MP,
    plane: str = PLANE_MP,
    alpha: float = HARD_ALPHA_MP,
    p: float = HARD_P_MP,
    alpha_t: float = ALPHA_T_MP,
    t0: float = T0_MP,
    eps_reg: float = 1e-12,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = le.d_matrix_np(E, nu, plane)
    eps_th = alpha_t * (float(temp_gp) - float(t0)) * np.array([1.0, 1.0, 0.0], dtype=np.float64)
    eps_eff = np.asarray(eps_total, dtype=np.float64) - eps_th
    d_eps = d @ eps_eff
    q = max(float(eps_eff @ d_eps), 0.0)
    kappa = np.sqrt(q + eps_reg**2)
    scale = 1.0 + alpha * kappa**p
    psi = 0.5 * q + alpha / (p + 2.0) * (kappa ** (p + 2.0) - eps_reg ** (p + 2.0))
    sig = scale * d_eps
    tangent = scale * d + alpha * p * (kappa ** (p - 2.0)) * np.outer(d_eps, d_eps)
    return float(psi), sig, tangent, eps_th, eps_eff


def thermo_hardening_response_torch(
    eps_total: Tensor,
    temp_gp: Tensor,
    d_matrix: Tensor,
    *,
    alpha: float = HARD_ALPHA_MP,
    p: float = HARD_P_MP,
    alpha_t: float = ALPHA_T_MP,
    t0: float = T0_MP,
    eps_reg: float = 1e-12,
) -> dict[str, Tensor]:
    theta = float(alpha_t) * (temp_gp - float(t0))
    eps_th = torch.zeros_like(eps_total)
    eps_th[..., 0] = theta
    eps_th[..., 1] = theta
    eps_eff = eps_total - eps_th
    d = d_matrix.to(device=eps_total.device, dtype=eps_total.dtype)
    d_eps = torch.einsum("ij,...j->...i", d, eps_eff)
    q = torch.sum(eps_eff * d_eps, dim=-1).clamp_min(0.0)
    kappa = torch.sqrt(q + eps_reg**2)
    psi = 0.5 * q + alpha / (p + 2.0) * (kappa ** (p + 2.0) - eps_reg ** (p + 2.0))
    scale = 1.0 + alpha * kappa**p
    sig = scale.unsqueeze(-1) * d_eps
    return {
        "psi": psi,
        "sig_vec": sig,
        "eps_total": eps_total,
        "eps_th": eps_th,
        "eps_eff": eps_eff,
        "q": q,
        "kappa": kappa,
        "scale": scale,
    }


def assemble_internal_tangent(
    u_vec: np.ndarray,
    temp_grid: np.ndarray,
    elements: np.ndarray,
    shape_vals: np.ndarray,
    b_mats: np.ndarray,
    det_j: np.ndarray,
    weights: np.ndarray,
    *,
    E: float = E_MP,
    nu: float = NU_MP,
    plane: str = PLANE_MP,
    alpha: float = HARD_ALPHA_MP,
    p: float = HARD_P_MP,
    alpha_t: float = ALPHA_T_MP,
    t0: float = T0_MP,
    thickness: float = THICK_MP,
) -> tuple[np.ndarray, np.ndarray, float]:
    ndof = u_vec.shape[0]
    temp_nodes = np.asarray(temp_grid, dtype=np.float64).reshape(-1)
    fint = np.zeros(ndof, dtype=np.float64)
    ktan = np.zeros((ndof, ndof), dtype=np.float64)
    energy = 0.0

    for e, conn in enumerate(elements):
        dofs = np.empty(8, dtype=np.int64)
        dofs[0::2] = 2 * conn
        dofs[1::2] = 2 * conn + 1
        ue = u_vec[dofs]
        te = temp_nodes[conn]
        fe = np.zeros(8, dtype=np.float64)
        ke = np.zeros((8, 8), dtype=np.float64)
        for g in range(weights.shape[0]):
            b = b_mats[e, g]
            eps_total = b @ ue
            temp_gp = float(shape_vals[g] @ te)
            psi, sig, tangent, _, _ = thermo_hardening_response_tangent_np(
                eps_total,
                temp_gp,
                E=E,
                nu=nu,
                plane=plane,
                alpha=alpha,
                p=p,
                alpha_t=alpha_t,
                t0=t0,
            )
            dvol = det_j[e, g] * weights[g] * thickness
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
    heat_case: str = ht.CASE_HT,
    temp_scale: float = TEMP_SCALE_MP,
    recenter_temperature: bool = True,
    E: float = E_MP,
    nu: float = NU_MP,
    thickness: float = THICK_MP,
    top_v: float = TOP_V_MP,
    plane: str = PLANE_MP,
    alpha: float = HARD_ALPHA_MP,
    p: float = HARD_P_MP,
    alpha_t: float = ALPHA_T_MP,
    t0_value: float = T0_MP,
    n_steps: int = 8,
    newton_tol: float = 1e-8,
    newton_max_iter: int = 25,
    damping: float = 1.0,
    ramp_temperature: bool = True,
) -> dict[str, object]:
    t_start = time.perf_counter()
    mesh = le.make_rect_mesh(nx, ny)
    coords = mesh["coords"]
    elems = mesh["elements"]
    coords_t = torch.tensor(coords, dtype=torch.float64)
    elems_t = torch.tensor(elems, dtype=torch.long)
    b_t, det_t, weights_t = q4_b_matrices(coords_t, elems_t)
    shape_t, _, _, _ = q4_scalar_grad_matrices(coords_t, elems_t)
    b_mats = b_t.cpu().numpy()
    det_j = det_t.cpu().numpy()
    weights = weights_t.cpu().numpy()
    shape_vals = shape_t.cpu().numpy()

    temp_pack = build_temperature(nx, ny, heat_case=heat_case, temp_scale=temp_scale, recenter=recenter_temperature)
    temp_full = np.asarray(temp_pack["T"], dtype=np.float64)
    u = np.zeros(2 * coords.shape[0], dtype=np.float64)
    hist = {"step": [], "load_factor": [], "iter": [], "res_norm": [], "res_rel": [], "energy": []}
    converged_all = True

    for step in range(1, n_steps + 1):
        load_factor = step / n_steps
        fixed_dofs, fixed_vals, free_dofs = le.make_dirichlet_dofs(mesh["xs"], mesh["ys"], top_v=top_v * load_factor)
        temp_step = load_factor * temp_full if ramp_temperature else temp_full
        u[fixed_dofs] = fixed_vals
        step_res0 = None
        step_ok = False
        for it in range(newton_max_iter + 1):
            u[fixed_dofs] = fixed_vals
            fint, ktan, energy = assemble_internal_tangent(
                u,
                temp_step,
                elems,
                shape_vals,
                b_mats,
                det_j,
                weights,
                E=E,
                nu=nu,
                plane=plane,
                alpha=alpha,
                p=p,
                alpha_t=alpha_t,
                t0=t0_value,
                thickness=thickness,
            )
            residual = fint[free_dofs]
            res_norm = float(np.linalg.norm(residual))
            if step_res0 is None:
                step_res0 = max(res_norm, 1.0)
            res_rel = res_norm / step_res0
            hist["step"].append(step)
            hist["load_factor"].append(load_factor)
            hist["iter"].append(it)
            hist["res_norm"].append(res_norm)
            hist["res_rel"].append(res_rel)
            hist["energy"].append(float(energy))
            if res_rel < newton_tol or res_norm < newton_tol:
                step_ok = True
                break
            du = np.linalg.solve(ktan[np.ix_(free_dofs, free_dofs)], -residual)
            u[free_dofs] += damping * du
        if not step_ok:
            converged_all = False

    disp = u.reshape(-1, 2)
    fields_gp = element_gauss_fields_np(
        disp,
        temp_full,
        elems,
        shape_vals,
        b_mats,
        E=E,
        nu=nu,
        plane=plane,
        alpha=alpha,
        p=p,
        alpha_t=alpha_t,
        t0=t0_value,
    )
    fields = project_q4_thermo_fields_np(fields_gp, elems, shape_vals, nx, ny)
    fields["internal_energy"] = float(np.sum(fields_gp["psi"] * det_j * weights.reshape(1, -1)) * thickness)
    mask, values = le.bc_mask_values(coords, mesh["xs"], mesh["ys"], top_v)
    free_full = le.make_dirichlet_dofs(mesh["xs"], mesh["ys"], top_v)[2]
    return {
        **mesh,
        "b_mats": b_mats,
        "shape_vals": shape_vals,
        "det_j": det_j,
        "weights": weights,
        "free_dofs": free_full,
        "bc_mask": mask,
        "bc_values": values,
        "disp": disp,
        "U": disp[:, 0].reshape(ny, nx),
        "V": disp[:, 1].reshape(ny, nx),
        "Umag": np.linalg.norm(disp, axis=1).reshape(ny, nx),
        "T": temp_full,
        "temperature_pack": temp_pack,
        "fields_gp": fields_gp,
        "fields": fields,
        "internal_energy": fields["internal_energy"],
        "hist": hist,
        "wall": time.perf_counter() - t_start,
        "converged_all": converged_all,
        "E": float(E),
        "nu": float(nu),
        "plane": plane,
        "top_v": float(top_v),
        "alpha": float(alpha),
        "p": float(p),
        "alpha_t": float(alpha_t),
        "T0": float(t0_value),
        "temp_scale": float(temp_scale),
        "recenter_temperature": bool(recenter_temperature),
        "ramp_temperature": bool(ramp_temperature),
    }


def element_gauss_fields_np(
    disp: np.ndarray,
    temp_grid: np.ndarray,
    elements: np.ndarray,
    shape_vals: np.ndarray,
    b_mats: np.ndarray,
    *,
    E: float = E_MP,
    nu: float = NU_MP,
    plane: str = PLANE_MP,
    alpha: float = HARD_ALPHA_MP,
    p: float = HARD_P_MP,
    alpha_t: float = ALPHA_T_MP,
    t0: float = T0_MP,
) -> dict[str, np.ndarray]:
    u_e = np.asarray(disp, dtype=np.float64)[elements].reshape(elements.shape[0], 8)
    temp_e = np.asarray(temp_grid, dtype=np.float64).reshape(-1)[elements]
    strain_total = np.einsum("egij,ej->egi", b_mats, u_e)
    temp_gp = np.einsum("gj,ej->eg", shape_vals, temp_e)
    d = le.d_matrix_np(E, nu, plane)
    theta = alpha_t * (temp_gp - t0)
    eps_th = np.zeros_like(strain_total)
    eps_th[..., 0] = theta
    eps_th[..., 1] = theta
    eps_eff = strain_total - eps_th
    d_eps = np.einsum("ij,egj->egi", d, eps_eff)
    q = np.maximum(np.einsum("egi,egi->eg", eps_eff, d_eps), 0.0)
    kappa = np.sqrt(q + 1e-24)
    psi = 0.5 * q + alpha / (p + 2.0) * (kappa ** (p + 2.0) - 1e-12 ** (p + 2.0))
    stress = (1.0 + alpha * kappa**p)[..., None] * d_eps
    return {
        "temp_gp": temp_gp,
        "strain_total": strain_total,
        "eps_th": eps_th,
        "eps_eff": eps_eff,
        "stress": stress,
        "psi": psi,
    }


def element_gauss_fields_torch(
    disp: Tensor,
    temperature: Tensor,
    elements: Tensor,
    shape_vals: Tensor,
    b_mats: Tensor,
    d_matrix: Tensor,
    *,
    alpha: float = HARD_ALPHA_MP,
    p: float = HARD_P_MP,
    alpha_t: float = ALPHA_T_MP,
    t0: float = T0_MP,
) -> dict[str, Tensor]:
    elems = elements.to(device=disp.device, dtype=torch.long)
    b = b_mats.to(device=disp.device, dtype=disp.dtype)
    n = shape_vals.to(device=disp.device, dtype=disp.dtype)
    strain_total = q4_element_strain(disp, elems, b)
    if disp.ndim == 2:
        strain_total = strain_total[0]
    temp_nodes = temperature.reshape(-1).to(device=disp.device, dtype=disp.dtype)
    temp_gp = torch.einsum("gj,ej->eg", n, temp_nodes[elems])
    return thermo_hardening_response_torch(
        strain_total,
        temp_gp,
        d_matrix,
        alpha=alpha,
        p=p,
        alpha_t=alpha_t,
        t0=t0,
    )


def project_q4_thermo_fields_np(
    fields_gp: dict[str, np.ndarray],
    elements: np.ndarray,
    shape_vals: np.ndarray,
    nx: int,
    ny: int,
) -> dict[str, object]:
    n_nodes = nx * ny
    acc = {
        "strain_total": np.zeros((n_nodes, 3), dtype=np.float64),
        "eps_th": np.zeros((n_nodes, 3), dtype=np.float64),
        "eps_eff": np.zeros((n_nodes, 3), dtype=np.float64),
        "stress": np.zeros((n_nodes, 3), dtype=np.float64),
        "psi": np.zeros(n_nodes, dtype=np.float64),
    }
    counts = np.zeros(n_nodes, dtype=np.float64)
    for e, conn in enumerate(elements):
        for g in range(shape_vals.shape[0]):
            for a, node in enumerate(conn):
                weight = float(shape_vals[g, a])
                acc["strain_total"][node] += weight * fields_gp["strain_total"][e, g]
                acc["eps_th"][node] += weight * fields_gp["eps_th"][e, g]
                acc["eps_eff"][node] += weight * fields_gp["eps_eff"][e, g]
                acc["stress"][node] += weight * fields_gp["stress"][e, g]
                acc["psi"][node] += weight * fields_gp["psi"][e, g]
                counts[node] += weight
    counts = np.maximum(counts, 1e-14)
    strain_total = (acc["strain_total"] / counts[:, None]).reshape(ny, nx, 3)
    eps_th = (acc["eps_th"] / counts[:, None]).reshape(ny, nx, 3)
    eps_eff = (acc["eps_eff"] / counts[:, None]).reshape(ny, nx, 3)
    stress = (acc["stress"] / counts[:, None]).reshape(ny, nx, 3)
    psi = (acc["psi"] / counts).reshape(ny, nx)
    return {
        "strain_total": {
            "eps_vec": strain_total,
            "exx": strain_total[..., 0],
            "eyy": strain_total[..., 1],
            "gamma_xy": strain_total[..., 2],
            "exy": 0.5 * strain_total[..., 2],
        },
        "eps_th": eps_th,
        "eps_eff": eps_eff,
        "stress": {
            "sig_vec": stress,
            "sxx": stress[..., 0],
            "syy": stress[..., 1],
            "sxy": stress[..., 2],
        },
        "energy_density": psi,
    }


def make_node_features(
    coords: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    bc_values: np.ndarray,
    temp_grid: np.ndarray,
    *,
    top_v: float = TOP_V_MP,
    nu: float = NU_MP,
    alpha: float = HARD_ALPHA_MP,
    p: float = HARD_P_MP,
    alpha_t: float = ALPHA_T_MP,
    t0: float = T0_MP,
) -> np.ndarray:
    n = coords.shape[0]
    disp_scale = max(abs(top_v), 1e-12)
    alpha_ref = max(abs(alpha), 1e-12)
    p_ref = max(abs(p), 1e-12)
    xnorm = (coords[:, 0] - xs[0]) / (xs[-1] - xs[0])
    ynorm = (coords[:, 1] - ys[0]) / (ys[-1] - ys[0])
    tol = 1e-12
    bottom = coords[:, 1] <= tol
    top_left = (coords[:, 1] >= 1.0 - tol) & (coords[:, 0] <= 0.5 + tol)
    temp = np.asarray(temp_grid, dtype=np.float64).reshape(-1)
    temp_norm = temp / max(float(np.max(np.abs(temp))), 1e-12)
    thermal = alpha_t * (temp - t0)
    thermal_norm = thermal / max(float(np.max(np.abs(thermal))), 1e-12)
    return np.stack(
        [
            xnorm,
            ynorm,
            bottom.astype(np.float64),
            top_left.astype(np.float64),
            bc_values[:, 0] / disp_scale,
            bc_values[:, 1] / disp_scale,
            temp_norm,
            thermal_norm,
            np.ones(n, dtype=np.float64),
            nu * np.ones(n, dtype=np.float64),
            (alpha / alpha_ref) * np.ones(n, dtype=np.float64),
            (p / p_ref) * np.ones(n, dtype=np.float64),
        ],
        axis=1,
    )


def rel_l2(pred: Tensor, ref: Tensor, eps: float = 1e-12) -> Tensor:
    p = pred.reshape(-1).to(dtype=torch.float64)
    r = ref.reshape(-1).to(device=p.device, dtype=p.dtype)
    return torch.linalg.norm(p - r) / torch.linalg.norm(r).clamp_min(eps)


def residual_norm_from_energy(energy_fn, disp: Tensor, free_dofs: Tensor, eps: float = 1e-12) -> Tensor:
    u_var = disp.detach().clone().requires_grad_(True)
    energy = energy_fn(u_var)
    grad = torch.autograd.grad(energy, u_var, create_graph=False)[0].reshape(-1)
    free = free_dofs.to(device=grad.device, dtype=torch.long)
    denom = torch.linalg.norm(grad).clamp_min(eps)
    return torch.linalg.norm(grad[free]) / denom
