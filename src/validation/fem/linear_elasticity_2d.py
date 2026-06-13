from __future__ import annotations

import numpy as np
import torch

from physics.derivatives import q4_b_matrices, q4_element_strain

Tensor = torch.Tensor

E_LE = 10.0
NU_LE = 0.2
THICK_LE = 1.0
LX_LE = 1.0
LY_LE = 1.0
TOP_V_LE = 0.1
PLANE_LE = "stress"


def d_matrix_np(E: float = E_LE, nu: float = NU_LE, plane: str = PLANE_LE) -> np.ndarray:
    if plane == "stress":
        return E / (1.0 - nu**2) * np.array(
            [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - nu)]],
            dtype=np.float64,
        )
    if plane == "strain":
        c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return c * np.array(
            [[1.0 - nu, nu, 0.0], [nu, 1.0 - nu, 0.0], [0.0, 0.0, 0.5 * (1.0 - 2.0 * nu)]],
            dtype=np.float64,
        )
    raise ValueError("plane must be 'stress' or 'strain'")


def d_matrix_torch(E: float = E_LE, nu: float = NU_LE, plane: str = PLANE_LE, *, device=None, dtype=torch.float32) -> Tensor:
    return torch.tensor(d_matrix_np(E, nu, plane), device=device, dtype=dtype)


def make_rect_mesh(
    nx: int = 31,
    ny: int = 31,
    lx: float = LX_LE,
    ly: float = LY_LE,
) -> dict[str, np.ndarray]:
    xs = np.linspace(0.0, lx, nx, dtype=np.float64)
    ys = np.linspace(0.0, ly, ny, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    coords = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)

    elems = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n1 = j * nx + i
            n2 = n1 + 1
            n4 = n1 + nx
            n3 = n4 + 1
            elems.append([n1, n2, n3, n4])

    return {
        "xs": xs,
        "ys": ys,
        "xx": xx,
        "yy": yy,
        "coords": coords,
        "elements": np.asarray(elems, dtype=np.int64),
    }


def make_dirichlet_dofs(xs: np.ndarray, ys: np.ndarray, top_v: float = TOP_V_LE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx = len(xs)
    ny = len(ys)
    xmid = 0.5 * (xs[0] + xs[-1])
    tol = 1e-12 * max(1.0, abs(xs[-1] - xs[0]))
    bc: dict[int, float] = {}

    for i in range(nx):
        node = i
        bc[2 * node] = 0.0
        bc[2 * node + 1] = 0.0

    top_j = ny - 1
    for i, x in enumerate(xs):
        if x <= xmid + tol:
            node = top_j * nx + i
            bc[2 * node] = 0.0
            bc[2 * node + 1] = top_v

    dofs = np.fromiter(sorted(bc), dtype=np.int64)
    vals = np.asarray([bc[int(d)] for d in dofs], dtype=np.float64)
    all_dofs = np.arange(2 * nx * ny, dtype=np.int64)
    free = np.setdiff1d(all_dofs, dofs, assume_unique=True)
    return dofs, vals, free


def bc_mask_values(coords: np.ndarray, xs: np.ndarray, ys: np.ndarray, top_v: float = TOP_V_LE) -> tuple[np.ndarray, np.ndarray]:
    nx = len(xs)
    ny = len(ys)
    xmid = 0.5 * (xs[0] + xs[-1])
    tol = 1e-12 * max(1.0, abs(xs[-1] - xs[0]))
    mask = np.zeros((nx * ny, 2), dtype=bool)
    values = np.zeros((nx * ny, 2), dtype=np.float64)
    for j in range(ny):
        for i, x in enumerate(xs):
            node = j * nx + i
            if j == 0:
                mask[node, :] = True
            if j == ny - 1 and x <= xmid + tol:
                mask[node, :] = True
                values[node, 1] = top_v
    return mask, values


def assemble_stiffness(
    coords: np.ndarray,
    elements: np.ndarray,
    E: float = E_LE,
    nu: float = NU_LE,
    thickness: float = THICK_LE,
    plane: str = PLANE_LE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coords_t = torch.tensor(coords, dtype=torch.float64)
    elems_t = torch.tensor(elements, dtype=torch.long)
    b_t, det_t, weights_t = q4_b_matrices(coords_t, elems_t)
    b = b_t.cpu().numpy()
    det_j = det_t.cpu().numpy()
    weights = weights_t.cpu().numpy()
    d = d_matrix_np(E, nu, plane)

    ndof = 2 * coords.shape[0]
    k_global = np.zeros((ndof, ndof), dtype=np.float64)
    for e, conn in enumerate(elements):
        ke = np.zeros((8, 8), dtype=np.float64)
        for g in range(4):
            bg = b[e, g]
            ke += bg.T @ d @ bg * det_j[e, g] * weights[g] * thickness
        dofs = np.empty(8, dtype=np.int64)
        dofs[0::2] = 2 * conn
        dofs[1::2] = 2 * conn + 1
        k_global[np.ix_(dofs, dofs)] += ke
    return k_global, b, det_j, weights


def solve_reference(nx: int = 31, ny: int = 31) -> dict[str, np.ndarray | float]:
    mesh = make_rect_mesh(nx, ny)
    coords = mesh["coords"]
    elems = mesh["elements"]
    k_global, b_mats, det_j, weights = assemble_stiffness(coords, elems)
    bc_dofs, bc_vals, free_dofs = make_dirichlet_dofs(mesh["xs"], mesh["ys"])

    u = np.zeros(2 * coords.shape[0], dtype=np.float64)
    u[bc_dofs] = bc_vals
    rhs = -k_global[np.ix_(free_dofs, bc_dofs)] @ bc_vals
    u[free_dofs] = np.linalg.solve(k_global[np.ix_(free_dofs, free_dofs)], rhs)
    disp = u.reshape(-1, 2)

    strain_gp, stress_gp = element_gauss_fields_np(disp, elems, b_mats)
    energy = 0.5 * float(u @ (k_global @ u))
    mask, values = bc_mask_values(coords, mesh["xs"], mesh["ys"])
    return {
        **mesh,
        "K": k_global,
        "b_mats": b_mats,
        "det_j": det_j,
        "weights": weights,
        "bc_dofs": bc_dofs,
        "bc_vals": bc_vals,
        "free_dofs": free_dofs,
        "bc_mask": mask,
        "bc_values": values,
        "disp": disp,
        "U": disp[:, 0].reshape(ny, nx),
        "V": disp[:, 1].reshape(ny, nx),
        "Umag": np.linalg.norm(disp, axis=1).reshape(ny, nx),
        "strain_gp": strain_gp,
        "stress_gp": stress_gp,
        "internal_energy": energy,
    }


def element_gauss_fields_np(disp: np.ndarray, elements: np.ndarray, b_mats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u_e = disp[elements].reshape(elements.shape[0], 8)
    strain = np.einsum("egij,ej->egi", b_mats, u_e)
    stress = np.einsum("ij,egj->egi", d_matrix_np(), strain)
    return strain, stress


def make_node_features(coords: np.ndarray, bc_mask: np.ndarray, bc_values: np.ndarray, top_v: float = TOP_V_LE) -> np.ndarray:
    scale = max(abs(top_v), 1e-12)
    tol = 1e-12
    bottom = coords[:, 1] <= tol
    top_left = (coords[:, 1] >= 1.0 - tol) & (coords[:, 0] <= 0.5 + tol)
    return np.stack(
        [
            coords[:, 0],
            coords[:, 1],
            bottom.astype(np.float64),
            top_left.astype(np.float64),
            bc_values[:, 0] / scale,
            bc_values[:, 1] / scale,
        ],
        axis=1,
    )


def make_envelope_nodes(X_nodes: Tensor, top_v: float = TOP_V_LE, xmid_n: float = 0.5, alpha_left: float = 30.0) -> tuple[Tensor, Tensor]:
    x = X_nodes[:, 0]
    y = X_nodes[:, 1]
    s_left = torch.where(
        x <= xmid_n,
        torch.ones_like(x),
        torch.exp(-alpha_left * (x - xmid_n) ** 2),
    )
    g = torch.stack([torch.zeros_like(x), top_v * y * s_left], dim=-1)
    right_dist = torch.relu(x - xmid_n)
    d_top_left = (1.0 - y) ** 2 + right_dist**2
    phi = y * d_top_left
    phi = phi / torch.max(phi.detach()).clamp_min(1e-12)
    return g, phi[:, None].expand(-1, 2)


def apply_envelope(raw: Tensor, g_nodes: Tensor, phi_nodes: Tensor) -> Tensor:
    if raw.ndim == 2:
        return g_nodes + phi_nodes * raw
    return g_nodes.unsqueeze(0) + phi_nodes.unsqueeze(0) * raw


def bc_error(disp: Tensor, bc_mask: Tensor, bc_values: Tensor) -> Tensor:
    if disp.ndim == 3:
        u = disp[0]
    else:
        u = disp
    mask = bc_mask.to(device=u.device, dtype=torch.bool)
    vals = bc_values.to(device=u.device, dtype=u.dtype)
    return torch.max(torch.abs(u[mask] - vals[mask]))


def rel_l2(pred: Tensor, ref: Tensor, eps: float = 1e-12) -> Tensor:
    p = pred.reshape(-1).to(dtype=torch.float64)
    r = ref.reshape(-1).to(device=p.device, dtype=p.dtype)
    return torch.linalg.norm(p - r) / torch.linalg.norm(r).clamp_min(eps)


def element_gauss_fields_torch(u: Tensor, elements: Tensor, b_mats: Tensor, d_matrix: Tensor) -> tuple[Tensor, Tensor]:
    strain = q4_element_strain(u, elements, b_mats)
    stress = torch.einsum("ij,begj->begi", d_matrix.to(device=strain.device, dtype=strain.dtype), strain)
    if u.ndim == 2:
        return strain[0], stress[0]
    return strain, stress


def residual_norm(u: Tensor, stiffness: Tensor, free_dofs: Tensor, eps: float = 1e-12) -> Tensor:
    if u.ndim == 3:
        u_vec = u[0].reshape(-1)
    else:
        u_vec = u.reshape(-1)
    k = stiffness.to(device=u_vec.device, dtype=u_vec.dtype)
    res = k @ u_vec
    reaction_scale = torch.linalg.norm(res).clamp_min(eps)
    return torch.linalg.norm(res[free_dofs]) / reaction_scale
