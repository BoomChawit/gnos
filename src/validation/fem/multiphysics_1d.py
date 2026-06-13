from __future__ import annotations

import math

import torch

from physics.derivatives import element_gradient_1d
from src.validation.fem import heat_transfer_1d as ht
from src.validation.fem import linear_elasticity_1d as le

Tensor = torch.Tensor

L_MP = ht.L_HT
A_MP = le.A_BAR
E_MP = le.E_STEEL
ALPHA_T_MP = 12.0e-6
T0_MP = ht.T_RIGHT_HT
MP_THERMAL_FACTOR = 0.05
MP_BODY_SCALE = 5.0e5
BETA_MP = 2.0e18
N_ELEM_MP = 100
MP_TEMP_SCALE = max(MP_THERMAL_FACTOR * abs(ht.T_LEFT_HT - ht.T_RIGHT_HT), 1e-12)
U_THERMAL_SCALE_MP = ALPHA_T_MP * MP_TEMP_SCALE * L_MP
U_MECH_SCALE_MP = MP_BODY_SCALE * L_MP**2 / (A_MP * E_MP)
U_SCALE_MP = max(U_THERMAL_SCALE_MP, U_MECH_SCALE_MP, 1e-12)
B_SCALE_MP = MP_BODY_SCALE


def make_mesh(n_elem: int = N_ELEM_MP, length: float = L_MP, *, device=None, dtype=torch.float32) -> tuple[Tensor, Tensor]:
    x = torch.linspace(0.0, length, n_elem + 1, device=device, dtype=dtype)
    x_normalized = (x / length).view(-1, 1)
    return x, x_normalized


def make_dirichlet(n_node: int, *, device=None, dtype=torch.float32) -> tuple[Tensor, Tensor, Tensor]:
    idx = torch.tensor([0, n_node - 1], device=device, dtype=torch.long)
    vals = torch.zeros(2, device=device, dtype=dtype)
    free = torch.arange(1, n_node - 1, device=device, dtype=torch.long)
    return idx, vals, free


def temperature(x: Tensor) -> Tensor:
    return T0_MP + MP_THERMAL_FACTOR * (ht.temperature_exact(x) - T0_MP)


def thermal_strain(temp: Tensor) -> Tensor:
    return ALPHA_T_MP * (temp - T0_MP)


def body_force(x: Tensor) -> Tensor:
    x_le = 10.0 * x / L_MP
    return le.body_force(x_le, load_scale=MP_BODY_SCALE)


def cumulative_trapezoid(y: Tensor, x: Tensor) -> Tensor:
    out = torch.zeros_like(y)
    out[1:] = torch.cumsum(0.5 * (y[1:] + y[:-1]) * torch.diff(x), dim=0)
    return out


def body_integral_0_to_x(x: Tensor, n_grid: int = 5001) -> Tensor:
    xq = torch.linspace(0.0, L_MP, n_grid, device=x.device, dtype=x.dtype)
    bq = body_force(xq)
    iq = cumulative_trapezoid(bq, xq)

    x_clamped = x.clamp(0.0, L_MP)
    idx = torch.searchsorted(xq, x_clamped)
    idx = torch.clamp(idx, 1, n_grid - 1)
    x0 = xq[idx - 1]
    x1 = xq[idx]
    i0 = iq[idx - 1]
    i1 = iq[idx]
    w = (x_clamped - x0) / (x1 - x0).clamp_min(1e-30)
    return i0 + w * (i1 - i0)


def assemble_load_vector(x: Tensor) -> Tensor:
    f = torch.zeros_like(x)
    gp = torch.tensor([-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)], device=x.device, dtype=x.dtype)
    for e in range(x.numel() - 1):
        xi = x[e]
        xj = x[e + 1]
        h = xj - xi
        for s in gp:
            n1 = 0.5 * (1.0 - s)
            n2 = 0.5 * (1.0 + s)
            xq = n1 * xi + n2 * xj
            bq = body_force(xq)
            f[e] += n1 * bq * h / 2.0
            f[e + 1] += n2 * bq * h / 2.0
    return f


def energy_density(mech_strain: Tensor) -> Tensor:
    return 0.5 * E_MP * mech_strain**2 + 0.25 * BETA_MP * mech_strain**4


def stress(mech_strain: Tensor) -> Tensor:
    return E_MP * mech_strain + BETA_MP * mech_strain**3


def tangent(mech_strain: Tensor) -> Tensor:
    return E_MP + 3.0 * BETA_MP * mech_strain**2


def inverse_stress(sig: Tensor, n_iter: int = 40) -> Tensor:
    e = sig / E_MP
    for _ in range(n_iter):
        r = stress(e) - sig
        k = tangent(e).clamp_min(1e-30)
        e = e - r / k
    return e


def element_centers(x: Tensor) -> Tensor:
    return 0.5 * (x[:-1] + x[1:])


def element_strain(u: Tensor, x: Tensor) -> Tensor:
    return element_gradient_1d(u, x)


def residual_norm(u: Tensor, x: Tensor, temp_elem: Tensor, f_ext: Tensor, free_idx: Tensor) -> Tensor:
    if u.ndim == 1:
        u_b = u.unsqueeze(0)
    else:
        u_b = u
    eps_th = thermal_strain(temp_elem).reshape(1, -1).to(device=u_b.device, dtype=u_b.dtype)
    eps = element_strain(u_b, x)
    nint = A_MP * stress(eps - eps_th)
    res = torch.zeros_like(u_b)
    res[:, :-1] += -nint
    res[:, 1:] += nint
    res -= f_ext.reshape(1, -1).to(device=u_b.device, dtype=u_b.dtype)

    e0 = -eps_th
    n0 = A_MP * stress(e0)
    rth = torch.zeros_like(u_b)
    rth[:, :-1] += -n0
    rth[:, 1:] += n0
    den = (
        torch.linalg.norm(f_ext.reshape(-1).to(u_b)[free_idx])
        + torch.linalg.norm(rth[:, free_idx])
    ).clamp_min(1e-30)
    return (torch.linalg.norm(res[:, free_idx], dim=1) / den).mean()


def reference_solution(x: Tensor, temp_node: Tensor) -> dict[str, Tensor]:
    eps_th = thermal_strain(temp_node)
    ib = body_integral_0_to_x(x)

    def constraint(c):
        n = c - ib
        sig = n / A_MP
        mech = inverse_stress(sig)
        eps_total = mech + eps_th
        return torch.trapz(eps_total, x)

    c_scale = (
        A_MP * E_MP * torch.max(torch.abs(eps_th)).item()
        + torch.max(torch.abs(ib)).item()
        + 1.0
    )
    c_low = torch.tensor(-10.0 * c_scale, device=x.device, dtype=x.dtype)
    c_high = torch.tensor(10.0 * c_scale, device=x.device, dtype=x.dtype)
    g_low = constraint(c_low)
    g_high = constraint(c_high)

    for _ in range(20):
        if g_low * g_high <= 0:
            break
        c_low *= 2.0
        c_high *= 2.0
        g_low = constraint(c_low)
        g_high = constraint(c_high)

    for _ in range(100):
        c_mid = 0.5 * (c_low + c_high)
        g_mid = constraint(c_mid)
        if g_low * g_mid <= 0:
            c_high = c_mid
            g_high = g_mid
        else:
            c_low = c_mid
            g_low = g_mid

    c = 0.5 * (c_low + c_high)
    n = c - ib
    sig = n / A_MP
    mech = inverse_stress(sig)
    eps_total = mech + eps_th
    u = cumulative_trapezoid(eps_total, x)
    return {
        "C": c,
        "Ib": ib,
        "u": u,
        "eps_total": eps_total,
        "eps_th": eps_th,
        "e_mech": mech,
        "sigma": sig,
        "N": n,
    }


def reference_center_fields(x: Tensor, c_ref: Tensor) -> dict[str, Tensor]:
    xc = element_centers(x)
    ib = body_integral_0_to_x(xc)
    n = c_ref - ib
    sig = n / A_MP
    mech = inverse_stress(sig)
    eps_th = thermal_strain(temperature(xc))
    return {
        "x": xc,
        "N": n,
        "sigma": sig,
        "e_mech": mech,
        "eps_th": eps_th,
        "eps_total": mech + eps_th,
    }


def make_node_features(x_normalized: Tensor, temp_node: Tensor, b_node: Tensor) -> Tensor:
    theta = temp_node - T0_MP
    eps_th = thermal_strain(temp_node)
    eps_scale = max(ALPHA_T_MP * MP_TEMP_SCALE, 1e-12)
    return torch.cat(
        [
            x_normalized,
            (theta / MP_TEMP_SCALE).view(-1, 1),
            (eps_th / eps_scale).view(-1, 1),
            (b_node / B_SCALE_MP).view(-1, 1),
        ],
        dim=1,
    )


def rel_l2_element(pred: Tensor, ref: Tensor, x_nodes: Tensor, eps: float = 1e-12) -> Tensor:
    p = pred.reshape(-1).to(dtype=torch.float64)
    r = ref.reshape(-1).to(device=p.device, dtype=p.dtype)
    x = element_centers(x_nodes).to(device=p.device, dtype=p.dtype)
    num = torch.trapezoid((p - r).square(), x).clamp_min(0.0).sqrt()
    den = torch.trapezoid(r.square(), x).clamp_min(eps).sqrt()
    return num / den

