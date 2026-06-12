from __future__ import annotations

import math

import torch

from src.validation.fem.linear_elasticity_1d import A_BAR

Tensor = torch.Tensor

L_HT = 1.0
A_HT = A_BAR
X1_HT = 0.30
X2_HT = 0.70
ELL_HT = X2_HT - X1_HT
K1_HT = 45.0
K2_HT = 1.5
K3_HT = 15.0
T_LEFT_HT = 100.0
T_RIGHT_HT = 20.0
Q0_HT = 20.0
N_ELEM_HT = 100
K_REF_HT = K1_HT
Q_SCALE_HT = Q0_HT
T_SCALE_HT = abs(T_LEFT_HT - T_RIGHT_HT)


def make_mesh(n_elem: int = N_ELEM_HT, length: float = L_HT, *, device=None, dtype=torch.float32) -> tuple[Tensor, Tensor]:
    x = torch.linspace(0.0, length, n_elem + 1, device=device, dtype=dtype)
    x_normalized = (x / length).view(-1, 1)
    return x, x_normalized


def conductivity(x: Tensor) -> Tensor:
    return torch.where(
        x < X1_HT,
        torch.as_tensor(K1_HT, device=x.device, dtype=x.dtype),
        torch.where(
            x <= X2_HT,
            torch.as_tensor(K2_HT, device=x.device, dtype=x.dtype),
            torch.as_tensor(K3_HT, device=x.device, dtype=x.dtype),
        ),
    )


def conductivity_area(x: Tensor) -> Tensor:
    return conductivity(x) * A_HT


def source(x: Tensor) -> Tensor:
    q = torch.zeros_like(x)
    mask = (x >= X1_HT) & (x <= X2_HT)
    pi = torch.as_tensor(math.pi, device=x.device, dtype=x.dtype)
    s = (x[mask] - X1_HT) / ELL_HT
    q[mask] = Q0_HT * torch.sin(pi * s)
    return q


def initial_heat_rate(device=None, dtype=torch.float32) -> Tensor:
    pi = torch.as_tensor(math.pi, device=device, dtype=dtype)
    r1 = X1_HT / (K1_HT * A_HT)
    r2 = ELL_HT / (K2_HT * A_HT)
    r3 = (L_HT - X2_HT) / (K3_HT * A_HT)
    rtot = r1 + r2 + r3
    s_mid = Q0_HT * ELL_HT**2 / (pi * K2_HT * A_HT)
    i2 = 2.0 * Q0_HT * ELL_HT / pi
    s_right = i2 * (L_HT - X2_HT) / (K3_HT * A_HT)
    return (T_LEFT_HT - T_RIGHT_HT - s_mid - s_right) / rtot


def heat_rate_exact(x: Tensor) -> Tensor:
    pi = torch.as_tensor(math.pi, device=x.device, dtype=x.dtype)
    h0 = initial_heat_rate(x.device, x.dtype)
    h = torch.zeros_like(x) + h0
    m2 = (x > X1_HT) & (x <= X2_HT)
    z = x[m2] - X1_HT
    h[m2] = h0 + Q0_HT * ELL_HT / pi * (1.0 - torch.cos(pi * z / ELL_HT))
    m3 = x > X2_HT
    h[m3] = h0 + 2.0 * Q0_HT * ELL_HT / pi
    return h


def temperature_exact(x: Tensor) -> Tensor:
    pi = torch.as_tensor(math.pi, device=x.device, dtype=x.dtype)
    h0 = initial_heat_rate(x.device, x.dtype)
    temp = torch.empty_like(x)
    t1 = T_LEFT_HT - h0 * X1_HT / (K1_HT * A_HT)
    mid_drop = (h0 * ELL_HT + Q0_HT * ELL_HT**2 / pi) / (K2_HT * A_HT)
    t2 = t1 - mid_drop
    h3 = h0 + 2.0 * Q0_HT * ELL_HT / pi

    m1 = x <= X1_HT
    m2 = (x > X1_HT) & (x <= X2_HT)
    m3 = x > X2_HT
    temp[m1] = T_LEFT_HT - h0 * x[m1] / (K1_HT * A_HT)

    z = x[m2] - X1_HT
    int_h = h0 * z + Q0_HT * ELL_HT / pi * (z - ELL_HT / pi * torch.sin(pi * z / ELL_HT))
    temp[m2] = t1 - int_h / (K2_HT * A_HT)
    temp[m3] = t2 - h3 * (x[m3] - X2_HT) / (K3_HT * A_HT)
    return temp


def gradient_exact(x: Tensor) -> Tensor:
    return -heat_rate_exact(x) / (conductivity(x) * A_HT)


def flux_exact(x: Tensor) -> Tensor:
    return heat_rate_exact(x) / A_HT


def resistance_coordinate(x: Tensor) -> Tensor:
    r1 = X1_HT / (K1_HT * A_HT)
    r2 = ELL_HT / (K2_HT * A_HT)
    r3 = (L_HT - X2_HT) / (K3_HT * A_HT)
    rtot = r1 + r2 + r3
    r = torch.empty_like(x)
    m1 = x <= X1_HT
    m2 = (x > X1_HT) & (x <= X2_HT)
    m3 = x > X2_HT
    r[m1] = x[m1] / (K1_HT * A_HT)
    r[m2] = r1 + (x[m2] - X1_HT) / (K2_HT * A_HT)
    r[m3] = r1 + r2 + (x[m3] - X2_HT) / (K3_HT * A_HT)
    return r / rtot


def resistance_lift(x: Tensor) -> Tensor:
    r = resistance_coordinate(x)
    return T_LEFT_HT + (T_RIGHT_HT - T_LEFT_HT) * r


def make_dirichlet(n_node: int, *, device=None, dtype=torch.float32) -> tuple[Tensor, Tensor]:
    idx = torch.tensor([0, n_node - 1], device=device, dtype=torch.long)
    vals = torch.tensor([T_LEFT_HT, T_RIGHT_HT], device=device, dtype=dtype)
    return idx, vals


def make_node_features(x_normalized: Tensor, x_resistance: Tensor, k_node: Tensor, q_node: Tensor) -> Tensor:
    dtype = x_normalized.dtype
    x = x_normalized.reshape(-1)
    layer1 = (x <= X1_HT).to(dtype).view(-1, 1)
    layer2 = ((x > X1_HT) & (x <= X2_HT)).to(dtype).view(-1, 1)
    layer3 = (x > X2_HT).to(dtype).view(-1, 1)
    return torch.cat(
        [
            x_normalized,
            x_resistance,
            (k_node / K_REF_HT).view(-1, 1),
            (q_node / Q_SCALE_HT).view(-1, 1),
            layer1,
            layer2,
            layer3,
        ],
        dim=1,
    )


def assemble_source_vector(x_nodes: Tensor) -> Tensor:
    f = torch.zeros_like(x_nodes)
    gp = torch.tensor([-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)], device=x_nodes.device, dtype=x_nodes.dtype)
    for e in range(x_nodes.numel() - 1):
        x1 = x_nodes[e]
        x2 = x_nodes[e + 1]
        h = x2 - x1
        for s in gp:
            n1 = 0.5 * (1.0 - s)
            n2 = 0.5 * (1.0 + s)
            xq = n1 * x1 + n2 * x2
            qq = source(xq)
            f[e] += n1 * qq * h / 2.0
            f[e + 1] += n2 * qq * h / 2.0
    return f


def element_centers(x_nodes: Tensor) -> Tensor:
    return 0.5 * (x_nodes[:-1] + x_nodes[1:])


def element_gradient(temp: Tensor, x_nodes: Tensor) -> Tensor:
    if temp.ndim == 1:
        temp = temp.unsqueeze(0)
    h = torch.diff(x_nodes.reshape(-1).to(device=temp.device, dtype=temp.dtype))
    return (temp[:, 1:] - temp[:, :-1]) / h.unsqueeze(0)


def element_heat_rate_from_temperature(temp: Tensor, x_nodes: Tensor) -> Tensor:
    xc = element_centers(x_nodes).to(device=temp.device, dtype=temp.dtype)
    return -conductivity(xc) * A_HT * element_gradient(temp, x_nodes)


def rel_l2_element(pred: Tensor, ref: Tensor, x_nodes: Tensor, eps: float = 1e-12) -> Tensor:
    p = pred.reshape(-1).to(dtype=torch.float64)
    r = ref.reshape(-1).to(device=p.device, dtype=p.dtype)
    x = element_centers(x_nodes).to(device=p.device, dtype=p.dtype)
    num = torch.trapezoid((p - r).square(), x).clamp_min(0.0).sqrt()
    den = torch.trapezoid(r.square(), x).clamp_min(eps).sqrt()
    return num / den

