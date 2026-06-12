from __future__ import annotations

import torch

from physics.derivatives import element_gradient_1d, node_gradient_1d
from src.validation.fem.linear_elasticity_1d import (
    A_BAR,
    E_STEEL,
    L_BAR,
    assemble_body_force_vector,
    body_force,
    make_mesh,
)

Tensor = torch.Tensor

E_NE = E_STEEL
A_NE = A_BAR
L_NE = L_BAR
NE_LOAD_SCALE = 5.0e6
ALPHA_CONVEX_NE = 1.0e15
EPS0_CONCAVE_NE = 1.8e-2


def stress(eps: Tensor, law: str = "convex") -> Tensor:
    if law == "linear":
        return E_NE * eps
    if law == "convex":
        return E_NE * eps + ALPHA_CONVEX_NE * eps**3
    if law == "concave":
        z = eps / EPS0_CONCAVE_NE
        return E_NE * EPS0_CONCAVE_NE * torch.tanh(z)
    raise ValueError(f"Unknown law: {law}")


def tangent(eps: Tensor, law: str = "convex") -> Tensor:
    if law == "linear":
        return torch.zeros_like(eps) + E_NE
    if law == "convex":
        return E_NE + 3.0 * ALPHA_CONVEX_NE * eps**2
    if law == "concave":
        z = eps / EPS0_CONCAVE_NE
        return E_NE / torch.cosh(z).clamp_min(1e-12) ** 2
    raise ValueError(f"Unknown law: {law}")


def energy_density(eps: Tensor, law: str = "convex") -> Tensor:
    if law == "linear":
        return 0.5 * E_NE * eps**2
    if law == "convex":
        return 0.5 * E_NE * eps**2 + 0.25 * ALPHA_CONVEX_NE * eps**4
    if law == "concave":
        z = eps / EPS0_CONCAVE_NE
        return E_NE * EPS0_CONCAVE_NE**2 * torch.log(torch.cosh(z))
    raise ValueError(f"Unknown law: {law}")


def assemble_load_vector(x_nodes: Tensor, load_scale: float = NE_LOAD_SCALE) -> Tensor:
    return assemble_body_force_vector(x_nodes, lambda x: body_force(x, load_scale=load_scale))


def make_node_features(x_normalized: Tensor, b_node: Tensor, *, law: str, load_scale: float = NE_LOAD_SCALE) -> Tensor:
    law_code = {"linear": 0.0, "convex": 1.0, "concave": -1.0}[law]
    return torch.cat(
        [
            x_normalized,
            (b_node / load_scale).view(-1, 1),
            torch.ones_like(x_normalized),
            torch.ones_like(x_normalized) * law_code,
        ],
        dim=1,
    )


def solve_reference_newton(
    x_nodes: Tensor,
    f_ext: Tensor,
    *,
    law: str,
    area: float = A_NE,
    tol: float = 1e-9,
    max_iter: int = 80,
) -> Tensor:
    x = x_nodes.detach().to(dtype=torch.float64)
    f = f_ext.detach().to(device=x.device, dtype=x.dtype)
    n = x.numel()
    u = torch.zeros(n, device=x.device, dtype=x.dtype)
    free = torch.arange(1, n - 1, device=x.device)

    for _ in range(max_iter):
        eps = element_gradient_1d(u, x).reshape(-1)
        sig = stress(eps, law=law)
        tan = tangent(eps, law=law)
        fint = torch.zeros_like(u)
        kt = torch.zeros(n, n, device=x.device, dtype=x.dtype)
        h = torch.diff(x)
        for e in range(n - 1):
            fint[e] -= area * sig[e]
            fint[e + 1] += area * sig[e]
            ke = area * tan[e] / h[e]
            kt[e, e] += ke
            kt[e, e + 1] -= ke
            kt[e + 1, e] -= ke
            kt[e + 1, e + 1] += ke
        r = fint - f
        if torch.linalg.norm(r[free]) <= tol * torch.linalg.norm(f[free]).clamp_min(1.0):
            break
        du = torch.linalg.solve(kt[free][:, free], -r[free])
        u[free] += du
    return u.to(dtype=x_nodes.dtype)


def reference_fields(x_nodes: Tensor, f_ext: Tensor, *, law: str) -> dict[str, Tensor]:
    u = solve_reference_newton(x_nodes, f_ext, law=law)
    eps = node_gradient_1d(u, x_nodes)
    sig = stress(eps, law=law)
    return {"u": u, "strain": eps, "stress": sig}

