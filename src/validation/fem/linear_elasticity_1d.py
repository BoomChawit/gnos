from __future__ import annotations

import math
from collections.abc import Callable

import torch

Tensor = torch.Tensor

L_BAR = 10.0
E_STEEL = 200.0e9
A_BAR = 5.0e-3
LOAD_SCALE = 1.0e5


def make_mesh(n_elem: int, length: float = L_BAR, *, device=None, dtype=torch.float32) -> tuple[Tensor, Tensor]:
    x = torch.linspace(0.0, length, n_elem + 1, device=device, dtype=dtype)
    x_normalized = (x / length).view(-1, 1)
    return x, x_normalized


def body_force_shape(x: Tensor) -> Tensor:
    pi = torch.as_tensor(math.pi, device=x.device, dtype=x.dtype)
    b1 = -(4.0 * pi**2 * (x - 2.5) ** 2 - 2.0 * pi) * torch.exp(-pi * (x - 2.5) ** 2)
    b2 = -(8.0 * pi**2 * (x - 7.5) ** 2 - 4.0 * pi) * torch.exp(-pi * (x - 7.5) ** 2)
    return b1 + b2


def body_force(x: Tensor, load_scale: float = LOAD_SCALE) -> Tensor:
    return load_scale * body_force_shape(x)


def exact_displacement(x: Tensor, *, area: float = A_BAR, young: float = E_STEEL, load_scale: float = LOAD_SCALE) -> Tensor:
    pi = torch.as_tensor(math.pi, device=x.device, dtype=x.dtype)
    ae = area * young
    u1 = (torch.exp(-pi * (x - 2.5) ** 2) - torch.exp(torch.as_tensor(-6.25 * math.pi, device=x.device, dtype=x.dtype))) / ae
    u2 = 2.0 * (torch.exp(-pi * (x - 7.5) ** 2) - torch.exp(torch.as_tensor(-56.25 * math.pi, device=x.device, dtype=x.dtype))) / ae
    u3 = -(torch.exp(torch.as_tensor(-6.25 * math.pi, device=x.device, dtype=x.dtype)) - torch.exp(torch.as_tensor(-56.25 * math.pi, device=x.device, dtype=x.dtype))) * x / (10.0 * ae)
    return load_scale * (u1 + u2 + u3)


def exact_strain(x: Tensor, *, area: float = A_BAR, young: float = E_STEEL, load_scale: float = LOAD_SCALE) -> Tensor:
    pi = torch.as_tensor(math.pi, device=x.device, dtype=x.dtype)
    ae = area * young
    dudx1 = -2.0 * pi * (x - 2.5) * torch.exp(-pi * (x - 2.5) ** 2) / ae
    dudx2 = -4.0 * pi * (x - 7.5) * torch.exp(-pi * (x - 7.5) ** 2) / ae
    dudx3 = -(torch.exp(torch.as_tensor(-6.25 * math.pi, device=x.device, dtype=x.dtype)) - torch.exp(torch.as_tensor(-56.25 * math.pi, device=x.device, dtype=x.dtype))) / (10.0 * ae)
    return load_scale * (dudx1 + dudx2 + dudx3)


def assemble_body_force_vector(x_nodes: Tensor, body_force_fn: Callable[[Tensor], Tensor]) -> Tensor:
    f = torch.zeros_like(x_nodes)
    gauss = torch.tensor([-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)], device=x_nodes.device, dtype=x_nodes.dtype)
    for e in range(x_nodes.numel() - 1):
        x1 = x_nodes[e]
        x2 = x_nodes[e + 1]
        h = x2 - x1
        for xi in gauss:
            n1 = 0.5 * (1.0 - xi)
            n2 = 0.5 * (1.0 + xi)
            xq = n1 * x1 + n2 * x2
            bq = body_force_fn(xq)
            wj = h / 2.0
            f[e] += n1 * bq * wj
            f[e + 1] += n2 * bq * wj
    return f


def make_node_features(x_normalized: Tensor, b_node: Tensor, *, load_scale: float = LOAD_SCALE) -> Tensor:
    return torch.cat([x_normalized, (b_node / load_scale).view(-1, 1)], dim=1)

