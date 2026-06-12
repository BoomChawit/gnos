from __future__ import annotations

import torch

Tensor = torch.Tensor


def _as_batch(u: Tensor) -> Tensor:
    if u.ndim == 1:
        return u.unsqueeze(0)
    if u.ndim == 2:
        return u
    if u.ndim == 3 and u.shape[-1] == 1:
        return u[..., 0]
    raise ValueError(f"Expected u with shape [N], [B,N], or [B,N,1], got {tuple(u.shape)}")


def element_gradient_1d(u: Tensor, x_nodes: Tensor) -> Tensor:
    """Piecewise-constant du/dx using the 1D two-node B-matrix calculus."""
    u_b = _as_batch(u)
    x = x_nodes.reshape(-1).to(device=u_b.device, dtype=u_b.dtype)
    h = torch.diff(x)
    if u_b.shape[-1] != x.numel():
        raise ValueError(f"u has {u_b.shape[-1]} nodes but x_nodes has {x.numel()}.")
    return (u_b[:, 1:] - u_b[:, :-1]) / h.unsqueeze(0)


def node_gradient_1d(u: Tensor, x_nodes: Tensor) -> Tensor:
    """Node strain by averaging adjacent element gradients."""
    elem = element_gradient_1d(u, x_nodes)
    out = torch.zeros(elem.shape[0], elem.shape[1] + 1, device=elem.device, dtype=elem.dtype)
    out[:, 0] = elem[:, 0]
    out[:, -1] = elem[:, -1]
    if elem.shape[1] > 1:
        out[:, 1:-1] = 0.5 * (elem[:, :-1] + elem[:, 1:])
    return out.squeeze(0) if u.ndim == 1 else out

