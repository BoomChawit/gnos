from __future__ import annotations

import math

import torch

Tensor = torch.Tensor


def q4_gauss_rule(*, device=None, dtype=torch.float32) -> tuple[Tensor, Tensor]:
    a = 1.0 / math.sqrt(3.0)
    points = torch.tensor(
        [[-a, -a], [a, -a], [a, a], [-a, a]],
        device=device,
        dtype=dtype,
    )
    weights = torch.ones(4, device=device, dtype=dtype)
    return points, weights


def q4_shape_gradients_nat(xi: Tensor, eta: Tensor) -> Tensor:
    """Natural-coordinate derivatives [dN/dxi, dN/deta] for Q4 nodes."""
    return 0.25 * torch.stack(
        [
            torch.stack([-(1.0 - eta), -(1.0 - xi)]),
            torch.stack([(1.0 - eta), -(1.0 + xi)]),
            torch.stack([(1.0 + eta), (1.0 + xi)]),
            torch.stack([-(1.0 + eta), (1.0 - xi)]),
        ],
        dim=0,
    )


def q4_shape_functions(xi: Tensor, eta: Tensor) -> Tensor:
    """Q4 shape functions in counter-clockwise node order."""
    return 0.25 * torch.stack(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ],
        dim=0,
    )


def q4_scalar_grad_matrices(coords: Tensor, elements: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return Q4 scalar shape values and physical gradients at 2x2 Gauss points.

    Returns:
        shape_vals: [4, 4] shape functions N_a at Gauss points.
        grad_mats: [E, 4, 2, 4] rows = d/dx, d/dy.
        det_j: [E, 4].
        weights: [4].
    """
    x_e = coords[elements]
    points, weights = q4_gauss_rule(device=coords.device, dtype=coords.dtype)
    n_all = []
    grad_all = []
    det_all = []
    for xi, eta in points:
        n_all.append(q4_shape_functions(xi, eta).to(device=coords.device, dtype=coords.dtype))
        d_nat = q4_shape_gradients_nat(xi, eta).to(device=coords.device, dtype=coords.dtype)
        jac = torch.einsum("ai,eaj->eij", d_nat, x_e)
        det_j = torch.linalg.det(jac)
        inv_j = torch.linalg.inv(jac)
        d_xy = torch.einsum("ai,eij->eaj", d_nat, inv_j)
        grad_all.append(d_xy.transpose(1, 2))
        det_all.append(det_j)

    return (
        torch.stack(n_all, dim=0),
        torch.stack(grad_all, dim=1),
        torch.stack(det_all, dim=1),
        weights,
    )


def q4_b_matrices(coords: Tensor, elements: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return Q4 strain-displacement matrices at 2x2 Gauss points.

    Args:
        coords: nodal coordinates [N, 2].
        elements: Q4 connectivity [E, 4] in counter-clockwise order.

    Returns:
        b_mats: [E, 4, 3, 8] with strains [exx, eyy, gxy].
        det_j: [E, 4].
        weights: [4].
    """
    x_e = coords[elements]
    points, weights = q4_gauss_rule(device=coords.device, dtype=coords.dtype)
    b_all = []
    det_all = []
    for xi, eta in points:
        d_nat = q4_shape_gradients_nat(xi, eta).to(device=coords.device, dtype=coords.dtype)
        jac = torch.einsum("ai,eaj->eij", d_nat, x_e)
        det_j = torch.linalg.det(jac)
        inv_j = torch.linalg.inv(jac)
        d_xy = torch.einsum("ai,eij->eaj", d_nat, inv_j)

        b = torch.zeros(
            (elements.shape[0], 3, 8),
            device=coords.device,
            dtype=coords.dtype,
        )
        b[:, 0, 0::2] = d_xy[:, :, 0]
        b[:, 1, 1::2] = d_xy[:, :, 1]
        b[:, 2, 0::2] = d_xy[:, :, 1]
        b[:, 2, 1::2] = d_xy[:, :, 0]
        b_all.append(b)
        det_all.append(det_j)

    return torch.stack(b_all, dim=1), torch.stack(det_all, dim=1), weights


def q4_element_strain(u: Tensor, elements: Tensor, b_mats: Tensor) -> Tensor:
    """Compute Q4 Gauss strains from nodal displacement."""
    if u.ndim == 2:
        u_b = u.unsqueeze(0)
    elif u.ndim == 3:
        u_b = u
    else:
        raise ValueError(f"Expected [N,2] or [B,N,2], got {tuple(u.shape)}")
    u_e = u_b[:, elements, :].reshape(u_b.shape[0], elements.shape[0], 8)
    return torch.einsum("egij,bej->begi", b_mats, u_e)


def q4_scalar_gradient(field: Tensor, elements: Tensor, grad_mats: Tensor) -> Tensor:
    """Compute Q4 Gauss gradients for a scalar nodal field."""
    if field.ndim == 1:
        f_b = field.unsqueeze(0)
    elif field.ndim == 2 and field.shape[-1] == 1:
        f_b = field[:, 0].unsqueeze(0)
    elif field.ndim == 2:
        f_b = field
    elif field.ndim == 3 and field.shape[-1] == 1:
        f_b = field[..., 0]
    else:
        raise ValueError(f"Expected scalar nodal field, got {tuple(field.shape)}")
    elems = elements.to(device=f_b.device, dtype=torch.long)
    grad = grad_mats.to(device=f_b.device, dtype=f_b.dtype)
    f_e = f_b[:, elems]
    return torch.einsum("egij,bej->begi", grad, f_e)
