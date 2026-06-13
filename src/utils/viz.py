from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_field_1d(x, ref, pred, path: str | Path, *, title: str, ylabel: str = "u") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x_np = torch.as_tensor(x).detach().cpu().reshape(-1)
    ref_np = torch.as_tensor(ref).detach().cpu().reshape(-1)
    pred_np = torch.as_tensor(pred).detach().cpu().reshape(-1)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(x_np, ref_np, "k-", lw=2, label="reference")
    ax.plot(x_np, pred_np, "r--", lw=1.8, label="GNOS")
    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _cmap(name: str):
    if name.startswith("cmo."):
        try:
            import cmocean

            return getattr(cmocean.cm, name.split(".", 1)[1])
        except Exception:
            return "viridis"
    return name


def plot_panel_displacement_2d(
    xs,
    ys,
    ref_disp,
    pred_disp,
    path: str | Path,
    *,
    title: str,
    cmap: str = "viridis",
    deform_scale: float = 1.0,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xs_np = np.asarray(xs, dtype=float)
    ys_np = np.asarray(ys, dtype=float)
    xx, yy = np.meshgrid(xs_np, ys_np, indexing="xy")
    ny, nx = yy.shape

    ref = torch.as_tensor(ref_disp).detach().cpu().numpy().reshape(ny, nx, 2)
    pred = torch.as_tensor(pred_disp).detach().cpu().numpy().reshape(ny, nx, 2)
    ref_mag = np.linalg.norm(ref, axis=-1)
    pred_mag = np.linalg.norm(pred, axis=-1)
    vmin = float(min(np.nanmin(ref_mag), np.nanmin(pred_mag)))
    vmax = float(max(np.nanmax(ref_mag), np.nanmax(pred_mag)))
    levels = np.linspace(vmin, vmax, 64)
    cm = _cmap(cmap)

    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.2), constrained_layout=True)
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    mesh_step = max(1, int(np.ceil(max(nx, ny) / 24)))
    ax = axes[0]
    for j in range(0, ny, mesh_step):
        ax.plot(xs_np, np.full_like(xs_np, ys_np[j]), color="0.82", lw=0.6)
    for i in range(0, nx, mesh_step):
        ax.plot(np.full_like(ys_np, xs_np[i]), ys_np, color="0.82", lw=0.6)
    xmid = 0.5 * (xs_np[0] + xs_np[-1])
    ax.plot(xs_np, np.full_like(xs_np, ys_np[0]), color="black", lw=3.0, solid_capstyle="butt")
    top_x = xs_np[xs_np <= xmid + 1e-12]
    ax.plot(top_x, np.full_like(top_x, ys_np[-1]), color="red", lw=3.0, solid_capstyle="butt")
    ax.set_title("mesh + BC")

    im = axes[1].contourf(xx, yy, ref_mag, levels=levels, cmap=cm)
    axes[1].set_title("|u| FEM")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02)

    im = axes[2].contourf(xx, yy, pred_mag, levels=levels, cmap=cm)
    axes[2].set_title("|u| GNOS")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.02)

    xd = xx + deform_scale * pred[:, :, 0]
    yd = yy + deform_scale * pred[:, :, 1]
    im = axes[3].contourf(xd, yd, pred_mag, levels=levels, cmap=cm)
    for j in range(0, ny, mesh_step):
        axes[3].plot(xd[j, :], yd[j, :], color="0.2", lw=0.35, alpha=0.55)
    for i in range(0, nx, mesh_step):
        axes[3].plot(xd[:, i], yd[:, i], color="0.2", lw=0.35, alpha=0.55)
    axes[3].set_title("deformation")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.02)

    fig.suptitle(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)
