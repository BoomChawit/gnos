from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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

