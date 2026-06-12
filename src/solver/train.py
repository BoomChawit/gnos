from __future__ import annotations

import copy
import time
from collections.abc import Callable

import numpy as np
import torch

from .metrics import to_float_dict

Tensor = torch.Tensor


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_energy_model(
    model: torch.nn.Module,
    predict_fn: Callable[[], Tensor],
    energy_fn: Callable[[Tensor], Tensor],
    metrics_fn: Callable[[Tensor], dict[str, object]],
    *,
    n_iter: int,
    lr: float,
    weight_decay: float = 1e-6,
    grad_clip: float = 1.0,
    print_every: int = 100,
    verbose: bool = True,
) -> dict[str, object]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, n_iter), eta_min=lr * 0.01)

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, object]] = []
    t0 = time.time()

    for it in range(n_iter):
        opt.zero_grad(set_to_none=True)
        u = predict_fn()
        loss = energy_fn(u)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        sched.step()

        loss_value = float(loss.detach().cpu())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = copy.deepcopy(model.state_dict())

        should_log = it == 0 or it == n_iter - 1 or (print_every > 0 and it % print_every == 0)
        if should_log:
            with torch.no_grad():
                u_eval = predict_fn()
            row = {"iter": it, "time_s": time.time() - t0, "loss": loss_value}
            row.update(to_float_dict(metrics_fn(u_eval)))
            history.append(row)
            if verbose:
                rel = row.get("rel_u", None)
                res = row.get("residual", None)
                parts = [f"it={it:04d}", f"loss={loss_value:+.3e}"]
                if isinstance(rel, float):
                    parts.append(f"rel_u={100.0 * rel:.3f}%")
                if isinstance(res, float):
                    parts.append(f"res={res:.2e}")
                print(" | ".join(parts))

    model.load_state_dict(best_state)
    with torch.no_grad():
        u_final = predict_fn()
    final_metrics = to_float_dict(metrics_fn(u_final))
    return {
        "best_loss": best_loss,
        "train_time_s": time.time() - t0,
        "history": history,
        "metrics": final_metrics,
    }

