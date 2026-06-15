from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import GAMFNO1D, count_trainable_params
from physics import bc_error, make_dirichlet_fixed_fixed
from physics.derivatives import node_gradient_1d
from physics.energy import linear_elastic_energy, residual_norm_from_energy
from src.solver import load_config, relative_l2, set_seed, train_energy_model
from src.utils.io import ensure_dir, save_json
from src.utils.viz import plot_field_1d
from src.validation.fem import linear_elasticity_1d as le


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def choose_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def main() -> None:
    parser = argparse.ArgumentParser(description="GNOS 1D linear-elastic truss demo")
    parser.add_argument("--config", default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    device = choose_device(cfg.get("device", "auto"))
    dtype = choose_dtype(cfg.get("dtype", "float32"))
    mat = cfg["material"]
    mesh = cfg["mesh"]
    train = cfg["train"]
    model_cfg = cfg["model"]

    length = float(mat["length"])
    area = float(mat["area"])
    young = float(mat["young"])
    load_scale = float(mat["load_scale"])
    n_elem = int(mesh["n_elem"])

    x, xg = le.make_mesh(n_elem, length, device=device, dtype=dtype)
    b_node = le.body_force(x, load_scale=load_scale)
    f_ext = le.assemble_body_force_vector(x, lambda xp: le.body_force(xp, load_scale=load_scale))
    c_node = le.make_node_features(xg, b_node, load_scale=load_scale)
    dirichlet_idx, dirichlet_val = make_dirichlet_fixed_fixed(x.numel(), device=device, dtype=dtype)

    u_ref = le.exact_displacement(x, area=area, young=young, load_scale=load_scale)
    strain_ref = le.exact_strain(x, area=area, young=young, load_scale=load_scale)

    output_scale = load_scale * length**2 / (area * young)
    model = GAMFNO1D(
        node_in_dim=c_node.shape[-1],
        out_dim=1,
        backbone=model_cfg["backbone"],
        latent_dim=int(model_cfg["latent_dim"]),
        n_latent=int(model_cfg["n_latent"]),
        radial_hidden=int(model_cfg["radial_hidden"]),
        sigma_enc=float(model_cfg["sigma_enc"]),
        sigma_dec=float(model_cfg["sigma_dec"]),
        width=int(model_cfg["width"]),
        modes=int(model_cfg["modes"]),
        layers=int(model_cfg["layers"]),
        fc_dim=int(model_cfg["fc_dim"]),
        fno_padding=int(model_cfg["fno_padding"]),
        bc_mode=model_cfg["bc_mode"],
        output_scale=output_scale,
    ).to(device=device)

    def predict():
        return model(xg, c_node, dirichlet_idx=dirichlet_idx, dirichlet_val=dirichlet_val)

    def energy(u):
        return linear_elastic_energy(u, x, f_ext, area=area, young=young)

    free_idx = torch.arange(1, x.numel() - 1, device=device)

    def metrics(u):
        strain = node_gradient_1d(u, x)
        return {
            "rel_u": relative_l2(u, u_ref, x),
            "rel_strain": relative_l2(strain, strain_ref, x),
            "residual": residual_norm_from_energy(energy, u, free_idx, f_ext),
            "bc_error": bc_error(u, dirichlet_idx, dirichlet_val),
        }

    n_iter = int(args.max_iter if args.max_iter is not None else train["n_iter"])
    print(f"linear-elasticity/1d | n_elem={n_elem} | iter={n_iter} | params={count_trainable_params(model):,}")
    result = train_energy_model(
        model,
        predict,
        energy,
        metrics,
        n_iter=n_iter,
        lr=float(train["lr"]),
        print_every=int(train["print_every"]),
    )

    with torch.no_grad():
        u_pred = predict()
    result["problem"] = "linear-elasticity/1d/truss"
    result["n_elem"] = n_elem

    output_dir = ensure_dir(ROOT / cfg["output_dir"])
    save_json(result, output_dir / "metrics.json")
    if not args.no_plots:
        plot_field_1d(x, u_ref, u_pred, output_dir / "displacement.png", title="1D Linear Elasticity", ylabel="u [m]")

    m = result["metrics"]
    print(f"saved={output_dir}")
    print(f"final | rel_u={100*m['rel_u']:.3f}% | rel_strain={100*m['rel_strain']:.3f}% | res={m['residual']:.2e}")


if __name__ == "__main__":
    main()
