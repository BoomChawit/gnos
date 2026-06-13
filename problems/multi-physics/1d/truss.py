from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import GAMFNO1D, count_trainable_params
from physics import bc_error
from physics.energy import thermo_nonlinear_elastic_energy_1d
from src.solver import load_config, relative_l2, set_seed, train_energy_model
from src.utils.io import ensure_dir, save_json
from src.utils.viz import plot_field_1d
from src.validation.fem import multiphysics_1d as mp


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def choose_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def main() -> None:
    parser = argparse.ArgumentParser(description="GNOS 1D thermo-nonlinear truss demo")
    parser.add_argument("--config", default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = choose_device(cfg.get("device", "auto"))
    dtype = choose_dtype(cfg.get("dtype", "float32"))
    mesh = cfg["mesh"]
    train = cfg["train"]
    model_cfg = cfg["model"]

    n_elem = int(mesh["n_elem"])
    x, xg = mp.make_mesh(n_elem, mp.L_MP, device=device, dtype=dtype)
    n_node = x.numel()
    dirichlet_idx, dirichlet_val, free_idx = mp.make_dirichlet(n_node, device=device, dtype=dtype)

    temp_node = mp.temperature(x)
    body_node = mp.body_force(x)
    f_ext = mp.assemble_load_vector(x)
    xc = mp.element_centers(x)
    temp_elem = mp.temperature(xc)
    eps_th_elem = mp.thermal_strain(temp_elem)

    ref = mp.reference_solution(x, temp_node)
    center_ref = mp.reference_center_fields(x, ref["C"])
    u_ref = ref["u"]
    c_node = mp.make_node_features(xg, temp_node, body_node)

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
        output_scale=mp.U_SCALE_MP,
    ).to(device=device)

    def predict():
        return model(xg, c_node, dirichlet_idx=dirichlet_idx, dirichlet_val=dirichlet_val)

    def energy(u):
        return thermo_nonlinear_elastic_energy_1d(
            u,
            x,
            f_ext,
            area=mp.A_MP,
            thermal_strain_elem=eps_th_elem,
            energy_density=mp.energy_density,
        )

    def metrics(u):
        eps = mp.element_strain(u, x)
        mech = eps - eps_th_elem.reshape(1, -1)
        sig = mp.stress(mech)
        axial = mp.A_MP * sig
        return {
            "rel_u": relative_l2(u, u_ref, x),
            "rel_strain": mp.rel_l2_element(eps, center_ref["eps_total"], x),
            "rel_mech_strain": mp.rel_l2_element(mech, center_ref["e_mech"], x),
            "rel_stress": mp.rel_l2_element(sig, center_ref["sigma"], x),
            "rel_axial": mp.rel_l2_element(axial, center_ref["N"], x),
            "residual": mp.residual_norm(u, x, temp_elem, f_ext, free_idx),
            "bc_error": bc_error(u, dirichlet_idx, dirichlet_val),
        }

    n_iter = int(args.max_iter if args.max_iter is not None else train["n_iter"])
    print(f"multi-physics/1d | n_elem={n_elem} | iter={n_iter} | params={count_trainable_params(model):,}")
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
    result["problem"] = "multi-physics/1d/thermo-nonlinear-truss"
    result["n_elem"] = n_elem

    output_dir = ensure_dir(ROOT / cfg["output_dir"])
    save_json(result, output_dir / "metrics.json")
    if not args.no_plots:
        plot_field_1d(x, u_ref, u_pred, output_dir / "displacement.png", title="1D Thermo-Nonlinear Elasticity", ylabel="u [m]")

    m = result["metrics"]
    print(f"saved={output_dir}")
    print(f"final | rel_u={100*m['rel_u']:.3f}% | rel_stress={100*m['rel_stress']:.3f}% | res={m['residual']:.2e}")


if __name__ == "__main__":
    main()

