from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import GAMFNO2D, count_trainable_params
from physics.energy import linear_elastic_q4_energy_2d
from physics.derivatives import q4_b_matrices
from src.solver import load_config, set_seed, train_energy_model
from src.utils.io import ensure_dir, save_json
from src.utils.viz import plot_panel_displacement_2d
from src.validation.fem import linear_elasticity_2d as le


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def choose_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def main() -> None:
    parser = argparse.ArgumentParser(description="GNOS 2D linear-elastic panel demo")
    parser.add_argument("--config", default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = choose_device(cfg.get("device", "auto"))
    dtype = choose_dtype(cfg.get("dtype", "float32"))
    mesh_cfg = cfg["mesh"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    nx = int(mesh_cfg["nx"])
    ny = int(mesh_cfg["ny"])
    ref = le.solve_reference(nx, ny)

    coords = torch.tensor(ref["coords"], device=device, dtype=dtype)
    elements = torch.tensor(ref["elements"], device=device, dtype=torch.long)
    bc_mask = torch.tensor(ref["bc_mask"], device=device, dtype=torch.bool)
    bc_values = torch.tensor(ref["bc_values"], device=device, dtype=dtype)
    features = le.make_node_features(ref["coords"], ref["bc_mask"], ref["bc_values"])
    c_nodes = torch.tensor(features[None, :, :], device=device, dtype=dtype)
    u_ref = torch.tensor(ref["disp"], device=device, dtype=dtype)
    strain_ref = torch.tensor(ref["strain_gp"], device=device, dtype=dtype)
    stress_ref = torch.tensor(ref["stress_gp"], device=device, dtype=dtype)
    stiffness = torch.tensor(ref["K"], device=device, dtype=dtype)
    free_dofs = torch.tensor(ref["free_dofs"], device=device, dtype=torch.long)

    b_mats, det_j, weights = q4_b_matrices(coords, elements)
    d_mat = le.d_matrix_torch(device=device, dtype=dtype)
    g_nodes, phi_nodes = le.make_envelope_nodes(coords, top_v=le.TOP_V_LE)

    model = GAMFNO2D(
        node_in_dim=c_nodes.shape[-1],
        out_dim=2,
        backbone=model_cfg["backbone"],
        latent_dim=int(model_cfg["latent_dim"]),
        n_latent=model_cfg["n_latent"],
        radial_hidden=int(model_cfg["radial_hidden"]),
        sigma_enc=float(model_cfg["sigma_enc"]),
        sigma_dec=float(model_cfg["sigma_dec"]),
        width=int(model_cfg["width"]),
        modes1=int(model_cfg["modes1"]),
        modes2=int(model_cfg["modes2"]),
        layers=int(model_cfg["layers"]),
        fc_dim=int(model_cfg["fc_dim"]),
        fno_padding=model_cfg["fno_padding"],
        append_latent_coords=True,
        bc_mode="none",
        mask_latent=True,
        latent_support_threshold=float(model_cfg["latent_support_threshold"]),
        output_scale=[le.TOP_V_LE, le.TOP_V_LE],
    ).to(device=device)

    def predict():
        raw = model(coords, c_nodes, bc_mode="none")
        return le.apply_envelope(raw, g_nodes, phi_nodes)

    def energy(u):
        return linear_elastic_q4_energy_2d(
            u,
            elements,
            b_mats,
            det_j,
            weights,
            d_mat,
            thickness=le.THICK_LE,
        )

    def metrics(u):
        u0 = u[0] if u.ndim == 3 else u
        strain, stress = le.element_gauss_fields_torch(u0, elements, b_mats, d_mat)
        return {
            "rel_u": le.rel_l2(u0, u_ref),
            "rel_umag": le.rel_l2(torch.linalg.norm(u0, dim=-1), torch.linalg.norm(u_ref, dim=-1)),
            "rel_strain": le.rel_l2(strain, strain_ref),
            "rel_stress": le.rel_l2(stress, stress_ref),
            "residual": le.residual_norm(u0, stiffness, free_dofs),
            "bc_error": le.bc_error(u0, bc_mask, bc_values),
        }

    n_iter = int(args.max_iter if args.max_iter is not None else train_cfg["n_iter"])
    print(f"linear-elasticity/2d | grid={ny}x{nx} | iter={n_iter} | params={count_trainable_params(model):,}")
    result = train_energy_model(
        model,
        predict,
        energy,
        metrics,
        n_iter=n_iter,
        lr=float(train_cfg["lr"]),
        print_every=int(train_cfg["print_every"]),
    )

    with torch.no_grad():
        u_pred = predict()
    result["problem"] = "linear-elasticity/2d/panel"
    result["grid"] = [ny, nx]
    result["reference_internal_energy"] = float(ref["internal_energy"])

    output_dir = ensure_dir(ROOT / cfg["output_dir"])
    save_json(result, output_dir / "metrics.json")
    if not args.no_plots:
        plot_panel_displacement_2d(
            ref["xs"],
            ref["ys"],
            u_ref,
            u_pred,
            output_dir / "displacement.png",
            title="2D Linear Elasticity Panel",
            cmap="viridis",
        )

    m = result["metrics"]
    print(f"saved={output_dir}")
    print(f"final | rel_u={100*m['rel_u']:.3f}% | rel_stress={100*m['rel_stress']:.3f}% | res={m['residual']:.2e}")


if __name__ == "__main__":
    main()
