from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import GAMFNO2D, count_trainable_params
from physics.derivatives import q4_b_matrices, q4_scalar_grad_matrices
from physics.energy import thermo_nonlinear_q4_energy_2d
from src.solver import load_config, set_seed, train_energy_model
from src.utils.io import ensure_dir, save_json
from src.utils.viz import plot_panel_displacement_2d
from src.validation.fem import linear_elasticity_2d as le
from src.validation.fem import multiphysics_2d as mp


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def choose_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def main() -> None:
    parser = argparse.ArgumentParser(description="GNOS 2D one-way thermo-nonlinear panel demo")
    parser.add_argument("--config", default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = choose_device(cfg.get("device", "auto"))
    dtype = choose_dtype(cfg.get("dtype", "float32"))
    mesh_cfg = cfg["mesh"]
    heat_cfg = cfg["heat"]
    mat_cfg = cfg["material"]
    ref_cfg = cfg["reference"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    nx = int(mesh_cfg["nx"])
    ny = int(mesh_cfg["ny"])
    E = float(mat_cfg["E"])
    nu = float(mat_cfg["nu"])
    top_v = float(mat_cfg["top_v"])
    alpha = float(mat_cfg["hard_alpha"])
    p = float(mat_cfg["hard_p"])
    alpha_t = float(mat_cfg["alpha_t"])
    t0 = float(mat_cfg["T0"])
    thickness = float(mat_cfg["thickness"])
    plane = str(mat_cfg["plane"])

    ref = mp.solve_reference(
        nx,
        ny,
        heat_case=str(heat_cfg["case"]),
        temp_scale=float(heat_cfg["temp_scale"]),
        recenter_temperature=bool(heat_cfg["recenter_temperature"]),
        E=E,
        nu=nu,
        thickness=thickness,
        top_v=top_v,
        plane=plane,
        alpha=alpha,
        p=p,
        alpha_t=alpha_t,
        t0_value=t0,
        n_steps=int(ref_cfg["n_steps"]),
        newton_tol=float(ref_cfg["newton_tol"]),
        newton_max_iter=int(ref_cfg["newton_max_iter"]),
        ramp_temperature=bool(ref_cfg["ramp_temperature"]),
    )

    coords = torch.tensor(ref["coords"], device=device, dtype=dtype)
    elements = torch.tensor(ref["elements"], device=device, dtype=torch.long)
    bc_mask = torch.tensor(ref["bc_mask"], device=device, dtype=torch.bool)
    bc_values = torch.tensor(ref["bc_values"], device=device, dtype=dtype)
    free_dofs = torch.tensor(ref["free_dofs"], device=device, dtype=torch.long)
    b_mats, det_j, weights = q4_b_matrices(coords, elements)
    shape_vals = q4_scalar_grad_matrices(coords, elements)[0]
    d_mat = le.d_matrix_torch(E, nu, plane, device=device, dtype=dtype)

    features = mp.make_node_features(
        ref["coords"],
        ref["xs"],
        ref["ys"],
        ref["bc_values"],
        ref["T"],
        top_v=top_v,
        nu=nu,
        alpha=alpha,
        p=p,
        alpha_t=alpha_t,
        t0=t0,
    )
    c_nodes = torch.tensor(features[None, :, :], device=device, dtype=dtype)
    u_ref = torch.tensor(ref["disp"], device=device, dtype=dtype)
    temp_nodes = torch.tensor(ref["T"].reshape(-1, 1), device=device, dtype=dtype)
    eps_eff_ref = torch.tensor(ref["fields_gp"]["eps_eff"], device=device, dtype=dtype)
    stress_ref = torch.tensor(ref["fields_gp"]["stress"], device=device, dtype=dtype)
    g_nodes, phi_nodes = le.make_envelope_nodes(coords, top_v=top_v)
    energy_scale = max(abs(float(ref["internal_energy"])), 1e-12)

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
        output_scale=[top_v, top_v],
        output_shift=[0.0, 0.0],
    ).to(device=device)

    def predict():
        raw = model(coords, c_nodes, bc_mode="none")
        return le.apply_envelope(raw, g_nodes, phi_nodes)

    def raw_energy(u):
        return thermo_nonlinear_q4_energy_2d(
            u,
            temp_nodes,
            elements,
            shape_vals,
            b_mats,
            det_j,
            weights,
            d_mat,
            alpha=alpha,
            p=p,
            alpha_t=alpha_t,
            t0=t0,
            thickness=thickness,
        )

    def energy(u):
        return raw_energy(u) / energy_scale

    def metrics(u):
        u0 = u[0] if u.ndim == 3 else u
        fields = mp.element_gauss_fields_torch(
            u0,
            temp_nodes,
            elements,
            shape_vals,
            b_mats,
            d_mat,
            alpha=alpha,
            p=p,
            alpha_t=alpha_t,
            t0=t0,
        )
        energy_value = raw_energy(u0)
        return {
            "rel_u": mp.rel_l2(u0, u_ref),
            "rel_umag": mp.rel_l2(torch.linalg.norm(u0, dim=-1), torch.linalg.norm(u_ref, dim=-1)),
            "rel_eff_strain": mp.rel_l2(fields["eps_eff"], eps_eff_ref),
            "rel_stress": mp.rel_l2(fields["sig_vec"], stress_ref),
            "energy_gap": (energy_value - energy_scale) / energy_scale,
            "residual": mp.residual_norm_from_energy(raw_energy, u0, free_dofs),
            "bc_error": le.bc_error(u0, bc_mask, bc_values),
        }

    n_iter = int(args.max_iter if args.max_iter is not None else train_cfg["n_iter"])
    print(f"multi-physics/2d | grid={ny}x{nx} | iter={n_iter} | params={count_trainable_params(model):,}")
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
    result["problem"] = "multi-physics/2d/panel"
    result["grid"] = [ny, nx]
    result["reference_internal_energy"] = float(ref["internal_energy"])
    result["reference_converged"] = bool(ref["converged_all"])
    result["temperature_source"] = ref["temperature_pack"]["source"]

    output_dir = ensure_dir(ROOT / cfg["output_dir"])
    save_json(result, output_dir / "metrics.json")
    if not args.no_plots:
        plot_panel_displacement_2d(
            ref["xs"],
            ref["ys"],
            u_ref,
            u_pred,
            output_dir / "displacement.png",
            title="2D One-way Thermo-Nonlinear Panel",
            cmap="cmo.balance",
        )

    m = result["metrics"]
    print(f"saved={output_dir}")
    print(f"final | rel_u={100*m['rel_u']:.3f}% | rel_stress={100*m['rel_stress']:.3f}% | res={m['residual']:.2e}")


if __name__ == "__main__":
    main()
