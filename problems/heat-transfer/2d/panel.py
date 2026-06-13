from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import GAMFNO2D, count_trainable_params
from physics.derivatives import q4_scalar_grad_matrices
from physics.energy import heat_q4_energy_2d
from src.solver import load_config, set_seed, train_energy_model
from src.utils.io import ensure_dir, save_json
from src.utils.viz import plot_panel_heat_2d
from src.validation.fem import heat_transfer_2d as ht


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def choose_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def main() -> None:
    parser = argparse.ArgumentParser(description="GNOS 2D heat-transfer panel demo")
    parser.add_argument("--config", default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = choose_device(cfg.get("device", "auto"))
    dtype = choose_dtype(cfg.get("dtype", "float32"))
    mesh_cfg = cfg["mesh"]
    ref_cfg = cfg["reference"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    nx = int(mesh_cfg["nx"])
    ny = int(mesh_cfg["ny"])
    case = str(ref_cfg["case"])
    k = float(ref_cfg["k"])
    source = float(ref_cfg["source"])
    thickness = float(ref_cfg["thickness"])
    ref = ht.solve_reference(nx, ny, case=case, k=k, source=source, thickness=thickness)

    coords = torch.tensor(ref["coords"], device=device, dtype=dtype)
    elements = torch.tensor(ref["elements"], device=device, dtype=torch.long)
    fixed_nodes = torch.tensor(ref["fixed_nodes"], device=device, dtype=torch.long)
    fixed_vals = torch.tensor(ref["fixed_vals"], device=device, dtype=dtype)
    free_nodes = torch.tensor(ref["free_nodes"], device=device, dtype=torch.long)
    shape_vals, grad_mats, det_j, weights = q4_scalar_grad_matrices(coords, elements)

    features, _, _, temp_scale = ht.make_node_features(
        ref["coords"],
        ref["xs"],
        ref["ys"],
        ref["T_left"],
        ref["T_right"],
        case=case,
        k=k,
        source=source,
    )
    c_nodes = torch.tensor(features[None, :, :], device=device, dtype=dtype)
    t_ref = torch.tensor(ref["T"][:, None], device=device, dtype=dtype)
    grad_ref = torch.tensor(ref["grad_gp"], device=device, dtype=dtype)
    flux_ref = torch.tensor(ref["flux_gp"], device=device, dtype=dtype)
    qmag_ref = torch.tensor(ref["qmag_gp"], device=device, dtype=dtype)
    g_nodes, phi_nodes = ht.make_envelope_nodes(
        ref["xs"],
        ref["ys"],
        ref["T_left"],
        ref["T_right"],
        device=device,
        dtype=dtype,
    )
    energy_scale = max(abs(float(ref["internal_energy"])), 1e-12)

    model = GAMFNO2D(
        node_in_dim=c_nodes.shape[-1],
        out_dim=1,
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
        output_scale=[temp_scale],
        output_shift=[0.0],
    ).to(device=device)

    def predict():
        raw = model(coords, c_nodes, bc_mode="none")
        return ht.apply_envelope(raw, g_nodes, phi_nodes)

    def raw_energy(temp):
        return heat_q4_energy_2d(
            temp,
            elements,
            shape_vals,
            grad_mats,
            det_j,
            weights,
            k=k,
            source=source,
            thickness=thickness,
        )

    def energy(temp):
        return raw_energy(temp) / energy_scale

    def metrics(temp):
        t0 = temp[0] if temp.ndim == 3 else temp
        fields = ht.element_gauss_fields_torch(t0, elements, shape_vals, grad_mats, k=k, source=source)
        energy_value = raw_energy(t0)
        return {
            "rel_u": ht.rel_l2(t0, t_ref),
            "rel_T": ht.rel_l2(t0, t_ref),
            "rel_grad": ht.rel_l2(fields["grad_gp"][0], grad_ref),
            "rel_flux": ht.rel_l2(fields["flux_gp"][0], flux_ref),
            "rel_qmag": ht.rel_l2(fields["qmag_gp"][0], qmag_ref),
            "energy_gap": (energy_value - energy_scale) / energy_scale,
            "residual": ht.residual_norm_from_energy(raw_energy, t0, free_nodes),
            "bc_error": ht.bc_error(t0, fixed_nodes, fixed_vals),
        }

    n_iter = int(args.max_iter if args.max_iter is not None else train_cfg["n_iter"])
    print(f"heat-transfer/2d | grid={ny}x{nx} | iter={n_iter} | params={count_trainable_params(model):,}")
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
        t_pred = predict()
    result["problem"] = "heat-transfer/2d/panel"
    result["grid"] = [ny, nx]
    result["case"] = case
    result["reference_internal_energy"] = float(ref["internal_energy"])

    output_dir = ensure_dir(ROOT / cfg["output_dir"])
    save_json(result, output_dir / "metrics.json")
    if not args.no_plots:
        plot_panel_heat_2d(
            ref["xs"],
            ref["ys"],
            t_ref,
            t_pred,
            output_dir / "temperature.png",
            title="2D Heat Transfer Panel",
        )

    m = result["metrics"]
    print(f"saved={output_dir}")
    print(f"final | rel_T={100*m['rel_T']:.3f}% | rel_flux={100*m['rel_flux']:.3f}% | res={m['residual']:.2e}")


if __name__ == "__main__":
    main()
