# Migration Tasks

Goal: migrate GPO+ from Colab/Drive into a public-demo GitHub repo with small, reviewable stacked PRs.

## Stack

1. `codex/1d-elasticity-scaffold`
   - Repo scaffold, GNOS 1D model, physics utilities, validation helpers.
   - 1D linear-elastic truss demo.
   - 1D nonlinear-elastic truss demo for convex and concave laws.

2. `codex/1d-heat-transfer`
   - Migrate the main 1D multilayer heat-transfer case from the source notebook.
   - Reuse `physics/energy.py`, `physics/boundary.py`, and 1D spatial gradients.
   - Leave physical-coordinate/resistance-coordinate ablations for a later optional PR.

3. `codex/1d-multiphysics`
   - Migrate the main 1D thermo-nonlinear truss case coupled to the heat-transfer field.
   - Leave multiphysics ablations for a later optional PR.

4. `codex/1d-plasticity`
   - Migrate 1D plasticity/history-dependent cases.
   - Use the existing `StatefulIncrementalWrapper` or a small external stepping loop.

5. `codex/2d-linear-elasticity`
   - Migrate the main 2D rectangular panel linear-elasticity case.
   - Use the downloaded 2D GAM-FNO model and FE-consistent Q4 energy.

6. `codex/3d-linear-elasticity`
   - Start 3D migration after 2D foundations settle.

Next 2D slices:
- `codex/2d-nonlinear-elasticity`
  - Migrate the convex-hardening rectangular panel with FE-consistent Q4 energy.
- `codex/2d-heat-transfer`
  - Migrate the FE-consistent 2D heat-transfer panel.
- `codex/2d-multiphysics`
  - Migrate the one-way thermo-nonlinear panel.

## PR Rules

- One branch per reviewable migration slice.
- Keep default runs fast-demo friendly.
- Keep full reproduction settings documented in YAML.
- Do not commit heavy outputs, checkpoints, Drive links, or private artifacts.
- Each PR should include smoke commands and the generated metrics path.
