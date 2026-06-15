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
   - Migrate 1D coupled hyperelasticity + heat-transfer cases.
   - Add shared multiphysics validation and plotting only where needed.

4. `codex/1d-plasticity`
   - Migrate 1D plasticity/history-dependent cases.
   - Use the existing `StatefulIncrementalWrapper` or a small external stepping loop.

5. `codex/2d-linear-elasticity`
   - Start 2D migration after 1D cases are reviewable.

6. `codex/3d-linear-elasticity`
   - Start 3D migration after 2D foundations settle.

## PR Rules

- One branch per reviewable migration slice.
- Keep default runs fast-demo friendly.
- Keep full reproduction settings documented in YAML.
- Do not commit heavy outputs, checkpoints, Drive links, or private artifacts.
- Each PR should include smoke commands and the generated metrics path.
