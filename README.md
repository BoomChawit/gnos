# GNOS

Public-demo repo for GPO+ neural operator experiments.

Phase 1 starts with 1D truss problems from the original Colab:

- `problems/linear-elasticity/1d/truss.py`
- `problems/nonlinear-elasticity/1d/truss.py`
- `problems/heat-transfer/1d/conduction.py`
- `problems/multi-physics/1d/truss.py`
- `problems/linear-elasticity/2d/panel.py`
- `problems/nonlinear-elasticity/2d/panel.py`
- `problems/heat-transfer/2d/panel.py`
- `problems/multi-physics/2d/panel.py`

The models are the downloaded `model-1d-v2.0.1.py` and `model-2d-v2.0.1.py`, copied into `models/`.

## Quick Start

```bash
python -m pip install -e .
bash scripts/run_1d_linear_elasticity.sh
bash scripts/run_1d_nonlinear_elasticity.sh
bash scripts/run_1d_heat_transfer.sh
bash scripts/run_1d_multiphysics.sh
bash scripts/run_2d_linear_elasticity.sh
bash scripts/run_2d_nonlinear_elasticity.sh
bash scripts/run_2d_heat_transfer.sh
bash scripts/run_2d_multiphysics.sh
```

For a tiny smoke run:

```bash
python problems/linear-elasticity/1d/truss.py --max-iter 2 --no-plots
python problems/nonlinear-elasticity/1d/truss.py --max-iter 2 --laws convex --no-plots
python problems/heat-transfer/1d/conduction.py --max-iter 2 --no-plots
python problems/multi-physics/1d/truss.py --max-iter 2 --no-plots
python problems/linear-elasticity/2d/panel.py --max-iter 2 --no-plots
python problems/nonlinear-elasticity/2d/panel.py --max-iter 2 --no-plots
python problems/heat-transfer/2d/panel.py --max-iter 2 --no-plots
python problems/multi-physics/2d/panel.py --max-iter 2 --no-plots
```

## 2D Mesh Refinement: GNOS vs FEM

Convergence/speed study for the four 2D panels across mesh resolutions. For each
problem and mesh we report the **speedup to reach a target accuracy**:

```
speedup@a% = (FEM reference solve time) / (GNOS time to first reach a% relative-L2 error vs that FEM solution)
```

Hardware: GNOS trained on **NVIDIA GeForce RTX 4090 (24 GB)**; FEM reference solved on
**Intel Core i9-13900KF (24C/32T)** CPU. Software: torch 2.12 (CUDA 13.0), scipy 1.16.
Each run is 3000 iterations of the repo's default 2D model (~2.7k params). FEM references
use a sparse direct solve (a drop-in for the repo's dense reference, identical results to
machine precision) so the elasticity problems reach 201² / 401².

| Problem | Mesh | FEM solve | →10% | →5% | →2.5% | final err |
|---|---|--:|--:|--:|--:|--:|
| Heat | 101² | 0.06 s | 0.10× | 0.08× | 0.04× | 0.07% |
| Heat | 201² | 0.40 s | 0.26× | 0.19× | 0.09× | 0.06% |
| Heat | 401² | 3.0 s | 0.60× | 0.42× | 0.19× | 0.06% |
| Linear elasticity | 101² | 0.43 s | 0.14× | – | – | 5.28% |
| Linear elasticity | 201² | 2.9 s | 0.26× | – | – | 5.56% |
| Linear elasticity | 401² | 23 s | 0.54× | – | – | 5.65% |
| Nonlinear elasticity | 101² | 37 s | 25× | 9.4× | – | 2.58% |
| Nonlinear elasticity | 201² | 268 s | 58× | 19× | – | 2.76% |
| Nonlinear elasticity | 401² | 2413 s | 140× | 39× | – | 2.78% |
| Multiphysics | 101² | 39 s | 50× | 8.9× | – | 2.73% |
| Multiphysics | 201² | 339 s | 155× | 22× | – | 2.87% |
| Multiphysics | 401² | 2191 s | 281× | 33× | – | 2.93% |

`→a%` cells are `speedup@a%` (FEM solve time ÷ GNOS time-to-reach). `–` = target not
reached within 3000 iters. 1% is never reached at these resolutions (work in progress).

- **Speedup grows with mesh.** For every problem the speedup-to-target rises monotonically
  with resolution: the FEM solve cost grows steeply (dense/Newton factorizations) while
  GNOS time-to-a-fixed-accuracy grows far more slowly.
- **Largest gains where FEM is expensive.** Nonlinear/multiphysics references need
  Newton + load-stepping (tens of factorizations), so GNOS reaches 10% error **25–280×**
  faster and 5% **9–39×** faster, increasing with mesh.
- **Heat/linear FEM is cheap** (one solve), so single-instance speedup is `<1×` but still
  climbs with mesh (heat 0.10×→0.60× at 10%).
- Heat converges to ~0.06%; the elasticity/nonlinear/multiphysics demos plateau at
  ~2.6–5.6% with the default small model, so 2.5% is reached only by heat and 1% by none.

## Layout

```text
models/      GNOS model and backbones
physics/     Energy, boundary constraints, spatial gradients
src/solver/  Shared training/config/metric helpers
src/validation/fem/  Reference truss solvers
src/utils/   IO and visualization helpers
problems/    Demo entrypoints grouped by physics and dimension
scripts/     Thin run scripts
```
