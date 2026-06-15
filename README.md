# GNOS

Public-demo repo for GPO+ neural operator experiments.

Phase 1 starts with 1D truss problems from the original Colab:

- `problems/linear-elasticity/1d/truss.py`
- `problems/nonlinear-elasticity/1d/truss.py`
- `problems/heat-transfer/1d/conduction.py`

The model is the downloaded `model-1d-v2.0.1.py`, copied into `models/gnos.py`.

## Quick Start

```bash
python -m pip install -e .
bash scripts/run_1d_linear_elasticity.sh
bash scripts/run_1d_nonlinear_elasticity.sh
bash scripts/run_1d_heat_transfer.sh
```

For a tiny smoke run:

```bash
python problems/linear-elasticity/1d/truss.py --max-iter 2 --no-plots
python problems/nonlinear-elasticity/1d/truss.py --max-iter 2 --laws convex --no-plots
python problems/heat-transfer/1d/conduction.py --max-iter 2 --no-plots
```

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
