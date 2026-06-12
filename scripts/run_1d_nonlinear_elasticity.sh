#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python problems/nonlinear-elasticity/1d/truss.py --config problems/nonlinear-elasticity/1d/config.yaml "$@"

