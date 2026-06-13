#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python problems/nonlinear-elasticity/2d/panel.py --config problems/nonlinear-elasticity/2d/config.yaml "$@"
