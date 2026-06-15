#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python problems/linear-elasticity/1d/truss.py --config problems/linear-elasticity/1d/config.yaml "$@"

