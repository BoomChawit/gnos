#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python problems/linear-elasticity/2d/panel.py --config problems/linear-elasticity/2d/config.yaml "$@"
