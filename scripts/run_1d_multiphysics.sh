#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python problems/multi-physics/1d/truss.py --config problems/multi-physics/1d/config.yaml "$@"

