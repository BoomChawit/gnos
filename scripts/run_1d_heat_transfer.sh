#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python problems/heat-transfer/1d/conduction.py --config problems/heat-transfer/1d/config.yaml "$@"

