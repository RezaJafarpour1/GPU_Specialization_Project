#!/usr/bin/env bash
set -euo pipefail
mkdir -p build
make -j
ts=$(date +"%Y%m%d_%H%M%S")
out="outputs/run_${ts}"
mkdir -p "$out"
./gpu_pipeline --input_dir data --output_dir "$out" --ops invert
echo "Outputs in $out"
