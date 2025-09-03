#!/usr/bin/env bash
set -euo pipefail

N="${1:-200}"
OUT="outputs/bench_$(date +%Y%m%d_%H%M%S)"
mkdir -p data "$OUT"

# Ensure a seed PGM exists: create a 256x256 gradient if missing
if [[ ! -f "data/small.pgm" ]]; then
  echo "Creating data/small.pgm (256x256 gradient)..."
  {
    printf "P5\n256 256\n255\n"
    python3 - <<'PY'
import sys
w=h=256
sys.stdout.buffer.write(bytes(((x+y)&0xff) for y in range(h) for x in range(w)))
PY
  } > data/small.pgm
fi

# Duplicate to N files
echo "Duplicating data/small.pgm to $N files..."
for i in $(seq -f "%03g" 1 "$N"); do
  cp -f data/small.pgm "data/small_${i}.pgm"
done

# Run streamed pipeline
echo "Running pipeline with 3 streams on $N images..."
./gpu_pipeline --input_dir data --output_dir "$OUT" --ops gauss,sobel,histeq --streams 3

echo
echo "Artifacts:"
echo "  $OUT/log.txt"
echo "  $OUT/metrics.csv"
echo "  $OUT/*.pgm"
echo
echo "Tail of log:"
tail -n 10 "$OUT/log.txt" || true
