#!/bin/bash
# Reduce one downloaded run to tables, then make the run overview plot and the forecast scores.
# Usage: scripts/run_pipeline.sh <run> <h5 dir> <tables dir> <results dir>
#   <h5 dir> is the folder given to fetch_h5.sh; set PY to choose the interpreter (default python3).
set -eu
r=$1; src=$2/acquisition_$r; T=$3; out=$4/$r; PY=${PY:-python3}; S=$(dirname "$0")
mkdir -p "$T" "$out"
$PY -I $S/extract_rf.py "$src" "$T/${r}_rf.npz" > /dev/null
$PY -I $S/extract_laser.py "$src" "$T/${r}_laser.npz" > /dev/null
$PY -I $S/reduce_chunks.py "$src" "$T/$r.parquet" | tail -1
$PY -I $S/plot_run.py "$T" "$r" "$out/run_overview.png" > "$out/run_summary.txt"
$PY -I $S/forecast_run.py "$T" "$r" "$out/next_shot.png" "$out/residuals_stable.png" > "$out/forecast.txt"
cat "$out/forecast.txt"
$PY -I $S/hybrid_forecast.py "$T" "$r" "$out/hybrid_next_shot.png" > "$out/hybrid.txt"
sed -n '1,12p' "$out/hybrid.txt"
