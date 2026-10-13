#!/bin/bash
# Download every HDF5 chunk of one run from the public CERNBox share; skips chunks already present.
# Usage: scripts/fetch_h5.sh <run, e.g. 2026-10-09_07-46-07> <out dir>
set -u
r=$1; out=$2/acquisition_$r
B=https://cernbox.cern.ch/remote.php/dav/public-files/yVioydShmcGKkto/measured_data/acquisition_$r/h5
mkdir -p "$out"
for c in $(curl -s -X PROPFIND -H "Depth: 1" "$B/" | grep -o 'chunk_[0-9]*\.h5' | sort -u); do
  [ -s "$out/$c" ] || { curl -s -f -m 1800 -o "$out/$c.part" "$B/$c" && mv "$out/$c.part" "$out/$c"; }
  echo "$(date +%T) $c $(stat -c %s "$out/$c" 2>/dev/null || echo FAILED)"
done
