#!/usr/bin/env bash
# Every NIfTI under data/input/<subject>/ as an OME-Zarr store under data/ome-zarr/<subject>/, through the
# ngff-zarr CLI. A store that exists and is newer than its NIfTI is left alone.
#   bash pipeline/convert.sh [INPUT_DIR] [OUTPUT_DIR]      (pixi run convert)
set -euo pipefail
cd "$(dirname "$0")/.."
input=${1:-data/input}; output=${2:-data/ome-zarr}
shopt -s nullglob
files=("$input"/*/*.nii.gz)
[ ${#files[@]} -gt 0 ] || { echo "no NIfTI files under $input/<subject>/"; exit 1; }
for nii in "${files[@]}"; do
  subject=$(basename "$(dirname "$nii")"); stem=$(basename "$nii" .nii.gz)
  store=$output/$subject/$stem.ome.zarr
  if [ -f "$store/zarr.json" ] && [ "$store/zarr.json" -nt "$nii" ]; then echo "== $store: up to date"; continue; fi
  echo "== $nii -> $store"
  mkdir -p "$output/$subject"; rm -rf "$store"
  ngff-zarr -i "$nii" -o "$store" --quiet
done
