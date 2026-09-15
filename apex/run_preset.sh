#!/bin/bash
# APEX: run a local preset on stage-2 pairs, one after the other. Output: apex/out/<subject>/stage2_<tag>_<preset>/P000
# (Moved.mha, Transform.h5, DVF.ome.zarr).
#   bash apex/run_preset.sh PRESET DEVICE "subject_v mirror" "subject_m nomirror_slidenorm_pad64" ...
#   DEVICE is "--cpu 1" or "--gpu 0"
set -o pipefail
cd /home/valentin/Documents/ImpactReg_OOC_Demo
V=/home/valentin/Documents/lib/venv-impactreg/bin; export PATH=$V:$PATH
export KONFAI_IMPACTREG_REPO=$PWD/presets
preset=$1; read -ra device <<< "$2"; shift 2
for job in "$@"; do
  read -r subject tag <<< "$job"
  pair=apex/out/$subject/pair_$tag; out=apex/out/$subject/stage2_${tag}_$preset; log=apex/out/$subject/run_stage2_${tag}_$preset.log
  [ -f "$pair/Fixed.mha" ] || { echo "== $subject $tag $preset: no pair at $pair"; continue; }
  echo "== $subject $tag $preset ${device[*]}"
  rm -rf "$out"
  /usr/bin/time -f "wall %e s, peak RSS %M kB" systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=0 --quiet \
    impact-reg-konfai register "$preset" -f "$pair/Fixed.mha" -m "$pair/Moving.mha" \
    --fixed-mask "$pair/FixedMask.mha" --moving-mask "$pair/MovingMask.mha" -o "$out" "${device[@]}" > "$log" 2>&1
  tr '\r' '\n' < "$log" | grep -E "^wall|elastix failed|Description|Error:|OutOfMemory" | grep -v Warning | tail -3 | cut -c1-220
  # the displacement field as an NGFF RFC-5 OME-Zarr beside the transform, checked by reading it back
  [ -f "$out/P000/Transform.h5" ] && python apex/dvf_to_rfc5.py "$out/P000/Transform.h5" "$pair/Fixed.mha"
  ls "$out/P000" 2>&1 | tr '\n' ' '; echo
done
