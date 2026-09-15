#!/bin/bash
# APEX, stage 2: the written preset APEX_FA_RET_BSPLINE on each pair, one run after the other on the CPU.
#   bash apex/stage2_run.sh "subject_v mirror" "subject_v nomirror" "subject_m nomirror"
set -o pipefail
cd /home/valentin/Documents/ImpactReg_OOC_Demo
V=/home/valentin/Documents/lib/venv-impactreg/bin; export PATH=$V:$PATH
export KONFAI_IMPACTREG_REPO=$PWD/presets
for job in "$@"; do
  read -r subject tag <<< "$job"
  pair=apex/out/$subject/pair_$tag; out=apex/out/$subject/stage2_$tag; log=apex/out/$subject/run_stage2_$tag.log
  [ -f "$pair/Fixed.mha" ] || { echo "== $subject $tag: no pair at $pair"; continue; }
  echo "== $subject $tag"
  rm -rf "$out"
  /usr/bin/time -f "wall %e s, peak RSS %M kB" systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=0 --quiet \
    impact-reg-konfai register APEX_FA_RET_BSPLINE -f "$pair/Fixed.mha" -m "$pair/Moving.mha" \
    --fixed-mask "$pair/FixedMask.mha" --moving-mask "$pair/MovingMask.mha" -o "$out" --cpu 1 > "$log" 2>&1
  tr '\r' '\n' < "$log" | grep -E "^wall|elastix failed|Description" | tail -2 | cut -c1-200
  ls "$out/P000" 2>&1 | tr '\n' ' '; echo
done
