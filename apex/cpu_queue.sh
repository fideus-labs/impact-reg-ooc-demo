#!/bin/bash
# APEX CPU queue: wait for the running MI-variant queue, then the IMPACT presets built from the ImpactLoss examples.
# Static presets run in full; each Jacobian preset first gets a 120 s probe whose iteration rate decides whether it
# is worth a full run (the elastix-IMPACT build here is CPU-only).
cd /home/valentin/Documents/ImpactReg_OOC_Demo
wait_pid=$(pgrep -of "run_preset[.]sh APEX_ANATOMIX_MI")
[ -n "$wait_pid" ] && { echo "waiting on the running CPU queue (pid $wait_pid)"; while kill -0 "$wait_pid" 2>/dev/null; do sleep 20; done; }
bash apex/run_preset.sh APEX_IMPACT_MIND_STATIC "--cpu 1" "subject_v mirror" "subject_m nomirror_slidenorm"
bash apex/run_preset.sh APEX_IMPACT_ANATOMIX_STATIC "--cpu 1" "subject_v mirror_pad64" "subject_m nomirror_slidenorm_pad64"
bash apex/run_preset.sh APEX_MIND3D_MI "--cpu 1" "subject_v mirror" "subject_m nomirror_slidenorm"
V=/home/valentin/Documents/lib/venv-impactreg/bin; export PATH=$V:$PATH; export KONFAI_IMPACTREG_REPO=$PWD/presets
for preset in APEX_IMPACT_MIND_JACOBIAN APEX_IMPACT_TS_JACOBIAN APEX_IMPACT_SAM_JACOBIAN; do
  pair=apex/out/subject_v/pair_mirror; log=apex/out/subject_v/probe_$preset.log
  echo "== probe $preset, 120 s"
  timeout 120 systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=0 --quiet impact-reg-konfai register $preset \
    -f $pair/Fixed.mha -m $pair/Moving.mha --fixed-mask $pair/FixedMask.mha --moving-mask $pair/MovingMask.mha \
    -o apex/out/subject_v/probe_$preset --cpu 1 > "$log" 2>&1
  pkill -f "probe_$preset" 2>/dev/null
  tr '\r' '\n' < "$log" | grep -E "res [0-3] \| metric|elastix failed|Description" | tail -1 | cut -c1-160
done
