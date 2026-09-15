#!/bin/bash
# APEX CPU queue 2: after cpu_queue.sh, rerun the IMPACT MIND Static preset now that its map writes the composite transform.
cd /home/valentin/Documents/ImpactReg_OOC_Demo
wait_pid=$(pgrep -of "cpu_queue[.]sh")
[ -n "$wait_pid" ] && { echo "waiting on cpu_queue.sh (pid $wait_pid)"; while kill -0 "$wait_pid" 2>/dev/null; do sleep 20; done; }
bash apex/run_preset.sh APEX_IMPACT_MIND_STATIC "--cpu 1" "subject_v mirror" "subject_m nomirror_slidenorm"
