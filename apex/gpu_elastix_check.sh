#!/bin/bash
# Once the GPU elastix-IMPACT build is installed: link check under impact-reg's own loader path, then one short IMPACT
# registration on the GPU through impact-reg, and proof from its log that the metric ran on CUDA.
#   bash apex/gpu_elastix_check.sh [DEVICE, default "--gpu 0"] [NAME, default gpucheck]
set -o pipefail
cd /home/valentin/Documents/ImpactReg_OOC_Demo
V=/home/valentin/Documents/lib/venv-impactreg/bin; export PATH=$V:$PATH
export KONFAI_ELASTIX_DIR=/home/valentin/Documents/lib/ImpactElastix-install-cu130
export KONFAI_IMPACTREG_REPO=$PWD/presets
$V/python - <<'PY'
import os, subprocess
from pathlib import Path
from impact_reg_konfai.models.elastix_install import get_elastix_bin, loader_env
root = Path(os.environ["KONFAI_ELASTIX_DIR"])
env = loader_env(root)
print(subprocess.run([str(get_elastix_bin(root)), "--version"], env=env, capture_output=True, text=True).stdout.strip())
so = next(root.rglob("libImpactMetric.so"))
ldd = subprocess.run(["ldd", str(so)], env=env, capture_output=True, text=True).stdout
print("ImpactMetric links:", sorted({l.split()[0] for l in ldd.splitlines() if "torch" in l or "c10" in l}), "| missing:", [l.split()[0] for l in ldd.splitlines() if "not found" in l])
PY
read -ra device <<< "${1:---gpu 0}"; name=${2:-gpucheck}
pair=apex/out/subject_v/pair_mirror; out=apex/out/subject_v/${name}_APEX_IMPACT_MIND_STATIC; log=$out.log
rm -rf $out
# GPU processes every 2 s while the registration runs: the elastix binary must appear here on --gpu
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader -l 2 > $out.gpu 2>&1 & smi=$!
/usr/bin/time -f "wall %e s, peak RSS %M kB" systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=0 --quiet \
  impact-reg-konfai register APEX_IMPACT_MIND_STATIC -f $pair/Fixed.mha -m $pair/Moving.mha \
  --fixed-mask $pair/FixedMask.mha --moving-mask $pair/MovingMask.mha -o $out "${device[@]}" > $log 2>&1
kill $smi
echo "GPU processes named elastix during the run:"; grep -i elastix $out.gpu | sort -u -t, -k1,1 | head -3
tr '\r' '\n' < $log | grep -iE "^wall|cuda|ImpactGPU|device|elastix failed|Description" | grep -viE "UserWarning" | sort -u | head -12 | cut -c1-200
tr '\r' '\n' < $log | grep -E "res [0-3] \| metric" | awk '{print $0}' | tail -1 | cut -c1-120
ls $out/P000 2>&1 | tr '\n' ' '; echo
