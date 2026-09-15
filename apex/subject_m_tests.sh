#!/bin/bash
# subject_m, two tests of why stage 2 did not help: (a) the mirror, (b) the slide-to-slide intensity steps of the
# retardance deck, removed by scaling every slide to the deck's median inside the tissue.
set -o pipefail
cd /home/valentin/Documents/ImpactReg_OOC_Demo
V=/home/valentin/Documents/lib/venv-impactreg/bin
cap=(systemd-run --user --scope -p MemoryMax=24G -p MemorySwapMax=0 --quiet)
"${cap[@]}" $V/python - <<'PY' 2>&1 | grep -vE "^WARNING|NiftiImageIO|^\s*$"
import numpy as np, SimpleITK as sitk
img = sitk.ReadImage("apex/data/subject_m/Ret_slide_deck.nii.gz", sitk.sitkFloat32)
a = sitk.GetArrayFromImage(img)                     # z, y, x; slides along y
target = np.median(a[a > 0])
for i in range(a.shape[1]):
    s = a[:, i]; m = s > 0
    if m.sum() > 100:
        a[:, i] = np.where(m, s * (target / np.median(s[m])), 0)
out = sitk.GetImageFromArray(a); out.CopyInformation(img)
sitk.WriteImage(out, "apex/out/subject_m/Ret_slidenorm.nii.gz", useCompression=True)
med = [np.median(a[:, i][a[:, i] > 0]) for i in range(a.shape[1]) if np.sum(a[:, i] > 0) > 100]
print(f"slide medians after scaling: {np.min(med):.2f} to {np.max(med):.2f} (target {target:.2f})")
PY
"${cap[@]}" $V/python apex/stage2_pair.py subject_m mirror 2>&1 | grep -vE "^WARNING|NiftiImageIO|^\s*$"
"${cap[@]}" $V/python apex/stage2_pair.py subject_m nomirror_slidenorm apex/out/subject_m/Ret_slidenorm.nii.gz nomirror 2>&1 | grep -vE "^WARNING|NiftiImageIO|^\s*$"
bash apex/stage2_run.sh "subject_m mirror" "subject_m nomirror_slidenorm"
for tag in mirror nomirror_slidenorm; do
  "${cap[@]}" $V/python apex/score.py subject_m $tag stage2_$tag 2>&1 | grep -vE "^WARNING|NiftiImageIO|^\s*$"
done
