#!/bin/bash
# The per-structure check: FLASH T2* through the same stages, then its segmentation against the consortium's.
set -eo pipefail
cd /home/valentin/Documents/ImpactReg_OOC_Demo
V=/home/valentin/Documents/lib/venv-impactreg/bin; export PATH=$V:$PATH
cap() { systemd-run --user --scope -p MemoryMax=$1 -p MemorySwapMax=0 --quiet "${@:2}"; }
R=Predictor.Model.RegistrationNet

echo "== stage 1: FLASH T2* at 0.24 mm, brain only, 24 orientations then the affine, against XPCT level 4"
cap 40G $V/python -u stage1_linc.py --level 4 --moving data/linc/flash_brain_024.nii.gz --name flash 2>&1 \
  | sed -u '/Warning/d' | tee out/stage1_flash.log

echo "== the coarse pair"
cap 20G $V/python evaluate_linc.py pair --name flash 2>&1 | sed -u '/Warning/d'

echo "== stage 2: the deformable, whole hemisphere at 0.32 mm, same settings as the dMRI"
rm -rf out/linc_flash
/usr/bin/time -f "wall %e s, peak RSS %M kB" systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=0 --quiet \
  impact-reg-konfai register Generic_Rigid_BSpline \
  -f data/linc/pair_flash/Fixed.ome.zarr -m data/linc/pair_flash/Moving.ome.zarr -o out/linc_flash --gpu 0 \
  --set "$R.parameter_maps=[Parameters_BSpline.txt]" --set "$R.final_grid_spacing=3.0" \
  --set "$R.max_iterations=600" --set "$R.spatial_samples=8192" > out/run_linc_flash.log 2>&1
tr '\r' '\n' < out/run_linc_flash.log | grep -E "^wall|Prediction.*100%" | tail -2

echo "== Dice per structure, against the consortium's segmentation in XPCT space"
cap 30G $V/python evaluate_linc.py dice --name flash --field out/linc_flash/P000/Transform.h5 2>&1 \
  | sed -u '/Warning/d' | tee out/dice_flash.txt
