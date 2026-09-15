#!/bin/bash
# The native run again, with a tile grid that fits the 768^3 window: stride 128, so the last tile ends on
# the face instead of being a 93-voxel sliver. Same pair, same masks, same settings otherwise.
set -eo pipefail
cd /home/valentin/Documents/ImpactReg_OOC_Demo
V=/home/valentin/Documents/lib/venv-impactreg/bin; export PATH=$V:$PATH
R=Predictor.Model.RegistrationNet

echo "== level-0 deformable, tiled, masked, overlap 128 (125 full tiles)"
rm -rf out/linc_native_o128
/usr/bin/time -f "wall %e s, peak RSS %M kB" systemd-run --user --scope -p MemoryMax=60G -p MemorySwapMax=0 --quiet impact-reg-konfai register Generic_Rigid_BSpline \
  -f data/linc/pair0/Fixed.ome.zarr -m data/linc/pair0/Moving.ome.zarr \
  --fixed-mask data/linc/pair0/FixedMask.ome.zarr --moving-mask data/linc/pair0/MovingMask.ome.zarr \
  -o out/linc_native_o128 --gpu 0 \
  --set "$R.parameter_maps=[Parameters_BSpline.txt]" --set "$R.final_grid_spacing=1.0" \
  --set "$R.max_iterations=200" --set "$R.spatial_samples=4096" \
  --set "$R.parameter_overrides=[RequiredRatioOfValidSamples=0.05]" \
  --set "Predictor.Dataset.Patch.patch_size=[256,256,256]" --set "Predictor.Dataset.Patch.overlap=128" \
  --set "Predictor.outputs_dataset.DisplacementField.OutputDataset.patch_combine=Cosinus" > out/run_linc_native_o128.log 2>&1
tr '\r' '\n' < out/run_linc_native_o128.log | grep -E "^wall|Prediction.*100%|held |elastix failed|Description" | tail -4

echo "== score"
systemd-run --user --scope -p MemoryMax=30G -p MemorySwapMax=0 --quiet $V/python evaluate_linc.py native \
  --run out/linc_native_o128/P000 --overlap 128 2>&1 | sed -u '/Warning/d'
