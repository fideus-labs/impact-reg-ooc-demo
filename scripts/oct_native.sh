#!/bin/bash
# Stage 3 of the OCT / light-sheet section through a given stage-2 run: the native window re-read through its field
# (the light-sheet box is cached, so only the resampling runs again), the 3 um plane pair, the 25 tiles, the flicker.
#   scripts/oct_native.sh RUN[,RUN2...] TAG   e.g. scripts/oct_native.sh plane_mind2dR2D2_mi_b0.1,plane_pass2_g0.2 pass2
#   several runs are a chain of stage-2 passes, in run order
set -o pipefail
cd /home/valentin/Documents/ImpactReg_OOC_Demo
V=/home/valentin/Documents/lib/venv-impactreg/bin; export PATH=$V:$PATH
R=Predictor.Model.RegistrationNet
run=$1; tag=$2
fields=(); IFS=, read -ra chain <<< "$run"; for r in "${chain[@]}"; do fields+=(out/$r/P000/Transform.h5); done
cap=(systemd-run --user --scope -p MemoryMax=24G -p MemorySwapMax=0 --quiet)
"${cap[@]}" $V/python -u stage2_native.py --x 23 --y 11 --size 2048 --depth 64 --moving-store NeuN --margin 4.0 \
  --skip-fixed --coarse-field "${fields[@]}" --out data/dandi/cortex_$tag 2>&1 | sed -u '/Warning/d' || exit 1
"${cap[@]}" $V/python -u stage2_plane.py --spacing 0.003 --window 23 11 6.1 \
  --moving-window data/dandi/cortex_$tag --out data/dandi/plane_native_$tag 2>&1 | sed -u '/Warning/d' || exit 1
rm -rf out/native_tiled_$tag
/usr/bin/time -f "wall %e s" systemd-run --user --scope -p MemoryMax=30G -p MemorySwapMax=0 --quiet \
  impact-reg-konfai register Generic_Rigid_BSpline \
  -f data/dandi/plane_native_$tag/Fixed.ome.zarr -m data/dandi/plane_native_$tag/Moving.ome.zarr \
  -o out/native_tiled_$tag --gpu 0 \
  --set "$R.parameter_maps=[Parameters_BSpline.txt]" --set "$R.final_grid_spacing=0.3" \
  --set "$R.parameter_overrides=[FinalGridSpacingInPhysicalUnits=0.3 0.3 1000, RequiredRatioOfValidSamples=0.05]" \
  --set "Predictor.Dataset.Patch.patch_size=[8, 512, 512]" --set "Predictor.Dataset.Patch.overlap=12%" \
  --set "Predictor.outputs_dataset.DisplacementField.OutputDataset.patch_combine=Cosinus" > out/run_native_tiled_$tag.log 2>&1
tr '\r' '\n' < out/run_native_tiled_$tag.log | grep -E "^wall|elastix failed|Error" | tail -2
"${cap[@]}" $V/python - "$tag" <<'PY' 2>&1 | sed -u '/Warning/d'
import sys
from figures import ROOT, read_zarr
from stage1_linc import mutual_information
tag = sys.argv[1]; d = ROOT / "data" / "dandi"
fixed = read_zarr(d / f"plane_native_{tag}" / "Fixed.ome.zarr")[0]
for name, path in (("old window, before the tiles", d / "plane_native" / "Moving.ome.zarr"),
                   ("old window, after the tiles", ROOT / "out" / "native_tiled" / "P000" / "Moved.ome.zarr"),
                   (f"{tag} window, before the tiles", d / f"plane_native_{tag}" / "Moving.ome.zarr"),
                   (f"{tag} window, after the tiles", ROOT / "out" / f"native_tiled_{tag}" / "P000" / "Moved.ome.zarr")):
    print(f"{name:32s} MI {mutual_information(fixed, read_zarr(path)[0]):.3f}")
PY
