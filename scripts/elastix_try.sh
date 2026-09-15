#!/bin/bash
# One masked elastix B-spline run on a FLASH pair, then the Dice through the whole chain of fields.
#   elastix_try.sh PAIR_DIR RUN_NAME "PRIOR_FIELDS" CONFIG
#   PRIOR_FIELDS: the fields already applied to PAIR_DIR's moving image, space-separated ("" for none)
#   CONFIG: base | bend10 | bend100 | grid6
set -o pipefail
cd /home/valentin/Documents/ImpactReg_OOC_Demo
V=/home/valentin/Documents/lib/venv-impactreg/bin; export PATH=$V:$PATH
R=Predictor.Model.RegistrationNet
pair=$1 name=$2 prior=$3 config=$4

grid=3.0; overrides="RequiredRatioOfValidSamples=0.05"
case $config in
  base) ;;
  grid6) grid=6.0 ;;
  bend*) overrides="Registration=\"MultiMetricMultiResolutionRegistration\", Metric=\"AdvancedMattesMutualInformation\" \"TransformBendingEnergyPenalty\", Metric0Weight=1.0, Metric1Weight=${config#bend}, RequiredRatioOfValidSamples=0.05" ;;
  *) echo "unknown config $config"; exit 1 ;;
esac

[ -d "$pair/FixedMask.ome.zarr" ] || systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 --quiet $V/python masks_linc.py "$pair" 2>&1 | sed -u '/Warning/d'

echo "== $name: B-spline, grid $grid mm, $config"
rm -rf "out/$name"
/usr/bin/time -f "wall %e s, peak RSS %M kB" systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=0 --quiet \
  impact-reg-konfai register Generic_Rigid_BSpline \
  -f "$pair/Fixed.ome.zarr" -m "$pair/Moving.ome.zarr" \
  --fixed-mask "$pair/FixedMask.ome.zarr" --moving-mask "$pair/MovingMask.ome.zarr" \
  -o "out/$name" --gpu 0 \
  --set "$R.parameter_maps=[Parameters_BSpline.txt]" --set "$R.final_grid_spacing=$grid" \
  --set "$R.max_iterations=600" --set "$R.spatial_samples=8192" \
  --set "$R.parameter_overrides=[$overrides]" > "out/run_$name.log" 2>&1
tr '\r' '\n' < "out/run_$name.log" | grep -E "^wall|elastix failed|Description|Traceback" | tail -3
[ -f "out/$name/P000/Transform.h5" ] || exit 1

# shellcheck disable=SC2086
systemd-run --user --scope -p MemoryMax=30G -p MemorySwapMax=0 --quiet $V/python evaluate_linc.py dice --name flash \
  --field $prior "out/$name/P000/Transform.h5" 2>&1 | sed -u '/Warning/d' | grep -v "^field " | tee "out/dice_$name.txt" | tail -1
