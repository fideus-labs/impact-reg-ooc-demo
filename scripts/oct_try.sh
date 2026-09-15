#!/bin/bash
# Stage 2 of the OCT / light-sheet section, variants: masked, with a bending penalty, finer grid, two passes.
#   scripts/oct_try.sh CONFIG...      CONFIG: m_g0.4_b10 | m_g0.2_b10 | m_g0.2 | m_g0.3_b30
set -o pipefail
cd /home/valentin/Documents/ImpactReg_OOC_Demo
V=/home/valentin/Documents/lib/venv-impactreg/bin; export PATH=$V:$PATH
R=Predictor.Model.RegistrationNet
pair=data/dandi/plane
[ -d $pair/FixedMask.ome.zarr ] || systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 --quiet $V/python scripts/oct_masks.py 2>&1 | sed -u '/Warning/d'
runs=()
for config in "$@"; do
  grid=${config#*_g}; grid=${grid%%_*}
  bend=${config##*_b}; [ "$bend" = "$config" ] && bend=0
  overrides="FinalGridSpacingInPhysicalUnits=$grid $grid 1000, RequiredRatioOfValidSamples=0.05"
  [ "$bend" != 0 ] && overrides="Registration=\"MultiMetricMultiResolutionRegistration\", Metric=\"AdvancedMattesMutualInformation\" \"TransformBendingEnergyPenalty\", Metric0Weight=1.0, Metric1Weight=$bend, $overrides"
  name=plane_$config
  echo "== $name: masked, grid $grid mm in plane, bending $bend, 1200 it, 8192 samples"
  rm -rf out/$name
  /usr/bin/time -f "wall %e s" systemd-run --user --scope -p MemoryMax=30G -p MemorySwapMax=0 --quiet \
    impact-reg-konfai register Generic_Rigid_BSpline \
    -f $pair/Fixed.ome.zarr -m $pair/Moving.ome.zarr \
    --fixed-mask $pair/FixedMask.ome.zarr --moving-mask $pair/MovingMask.ome.zarr \
    -o out/$name --gpu 0 \
    --set "$R.parameter_maps=[Parameters_BSpline.txt]" --set "$R.final_grid_spacing=$grid" \
    --set "$R.max_iterations=1200" --set "$R.spatial_samples=8192" \
    --set "$R.parameter_overrides=[$overrides]" > out/run_$name.log 2>&1
  tr '\r' '\n' < out/run_$name.log | grep -E "^wall|elastix failed|Description|Traceback" | tail -2
  [ -f out/$name/P000/Moved.ome.zarr/.zattrs ] || [ -d out/$name/P000/Moved.ome.zarr ] && runs+=("$name")
done
systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 --quiet $V/python scripts/oct_score.py plane_g0.4 "${runs[@]}" 2>&1 | sed -u '/Warning/d'
