#!/bin/bash
# Stage 2 of the OCT / light-sheet section, second sweep: light penalties, more samples, an affine refit,
# and two passes. One GPU job at a time; each run is scored and drawn (P3_crops_<run>.png).
#   scripts/oct_sweep.sh CONFIG...
#   CONFIG = [m_]g<grid>[_b<bend>][_it<iters>][_s<samples>][_from<run>][_aff]
#     m_       masked (Otsu tissue mask)        _from<run>  start from that run's Moved image (a second pass)
#     _aff     an elastix affine before the B-spline, in the same run
set -o pipefail
cd /home/valentin/Documents/ImpactReg_OOC_Demo
V=/home/valentin/Documents/lib/venv-impactreg/bin; export PATH=$V:$PATH
R=Predictor.Model.RegistrationNet
base=data/dandi/plane
[ -d $base/FixedMask.ome.zarr ] || systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 --quiet $V/python scripts/oct_masks.py 2>&1 | sed -u '/Warning/d'
runs=()
for config in "$@"; do
  grid=${config#*g}; grid=${grid%%_*}
  bend=0; [[ $config =~ _b([0-9.]+) ]] && bend=${BASH_REMATCH[1]}
  iters=1500; [[ $config =~ _it([0-9]+) ]] && iters=${BASH_REMATCH[1]}
  samples=16384; [[ $config =~ _s([0-9]+) ]] && samples=${BASH_REMATCH[1]}
  from=""; [[ $config =~ _from(.+)$ ]] && from=${BASH_REMATCH[1]}
  maps="Parameters_BSpline.txt"; [[ $config == *_aff* ]] && maps="Parameters_Rigid.txt, Parameters_BSpline.txt"
  pair=$base
  if [ -n "$from" ]; then
    pair=data/dandi/plane_from_$from
    mkdir -p $pair && ln -sfn "$PWD/$base/Fixed.ome.zarr" $pair/Fixed.ome.zarr
    ln -sfn "$PWD/$base/FixedMask.ome.zarr" $pair/FixedMask.ome.zarr && ln -sfn "$PWD/$base/MovingMask.ome.zarr" $pair/MovingMask.ome.zarr
    ln -sfn "$PWD/out/$from/P000/Moved.ome.zarr" $pair/Moving.ome.zarr
  fi
  overrides="FinalGridSpacingInPhysicalUnits=$grid $grid 1000, RequiredRatioOfValidSamples=0.05"
  [[ $config == *_aff* ]] && overrides="Transform=\"AffineTransform\", $overrides"
  [ "$bend" != 0 ] && overrides="Registration=\"MultiMetricMultiResolutionRegistration\", Metric=\"AdvancedMattesMutualInformation\" \"TransformBendingEnergyPenalty\", Metric0Weight=1.0, Metric1Weight=$bend, $overrides"
  masks=(); [[ $config == m_* ]] && masks=(--fixed-mask $pair/FixedMask.ome.zarr --moving-mask $pair/MovingMask.ome.zarr)
  name=plane_$config
  echo "== $name: grid $grid, bending $bend, $iters it, $samples samples, maps [$maps], from '${from:-affine}', ${#masks[@]} mask args"
  rm -rf out/$name
  /usr/bin/time -f "wall %e s" systemd-run --user --scope -p MemoryMax=30G -p MemorySwapMax=0 --quiet \
    impact-reg-konfai register Generic_Rigid_BSpline \
    -f $pair/Fixed.ome.zarr -m $pair/Moving.ome.zarr "${masks[@]}" -o out/$name --gpu 0 \
    --set "$R.parameter_maps=[$maps]" --set "$R.final_grid_spacing=$grid" \
    --set "$R.max_iterations=$iters" --set "$R.spatial_samples=$samples" \
    --set "$R.parameter_overrides=[$overrides]" > out/run_$name.log 2>&1
  tr '\r' '\n' < out/run_$name.log | grep -E "^wall|elastix failed|Description|Error" | tail -2
  [ -d out/$name/P000/Moved.ome.zarr ] || continue
  runs+=("$name")
  systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 --quiet $V/python figures_pair.py --run $name --stem _$config 2>&1 | sed -u '/Warning/d' | grep P3
done
systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 --quiet $V/python scripts/oct_score.py plane_g0.4 "${runs[@]}" 2>&1 | sed -u '/Warning/d'
