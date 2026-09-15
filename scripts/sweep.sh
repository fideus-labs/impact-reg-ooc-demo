#!/bin/bash
# One level-3 tile, one configuration change at a time:  sweep.sh TILE Z Y X CONFIG...
cd /home/valentin/Documents/ImpactReg_OOC_Demo
V=/home/valentin/Documents/lib/venv-impactreg/bin; export PATH=$V:$PATH
S=/home/valentin/Documents/ImpactReg_OOC_Demo/scripts
tile=$1 z=$2 y=$3 x=$4; shift 4
T=data/linc/tile3_$tile R=Predictor.Model.RegistrationNet
runs=()
for c in "$@"; do
  grid=2.0; extra=()
  case $c in
    baseline) ;;
    grid4) grid=4.0 ;;
    mask*) extra=(--fixed-mask $T/FixedMask.ome.zarr --moving-mask $T/MovingMask.ome.zarr) ;;&
    *bend*) extra+=(--set "$R.parameter_overrides=[Registration=\"MultiMetricMultiResolutionRegistration\", Metric=\"AdvancedMattesMutualInformation\" \"TransformBendingEnergyPenalty\", Metric0Weight=1.0, Metric1Weight=${c##*bend}]") ;;
    maskratio) extra+=(--set "$R.parameter_overrides=[RequiredRatioOfValidSamples=0.05]") ;;
    masksparse) extra+=(--set "$R.parameter_overrides=[ImageSampler=\"RandomSparseMask\", RequiredRatioOfValidSamples=0.05]") ;;
    mask) ;;
    *) echo "unknown config $c"; exit 1 ;;
  esac
  out=out/tile3_${tile}_$c; rm -rf $out; runs+=($out)
  /usr/bin/time -f "$c: %e s wall, peak RSS %M kB" -o $out.time systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=0 --quiet \
    $V/impact-reg-konfai register Generic_Rigid_BSpline -f $T/Fixed.ome.zarr -m $T/Moving.ome.zarr -o $out --gpu 0 \
    --set "$R.parameter_maps=[Parameters_BSpline.txt]" --set "$R.final_grid_spacing=$grid" --set "$R.max_iterations=200" \
    --set "$R.spatial_samples=4096" "${extra[@]}" > $out.log 2>&1
  cat $out.time; tr '\r' '\n' < $out.log | grep -E "elastix failed|Too many|Traceback|matched no entry|Error" | head -3
done
systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0 --quiet $V/python $S/tile3.py score $tile $z $y $x "${runs[@]}" 2>&1 | grep -v Warning
