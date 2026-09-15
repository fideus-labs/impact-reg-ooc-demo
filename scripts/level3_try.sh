#!/bin/bash
# Level 3, tiled and masked, from the regularised level-4 chain: without and with bending, then scored.
#   level3_try.sh CONFIG...      CONFIG: base | bend100
set -o pipefail
cd /home/valentin/Documents/ImpactReg_OOC_Demo
V=/home/valentin/Documents/lib/venv-impactreg/bin; export PATH=$V:$PATH
S=/home/valentin/Documents/ImpactReg_OOC_Demo/scripts
R=Predictor.Model.RegistrationNet
pair=data/linc/pair3_bend
runs=()
for config in "$@"; do
  overrides="CheckNumberOfSamples=\"false\""
  case $config in
    base) ;;
    bend*) overrides="Registration=\"MultiMetricMultiResolutionRegistration\", Metric=\"AdvancedMattesMutualInformation\" \"TransformBendingEnergyPenalty\", Metric0Weight=1.0, Metric1Weight=${config#bend}, CheckNumberOfSamples=\"false\"" ;;
    *) echo "unknown config $config"; exit 1 ;;
  esac
  name=linc_tiled3_$config
  echo "== $name: level 3, 125 tiles of 256^3, masked, $config"
  rm -rf "out/$name"
  /usr/bin/time -f "wall %e s, peak RSS %M kB" systemd-run --user --scope -p MemoryMax=60G -p MemorySwapMax=0 --quiet \
    impact-reg-konfai register Generic_Rigid_BSpline \
    -f "$pair/Fixed.ome.zarr" -m "$pair/Moving.ome.zarr" \
    --fixed-mask "$pair/FixedMask.ome.zarr" --moving-mask "$pair/MovingMask.ome.zarr" \
    -o "out/$name" --gpu 0 \
    --set "$R.parameter_maps=[Parameters_BSpline.txt]" --set "$R.final_grid_spacing=2.0" \
    --set "$R.max_iterations=200" --set "$R.spatial_samples=4096" \
    --set "$R.parameter_overrides=[$overrides]" \
    --set "Predictor.Dataset.Patch.patch_size=[256,256,256]" --set "Predictor.Dataset.Patch.overlap=12%" \
    --set "Predictor.outputs_dataset.DisplacementField.OutputDataset.patch_combine=Cosinus" > "out/run_$name.log" 2>&1
  tr '\r' '\n' < "out/run_$name.log" | grep -E "^wall|elastix failed|Description|Traceback|held " | tail -3
  [ -f "out/$name/P000/Transform.h5" ] && runs+=("$name")
done
[ ${#runs[@]} -gt 0 ] || { echo "no run finished, nothing to score"; exit 1; }
echo "== scored against the consortium, level 3"
PAIR=pair3_bend systemd-run --user --scope -p MemoryMax=30G -p MemorySwapMax=0 --quiet $V/python $S/diag3.py "${runs[@]}" 2>&1 \
  | sed -u '/Warning/d' | grep -E "^grid|^  [a-z]"
