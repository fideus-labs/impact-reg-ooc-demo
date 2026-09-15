#!/bin/bash
# Stage 3 of the OCT / light-sheet section with different losses on the same 25 tiles (512² at 3 um, overlap 12 %,
# cosine blend, grid 0.3 mm, locked to the plane), on a native pair built by oct_native.sh; each run is scored by
# scripts/oct_native_score.py (the start warped by hand, MI at 3/12/48 um, field size, folds).
#   scripts/oct_native_tiles.sh TAG CONFIG...
#   mi_b<w>        mutual information + bending w (GPU)
#   mind_mi_b<w>   2-D MIND (presets/NATIVE_MIND, 24 -> 3 um) + MI + bending w (CPU build)
set -o pipefail
cd /home/valentin/Documents/ImpactReg_OOC_Demo
V=/home/valentin/Documents/lib/venv-impactreg/bin; export PATH=$V:$PATH
R=Predictor.Model.RegistrationNet
tag=$1; shift
pair=data/dandi/plane_native_$tag
tiles=(--set "Predictor.Dataset.Patch.patch_size=[8, 512, 512]" --set "Predictor.Dataset.Patch.overlap=12%"
       --set "Predictor.outputs_dataset.DisplacementField.OutputDataset.patch_combine=Cosinus")
runs=()
# with a tissue mask (scripts/oct_native_masks.py) the run names end in _m
masks=(); suffix=""
[ -d $pair/FixedMask.ome.zarr ] && masks=(--fixed-mask $pair/FixedMask.ome.zarr --moving-mask $pair/MovingMask.ome.zarr) && suffix=_m
for config in "$@"; do
  w=0; [[ $config =~ _b([0-9.]+) ]] && w=${BASH_REMATCH[1]}
  unset KONFAI_IMPACTREG_REPO
  common="FinalGridSpacingInPhysicalUnits=0.3 0.3 1000, RequiredRatioOfValidSamples=0.05, CheckNumberOfSamples=\"false\""
  case $config in
    mi_b*) preset=Generic_Rigid_BSpline; device=(--gpu 0)
      sets=(--set "$R.parameter_maps=[Parameters_BSpline.txt]" --set "$R.final_grid_spacing=0.3"
            --set "$R.parameter_overrides=[Registration=\"MultiMetricMultiResolutionRegistration\", Metric=\"AdvancedMattesMutualInformation\" \"TransformBendingEnergyPenalty\", Metric0Weight=1.0, Metric1Weight=$w, $common]") ;;
    mind_mi_b*) preset=NATIVE_MIND; device=(--cpu 1); export KONFAI_IMPACTREG_REPO=$PWD/presets
      sets=(--set "$R.parameter_overrides=[Metric0Weight=1.0, Metric1Weight=1.0, Metric2Weight=$w, $common, ImpactUseMixedPrecision=\"false\"]") ;;
    *) echo "unknown config $config"; exit 1 ;;
  esac
  name=native_tiled_${tag}_$config$suffix
  echo "== $name: $preset ${device[*]}"
  rm -rf out/$name
  /usr/bin/time -f "wall %e s" systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=0 --quiet \
    impact-reg-konfai register $preset -f $pair/Fixed.ome.zarr -m $pair/Moving.ome.zarr -o out/$name "${device[@]}" "${masks[@]}" \
    "${sets[@]}" "${tiles[@]}" > out/run_$name.log 2>&1
  tr '\r' '\n' < out/run_$name.log | grep -E "^wall|Description|elastix failed" | tail -2 | cut -c1-200
  [ -f out/$name/P000/Transform.h5 ] && runs+=("$name")
done
systemd-run --user --scope -p MemoryMax=24G -p MemorySwapMax=0 --quiet $V/python scripts/oct_native_score.py plane_native_$tag native_tiled_$tag "${runs[@]}" 2>&1 | sed -u '/Warning/d'
