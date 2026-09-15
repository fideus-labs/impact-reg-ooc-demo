#!/bin/bash
# The loss, on the OCT / light-sheet section: the same elastix schedule (four resolutions, grid 3.2 -> 0.4 mm,
# 3000 iterations each, 32768 samples, locked to the plane), different losses. One CPU job at a time; every run
# is scored (MI, edge distance, folding) and drawn in the four windows.
#   scripts/oct_loss.sh CONFIG...
#   mi_b<w>         mutual information + bending penalty w
#   ncc_inv_b<w>    normalised correlation on the INVERTED light-sheet (data/dandi/plane_inv) + bending w
#   mind3d_mi_b<w>  IMPACT with 3-D MIND (anisotropic voxel, 0.4 mm in z) + MI + bending w, CPU build
#   mind3d_b<w>     the same without MI
#   mind2d_mi_b<w>  IMPACT with 2-D MIND (in plane; Static mode still takes three voxel sizes) + MI + bending w
#   mind{2d,3d}[R<r>D<d>][_mi[<w>]]_b<w>[_it<n>][_s<n>]  the general form: MIND radius/dilation, MI weight (bare _mi = 1),
#                   bending w, iterations per level (800) and samples (16384); a preset copy is made under presets/
set -o pipefail
cd /home/valentin/Documents/ImpactReg_OOC_Demo
V=/home/valentin/Documents/lib/venv-impactreg/bin; export PATH=$V:$PATH
R=Predictor.Model.RegistrationNet
runs=()
for config in "$@"; do
  w=0; [[ $config =~ _b([0-9.]+) ]] && w=${BASH_REMATCH[1]}
  pair=data/dandi/plane; preset=Generic_Rigid_BSpline; device=(--gpu 0); extra=()
  metric='"AdvancedMattesMutualInformation"'; weights="Metric0Weight=1.0, Metric1Weight=$w"
  unset KONFAI_IMPACTREG_REPO
  case $config in
    mi_b*) ;;
    ncc_inv_b*) pair=data/dandi/plane_inv; metric='"AdvancedNormalizedCorrelation"' ;;
    mind2d*|mind3d*)
      base=SECTION_MIND; [[ $config == mind3d* ]] && base=SECTION_MIND3D
      model=R1D2; [[ $config =~ ^mind[23]d(R[12]D[12]) ]] && model=${BASH_REMATCH[1]}
      mi=0.0; [[ $config =~ _mi([0-9.]*)_ ]] && mi=${BASH_REMATCH[1]:-1.0}
      it=800; [[ $config =~ _it([0-9]+) ]] && it=${BASH_REMATCH[1]}
      s=16384; [[ $config =~ _s([0-9]+) ]] && s=${BASH_REMATCH[1]}
      preset=${base}_${config//./p}; rm -rf presets/$preset; cp -r presets/$base presets/$preset
      sed -i "s/R1D2_/${model}_/; s/max_iterations: 800/max_iterations: $it/; s/spatial_samples: 16384/spatial_samples: $s/" presets/$preset/Prediction.yml
      device=(--cpu 1); export KONFAI_IMPACTREG_REPO=$PWD/presets
      weights="Metric0Weight=1.0, Metric1Weight=$mi, Metric2Weight=$w"
      echo "   MIND $model, MI weight $mi, $it it/level, $s samples" ;;
    *) echo "unknown config $config"; exit 1 ;;
  esac
  if [ $preset = Generic_Rigid_BSpline ]; then
    overrides="Registration=\"MultiMetricMultiResolutionRegistration\", Metric=$metric \"TransformBendingEnergyPenalty\", $weights, FinalGridSpacingInPhysicalUnits=0.4 0.4 1000, RequiredRatioOfValidSamples=0.05"
    extra=(--set "$R.parameter_maps=[Parameters_BSpline.txt]" --set "$R.final_grid_spacing=0.4" --set "$R.max_iterations=3000" --set "$R.spatial_samples=32768")
  else
    overrides="$weights, FinalGridSpacingInPhysicalUnits=0.4 0.4 1000, RequiredRatioOfValidSamples=0.05, ImpactUseMixedPrecision=\"false\""
  fi
  name=plane_$config
  echo "== $name: $preset on $pair, $overrides"
  rm -rf out/$name
  /usr/bin/time -f "wall %e s" systemd-run --user --scope -p MemoryMax=60G -p MemorySwapMax=0 --quiet \
    impact-reg-konfai register $preset -f $pair/Fixed.ome.zarr -m $pair/Moving.ome.zarr \
    --fixed-mask $pair/FixedMask.ome.zarr --moving-mask $pair/MovingMask.ome.zarr -o out/$name "${device[@]}" \
    "${extra[@]}" --set "$R.parameter_overrides=[$overrides]" > out/run_$name.log 2>&1
  tr '\r' '\n' < out/run_$name.log | grep -E "^wall|elastix failed|Description|Error" | grep -v Warning | tail -2 | cut -c1-200
  [[ $preset == SECTION_MIND_* ]] && rm -rf presets/$preset
  [ -d out/$name/P000/Moved.ome.zarr ] || continue
  runs+=("$name")
  systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 --quiet $V/python figures_pair.py --run $name --stem _$config 2>&1 | sed -u '/Warning/d' | grep P3
done
systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 --quiet $V/python scripts/oct_score.py plane_g0.4 plane_g0.4_it3000_s32768 "${runs[@]}" 2>&1 | sed -u '/Warning/d'
