#!/bin/bash
# Stage 2 of the OCT / light-sheet section with other losses and engines: a 2-D MIND feature loss under elastix
# (CPU build), ConvexAdam with MIND, FireANTs SyN with local correlation and with MIND. Scored and drawn like the
# elastix variants (scripts/oct_score.py, figures_pair.py --run).
#   scripts/oct_engines.sh mind_mi mind convexadam fireants fireants_mind
set -o pipefail
cd /home/valentin/Documents/ImpactReg_OOC_Demo
V=/home/valentin/Documents/lib/venv-impactreg/bin; export PATH=$V:$PATH
R=Predictor.Model.RegistrationNet
pair=data/dandi/plane
[ -d $pair/FixedMask.ome.zarr ] || systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 --quiet $V/python scripts/oct_masks.py 2>&1 | sed -u '/Warning/d'
inputs=(-f $pair/Fixed.ome.zarr -m $pair/Moving.ome.zarr --fixed-mask $pair/FixedMask.ome.zarr --moving-mask $pair/MovingMask.ome.zarr)
runs=()
for config in "$@"; do
  name=plane_$config
  case $config in
    mind_mi|mind)
      mi=1.0; [ "$config" = mind ] && mi=0.0
      preset=SECTION_MIND; device=(--cpu 1); export KONFAI_IMPACTREG_REPO=$PWD/presets
      sets=(--set "$R.parameter_overrides=[Metric0Weight=1.0, Metric1Weight=$mi, Metric2Weight=0, FinalGridSpacingInPhysicalUnits=0.4 0.4 1000, RequiredRatioOfValidSamples=0.05, ImpactUseMixedPrecision=\"false\"]") ;;
    convexadam)
      unset KONFAI_IMPACTREG_REPO; preset=ConvexAdam_Composite; device=(--gpu 0)
      sets=(--set "$R.linear=false" --set "$R.regularization_weight=3.0") ;;
    fireants)
      unset KONFAI_IMPACTREG_REPO; preset=FireANTs_SyN; device=(--gpu 0)
      sets=(--set "$R.affine_lr=0.0") ;;
    fireants_mind)
      unset KONFAI_IMPACTREG_REPO; preset=FireANTs_IMPACT; device=(--gpu 0)
      sets=(--set "$R.affine_lr=0.0" --set "$R.models.0.ref=VBoussot/impact-torchscript-models:MIND/R1D2_3D.pt" --set "$R.models.0.layers_mask=1") ;;
    *) echo "unknown config $config"; exit 1 ;;
  esac
  echo "== $name: preset $preset ${device[*]}"
  rm -rf out/$name
  /usr/bin/time -f "wall %e s" systemd-run --user --scope -p MemoryMax=60G -p MemorySwapMax=0 --quiet \
    impact-reg-konfai register $preset "${inputs[@]}" -o out/$name "${device[@]}" "${sets[@]}" > out/run_$name.log 2>&1
  tr '\r' '\n' < out/run_$name.log | grep -E "^wall|elastix failed|Description|Error|error" | grep -v Warning | tail -3 | cut -c1-300
  [ -d out/$name/P000/Moved.ome.zarr ] || continue
  runs+=("$name")
  systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 --quiet $V/python figures_pair.py --run $name --stem _$config 2>&1 | sed -u '/Warning/d' | grep P3
done
systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 --quiet $V/python scripts/oct_score.py plane_g0.4 "${runs[@]}" 2>&1 | sed -u '/Warning/d'
