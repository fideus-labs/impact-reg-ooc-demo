#!/bin/bash
# The native window through the regularised chain, then the tiled native run and its score.
#   native_try.sh LEVEL3_RUN CONFIG...     e.g. native_try.sh linc_tiled3_bend100 base bend100
set -o pipefail
cd /home/valentin/Documents/ImpactReg_OOC_Demo
V=/home/valentin/Documents/lib/venv-impactreg/bin; export PATH=$V:$PATH
R=Predictor.Model.RegistrationNet
level3=$1; shift
coarse=out/linc_coarse_bend100/P000/Transform.h5
fine=out/$level3/P000/Transform.h5
pair=data/linc/pair0_bend

echo "== native pair through $coarse and $fine"
mkdir -p $pair && ln -sfn "$PWD/data/linc/pair0/Fixed.ome.zarr" $pair/Fixed.ome.zarr
systemd-run --user --scope -p MemoryMax=30G -p MemorySwapMax=0 --quiet $V/python -u stage4_native_linc.py --reuse-fixed \
  --coarse-field $coarse --fine-field $fine --out $pair 2>&1 | sed -u '/Warning/d' || exit 1
rm -rf $pair/FixedMask.ome.zarr $pair/MovingMask.ome.zarr
systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0 --quiet $V/python masks_linc.py $pair 2>&1 | sed -u '/Warning/d'

for config in "$@"; do
  overrides="CheckNumberOfSamples=\"false\""
  case $config in
    base) ;;
    bend*) overrides="Registration=\"MultiMetricMultiResolutionRegistration\", Metric=\"AdvancedMattesMutualInformation\" \"TransformBendingEnergyPenalty\", Metric0Weight=1.0, Metric1Weight=${config#bend}, CheckNumberOfSamples=\"false\"" ;;
    *) echo "unknown config $config"; exit 1 ;;
  esac
  name=linc_native_$config
  echo; echo "== $name: level 0, 125 tiles of 256^3, overlap 128, masked, $config"
  rm -rf "out/$name"
  /usr/bin/time -f "wall %e s, peak RSS %M kB" systemd-run --user --scope -p MemoryMax=60G -p MemorySwapMax=0 --quiet \
    impact-reg-konfai register Generic_Rigid_BSpline \
    -f $pair/Fixed.ome.zarr -m $pair/Moving.ome.zarr \
    --fixed-mask $pair/FixedMask.ome.zarr --moving-mask $pair/MovingMask.ome.zarr \
    -o "out/$name" --gpu 0 \
    --set "$R.parameter_maps=[Parameters_BSpline.txt]" --set "$R.final_grid_spacing=1.0" \
    --set "$R.max_iterations=200" --set "$R.spatial_samples=4096" \
    --set "$R.parameter_overrides=[$overrides]" \
    --set "Predictor.Dataset.Patch.patch_size=[256,256,256]" --set "Predictor.Dataset.Patch.overlap=128" \
    --set "Predictor.outputs_dataset.DisplacementField.OutputDataset.patch_combine=Cosinus" > "out/run_$name.log" 2>&1
  tr '\r' '\n' < "out/run_$name.log" | grep -E "^wall|elastix failed|Description|Traceback|held " | tail -3
  [ -f "out/$name/P000/Transform.h5" ] || continue
  systemd-run --user --scope -p MemoryMax=30G -p MemorySwapMax=0 --quiet $V/python evaluate_linc.py native \
    --run "out/$name/P000" --coarse-field $coarse --fine-field $fine --overlap 128 2>&1 | sed -u '/Warning/d' | grep -v "^field "
done
