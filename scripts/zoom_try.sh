#!/bin/bash
# Zoom 1 on the overview at 40 um: the pair through the hand placement and the rigid correction, the tiled
# deformable, then the 20 um windows no run has seen.
#   zoom_try.sh CONFIG      CONFIG: bend100 | base ; checked on 8 windows of 4 mm spread over the zoom
set -o pipefail
cd /home/valentin/Documents/ImpactReg_OOC_Demo
V=/home/valentin/Documents/lib/venv-impactreg/bin; export PATH=$V:$PATH
R=Predictor.Model.RegistrationNet
config=$1; shift
pair=data/linc/zoom1_pair1
name=zoom1_level1_$config

if [ ! -d $pair/FixedMask.ome.zarr ]; then
  echo "== pair: overview level 1 over zoom 1, 1024 x 1024 x 896"
  /usr/bin/time -f "wall %e s, peak RSS %M kB" systemd-run --user --scope -p MemoryMax=45G -p MemorySwapMax=0 --quiet \
    $V/python -u zoom_linc.py pair --level 1 --size 1024 1024 896 --out $pair 2>&1 | sed -u '/Warning/d' || exit 1
fi

overrides="CheckNumberOfSamples=\"false\""
case $config in
  base) ;;
  bend*) overrides="Registration=\"MultiMetricMultiResolutionRegistration\", Metric=\"AdvancedMattesMutualInformation\" \"TransformBendingEnergyPenalty\", Metric0Weight=1.0, Metric1Weight=${config#bend}, CheckNumberOfSamples=\"false\"" ;;
  *) echo "unknown config $config"; exit 1 ;;
esac
echo "== $name: 294 tiles of 256^3, overlap 128, masked, $config"
rm -rf "out/$name"
/usr/bin/time -f "wall %e s, peak RSS %M kB" systemd-run --user --scope -p MemoryMax=90G -p MemorySwapMax=0 --quiet \
  impact-reg-konfai register Generic_Rigid_BSpline \
  -f $pair/Fixed.ome.zarr -m $pair/Moving.ome.zarr \
  --fixed-mask $pair/FixedMask.ome.zarr --moving-mask $pair/MovingMask.ome.zarr \
  -o "out/$name" --gpu 0 --fields-only \
  --set "$R.parameter_maps=[Parameters_BSpline.txt]" --set "$R.final_grid_spacing=2.0" \
  --set "$R.max_iterations=200" --set "$R.spatial_samples=4096" \
  --set "$R.parameter_overrides=[$overrides]" \
  --set "Predictor.Dataset.Patch.patch_size=[256,256,256]" --set "Predictor.Dataset.Patch.overlap=128" \
  --set "Predictor.outputs_dataset.DisplacementField.OutputDataset.patch_combine=Cosinus" > "out/run_$name.log" 2>&1
tr '\r' '\n' < "out/run_$name.log" | grep -E "^wall|elastix failed|Description|Traceback|held " | tail -3
[ -f "out/$name/P000/Transform.h5" ] || exit 1

echo "== held out: 20 um windows"
systemd-run --user --scope -p MemoryMax=30G -p MemorySwapMax=0 --quiet $V/python -u zoom_linc.py check \
  --field "out/$name/P000/Transform.h5" 2>&1 | sed -u '/Warning/d' | grep -v "^  level"
