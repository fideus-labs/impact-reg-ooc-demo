#!/bin/bash
# elastix with a MIND feature loss (the IMPACT metric) on a level-4 pair: the same masks, grid, iterations,
# samples and bending penalty as the elastix + mutual information reference, then the same scores.
#   mind_try.sh PAIR VARIANT      PAIR: flash | dmri      VARIANT: mind | mind_bendW | mind_mi
#   mind     IMPACT(MIND) + bending 100                    (mutual information weighted 0)
#   mind_bendW   IMPACT(MIND) + bending W, e.g. mind_bend10, mind_bend1, mind_bend0
#   mind_mi  IMPACT(MIND) + mutual information + bending 100
# Runs on the CPU: the elastix-IMPACT install here is the CPU flavour (ImpactGPU -1), and half precision has no 3D pooling there.
# The reference to beat: FLASH Dice 0.834 (out/linc_flash_bend100), dMRI correlation 0.888 (out/linc_coarse_bend100).
set -o pipefail
cd /home/valentin/Documents/ImpactReg_OOC_Demo
V=/home/valentin/Documents/lib/venv-impactreg/bin; export PATH=$V:$PATH
export KONFAI_IMPACTREG_REPO=$PWD/presets
R=Predictor.Model.RegistrationNet
pair_name=$1 variant=$2
case $pair_name in
  flash) pair=data/linc/pair_flash; name=linc_flash_$variant ;;
  dmri) pair=data/linc/pair; name=linc_coarse_$variant ;;
  *) echo "unknown pair $pair_name"; exit 1 ;;
esac
case $variant in
  mind) weights="Metric0Weight=1.0, Metric1Weight=0.0, Metric2Weight=100" ;;
  mind_bend*) weights="Metric0Weight=1.0, Metric1Weight=0.0, Metric2Weight=${variant#mind_bend}" ;;
  mind_mi) weights="Metric0Weight=1.0, Metric1Weight=1.0, Metric2Weight=100" ;;
  *) echo "unknown variant $variant"; exit 1 ;;
esac

echo "== $name: elastix + IMPACT(MIND R1D2), $variant, grid 3 mm, 600 it x 4 levels, masked, bending 100"
rm -rf "out/$name"
/usr/bin/time -f "wall %e s, peak RSS %M kB" systemd-run --user --scope -p MemoryMax=80G -p MemorySwapMax=0 --quiet \
  impact-reg-konfai register LINC_MIND \
  -f $pair/Fixed.ome.zarr -m $pair/Moving.ome.zarr \
  --fixed-mask $pair/FixedMask.ome.zarr --moving-mask $pair/MovingMask.ome.zarr \
  -o "out/$name" --cpu 1 \
  --set "$R.parameter_overrides=[$weights, RequiredRatioOfValidSamples=0.05, ImpactUseMixedPrecision=\"false\"]" > "out/run_$name.log" 2>&1
tr '\r' '\n' < "out/run_$name.log" | grep -E "^wall|elastix failed|Description|Traceback|Error" | tail -4
[ -f "out/$name/P000/Transform.h5" ] || exit 1

if [ "$pair_name" = flash ]; then
  systemd-run --user --scope -p MemoryMax=30G -p MemorySwapMax=0 --quiet $V/python evaluate_linc.py dice --name flash \
    --field "out/$name/P000/Transform.h5" 2>&1 | sed -u '/Warning/d' | grep -v "^field " | tee "out/dice_$name.txt" | tail -1
else
  # --run takes a CHAIN of fields run one after the other: score one run per call
  systemd-run --user --scope -p MemoryMax=30G -p MemorySwapMax=0 --quiet $V/python evaluate_linc.py coarse \
    --run "out/$name/P000" 2>&1 | sed -u '/Warning/d' | grep -v "^field "
fi
