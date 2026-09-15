#!/bin/bash
# Level 0, refined from the masked level-3 result. One step at a time, each in a memory-capped scope.
set -eo pipefail
cd /home/valentin/Documents/ImpactReg_OOC_Demo
V=/home/valentin/Documents/lib/venv-impactreg/bin; export PATH=$V:$PATH
cap() { systemd-run --user --scope -p MemoryMax=$1 -p MemorySwapMax=0 --quiet "${@:2}"; }
R=Predictor.Model.RegistrationNet

echo "== the native window's moving image, through affine, level-4 field and masked level-3 field"
cap 30G $V/python -u stage4_native_linc.py --reuse-fixed --fine-field out/linc_tiled3_mask/P000/Transform.h5 2>&1 | sed '/Warning/d'

echo "== native masks"
cap 20G $V/python - <<'EOF' 2>&1 | sed '/Warning/d'
import numpy as np, zarr
from scipy.ndimage import maximum_filter1d
from konfai.utils.ome_zarr import write_ome_zarr
g = zarr.open_group("data/linc/pair0/Moving.ome.zarr", mode="r"); d = dict(g.attrs)["multiscales"][0]["datasets"][0]
support = (np.asarray(g[d["path"]][0]) > 0.02).astype(np.uint8)
for axis in range(3):  # ~1 mm either side at 20 um, like level 3: the tissue edge stays in the metric
    support = maximum_filter1d(support, 101, axis=axis)
scale = d["coordinateTransformations"][0]["scale"][-3:]
origin = next(t["translation"][-3:] for t in d["coordinateTransformations"] if t["type"] == "translation")
geometry = dict(spacing=tuple(scale[::-1]), origin=tuple(origin[::-1]), chunks=(1, 64, 256, 256))
write_ome_zarr("data/linc/pair0/FixedMask.ome.zarr", support[None], **geometry)
write_ome_zarr("data/linc/pair0/MovingMask.ome.zarr", np.ones_like(support)[None], **geometry)
print(f"fixed mask covers {support.mean():.3f}")
EOF

echo "== level-0 deformable, tiled, masked"
rm -rf out/linc_native
/usr/bin/time -f "wall %e s, peak RSS %M kB" systemd-run --user --scope -p MemoryMax=60G -p MemorySwapMax=0 --quiet impact-reg-konfai register Generic_Rigid_BSpline \
  -f data/linc/pair0/Fixed.ome.zarr -m data/linc/pair0/Moving.ome.zarr \
  --fixed-mask data/linc/pair0/FixedMask.ome.zarr --moving-mask data/linc/pair0/MovingMask.ome.zarr \
  -o out/linc_native --gpu 0 \
  --set "$R.parameter_maps=[Parameters_BSpline.txt]" --set "$R.final_grid_spacing=1.0" \
  --set "$R.max_iterations=200" --set "$R.spatial_samples=4096" \
  --set "$R.parameter_overrides=[RequiredRatioOfValidSamples=0.05]" \
  --set "Predictor.Dataset.Patch.patch_size=[256,256,256]" --set "Predictor.Dataset.Patch.overlap=12%" \
  --set "Predictor.outputs_dataset.DisplacementField.OutputDataset.patch_combine=Cosinus" > out/run_linc_native.log 2>&1
tr '\r' '\n' < out/run_linc_native.log | grep -E "^wall|Prediction.*100%|held " | tail -3

echo "== score"
cap 30G $V/python evaluate_linc.py native 2>&1 | sed '/Warning/d' | tail -6
