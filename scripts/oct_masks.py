"""The tissue masks for the OCT / light-sheet section: Otsu on the light-sheet after the affine, widened by 0.5 mm.
The mounting medium is not black in either image, so a threshold near zero would keep the whole frame."""
import sys
from pathlib import Path

import numpy as np
import zarr
from konfai.utils.ome_zarr import write_ome_zarr
from scipy.ndimage import binary_fill_holes, maximum_filter1d
from skimage.filters import threshold_otsu

ROOT = Path(__file__).resolve().parent.parent
pair = ROOT / "data" / "dandi" / "plane"
group = zarr.open_group(str(pair / "Moving.ome.zarr"), mode="r")
dataset = dict(group.attrs)["multiscales"][0]["datasets"][0]
scale = dataset["coordinateTransformations"][0]["scale"][-3:]
origin = next(t["translation"][-3:] for t in dataset["coordinateTransformations"] if t["type"] == "translation")
moving = np.asarray(group[dataset["path"]][0])
plane = moving[moving.shape[0] // 2]
level = threshold_otsu(plane[plane > 0.02])
tissue = binary_fill_holes(plane > level)
width = 2 * round(0.5 / scale[-1]) + 1
for axis in range(2):
    tissue = maximum_filter1d(tissue.astype(np.uint8), width, axis=axis)
support = np.repeat(tissue[None], moving.shape[0], axis=0).astype(np.uint8)
kw = {"spacing": tuple(scale[::-1]), "origin": tuple(origin[::-1]), "chunks": (1, 8, 256, 256)}
write_ome_zarr(pair / "FixedMask.ome.zarr", support[None], **kw)
write_ome_zarr(pair / "MovingMask.ome.zarr", np.ones_like(support)[None], **kw)
print(f"Otsu level {level:.3f}; the mask covers {support.mean():.2f} of the frame", file=sys.stderr)
