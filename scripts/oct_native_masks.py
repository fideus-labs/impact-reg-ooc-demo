"""The tissue mask for a native OCT / light-sheet window: a three-class Otsu on the smoothed OCT, keeping all but the
darkest class, the mounting medium (the light-sheet cannot be used: NeuN is as dark in white matter as in the medium), holes filled, widened by 0.1 mm, repeated over the planes. The moving mask is all ones.
Tiles that fall in the mounting medium then get an empty mask, and a zero field, instead of fitting its noise.

    python scripts/oct_native_masks.py plane_native_r2d2
"""
import sys
from pathlib import Path

import numpy as np
import zarr
from konfai.utils.ome_zarr import write_ome_zarr
from scipy.ndimage import binary_fill_holes, binary_opening, gaussian_filter, maximum_filter1d
from skimage.filters import threshold_multiotsu

ROOT = Path(__file__).resolve().parent.parent
pair = ROOT / "data" / "dandi" / sys.argv[1]
group = zarr.open_group(str(pair / "Fixed.ome.zarr"), mode="r")
dataset = dict(group.attrs)["multiscales"][0]["datasets"][0]
scale = dataset["coordinateTransformations"][0]["scale"][-3:]
origin = next(t["translation"][-3:] for t in dataset["coordinateTransformations"] if t["type"] == "translation")
moving = np.asarray(group[dataset["path"]][0])  # the fixed OCT, named for the shape it lends the masks
plane = gaussian_filter(moving.mean(0), 8)
level = threshold_multiotsu(plane, classes=3)[0]
tissue = binary_fill_holes(binary_opening(plane > level, np.ones((9, 9))))
width = 2 * round(0.1 / scale[-1]) + 1
for axis in range(2):
    tissue = maximum_filter1d(tissue.astype(np.uint8), width, axis=axis)
support = np.repeat(tissue[None], moving.shape[0], axis=0).astype(np.uint8)
kw = {"spacing": tuple(scale[::-1]), "origin": tuple(origin[::-1]), "chunks": (1, 8, 256, 256)}
write_ome_zarr(pair / "FixedMask.ome.zarr", support[None], **kw)
write_ome_zarr(pair / "MovingMask.ome.zarr", np.ones_like(support)[None], **kw)
tiles = [support[0, y:y + 512, x:x + 512].mean() for y in range(0, 2048, 451) for x in range(0, 2048, 451)]
print(f"Otsu level {level:.3f}; the mask covers {support.mean():.2f} of the window; "
      f"tissue per tile: min {min(tiles):.2f}, median {np.median(tiles):.2f}, empty tiles {sum(t == 0 for t in tiles)}")
if len(sys.argv) > 2:  # a preview: the light-sheet with the mask's edge
    from PIL import Image
    grey = np.clip(plane / np.percentile(plane, 99.5), 0, 1)
    rgb = np.dstack([grey] * 3)
    from scipy.ndimage import binary_erosion
    edge = tissue.astype(bool) & ~binary_erosion(tissue.astype(bool), iterations=6)
    rgb[maximum_filter1d(maximum_filter1d(edge.astype(np.uint8), 7, axis=0), 7, axis=1) > 0] = (0, 0.8, 1)
    Image.fromarray((rgb[::2, ::2] * 255).astype(np.uint8)).save(sys.argv[2])
