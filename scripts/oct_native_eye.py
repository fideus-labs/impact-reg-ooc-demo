"""Stage 3 by eye: three 1.5 mm crops of a native OCT / light-sheet window at 3 um, red OCT and green light-sheet,
one row for the start and one per tiled run (the start warped through the run's field by hand, so faces stay filled).

    python scripts/oct_native_eye.py TAG RUN...      e.g. r2d2 native_tiled_r2d2 native_tiled_r2d2_mi_b1_m
"""
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import linc
from figures import pair, read_zarr, save
from figures_linc import bands

CROPS, HALF = ((800, 760), (1400, 460), (1750, 1400)), 256  # (x, y) centres in the 2048² window

tag, runs = sys.argv[1], sys.argv[2:]
d = ROOT / "data" / "dandi" / f"plane_native_{tag}"
fixed_volume = read_zarr(d / "Fixed.ome.zarr")[0]
k = fixed_volume.shape[0] // 2
fixed, start = fixed_volume[k], read_zarr(d / "Moving.ome.zarr")[0][k]
yy, xx = np.mgrid[: start.shape[0], : start.shape[1]].astype(np.float32)


def row(moving: np.ndarray) -> list[np.ndarray]:
    return [pair(fixed[cy - HALF : cy + HALF, cx - HALF : cx + HALF], moving[cy - HALF : cy + HALF, cx - HALF : cx + HALF], gamma=0.55)
            for cx, cy in CROPS]


rows = [("start, from stage 2 · red: OCT · green: light-sheet · three 1.5 mm crops at 3 um", row(start))]
for run in runs:
    path = ROOT / "out" / run / "P000" / "Transform.h5"
    size, _, sp = linc.field_header(path)
    u = linc.read_field(path, (0, 0, 0), size)
    warped = map_coordinates(start, [yy + u[k, ..., 1] / sp[1], xx + u[k, ..., 0] / sp[0]], order=1, mode="nearest")
    rows.append((f"after {run}", row(warped)))
save(bands(rows, size=22), f"P5_native_eye_{tag}.png", scale=0.5)
