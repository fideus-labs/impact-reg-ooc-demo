"""Score stage 3 of the OCT / light-sheet section: the native start warped through each tiled run's field by hand
(KonfAI's Moved image agrees with it to r 0.99), mutual information at 3, 12 and 48 um away from the faces, the
field's size and how much of the window it folds. A tiled run that lowers the MI at every scale made things worse.

    python scripts/oct_native_score.py PAIR RUN...     e.g. plane_native_mind native_tiled_mind native_tiled_mind_mi_b1
"""
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import linc
from figures import read_zarr
from stage1_linc import mutual_information

MARGIN = 100  # voxels kept away from the window's faces


def block(a: np.ndarray, f: int) -> np.ndarray:
    n0, n1 = a.shape[0] // f * f, a.shape[1] // f * f
    return a[:n0, :n1].reshape(n0 // f, f, n1 // f, f).mean((1, 3))


def mis(fixed: np.ndarray, moving: np.ndarray) -> str:
    inner = np.s_[MARGIN:-MARGIN, MARGIN:-MARGIN]
    return "  ".join(f"{3 * f} um {mutual_information(block(fixed[inner], f), block(moving[inner], f)):.3f}" for f in (1, 4, 16))


pair = ROOT / "data" / "dandi" / sys.argv[1]
fixed_volume = read_zarr(pair / "Fixed.ome.zarr")[0]
k = fixed_volume.shape[0] // 2
fixed, start = fixed_volume[k], read_zarr(pair / "Moving.ome.zarr")[0][k]
print(f"{'start':28s} MI {mis(fixed, start)}")
yy, xx = np.mgrid[: start.shape[0], : start.shape[1]].astype(np.float32)
for run in sys.argv[2:]:
    path = ROOT / "out" / run / "P000" / "Transform.h5"
    if not path.exists():
        print(f"{run:28s} no field")
        continue
    size, _, sp = linc.field_header(path)
    u = linc.read_field(path, (0, 0, 0), size)
    ux, uy = u[k, ..., 0], u[k, ..., 1]
    warped = map_coordinates(start, [yy + uy / sp[1], xx + ux / sp[0]], order=1, mode="nearest")
    det = ((1 + np.gradient(ux, sp[0], axis=1)) * (1 + np.gradient(uy, sp[1], axis=0))
           - np.gradient(ux, sp[1], axis=0) * np.gradient(uy, sp[0], axis=1))
    size_um = np.hypot(ux, uy) * 1000
    print(f"{run:28s} MI {mis(fixed, warped)}   |u| median {np.median(size_um):.0f} um, 95 % {np.percentile(size_um, 95):.0f} um"
          f"   folded {np.mean(det[MARGIN:-MARGIN, MARGIN:-MARGIN] <= 0):.2%}")
