"""Residual misalignment of the OCT / light-sheet section, measured on the images themselves: in 6 mm windows over
the tissue, the in-plane shift of the light-sheet (±1 mm, 50 um steps) that maximises mutual information with the
OCT. An aligned window peaks at zero; the peak's offset is the local residual. No edges, no thresholds.

    python scripts/oct_residual_shift.py RUN...        (RUN under out/, or 'affine')
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from figures import read_zarr
from stage1_linc import mutual_information

VOXEL_UM, HALF, SEARCH, STEP = 25, 120, 40, 2        # 6 mm windows, ±1 mm search, 50 um steps (in voxels)
fixed = read_zarr(ROOT / "data/dandi/plane/Fixed.ome.zarr")[0][0]
moving_mask = read_zarr(ROOT / "data/dandi/plane/FixedMask.ome.zarr")[0][0] > 0
H, W = fixed.shape
centres = [(x, y) for y in range(HALF + SEARCH, H - HALF - SEARCH, 2 * HALF) for x in range(HALF + SEARCH, W - HALF - SEARCH, 2 * HALF)
           if moving_mask[y - HALF:y + HALF, x - HALF:x + HALF].mean() > 0.8]


def residual(moving: np.ndarray, cx: int, cy: int) -> tuple[float, float, float]:
    ref = fixed[cy - HALF:cy + HALF, cx - HALF:cx + HALF]
    best = (-1.0, 0, 0)
    for dy in range(-SEARCH, SEARCH + 1, STEP):
        for dx in range(-SEARCH, SEARCH + 1, STEP):
            win = moving[cy - HALF + dy:cy + HALF + dy, cx - HALF + dx:cx + HALF + dx]
            mi = mutual_information(ref, win)
            if mi > best[0]:
                best = (mi, dx, dy)
    return best[1] * VOXEL_UM, best[2] * VOXEL_UM, best[0]


print(f"{len(centres)} windows of 6 mm fully inside the tissue")
for run in sys.argv[1:]:
    path = ROOT / "data/dandi/plane/Moving.ome.zarr" if run == "affine" else ROOT / "out" / run / "P000" / "Moved.ome.zarr"
    moving = read_zarr(path)[0][0]
    shifts = np.array([residual(moving, cx, cy)[:2] for cx, cy in centres])
    norm = np.hypot(shifts[:, 0], shifts[:, 1])
    cells = "  ".join(f"({cx * VOXEL_UM / 1000:.0f},{cy * VOXEL_UM / 1000:.0f} mm) {n:.0f}" for (cx, cy), n in zip(centres, norm))
    print(f"{run:26s} residual shift median {np.median(norm):4.0f} um, 90 % under {np.percentile(norm, 90):4.0f} um, "
          f"max {norm.max():4.0f} um, windows at zero {np.mean(norm == 0):.0%}")
    print(f"   per window: {cells}")
