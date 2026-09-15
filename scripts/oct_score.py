"""Score a stage-2 run of the OCT / light-sheet section: mutual information, how far the light-sheet's edges
sit from the OCT's (the median distance from each light-sheet edge pixel to the nearest OCT edge pixel, in um),
and how much of the tissue the field folds (in-plane Jacobian determinant at or below zero, 0.5 mm in from the
tissue's edge). A higher MI bought with folds is not a better registration.

    python scripts/oct_score.py plane_g0.4 plane_m_b10 ...      # run names under out/, the affine first
"""
import sys
from pathlib import Path

import numpy as np
import zarr
from scipy.ndimage import binary_erosion, distance_transform_edt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import linc
from figures import outline, read_zarr
from stage1_linc import mutual_information

VOXEL_UM = 25.0
WINDOWS = ((200, 330), (720, 210), (690, 1380), (1330, 480))  # (x, y) centres, as in figures_pair.py
fixed = read_zarr(ROOT / "data" / "dandi" / "plane" / "Fixed.ome.zarr")[0]
oct_edges = outline(fixed[fixed.shape[0] // 2])
to_oct_edge = distance_transform_edt(~oct_edges) * VOXEL_UM
tissue = fixed[fixed.shape[0] // 2] > np.percentile(fixed[fixed.shape[0] // 2], 30)
mask_store = ROOT / "data" / "dandi" / "plane" / "FixedMask.ome.zarr"
inside = None
if mask_store.exists():
    group = zarr.open_group(str(mask_store), mode="r")
    level = dict(group.attrs)["multiscales"][0]["datasets"][0]["path"]
    inside = binary_erosion(np.asarray(group[level][0, group[level].shape[1] // 2]) > 0, iterations=20)


def folding(field_path: Path) -> str:
    if inside is None or not field_path.exists():
        return ""
    size, _, spacing = linc.field_header(field_path)
    u = linc.read_field(field_path, (0, 0, 0), size)
    k = u.shape[0] // 2
    ux, uy = u[k, ..., 0], u[k, ..., 1]
    det = ((1 + np.gradient(ux, spacing[0], axis=1)) * (1 + np.gradient(uy, spacing[1], axis=0))
           - np.gradient(ux, spacing[1], axis=0) * np.gradient(uy, spacing[0], axis=1))
    d = det[inside]
    return f"   folded {np.mean(d <= 0):.2%} of the tissue, det J 1 % {np.percentile(d, 1):.2f}"


def score(name: str, moving: np.ndarray, field_path: Path | None = None) -> None:
    plane = moving[moving.shape[0] // 2]
    edges = outline(plane) & tissue
    d = to_oct_edge[edges]
    windows = []
    for cx, cy in WINDOWS:  # the deck's four 12 mm windows: a local failure does not hide in the global median
        x0, y0 = min(max(cx - 240, 0), plane.shape[1] - 480), min(max(cy - 240, 0), plane.shape[0] - 480)
        local = np.zeros_like(edges); local[y0:y0 + 480, x0:x0 + 480] = True
        windows.append(f"{np.median(to_oct_edge[edges & local]):.0f} um, MI {mutual_information(fixed[:, y0:y0 + 480, x0:x0 + 480], moving[:, y0:y0 + 480, x0:x0 + 480]):.2f}")
    print(f"{name:22s} MI {mutual_information(fixed, moving):.3f}   light-sheet edge to OCT edge: "
          f"median {np.median(d):.0f} um, 90 % under {np.percentile(d, 90):.0f} um   windows 1-4: {' | '.join(windows)}"
          + (folding(field_path) if field_path else ""))


score("affine", read_zarr(ROOT / "data" / "dandi" / "plane" / "Moving.ome.zarr")[0])
for run in sys.argv[1:]:
    path = ROOT / "out" / run / "P000" / "Moved.ome.zarr"
    if path.exists():
        score(run, read_zarr(path)[0], ROOT / "out" / run / "P000" / "Transform.h5")
    else:
        print(f"{run:22s} no Moved image")
