"""Where each tiled stage moved the MRI, and by how much: the field's length over the XPCT, inside the tissue.

    python figures_motion.py
"""

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

import linc
from evaluate_linc import NATIVE_RUN, native_states
from figures_demo import zarr_plane
from figures_linc import DATA, ROOT, bands, grey, save
from stage4_native_linc import written_window


def heat(background: np.ndarray, length: np.ndarray, top: float) -> np.ndarray:
    """The field's length in colour over the grey XPCT: dark where nothing moved, yellow at ``top`` mm."""
    t = np.clip(length / top, 0, 1)
    colour = np.dstack([t ** 0.7, t ** 1.6, 0.9 * (1 - t) ** 2 + 0.1 * t])          # blue -> magenta -> yellow
    alpha = np.clip(t * 3, 0, 0.75)[..., None]
    return np.dstack([grey(background)] * 3) * (1 - alpha) + colour * alpha


def crop(centre, half: int, *planes: np.ndarray) -> list[np.ndarray]:
    y, x = centre
    return [p[max(y - half, 0) : y + half, max(x - half, 0) : x + half] for p in planes]


def stage(name: str, fixed: np.ndarray, before: np.ndarray, length: np.ndarray, voxel_mm: float, scale: float) -> None:
    """The field's length over the XPCT, inside the tissue: a B-spline tile extrapolates freely past the mask,
    and those values outside the MRI say nothing about the registration."""
    tissue = ndimage.binary_dilation(before > 0.02, iterations=round(1.0 / voxel_mm))
    inside = length * tissue
    top = float(np.percentile(inside[tissue], 99.5))
    print(f"  {name}: inside the tissue, median {np.median(inside[tissue]):.2f} mm, max {inside.max():.2f} mm")
    save(bands([(f"{name} · how far the tiles moved the map · dark 0, yellow {top:.1f} mm",
                 [heat(fixed, inside, top)])], size=20), f"M_{name.split()[0]}_map.png", scale=scale)


def main() -> None:
    # level 3: the tiles over the whole hemisphere
    pair3 = DATA / "pair3_bend"
    tiled3 = ROOT / "out" / "linc_tiled3_bend100" / "P000"
    image3 = written_window(pair3 / "Fixed.ome.zarr")
    voxel3 = image3.GetSpacing()[0]
    j = image3.GetSize()[1] // 2
    fixed3 = sitk.GetArrayFromImage(image3[:, j : j + 1, :]).squeeze()
    size = linc.field_header(tiled3 / "Transform.h5")[0]
    field3 = linc.read_field(tiled3 / "Transform.h5", (0, j, 0), (size[0], j + 1, size[2]))[:, 0]       # (z, x, 3)
    length3 = np.linalg.norm(field3, axis=-1)
    stage("level3 (0.16 mm)", fixed3, zarr_plane(pair3 / "Moving.ome.zarr", np.s_[:, j, :]), length3, voxel3, scale=0.6)

    # the native window: the tiles at 20 um
    window = written_window(DATA / "pair0" / "Fixed.ome.zarr")
    voxel0 = window.GetSpacing()[0]
    z = window.GetSize()[2] // 2
    plane = window[:, :, z : z + 1]
    start, _ = native_states(plane)
    fixed0 = sitk.GetArrayFromImage(plane)[0]
    size = linc.field_header(NATIVE_RUN / "Transform.h5")[0]
    field0 = linc.read_field(NATIVE_RUN / "Transform.h5", (0, 0, z), (size[0], size[1], z + 1))[0]     # (y, x, 3)
    length0 = np.linalg.norm(field0, axis=-1)
    stage("native (20 um)", fixed0, start[0], length0, voxel0, scale=0.8)


if __name__ == "__main__":
    main()
