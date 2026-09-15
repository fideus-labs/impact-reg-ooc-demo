"""The short deck's figures: the same planes as L6 to L9, with only this pipeline's result on them.

    python figures_demo.py
"""

from pathlib import Path

import numpy as np
import SimpleITK as sitk
import zarr
from scipy import ndimage

from evaluate_linc import NATIVE_RUN, native_states
from figures_linc import (
    DATA,
    ROOT,
    bands,
    checkerboard,
    grey,
    iso_contours_over,
    read_zarr,
    save,
)
from stage1_linc import LPS, normalise
from stage4_native_linc import written_window


def checker_rows(planes, tile: int) -> np.ndarray:
    """One labelled row per plane: the XPCT | MRI checkerboard before, then after, side by side."""
    return bands([(label, [checkerboard(fixed, before, tile), checkerboard(fixed, after, tile)])
                  for label, fixed, before, after in planes])


def zarr_plane(path: Path, index) -> np.ndarray:
    group = zarr.open_group(str(path), mode="r")
    array = group[dict(group.attrs)["multiscales"][0]["datasets"][0]["path"]]
    return np.asarray(array[(0, *index)] if array.ndim == 4 else array[index])


def main() -> None:
    # level 4: after the affine, after the deformable
    fixed_image = sitk.ReadImage(str(DATA / "XPCT_level4.mha"))
    fixed_image.SetDirection(LPS)
    fixed = sitk.GetArrayFromImage(normalise(fixed_image))
    affine = read_zarr(DATA / "pair" / "Moving.ome.zarr")
    deformable = read_zarr(ROOT / "out" / "linc_coarse_bend100" / "P000" / "Moved.ome.zarr")
    z, y = fixed.shape[0] // 2, fixed.shape[1] // 2
    save(checker_rows([
        ("axial · after the affine | after the deformable", fixed[z], affine[z], deformable[z]),
        ("coronal · after the affine | after the deformable", fixed[:, y], affine[:, y], deformable[:, y]),
    ], tile=32), "D7_checker_level4.png", scale=0.7)

    # level 3: before and after the tiled run
    pair3 = DATA / "pair3_bend"
    tiled3 = ROOT / "out" / "linc_tiled3_bend100" / "P000" / "Moved.ome.zarr"
    image3 = written_window(pair3 / "Fixed.ome.zarr")
    k, j = image3.GetSize()[2] // 2, image3.GetSize()[1] // 2
    planes3 = []
    for name, slab, index in (("axial", image3[:, :, k : k + 1], np.s_[k, :, :]),
                              ("coronal", image3[:, j : j + 1, :], np.s_[:, j, :])):
        fixed3 = sitk.GetArrayFromImage(slab).squeeze()
        planes3.append((f"{name} · before the tiled run | after", fixed3,
                        zarr_plane(pair3 / "Moving.ome.zarr", index), zarr_plane(tiled3, index)))
    save(checker_rows(planes3, tile=64), "D8_checker_level3.png", scale=0.45)

    # the native window: one plane and its central 5 mm, then contours and checkerboards, before and after
    window = written_window(DATA / "pair0" / "Fixed.ome.zarr")
    z = window.GetSize()[2] // 2
    plane_array = sitk.GetArrayFromImage(window)[z]
    inner = plane_array[256:512, 256:512]
    save(np.hstack([np.dstack([grey(plane_array)] * 3), np.dstack([grey(ndimage.zoom(inner, 3, order=1))] * 3)]),
         "L5_native_window.png")
    plane = window[:, :, z : z + 1]
    start, after = native_states(plane)
    fixed_plane = sitk.GetArrayFromImage(plane)[0]
    level = float(np.median(after[0][after[0] > 0.02]))
    save(np.hstack([iso_contours_over(fixed_plane, start[0], level), iso_contours_over(fixed_plane, after[0], level)]),
         "D6_native_contours.png", scale=0.6)
    j = window.GetSize()[1] // 2
    across = window[:, j : j + 1, :]
    before_across, after_across = native_states(across)
    fixed_across = sitk.GetArrayFromImage(across).squeeze()
    save(checker_rows([
        ("one plane · before the native tiles | after", fixed_plane, start[0], after[0]),
        ("the perpendicular plane · before | after", fixed_across, before_across.squeeze(), after_across.squeeze()),
    ], tile=96), "D9_checker_native.png", scale=0.5)
    assert (NATIVE_RUN / "Transform.h5").exists()


if __name__ == "__main__":
    main()
