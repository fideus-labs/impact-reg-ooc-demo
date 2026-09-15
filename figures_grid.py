"""The warped grid: a regular lattice drawn in the affine-aligned space and carried through the deformable onto the
XPCT, so bent lines show where and how the deformable pushed the MRI; each line coloured by how far it moved
(grey still, orange to red the most), and the gold contour of the MRI after the deformable over it.

    python figures_grid.py
"""

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_dilation, binary_fill_holes

import linc
from figures_compare import GOLD, contour
from figures_linc import DATA, FIG, ROOT, bands, grey, read_zarr, save
from stage1_linc import LPS, normalise

STOPS = np.array([[0.6, 0.66, 0.76], [1.0, 0.5, 0.1], [1.0, 0.1, 0.1]])   # still -> orange -> red
WAS = np.array([0.12, 0.12, 0.14])                                          # the lattice where it started


def points(slab: sitk.Image) -> np.ndarray:
    """The physical LPS coordinate of every voxel of a one-voxel-thick slab, (rows, cols, 3)."""
    nx, ny, nz = slab.GetSize()
    k, j, i = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    index = np.stack([i, j, k], -1) * np.array(slab.GetSpacing())
    return (index @ np.array(slab.GetDirection()).reshape(3, 3).T + np.array(slab.GetOrigin())).squeeze()


def displacement(transform: sitk.Transform, slab: sitk.Image) -> np.ndarray:
    """Where each slab voxel reads the affine-aligned MRI from, relative to itself: (rows, cols, 3) in mm."""
    to_field = sitk.TransformToDisplacementFieldFilter()
    to_field.SetReferenceImage(slab)
    return sitk.GetArrayFromImage(to_field.Execute(transform)).squeeze()


def lattice(q: np.ndarray, axes: list[int], pitch: float, step: float, width: float = 1.5) -> np.ndarray:
    """Anti-aliased lines where q (mm) sits on a multiple of ``pitch`` along either in-plane axis."""
    d = np.min([np.abs(q[..., a] / pitch - np.round(q[..., a] / pitch)) for a in axes], axis=0) * pitch / step
    return np.clip(width - d, 0, 1)


def tissue(after: np.ndarray, step: float) -> np.ndarray:
    """Where the moved MRI has data, 1.5 mm wider: outside it the B-spline only extrapolates."""
    return binary_fill_holes(binary_dilation(after > 0.02, iterations=round(1.5 / step)))


def warped_grid(background: np.ndarray, after: np.ndarray, u: np.ndarray, slab: sitk.Image, pitch: float,
                top: float, level: float, width: int) -> np.ndarray:
    """The lattice of the affine-aligned space, thin and dark where it started and in colour where the deformable
    carried it: a line that was at q lands on the voxel p that reads from q = p + u(p), which is how the MRI
    itself is resampled. Where nothing moved the two coincide."""
    step = slab.GetSpacing()[0]
    axes = [a for a, n in enumerate(slab.GetSize()) if n > 1]
    p, inside = points(slab), tissue(after, step)
    was = (lattice(p, axes, pitch, step, width=1.0) * inside)[..., None]
    lines = (lattice(p + u, axes, pitch, step) * inside)[..., None]
    t = np.clip(np.linalg.norm(u, axis=-1) / top, 0, 1)
    colour = np.stack([np.interp(t, [0, 0.5, 1], STOPS[:, c]) for c in range(3)], -1)
    canvas = np.dstack([grey(background)] * 3) * 0.75 * (1 - was) + WAS * was
    canvas = canvas * (1 - lines) + colour * lines
    canvas[contour(after, level, width)] = GOLD
    return canvas


def top_of_scale(panels: list[tuple]) -> float:
    """The 99th percentile of the displacement inside the tissue, over the panels, to the half millimetre above."""
    lengths = [np.linalg.norm(u, axis=-1)[tissue(after, slab.GetSpacing()[0])] for _, after, u, slab in panels]
    return float(np.ceil(2 * np.percentile(np.concatenate(lengths), 99)) / 2)


def main() -> None:
    assert lattice(np.array([[[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]]), [0, 1], 2.0, 0.1).tolist() == [[1.0, 0.0]]
    FIG.mkdir(exist_ok=True)
    coarse_field = ROOT / "out" / "linc_coarse_bend100" / "P000" / "Transform.h5"

    # level 4, one coronal plane of the whole hemisphere, lattice every 4 mm
    fixed_image = sitk.ReadImage(str(DATA / "XPCT_level4.mha"))
    fixed_image.SetDirection(LPS)
    fixed = sitk.GetArrayFromImage(normalise(fixed_image))
    deformable = read_zarr(ROOT / "out" / "linc_coarse_bend100" / "P000" / "Moved.ome.zarr")
    y = fixed.shape[1] // 2
    level = float(np.median(deformable[deformable > 0.02]))
    slab = fixed_image[:, y : y + 1, :]
    u = displacement(linc.field_in_store_frame(coarse_field, slab), slab)
    top = top_of_scale([("", deformable[:, y], u, slab)])
    save(bands([(f"level 4 · 4 mm grid · grey still · red {top:g} mm",
                 [warped_grid(fixed[:, y], deformable[:, y], u, slab, 4.0, top, level, width=1)])], size=16),
         "B_grid_plane.png", scale=1.6)

    # 40 mm over the deep nuclei at 0.1 mm, axial and coronal, lattice every 4 mm (2 mm is a mesh at this move)
    consortium_seg = sitk.ReadImage(str(DATA / "dseg_spaceXPCT.nii.gz"))          # only to centre the window
    where = np.argwhere(np.isin(sitk.GetArrayFromImage(consortium_seg), (12, 13)))
    cx, cy, cz = consortium_seg.TransformContinuousIndexToPhysicalPoint(where.mean(0)[::-1].tolist())
    step, half = 0.1, 200
    xpct3 = sitk.ReadImage(str(DATA / "XPCT_level3.mha"))
    xpct3.SetDirection(LPS)
    source = normalise(sitk.ReadImage(str(DATA / "dmri_oriented.mha")))
    affine_t = sitk.ReadTransform(str(DATA / "affine.tfm"))
    panels = []
    for name, size, origin in (("axial", (2 * half, 2 * half, 1), (cx + half * step, cy + half * step, cz)),
                               ("coronal", (2 * half, 1, 2 * half), (cx + half * step, cy, cz - half * step))):
        slab = sitk.Image(size, sitk.sitkFloat32)
        slab.SetSpacing((step,) * 3)
        slab.SetDirection(LPS)
        slab.SetOrigin(origin)
        field = linc.field_in_store_frame(coarse_field, slab)
        chain = sitk.CompositeTransform(3)
        chain.AddTransform(affine_t)
        chain.AddTransform(field)
        background = sitk.GetArrayFromImage(sitk.Resample(xpct3, slab, sitk.Transform(), sitk.sitkLinear, 0.0)).squeeze()
        after = sitk.GetArrayFromImage(sitk.Resample(source, slab, chain, sitk.sitkLinear, 0.0)).squeeze()
        panels.append((name, background, after, displacement(field, slab), slab))
    top = top_of_scale([p[1:] for p in panels])
    rows = []
    for name, background, after, u, slab in panels:
        level = float(np.median(after[after > 0.02]))
        rows.append((f"{name} · 4 mm grid · grey still · red {top:g} mm",
                     [warped_grid(background, after, u, slab, 4.0, top, level, width=2)]))
    save(bands(rows, size=16), "B_grid_deep.png", scale=1.2)


if __name__ == "__main__":
    main()
