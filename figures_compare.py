"""Affine against deformable on ONE image: both contours over the XPCT, blue after the affine and gold after the
deformable, so the gap between the two lines is the move; and the same two states as a flicker.

    python figures_compare.py
"""

import numpy as np
import SimpleITK as sitk
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter

import linc
from figures_linc import DATA, FIG, ROOT, bands, grey, read_zarr, save
from stage1_linc import LPS, normalise

BLUE, GOLD = (0.35, 0.65, 1.0), (1.0, 0.85, 0.1)


def contour(image: np.ndarray, level: float, width: int = 1) -> np.ndarray:
    inside = gaussian_filter(image, 2) > level
    return binary_dilation(inside ^ binary_dilation(inside), np.ones((3, 3)), iterations=width)


def both_over(fixed: np.ndarray, affine: np.ndarray, deformable: np.ndarray, level: float, width: int = 1) -> np.ndarray:
    canvas = np.dstack([grey(fixed)] * 3) * 0.85
    canvas[contour(affine, level, width)] = BLUE
    canvas[contour(deformable, level, width)] = GOLD
    return canvas


def one_over(fixed: np.ndarray, moving: np.ndarray, level: float, width: int = 1) -> np.ndarray:
    canvas = np.dstack([grey(fixed)] * 3) * 0.85
    canvas[contour(moving, level, width)] = GOLD
    return canvas


def flicker(name: str, frames: list[tuple[str, np.ndarray]], scale: float = 1.0, ms: int = 900) -> None:
    images = []
    for label, array in frames:
        image = Image.fromarray((np.clip(bands([(label, [array])], size=20), 0, 1) * 255).astype(np.uint8))
        if scale != 1.0:
            image = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)
        images.append(image)
    images[0].save(FIG / name, save_all=True, append_images=images[1:], duration=ms, loop=0)
    print(f"  {name}  {images[0].width} x {images[0].height}, {len(images)} frames")


def main() -> None:
    FIG.mkdir(exist_ok=True)
    # level 4, one coronal plane of the whole hemisphere
    fixed_image = sitk.ReadImage(str(DATA / "XPCT_level4.mha"))
    fixed_image.SetDirection(LPS)
    fixed = sitk.GetArrayFromImage(normalise(fixed_image))
    affine = read_zarr(DATA / "pair" / "Moving.ome.zarr")
    deformable = read_zarr(ROOT / "out" / "linc_coarse_bend100" / "P000" / "Moved.ome.zarr")
    y = fixed.shape[1] // 2
    level = float(np.median(deformable[deformable > 0.02]))
    plane = (fixed[:, y], affine[:, y], deformable[:, y])
    save(bands([("level 4 · blue: affine · gold: deformable", [both_over(*plane, level)])],
               size=20), "D11_plane_both.png", scale=1.6)
    flicker("D12_flicker_plane.gif", [("after the affine", one_over(plane[0], plane[1], level)),
                                      ("after the deformable", one_over(plane[0], plane[2], level))], scale=1.6)

    # 40 mm over the deep nuclei at 0.1 mm, axial and coronal
    consortium_seg = sitk.ReadImage(str(DATA / "dseg_spaceXPCT.nii.gz"))          # only to centre the window
    where = np.argwhere(np.isin(sitk.GetArrayFromImage(consortium_seg), (12, 13)))
    cx, cy, cz = consortium_seg.TransformContinuousIndexToPhysicalPoint(where.mean(0)[::-1].tolist())
    step, half = 0.1, 200
    xpct3 = sitk.ReadImage(str(DATA / "XPCT_level3.mha"))
    xpct3.SetDirection(LPS)
    source = normalise(sitk.ReadImage(str(DATA / "dmri_oriented.mha")))
    affine_t = sitk.ReadTransform(str(DATA / "affine.tfm"))
    coarse_field = ROOT / "out" / "linc_coarse_bend100" / "P000" / "Transform.h5"
    rows, frames = [], []
    for name, size, origin in (("axial", (2 * half, 2 * half, 1), (cx + half * step, cy + half * step, cz)),
                               ("coronal", (2 * half, 1, 2 * half), (cx + half * step, cy, cz - half * step))):
        slab = sitk.Image(size, sitk.sitkFloat32)
        slab.SetSpacing((step,) * 3)
        slab.SetDirection(LPS)
        slab.SetOrigin(origin)
        chain = sitk.CompositeTransform(3)
        chain.AddTransform(affine_t)
        chain.AddTransform(linc.field_in_store_frame(coarse_field, slab))

        def on(image: sitk.Image, transform: sitk.Transform, slab: sitk.Image = slab) -> np.ndarray:
            return sitk.GetArrayFromImage(sitk.Resample(image, slab, transform, sitk.sitkLinear, 0.0)).squeeze()

        background, before, after = on(xpct3, sitk.Transform()), on(source, affine_t), on(source, chain)
        level = float(np.median(after[after > 0.02]))
        rows.append((f"{name} · blue: affine · gold: deformable",
                     [both_over(background, before, after, level, width=2)]))
        if name == "axial":
            frames = [("after the affine", one_over(background, before, level, width=2)),
                      ("after the deformable", one_over(background, after, level, width=2))]
    save(bands(rows, size=20), "D11_deep_both.png", scale=1.2)
    flicker("D12_flicker_deep.gif", frames, scale=1.2)


if __name__ == "__main__":
    main()
