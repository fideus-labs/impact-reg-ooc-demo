"""The deformable's move as arrows: each arrow goes from where the affine had left a piece of MRI to where the
deformable put it, coloured by its length, over the XPCT with both contours faint (blue affine, gold deformable).

    python figures_arrows.py
"""

import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, gaussian_filter

import linc
from figures_compare import BLUE, GOLD
from figures_linc import DATA, FIG, ROOT, bands, grey, read_zarr, save
from stage1_linc import LPS, normalise
from video import font

S = 4                                                                 # drawn at 4x, saved at 3x (anti-aliased)
RAMP = ((0.0, (0.55, 0.60, 0.70)), (1.5, (1.0, 0.80, 0.20)), (3.0, (1.0, 0.30, 0.05)))   # mm -> muted, yellow, warm


def colour(length_mm: float) -> tuple[int, int, int]:
    stops = [m for m, _ in RAMP]
    return tuple(int(255 * np.interp(length_mm, stops, [c[k] for _, c in RAMP])) for k in range(3))


def faint_both(fixed: np.ndarray, affine: np.ndarray, deformable: np.ndarray, level: float) -> np.ndarray:
    """Both contours as one-voxel rings, half blended into the XPCT: the arrows have to read first."""
    canvas = np.dstack([grey(fixed)] * 3) * 0.85
    for image, col in ((affine, BLUE), (deformable, GOLD)):
        inside = gaussian_filter(image, 2) > level
        ring = inside ^ binary_dilation(inside)
        canvas[ring] = 0.5 * canvas[ring] + 0.5 * np.array(col)
    return canvas


def arrow(draw: ImageDraw.ImageDraw, tail, head, col, width: int) -> None:
    d = np.subtract(head, tail)
    n = float(np.hypot(*d))
    if n < 1.5 * width:                                               # did not move: a dot
        draw.ellipse([head[0] - width, head[1] - width, head[0] + width, head[1] + width], fill=col)
        return
    u = d / n
    size = min(0.45 * n, 3.5 * width)
    base = np.subtract(head, u * size)
    perp = np.array([-u[1], u[0]]) * size * 0.5
    draw.line([tuple(tail), tuple(base)], fill=col, width=width)
    draw.polygon([tuple(head), tuple(base + perp), tuple(base - perp)], fill=col)


def samples(slab: sitk.Image, field: sitk.Transform, mask: np.ndarray, step_px: int):
    """(row, col) of every step_px-th tissue voxel of the plane, and the (row, col) the affine had it at."""
    axes = [a for a in (2, 1, 0) if slab.GetSize()[a] > 1]            # the plane's (row, col) axes in index space
    for r in range(step_px // 2, mask.shape[0], step_px):
        for c in range(step_px // 2, mask.shape[1], step_px):
            if not mask[r, c]:
                continue
            index = [0, 0, 0]
            index[axes[0]], index[axes[1]] = r, c
            p = slab.TransformIndexToPhysicalPoint(index)
            q = slab.TransformPhysicalPointToContinuousIndex(field.TransformPoint(p))   # where the affine had it
            yield (r, c), (q[axes[0]], q[axes[1]])


def arrows_over(canvas: np.ndarray, slab: sitk.Image, field: sitk.Transform, mask: np.ndarray, step_mm: float) -> np.ndarray:
    spacing = slab.GetSpacing()[0]
    image = Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8)).resize(
        (canvas.shape[1] * S, canvas.shape[0] * S), Image.NEAREST)
    draw = ImageDraw.Draw(image)
    width = max(3, round(0.06 / spacing * S))                         # about 60 um wide, 3 px at least
    lengths = []
    for (r, c), (tr, tc) in samples(slab, field, mask, round(step_mm / spacing)):
        length = float(np.hypot(tr - r, tc - c) * spacing)
        lengths.append(length)
        arrow(draw, ((tc + 0.5) * S, (tr + 0.5) * S), ((c + 0.5) * S, (r + 0.5) * S), colour(length), width)
    print(f"  {len(lengths)} arrows, in-plane median {np.median(lengths):.2f} mm, max {np.max(lengths):.2f} mm")

    # legend: a 2 mm reference arrow and the colour ramp
    ref, pad, text = 2.0 / spacing * S, 6 * S, font(9 * S)
    x0, y0 = pad, image.height - pad - 15 * S
    draw.rectangle([x0 - 3 * S, y0 - 3 * S, x0 + ref + 30 * S, image.height - pad + 3 * S], fill=(12, 16, 22))
    arrow(draw, (x0, y0), (x0 + ref, y0), (255, 255, 255), width)
    draw.text((x0 + ref + 3 * S, y0), "2 mm", font=text, fill=(231, 237, 243), anchor="lm")
    y1 = y0 + 8 * S
    for i in range(int(ref)):
        draw.line([(x0 + i, y1 - S), (x0 + i, y1 + S)], fill=colour(3.0 * i / ref))
    draw.text((x0 + ref + 3 * S, y1), "0 → 3 mm", font=text, fill=(231, 237, 243), anchor="lm")
    return np.asarray(image) / 255.0


def main() -> None:
    FIG.mkdir(exist_ok=True)
    coarse_field = ROOT / "out" / "linc_coarse_bend100" / "P000" / "Transform.h5"

    # level 4, one coronal plane of the whole hemisphere, an arrow every 4 mm
    fixed_image = sitk.ReadImage(str(DATA / "XPCT_level4.mha"))
    fixed_image.SetDirection(LPS)
    fixed = sitk.GetArrayFromImage(normalise(fixed_image))
    affine = read_zarr(DATA / "pair" / "Moving.ome.zarr")
    deformable = read_zarr(ROOT / "out" / "linc_coarse_bend100" / "P000" / "Moved.ome.zarr")
    y = fixed.shape[1] // 2
    level = float(np.median(deformable[deformable > 0.02]))
    slab = fixed_image[:, y : y + 1, :]
    panel = arrows_over(faint_both(fixed[:, y], affine[:, y], deformable[:, y], level), slab,
                        linc.field_in_store_frame(coarse_field, slab), gaussian_filter(deformable[:, y], 2) > 0.02, 4.0)
    save(bands([("level 4 · arrows: affine → deformable", [panel])], size=44), "C_arrows_plane.png", scale=0.75)

    # 40 mm over the deep nuclei at 0.1 mm, axial and coronal, an arrow every 2 mm
    consortium_seg = sitk.ReadImage(str(DATA / "dseg_spaceXPCT.nii.gz"))          # only to centre the window
    where = np.argwhere(np.isin(sitk.GetArrayFromImage(consortium_seg), (12, 13)))
    cx, cy, cz = consortium_seg.TransformContinuousIndexToPhysicalPoint(where.mean(0)[::-1].tolist())
    step, half = 0.1, 200
    xpct3 = sitk.ReadImage(str(DATA / "XPCT_level3.mha"))
    xpct3.SetDirection(LPS)
    source = normalise(sitk.ReadImage(str(DATA / "dmri_oriented.mha")))
    affine_t = sitk.ReadTransform(str(DATA / "affine.tfm"))
    panels = []
    for size, origin in (((2 * half, 2 * half, 1), (cx + half * step, cy + half * step, cz)),
                         ((2 * half, 1, 2 * half), (cx + half * step, cy, cz - half * step))):
        slab = sitk.Image(size, sitk.sitkFloat32)
        slab.SetSpacing((step,) * 3)
        slab.SetDirection(LPS)
        slab.SetOrigin(origin)
        field = linc.field_in_store_frame(coarse_field, slab)
        chain = sitk.CompositeTransform(3)
        chain.AddTransform(affine_t)
        chain.AddTransform(field)

        def on(image: sitk.Image, transform: sitk.Transform, slab: sitk.Image = slab) -> np.ndarray:
            return sitk.GetArrayFromImage(sitk.Resample(image, slab, transform, sitk.sitkLinear, 0.0)).squeeze()

        background, before, after = on(xpct3, sitk.Transform()), on(source, affine_t), on(source, chain)
        level = float(np.median(after[after > 0.02]))
        panels.append(arrows_over(faint_both(background, before, after, level), slab, field,
                                  gaussian_filter(after, 2) > 0.02, 2.0))
    save(bands([("axial · coronal · arrows: affine → deformable", panels)], size=44), "C_arrows_deep.png", scale=0.75)


if __name__ == "__main__":
    main()
