"""A morph from the affine state to the deformable one: the MRI's contour (gold) over the XPCT while the
field goes from 0 to its full length, held at both ends and played back and forth, so the eye sees the
contour slide onto the anatomy instead of comparing two stills.

    python figures_morph.py
"""

import numpy as np
import SimpleITK as sitk
from PIL import Image

import linc
from figures_compare import one_over
from figures_linc import DATA, FIG, ROOT, bands, read_zarr
from stage1_linc import LPS, normalise

STEPS, MS, HOLD_MS = 14, 80, 1000
FIELD = ROOT / "out" / "linc_coarse_bend100" / "P000" / "Transform.h5"


def label(t: float) -> str:
    n = round(10 * t)
    return f"affine {'▸' * n}{'▹' * (10 - n)} deformable"


def states(slab: sitk.Image, source: sitk.Image, affine: sitk.Transform, ts: np.ndarray) -> list[np.ndarray]:
    """The source MRI on the slab through the affine then t times the field, one plane per t."""
    field = linc.field_in_store_frame(FIELD, slab, margin_mm=2.0).GetDisplacementField()
    vectors = sitk.GetArrayFromImage(field)
    out = []
    for t in ts:
        scaled = sitk.GetImageFromArray(vectors * t, isVector=True)
        scaled.CopyInformation(field)
        chain = sitk.CompositeTransform(3)
        chain.AddTransform(affine)
        chain.AddTransform(sitk.DisplacementFieldTransform(scaled))
        out.append(sitk.GetArrayFromImage(sitk.Resample(source, slab, chain, sitk.sitkLinear, 0.0)).squeeze())
    return out


def morph(name: str, frames: list[tuple[str, np.ndarray]], scale: float) -> None:
    """Frames 0..n-1 then n-2..1 (ping-pong), the two ends held; one shared palette so the GIF stores only
    what moves between frames."""
    images = []
    for text, array in frames:
        image = Image.fromarray((np.clip(bands([(text, [array])], size=20), 0, 1) * 255).astype(np.uint8))
        if scale != 1.0:
            image = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)
        images.append(image)
    order = list(range(len(images))) + list(range(len(images) - 2, 0, -1))
    palette = images[-1].quantize(254)
    quantised = [images[i].quantize(palette=palette, dither=Image.Dither.NONE) for i in order]
    durations = [HOLD_MS if i in (0, len(images) - 1) else MS for i in order]
    quantised[0].save(FIG / name, save_all=True, append_images=quantised[1:], duration=durations, loop=0, disposal=1)
    size = (FIG / name).stat().st_size
    print(f"  {name}  {images[0].width} x {images[0].height}, {len(order)} frames, {size / 2**20:.2f} MB")
    assert size < 4 * 2**20, f"{name} is over 4 MB"


def main() -> None:
    FIG.mkdir(exist_ok=True)
    ts = np.linspace(0, 1, STEPS)
    source = normalise(sitk.ReadImage(str(DATA / "dmri_oriented.mha")))
    affine = sitk.ReadTransform(str(DATA / "affine.tfm"))

    # 40 mm over the deep nuclei at 0.1 mm, axial
    consortium_seg = sitk.ReadImage(str(DATA / "dseg_spaceXPCT.nii.gz"))          # only to centre the window
    where = np.argwhere(np.isin(sitk.GetArrayFromImage(consortium_seg), (12, 13)))
    cx, cy, cz = consortium_seg.TransformContinuousIndexToPhysicalPoint(where.mean(0)[::-1].tolist())
    step, half = 0.1, 200
    slab = sitk.Image((2 * half, 2 * half, 1), sitk.sitkFloat32)
    slab.SetSpacing((step,) * 3)
    slab.SetDirection(LPS)
    slab.SetOrigin((cx + half * step, cy + half * step, cz))
    xpct3 = sitk.ReadImage(str(DATA / "XPCT_level3.mha"))
    xpct3.SetDirection(LPS)
    background = sitk.GetArrayFromImage(sitk.Resample(xpct3, slab, sitk.Transform(), sitk.sitkLinear, 0.0)).squeeze()
    moved = states(slab, source, affine, ts)
    level = float(np.median(moved[-1][moved[-1] > 0.02]))
    morph("A_morph_deep.gif", [(label(t), one_over(background, m, level, width=1)) for t, m in zip(ts, moved)], scale=1.2)

    # level 4, one coronal plane of the whole hemisphere
    fixed_image = sitk.ReadImage(str(DATA / "XPCT_level4.mha"))
    fixed_image.SetDirection(LPS)
    fixed = sitk.GetArrayFromImage(normalise(fixed_image))
    y = fixed.shape[1] // 2
    deformable = read_zarr(ROOT / "out" / "linc_coarse_bend100" / "P000" / "Moved.ome.zarr")
    level = float(np.median(deformable[deformable > 0.02]))
    moved = states(fixed_image[:, y : y + 1, :], source, affine, ts)
    # the two ends of the morph are the two stored states
    for ours, stored, name in ((moved[0], read_zarr(DATA / "pair" / "Moving.ome.zarr")[:, y], "affine"),
                               (moved[-1], deformable[:, y], "deformable")):
        r = np.corrcoef(ours.ravel(), stored.ravel())[0, 1]
        print(f"  {name}: correlation with the stored state {r:.4f}")
        assert r > 0.95, name
    morph("A_morph_plane.gif", [(label(t), one_over(fixed[:, y], m, level, width=2)) for t, m in zip(ts, moved)],
          scale=1.6)


if __name__ == "__main__":
    main()
