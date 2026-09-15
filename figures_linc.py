"""The figures for the LINC pair: X-ray phase contrast against diffusion MRI, one human hemisphere.

    python figures_linc.py
"""

from pathlib import Path

import numpy as np
import SimpleITK as sitk
import zarr
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter, sobel

from stage1_linc import LPS, normalise

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "linc"
FIG = ROOT / "figures"
XPCT_COLOUR, MRI_COLOUR = (1.0, 0.32, 0.22), (0.35, 1.0, 0.45)


def read_zarr(path: Path) -> np.ndarray:
    group = zarr.open_group(str(path), mode="r")
    array = group[dict(group.attrs)["multiscales"][0]["datasets"][0]["path"]]
    return np.asarray(array[0] if array.ndim == 4 else array)


def grey(volume: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    lo, hi = np.percentile(volume, [1, 99.5])
    return np.clip((volume - lo) / (hi - lo + 1e-9), 0, 1) ** gamma


def three_views(volume: np.ndarray) -> list[np.ndarray]:
    return [np.take(volume, volume.shape[axis] // 2, axis=axis) for axis in (0, 1, 2)]


def strip(panels: list[np.ndarray], colour=None) -> np.ndarray:
    """Lay planes side by side, padded to a common height."""
    height = max(p.shape[0] for p in panels)
    width = sum(p.shape[1] for p in panels)
    channels = 3 if panels[0].ndim == 3 else 1
    canvas = np.zeros((height, width, 3) if channels == 3 else (height, width))
    x = 0
    for panel in panels:
        canvas[: panel.shape[0], x : x + panel.shape[1]] = panel
        x += panel.shape[1]
    return canvas


def overlay(fixed: np.ndarray, moving: np.ndarray) -> np.ndarray:
    return np.dstack([grey(fixed), grey(moving), np.zeros(fixed.shape[-2:])])


def save(array: np.ndarray, name: str, scale: float = 1.0) -> None:
    FIG.mkdir(exist_ok=True)
    image = Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8))
    if scale != 1.0:
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)
    image.save(FIG / name)
    print(f"  {name}  {image.width} x {image.height}")


def iso_contours_over(fixed: np.ndarray, moving: np.ndarray, level: float, reference: np.ndarray | None = None) -> np.ndarray:
    """One intensity level of the moving image (gold) drawn over the fixed one, and the same level of a
    reference alignment of that same image (cyan): panels then differ only by where the level sits."""
    canvas = np.dstack([grey(fixed)] * 3) * 0.85
    for image, colour in ((reference, (0.25, 0.85, 1.0)), (moving, (1.0, 0.85, 0.1))):
        if image is None:
            continue
        inside = gaussian_filter(image, 2) > level
        canvas[binary_dilation(inside ^ binary_dilation(inside), np.ones((3, 3)))] = colour
    return canvas


def contours_over(fixed: np.ndarray, moving: np.ndarray) -> np.ndarray:
    smooth = gaussian_filter(grey(moving), 2)
    magnitude = np.hypot(sobel(smooth, 0), sobel(smooth, 1))
    edges = binary_dilation(magnitude > np.percentile(magnitude, 98), np.ones((2, 2)))
    canvas = np.dstack([grey(fixed)] * 3) * 0.85
    canvas[edges] = (1.0, 0.85, 0.1)
    return canvas


def checkerboard(first: np.ndarray, second: np.ndarray, tile: int) -> np.ndarray:
    """Squares taken alternately from each image, each on its own grey scale: structures continue or step."""
    y, x = np.mgrid[0 : first.shape[0], 0 : first.shape[1]]
    picked = np.where(((y // tile + x // tile) % 2) == 0, grey(first), grey(second))
    return np.dstack([picked] * 3)


def agreement(ours: np.ndarray, consortium: np.ndarray) -> np.ndarray:
    """Two alignments of the same map, ours in green and the consortium's in magenta: grey where they agree."""
    a, b = grey(ours), grey(consortium)
    return np.dstack([b, a, b])


def bands(rows, size: int = 38) -> np.ndarray:
    """Rows of panels stacked, each under a dark band carrying its label."""
    from PIL import ImageDraw

    from video import font

    stacked = []
    for label, panels in rows:
        row = strip(panels)
        band = Image.new("RGB", (row.shape[1], size + 26), (12, 16, 22))
        ImageDraw.Draw(band).text((14, 10), label, font=font(size), fill=(231, 237, 243))
        stacked += [np.asarray(band) / 255.0, row]
    return np.vstack(stacked)


def qc_rows(planes, tile: int) -> np.ndarray:
    """One labelled row per (label, XPCT, this MRI, the consortium's MRI): the XPCT | MRI checkerboard,
    this MRI over the consortium's, and the checkerboard of the two MRI alignments."""
    return bands([(label, [checkerboard(fixed, moving, tile), agreement(moving, consortium), checkerboard(moving, consortium, tile)])
                  for label, fixed, moving, consortium in planes])


def outlines(background: np.ndarray, ours: np.ndarray, consortium: np.ndarray, names: dict[int, str]) -> np.ndarray:
    """Each structure's outline over the background, this pipeline in gold and the consortium in cyan, named where it sits."""
    from PIL import ImageDraw

    from video import font

    canvas = np.dstack([grey(background)] * 3) * 0.8
    for segmentation, colour in ((consortium, (0.25, 0.85, 1.0)), (ours, (1.0, 0.85, 0.1))):
        for label in names:
            inside = segmentation == label
            canvas[binary_dilation(inside, np.ones((3, 3)), iterations=2) & ~inside] = colour
    image = Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for label, name in names.items():
        where = np.argwhere(consortium == label)
        if len(where) > 200:
            y, x = where.mean(0)
            draw.text((x, y), name, font=font(20), fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0), anchor="mm")
    return np.asarray(image) / 255.0


def main() -> None:
    fixed_image = sitk.ReadImage(str(DATA / "XPCT_level4.mha"))
    fixed_image.SetDirection(LPS)
    fixed = sitk.GetArrayFromImage(normalise(fixed_image))
    affine = read_zarr(DATA / "pair" / "Moving.ome.zarr")
    deformable = read_zarr(ROOT / "out" / "linc_coarse_bend100" / "P000" / "Moved.ome.zarr")
    reference = sitk.GetArrayFromImage(sitk.Resample(
        normalise(sitk.ReadImage(str(DATA / "dmri_reference.nii.gz"))), normalise(fixed_image),
        sitk.Transform(), sitk.sitkLinear, 0.0))

    # 1 . the two modalities, three views each
    mri_native = sitk.GetArrayFromImage(normalise(sitk.ReadImage(str(DATA / "dmri.nii.gz"))))
    top = strip([np.dstack([grey(p)] * 3) for p in three_views(fixed)])
    bottom = strip([np.dstack([grey(p)] * 3) for p in three_views(mri_native)])
    width = max(top.shape[1], bottom.shape[1])
    stacked = np.zeros((top.shape[0] + bottom.shape[0], width, 3))
    stacked[: top.shape[0], : top.shape[1]] = top
    stacked[top.shape[0] :, : bottom.shape[1]] = bottom
    save(stacked, "L1_modalities.png", scale=0.8)

    # 2 . the three states against the reference
    rows = [strip([overlay(f, m) for f, m in zip(three_views(fixed), three_views(volume))])
            for volume in (affine, deformable, reference)]
    save(np.vstack(rows), "L2_affine_deformable_reference.png", scale=0.55)

    # 3 . the readable check, on the coronal plane
    plane = fixed.shape[1] // 2
    save(np.hstack([contours_over(fixed[:, plane], affine[:, plane]),
                    contours_over(fixed[:, plane], deformable[:, plane])]), "L3_contours.png")

    # 6 . checkerboards and overlays, level 4: after the affine and after the deformable, on two planes
    z, y = fixed.shape[0] // 2, fixed.shape[1] // 2
    save(qc_rows([
        ("axial, after the affine", fixed[z], affine[z], reference[z]),
        ("axial, after the regularised deformable", fixed[z], deformable[z], reference[z]),
        ("coronal, after the affine", fixed[:, y], affine[:, y], reference[:, y]),
        ("coronal, after the regularised deformable", fixed[:, y], deformable[:, y], reference[:, y]),
    ], tile=32), "L7_qc_level4.png", scale=0.7)

    # 4 . where the data was read from
    tiled = ROOT / "out" / "linc_tiled3_bend100" / "P000" / "Moved.ome.zarr"
    if tiled.exists():
        level3 = read_zarr(ROOT / "data" / "linc" / "pair3_bend" / "Fixed.ome.zarr")
        moved = read_zarr(tiled)
        rows = [strip([overlay(f, m) for f, m in zip(three_views(level3), three_views(v))])
                for v in (read_zarr(ROOT / "data" / "linc" / "pair3_bend" / "Moving.ome.zarr"), moved)]
        save(np.vstack(rows), "L4_tiled.png", scale=0.45)

    # 7 . the same, level 3: before and after the tiled run, on two planes
    from stage4_native_linc import written_window

    tiled3 = ROOT / "out" / "linc_tiled3_bend100" / "P000" / "Moved.ome.zarr"
    if tiled3.exists():
        pair3 = DATA / "pair3_bend"
        image3 = written_window(pair3 / "Fixed.ome.zarr")      # the level-3 grid, back in the store's frame
        k, j = image3.GetSize()[2] // 2, image3.GetSize()[1] // 2
        consortium3 = normalise(sitk.ReadImage(str(DATA / "dmri_reference.nii.gz")))

        def zarr_plane(path: Path, index) -> np.ndarray:
            group = zarr.open_group(str(path), mode="r")
            array = group[dict(group.attrs)["multiscales"][0]["datasets"][0]["path"]]
            return np.asarray(array[(0, *index)] if array.ndim == 4 else array[index])

        planes3 = []
        for name, slab, index in (("axial", image3[:, :, k : k + 1], np.s_[k, :, :]),
                                  ("coronal", image3[:, j : j + 1, :], np.s_[:, j, :])):
            fixed3 = sitk.GetArrayFromImage(slab).squeeze()
            reference3 = sitk.GetArrayFromImage(sitk.Resample(consortium3, slab, sitk.Transform(), sitk.sitkLinear, 0.0)).squeeze()
            planes3 += [(f"{name}, before the tiled run", fixed3, zarr_plane(pair3 / "Moving.ome.zarr", index), reference3),
                        (f"{name}, after the tiled run", fixed3, zarr_plane(tiled3, index), reference3)]
        save(qc_rows(planes3, tile=64), "L8_qc_level3.png", scale=0.45)

    # 5 . the native window at 20 um: the MRI's edges over the XPCT, before and after the tiled run
    from evaluate_linc import NATIVE_RUN, native_states

    if (NATIVE_RUN / "Transform.h5").exists():
        window = written_window(DATA / "pair0" / "Fixed.ome.zarr")
        z = window.GetSize()[2] // 2
        plane = window[:, :, z : z + 1]
        start, after = native_states(plane)
        fixed_plane = sitk.GetArrayFromImage(plane)[0]
        # the consortium's alignment of the same MRI map, at the same level: gold should close on cyan
        reference = sitk.GetArrayFromImage(sitk.Resample(normalise(sitk.ReadImage(str(DATA / "dmri_reference.nii.gz"))),
                                                         plane, sitk.Transform(), sitk.sitkLinear, 0.0))[0]
        level = float(np.median(reference[reference > 0.02]))
        save(np.hstack([iso_contours_over(fixed_plane, start[0], level, reference),
                        iso_contours_over(fixed_plane, after[0], level, reference)]), "L6_native_contours.png", scale=0.6)

        # 8 . the same, native: before and after the native tiles, on two perpendicular planes
        j = window.GetSize()[1] // 2
        across = window[:, j : j + 1, :]
        before_across, after_across = native_states(across)
        fixed_across = sitk.GetArrayFromImage(across).squeeze()
        reference_across = sitk.GetArrayFromImage(sitk.Resample(
            normalise(sitk.ReadImage(str(DATA / "dmri_reference.nii.gz"))), across, sitk.Transform(), sitk.sitkLinear, 0.0)).squeeze()
        save(qc_rows([
            ("one plane, before the native tiles", fixed_plane, start[0], reference),
            ("one plane, after the native tiles", fixed_plane, after[0], reference),
            ("the perpendicular plane, before", fixed_across, before_across.squeeze(), reference_across),
            ("the perpendicular plane, after", fixed_across, after_across.squeeze(), reference_across),
        ], tile=96), "L9_qc_native.png", scale=0.5)


    # 9 . closer: the deep grey nuclei, after the affine and after the level-4 deformable
    flash_field = ROOT / "out" / "linc_flash_bend100" / "P000" / "Transform.h5"
    coarse_field = ROOT / "out" / "linc_coarse_bend100" / "P000" / "Transform.h5"
    if flash_field.exists() and coarse_field.exists():
        import linc
        from evaluate_linc import carried_segmentation

        consortium_seg = sitk.ReadImage(str(DATA / "dseg_spaceXPCT.nii.gz"))
        where = np.argwhere(np.isin(sitk.GetArrayFromImage(consortium_seg), (12, 13)))   # putamen and pallidum
        cx, cy, cz = consortium_seg.TransformContinuousIndexToPhysicalPoint(where.mean(0)[::-1].tolist())
        step, half = 0.1, 200                                                             # 40 mm at 0.1 mm
        xpct3 = sitk.ReadImage(str(DATA / "XPCT_level3.mha"))
        xpct3.SetDirection(LPS)
        _, oriented, flash_affine = carried_segmentation("flash", DATA / "flash_brain_024.nii.gz")
        source = normalise(sitk.ReadImage(str(DATA / "dmri_oriented.mha")))
        consortium_mri = normalise(sitk.ReadImage(str(DATA / "dmri_reference.nii.gz")))
        dmri_affine = sitk.ReadTransform(str(DATA / "affine.tfm"))
        names = {12: "putamen", 13: "pallidum", 10: "thalamus", 11: "caudate"}
        rows = []
        for plane, size, origin in (("axial", (2 * half, 2 * half, 1), (cx + half * step, cy + half * step, cz)),
                                    ("coronal", (2 * half, 1, 2 * half), (cx + half * step, cy, cz - half * step))):
            slab = sitk.Image(size, sitk.sitkFloat32)
            slab.SetSpacing((step,) * 3)
            slab.SetDirection(LPS)
            slab.SetOrigin(origin)

            def on(image: sitk.Image, transform: sitk.Transform, labels: bool = False) -> np.ndarray:
                interpolator = sitk.sitkNearestNeighbor if labels else sitk.sitkLinear
                return sitk.GetArrayFromImage(sitk.Resample(image, slab, transform, interpolator, 0.0)).squeeze()

            flash_chain = sitk.CompositeTransform(3)
            flash_chain.AddTransform(flash_affine)
            flash_chain.AddTransform(linc.field_in_store_frame(flash_field, slab))
            dmri_chain = sitk.CompositeTransform(3)
            dmri_chain.AddTransform(dmri_affine)
            dmri_chain.AddTransform(linc.field_in_store_frame(coarse_field, slab))
            background = on(xpct3, sitk.Transform())
            reference_seg, reference_mri = on(consortium_seg, sitk.Transform(), True), on(consortium_mri, sitk.Transform())
            rows.append((f"{plane} · outlines: gold this pipeline, cyan consortium",
                         [outlines(background, on(oriented, flash_affine, True), reference_seg, names),
                          outlines(background, on(oriented, flash_chain, True), reference_seg, names)]))
            rows.append((f"{plane} · MRI: green this pipeline, magenta consortium",
                         [agreement(on(source, dmri_affine), reference_mri), agreement(on(source, dmri_chain), reference_mri)]))
        save(bands(rows, size=20), "L10_deep_nuclei.png")


if __name__ == "__main__":
    main()
