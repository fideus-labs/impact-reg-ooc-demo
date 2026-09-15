"""Four 12 mm crops at 0.05 mm where the deformable moved the MRI the most onto a sharp XPCT edge: the MRI's
iso-contour after the affine (blue) and after the deformable (gold) over the XPCT, with a 2 mm scale bar.

Candidates are the voxels of the MRI iso-surface at level 4, ranked by the displacement projected on the XPCT
gradient (a move along an edge shows nothing) summed over 2 mm around them, kept 15 mm apart and 6 mm inside
the tissue. Each crop lies in the axis plane holding most
of its displacement, and the band prints where it is in the XPCT volume (mm from the store's corner).

    python figures_landmarks.py            # D_landmarks.png
"""

import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter

import linc
from figures_compare import BLUE, GOLD, contour
from figures_linc import DATA, FIG, ROOT, bands, grey, read_zarr, save
from stage1_linc import LPS, normalise
from video import font

COARSE_FIELD = ROOT / "out" / "linc_coarse_bend100" / "P000" / "Transform.h5"
STEP, SIZE, MARGIN_MM, APART_MM, XPCT_LEVEL = 0.05, 240, 6.0, 15.0, "1"   # 12 mm crops; XPCT level 1 = 0.04 mm
PLANES = {0: "sagittal", 1: "coronal", 2: "axial"}                       # by the axis normal to the crop, (x, y, z)


def pick_landmarks(fixed: np.ndarray, deformable: np.ndarray, voxel: float, count: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    """(index z, y, x ; displacement vector x, y, z in mm) of the best crop centres, best first."""
    level = float(np.median(deformable[deformable > 0.02]))
    smooth = gaussian_filter(deformable, 2)
    inside = smooth > level
    surface = inside ^ binary_erosion(inside)
    deep = distance_transform_edt(smooth > 0.02) * voxel > MARGIN_MM
    # the field on every second voxel: 360 MiB instead of 2.9 GiB, and a B-spline field is smooth at that scale
    size = linc.field_header(COARSE_FIELD)[0]
    field = linc.read_field(COARSE_FIELD, np.zeros(3, int), size, step=2)
    half = tuple(slice(None, None, 2) for _ in range(3))
    # the field is (x, y, z) in the store's frame, the array (z, y, x): axis 2 - k carries component k
    across = sum(field[..., k] * gaussian_filter(fixed, 1, order=tuple(int(a == 2 - k) for a in range(3)))[half]
                 for k in range(3))
    # a crop is 12 mm: score a 2 mm neighbourhood of the contour, not one voxel, so the edge holds over its length
    on_contour = (surface & deep)[half]
    score = (gaussian_filter(np.abs(across) * on_contour, 3) * on_contour).ravel()
    picked: list[tuple[np.ndarray, np.ndarray]] = []
    for flat in np.argsort(score)[::-1]:
        if score[flat] <= 0 or len(picked) == count:
            break
        index = np.array(np.unravel_index(flat, across.shape)) * 2
        if all(np.linalg.norm(index - p) * voxel >= APART_MM for p, _ in picked):
            picked.append((index, field[tuple(index // 2)]))
    return picked


def crop_slab(centre_lps: np.ndarray, normal: int) -> sitk.Image:
    """A SIZE x SIZE plane at STEP mm through centre, normal to one axis, in the LPS frame."""
    mirror = np.diag(np.array(LPS).reshape(3, 3))
    size, origin = [SIZE] * 3, np.array(centre_lps) - mirror * (SIZE / 2 * STEP)
    size[normal], origin[normal] = 1, centre_lps[normal]
    slab = sitk.Image(size, sitk.sitkFloat32)
    slab.SetSpacing((STEP,) * 3)
    slab.SetDirection(LPS)
    slab.SetOrigin(origin.tolist())
    return slab


def xpct_plane(centre_lps: np.ndarray, normal: int, size_mm: float) -> sitk.Image:
    """The XPCT store at XPCT_LEVEL around one plane through centre: three voxels thick, so one layer of chunks."""
    voxel, array = linc.levels()[XPCT_LEVEL], linc.store()[XPCT_LEVEL]
    mirror = np.diag(np.array(LPS).reshape(3, 3))
    index = np.round(mirror * centre_lps / voxel).astype(int)              # the store's (x, y, z) index
    half = np.full(3, round(size_mm / voxel / 2))
    half[normal] = 1
    lo, hi = np.maximum(index - half, 0), np.minimum(index + half + 1, array.shape)
    image = linc.as_itk(np.asarray(array[tuple(slice(a, b) for a, b in zip(lo, hi))]), voxel)
    image.SetOrigin((mirror * lo * voxel).tolist())
    return image


def both_thin(background: np.ndarray, before: np.ndarray, after: np.ndarray, level: float) -> np.ndarray:
    """Blue two voxels wide, gold one: the XPCT's own edge stays visible beside the line that sits on it."""
    canvas = np.dstack([grey(background)] * 3) * 0.85
    canvas[contour(before, level, 2)] = BLUE
    canvas[contour(after, level, 1)] = GOLD
    return canvas


def annotate(panel: np.ndarray, number: int, moved: str, zoom: int = 2) -> np.ndarray:
    """The crop enlarged ``zoom`` times: its number at the top left, the move at the bottom right, a 2 mm bar."""
    image = Image.fromarray((np.clip(panel, 0, 1) * 255).astype(np.uint8)).resize((SIZE * zoom,) * 2, Image.LANCZOS)
    draw = ImageDraw.Draw(image)
    halo = {"fill": (255, 255, 255), "stroke_width": 2 * zoom, "stroke_fill": (0, 0, 0)}
    draw.text((8 * zoom, 4 * zoom), str(number), font=font(26 * zoom), **halo)
    draw.text(((SIZE - 8) * zoom, (SIZE - 8) * zoom), moved, font=font(12 * zoom), anchor="rb", **halo)
    bar, y = round(2.0 / STEP) * zoom, (SIZE - 14) * zoom
    draw.rectangle((10 * zoom - zoom, y - zoom, 10 * zoom + bar + zoom, y + 4 * zoom), fill=(0, 0, 0))
    draw.rectangle((10 * zoom, y, 10 * zoom + bar, y + 3 * zoom), fill=(255, 255, 255))
    draw.text((10 * zoom + bar + 6 * zoom, y + 2 * zoom), "2 mm", font=font(13 * zoom), anchor="lm", **halo)
    return np.asarray(image) / 255.0


def locator(fixed_image: sitk.Image, fixed: np.ndarray, affine: np.ndarray, deformable: np.ndarray, level: float,
            centres: list[np.ndarray], scale: float = 1.6) -> None:
    """The coronal mid-plane of level 4 with both contours, and the four crops as numbered boxes projected on it."""
    y = fixed.shape[1] // 2
    canvas = both_thin(fixed[:, y], affine[:, y], deformable[:, y], level)
    image = Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8))
    image = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)
    draw = ImageDraw.Draw(image)
    half = SIZE * STEP / 2 / fixed_image.GetSpacing()[0] * scale
    for number, centre in enumerate(centres, 1):
        ix, _, iz = fixed_image.TransformPhysicalPointToIndex(centre.tolist())
        cx, cz = ix * scale, iz * scale
        draw.rectangle((cx - half, cz - half, cx + half, cz + half), outline=(255, 255, 255), width=2)
        draw.text((cx + half + 4, cz - half - 2), str(number), font=font(22), fill=(255, 255, 255),
                  stroke_width=3, stroke_fill=(0, 0, 0))
    save(bands([("level 4 · blue: affine · gold: deformable · boxes: the four crops", [np.asarray(image) / 255.0])],
               size=20), "D_locator.png")


def main() -> None:
    FIG.mkdir(exist_ok=True)
    fixed_image = sitk.ReadImage(str(DATA / "XPCT_level4.mha"))
    fixed_image.SetDirection(LPS)
    voxel = fixed_image.GetSpacing()[0]
    fixed = sitk.GetArrayFromImage(normalise(fixed_image))
    deformable = read_zarr(ROOT / "out" / "linc_coarse_bend100" / "P000" / "Moved.ome.zarr")
    level = float(np.median(deformable[deformable > 0.02]))
    landmarks = pick_landmarks(fixed, deformable, voxel)
    centres = [np.array(fixed_image.TransformIndexToPhysicalPoint([int(i) for i in index[::-1]])) for index, _ in landmarks]
    locator(fixed_image, fixed, read_zarr(DATA / "pair" / "Moving.ome.zarr"), deformable, level, centres)
    del deformable

    source = normalise(sitk.ReadImage(str(DATA / "dmri_oriented.mha")))
    affine_t = sitk.ReadTransform(str(DATA / "affine.tfm"))
    panels = []
    for index, vector in landmarks:
        store_mm = index[::-1] * voxel                                     # (x, y, z) from the store's corner
        centre = np.array(fixed_image.TransformIndexToPhysicalPoint([int(i) for i in index[::-1]]))
        normal = int(np.argmin(np.abs(vector)))
        slab = crop_slab(centre, normal)
        try:
            xpct = xpct_plane(centre, normal, SIZE * STEP + 2.0)
        except Exception as error:                                          # noqa: BLE001  offline: level 3 from disk
            print(f"  S3 unavailable ({error!r}); XPCT from level 3")
            xpct = sitk.ReadImage(str(DATA / "XPCT_level3.mha"))
            xpct.SetDirection(LPS)
        chain = sitk.CompositeTransform(3)
        chain.AddTransform(affine_t)
        chain.AddTransform(linc.field_in_store_frame(COARSE_FIELD, slab, margin_mm=3.0))

        def on(image: sitk.Image, transform: sitk.Transform, slab: sitk.Image = slab) -> np.ndarray:
            return sitk.GetArrayFromImage(sitk.Resample(image, slab, transform, sitk.sitkLinear, 0.0)).squeeze()

        background, before, after = on(xpct, sitk.Transform()), on(source, affine_t), on(source, chain)
        print(f"  level {level:.3f} global, {np.median(after[after > 0.02]):.3f} in this crop")
        label = f"{PLANES[normal]} · x {store_mm[0]:.0f} y {store_mm[1]:.0f} z {store_mm[2]:.0f} mm"
        moved = f"moved {np.linalg.norm(vector):.1f} mm"
        print(f"  crop {len(panels) + 1}: {label} · {moved}")
        panels.append(annotate(both_thin(background, before, after, level), len(panels) + 1, moved))
    save(bands([("four 12 mm windows · blue: affine · gold: deformable", panels[:2]), ("", panels[2:])], size=24),
         "D_landmarks.png")


if __name__ == "__main__":
    main()
