"""The figures: what each stage did, in pictures.

    python figures.py
"""

import argparse
import colorsys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import zarr
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, gaussian_filter, sobel

ROOT = Path(__file__).parent
FIG = ROOT / "figures"
OCT_COLOUR, SPIM_COLOUR = (1.0, 0.30, 0.22), (0.35, 1.0, 0.45)


def read_zarr(path: Path):
    group = zarr.open_group(str(path), mode="r")
    dataset = dict(group.attrs)["multiscales"][0]["datasets"][0]
    array = group[dataset["path"]]
    array = np.asarray(array[0] if array.ndim == 4 else array)
    scale = dataset["coordinateTransformations"][0]["scale"][-3:]
    translation = next((t["translation"][-3:] for t in dataset["coordinateTransformations"]
                        if t["type"] == "translation"), [0.0, 0.0, 0.0])
    return array, scale, translation


def grey(plane: np.ndarray, low: float = 1.0, high: float = 99.6, gamma: float = 1.0) -> np.ndarray:
    lo, hi = np.percentile(plane, [low, high])
    return np.clip((plane - lo) / (hi - lo + 1e-9), 0, 1) ** gamma


def tint(plane: np.ndarray, colour) -> np.ndarray:
    return np.clip(plane[..., None] * np.array(colour), 0, 1)


def pair(fixed: np.ndarray, moving: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    return tint(grey(fixed), OCT_COLOUR) + tint(grey(moving, gamma=gamma), SPIM_COLOUR)


def outline(plane: np.ndarray, sigma: float = 3.0, quantile: float = 97.0) -> np.ndarray:
    """The edges of one image, as a thin mask: what to draw over the other one."""
    smooth = gaussian_filter(grey(plane), sigma)
    magnitude = np.hypot(sobel(smooth, 0), sobel(smooth, 1))
    return binary_dilation(magnitude > np.percentile(magnitude, quantile), np.ones((2, 2)))


def contours_over(fixed: np.ndarray, moving: np.ndarray, colour=(1.0, 0.85, 0.1)) -> np.ndarray:
    """The moving image's edges, drawn over the fixed image. Alignment is then a yes or a no."""
    canvas = np.dstack([grey(fixed)] * 3) * 0.85
    canvas[outline(moving)] = colour
    return canvas


def checkerboard(fixed: np.ndarray, moving: np.ndarray, tile: int = 96) -> np.ndarray:
    """Squares taken alternately from each image: structures either continue across them or step."""
    y, x = np.mgrid[0:fixed.shape[0], 0:fixed.shape[1]]
    picked = np.where((((y // tile) + (x // tile)) % 2) == 0, grey(fixed), grey(moving))
    return np.dstack([picked] * 3)


def save(array: np.ndarray, name: str, scale: float = 1.0) -> None:
    FIG.mkdir(exist_ok=True)
    image = Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8))
    if scale != 1.0:
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)
    image.save(FIG / name)
    print(f"  {name}  {image.width} x {image.height}")


def field_colours(plane: np.ndarray) -> np.ndarray:
    """Hue for direction, brightness for magnitude, on the in-plane components."""
    angle = (np.arctan2(plane[..., 1], plane[..., 0]) + np.pi) / (2 * np.pi)
    magnitude = np.linalg.norm(plane[..., :2], axis=-1)
    magnitude = magnitude / (np.percentile(magnitude, 99) + 1e-9)
    lut = np.array([colorsys.hsv_to_rgb(h / 255.0, 0.85, 1.0) for h in range(256)])
    return lut[(angle * 255).astype(np.uint8)] * np.clip(magnitude, 0, 1)[..., None]


def main() -> None:
    parser = argparse.ArgumentParser()
    data = ROOT / "data" / "dandi"
    parser.add_argument("--section", type=Path, default=data / "plane")
    parser.add_argument("--section-out", type=Path, default=ROOT / "out" / "plane_g0.4" / "P000")
    parser.add_argument("--native", type=Path, default=data / "plane_native")
    parser.add_argument("--native-out", type=Path, default=ROOT / "out" / "native_tiled" / "P000")
    parser.add_argument("--window", type=Path, default=data / "cortex")
    parser.add_argument("--patch", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=61)
    args = parser.parse_args()
    coarse_voxel = 0.05  # mm, the stage-1 grid

    # 1 . the section in both modalities
    reference = sitk.ReadImage(str(data / "coarse_OCT.mha"))
    section = grey(sitk.GetArrayFromImage(reference))
    light = grey(sitk.GetArrayFromImage(sitk.ReadImage(str(data / "coarse_SPIM_moved.mha"))))
    save(np.hstack([tint(section, (1, 1, 1)), tint(light, (1, 1, 1))]), "1_modalities.png")

    # 2 . after the affine, and after the deformable that is constrained to the plane
    fixed_section = read_zarr(args.section / "Fixed.ome.zarr")[0][0]
    affine_only = read_zarr(args.section / "Moving.ome.zarr")[0][0]
    save(pair(fixed_section, affine_only), "2_coarse.png", scale=0.6)
    if (args.section_out / "Moved.ome.zarr").exists():
        deformable = read_zarr(args.section_out / "Moved.ome.zarr")[0][0]
        save(np.hstack([pair(fixed_section, affine_only), pair(fixed_section, deformable)]),
             "2b_deformable.png", scale=0.55)
        # The readable check. A red/green overlay lies on this pair: the contrasts are inverted, so
        # a correct alignment puts red beside green, not yellow on top of yellow.
        save(np.hstack([contours_over(fixed_section, affine_only),
                        contours_over(fixed_section, deformable)]), "2c_contours.png", scale=0.55)
        save(np.hstack([checkerboard(fixed_section, affine_only),
                        checkerboard(fixed_section, deformable)]), "2d_checkerboard.png", scale=0.55)

    # 3 . where the native window sits, and what it holds
    window, scale, origin = read_zarr(args.window / "Fixed.ome.zarr")
    image = Image.fromarray((section * 255).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    x0, y0 = origin[2] / coarse_voxel, origin[1] / coarse_voxel
    w, h = window.shape[2] * scale[2] / coarse_voxel, window.shape[1] * scale[1] / coarse_voxel
    draw.rectangle([x0, y0, x0 + w, y0 + h], outline=(255, 200, 70), width=4)
    save(np.asarray(image) / 255.0, "3_window_in_section.png")

    moving_native = read_zarr(args.window / "Moving.ome.zarr")[0]
    z = window.shape[0] // 2
    box = (slice(768, 1280), slice(768, 1280))
    save(np.hstack([tint(grey(window[z][box]), (1, 1, 1)),
                    tint(grey(moving_native[z][box], gamma=0.55), (1, 1, 1))]), "4b_cells.png")

    # 4 . the tiling, over the native window
    plane = (grey(window[z]) * 255).astype(np.uint8)
    image = Image.fromarray(plane).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    step = args.patch - args.overlap
    for y in range(0, max(plane.shape[0] - args.overlap, 1), step):
        for x in range(0, max(plane.shape[1] - args.overlap, 1), step):
            draw.rectangle([x, y, min(x + args.patch, plane.shape[1] - 1),
                            min(y + args.patch, plane.shape[0] - 1)],
                           outline=(255, 200, 70, 210), width=3)
    save(np.asarray(image) / 255.0, "5_tiles.png", scale=0.5)

    # 5 . the native window, before and after the tiled run
    if (args.native_out / "Moved.ome.zarr").exists():
        fixed_native = read_zarr(args.native / "Fixed.ome.zarr")[0][0]
        start = read_zarr(args.native / "Moving.ome.zarr")[0][0]
        moved = read_zarr(args.native_out / "Moved.ome.zarr")[0][0]
        save(np.hstack([pair(fixed_native, start, gamma=0.55), pair(fixed_native, moved, gamma=0.55)]),
             "6_before_after.png", scale=0.5)
        field = sitk.GetArrayFromImage(
            sitk.DisplacementFieldTransform(
                sitk.ReadTransform(str(args.native_out / "Transform.h5"))).GetDisplacementField())
        save(field_colours(field[field.shape[0] // 2]), "7_field.png", scale=0.5)

    # 6 . why the coarse stage has to come first
    naive = ROOT / "out" / "native_tiled_affine" / "P000" / "Moved.ome.zarr"
    if naive.exists() and (args.native_out / "Moved.ome.zarr").exists():
        fixed_native = read_zarr(args.native / "Fixed.ome.zarr")[0][0]
        without = read_zarr(naive)[0][0]
        with_coarse = read_zarr(args.native_out / "Moved.ome.zarr")[0][0]
        save(np.hstack([pair(fixed_native, without, gamma=0.55),
                        pair(fixed_native, with_coarse, gamma=0.55)]), "8_why_coarse.png", scale=0.5)


if __name__ == "__main__":
    main()
