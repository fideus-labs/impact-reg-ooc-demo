"""The video for the LINC pair: one human hemisphere, X-ray phase contrast against diffusion MRI.

Every frame is built from the run's own outputs. The scenes for the native window and for the
per-structure Dice appear only once their runs have written something.

    python video_linc.py --out video/linc.mp4
"""

import argparse
import colorsys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw

from figures_linc import contours_over, iso_contours_over, read_zarr, strip, three_views
from linc import field_header, read_field
from stage1_linc import LPS, mutual_information, normalise
from video import (
    GOLD,
    INK,
    MUTED,
    Film,
    canvas,
    caption,
    font,
    grey,
    mono,
    place,
    scene_terminal,
    scene_tiles,
    scene_two,
    to_image,
)

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "linc"
OUT = ROOT / "out"


def rgb(array: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8))


def views_overlay(fixed: np.ndarray, moving: np.ndarray) -> Image.Image:
    panels = [np.dstack([grey(f), grey(m), np.zeros_like(f)]) for f, m in zip(three_views(fixed), three_views(moving))]
    return rgb(strip(panels))


def field_colours(plane: np.ndarray) -> np.ndarray:
    angle = (np.arctan2(plane[..., 1], plane[..., 0]) + np.pi) / (2 * np.pi)
    magnitude = np.linalg.norm(plane[..., :2], axis=-1)
    magnitude = magnitude / (np.percentile(magnitude, 99) + 1e-9)
    lut = np.array([colorsys.hsv_to_rgb(h / 255.0, 0.85, 1.0) for h in range(256)])
    return lut[(angle * 255).astype(np.uint8)] * np.clip(magnitude, 0, 1)[..., None]


def scene_table(film: Film, title: str, subtitle: str, rows, hold: float = 6.0) -> None:
    frame = canvas(title, subtitle)
    draw = ImageDraw.Draw(frame)
    y = 290
    for label, value, highlight in rows:
        colour = GOLD if highlight else INK
        draw.text((160, y), label, font=font(34), fill=colour if highlight else MUTED)
        draw.text((1300, y), value, font=mono(38), fill=colour)
        y += 92
    film.add(frame, hold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "video" / "linc.mp4")
    parser.add_argument("--short", action="store_true",
                        help="the demo cut: no orientation table, no reference alignment, no per-structure table")
    args = parser.parse_args()
    film = Film(ROOT / "video" / "frames_linc")

    fixed_image = sitk.ReadImage(str(DATA / "XPCT_level4.mha"))
    fixed_image.SetDirection(LPS)
    fixed = sitk.GetArrayFromImage(normalise(fixed_image))
    mri = sitk.GetArrayFromImage(normalise(sitk.ReadImage(str(DATA / "dmri.nii.gz"))))

    # 1 . the pair
    frame = canvas("Out-of-core registration", "one human hemisphere, imaged twice . BRAIN CONNECTS, LINC")
    place(frame, rgb(strip([np.dstack([grey(p)] * 3) for p in three_views(fixed)])), (90, 200, 1740, 400))
    place(frame, rgb(strip([np.dstack([grey(p)] * 3) for p in three_views(mri)])), (90, 620, 1740, 330))
    caption(frame, "synchrotron X-ray phase contrast   7569 x 7569 x 8514   20.07 µm   908 GiB", (90, 965), GOLD, code=True)
    caption(frame, "diffusion MRI, b-ratio map   0.4 mm   24 MiB   .   dandiset 001278, sub-Hb1", (90, 1010), MUTED, code=True)
    film.add(frame, 5.0)

    # 2 . why it is hard
    if not args.short:
        frame = canvas("Two contrasts from unrelated physics", "X-ray phase against the anisotropy of water diffusion")
        draw = ImageDraw.Draw(frame)
        lines = ["Nothing makes a bright voxel in one a bright voxel in the other.",
                 "And the fixed volume does not fit in a workstation, let alone on a card.",
                 "",
                 "The dandiset ships the consortium's own alignment:",
                 "that is the reference everything below is measured against."]
        for i, line in enumerate(lines):
            draw.text((160, 330 + i * 80), line, font=font(40), fill=INK if i < 2 else MUTED)
        film.add(frame, 6.0)

    # 3 . the coarse stages, whole hemisphere
    affine = read_zarr(DATA / "pair" / "Moving.ome.zarr")
    deformable = read_zarr(OUT / "linc_coarse_bend100" / "P000" / "Moved.ome.zarr")
    reference = sitk.GetArrayFromImage(sitk.Resample(normalise(sitk.ReadImage(str(DATA / "dmri_reference.nii.gz"))),
                                                     normalise(fixed_image), sitk.Transform(), sitk.sitkLinear, 0.0))
    if args.short:
        scene_table(film, "Stages 1 and 2, on a coarse level of the pyramid", "a global affine, then a coarse deformable . mutual information",
                    [("as the store and the NIfTI hand them over", "0.006", False),
                     ("after the affine", f"{mutual_information(fixed, affine):.3f}", False),
                     ("after the deformable, whole hemisphere", f"{mutual_information(fixed, deformable):.3f}", True)])
    else:
        scene_table(film, "The store says nothing about anatomy", "so stage 1 screens 24 orientations, then fits an affine",
                    [("as the store and the NIfTI hand them over", "0.006", False),
                     ("best of the 24 orientations, rigid", "0.247", False),
                     ("after the affine", f"{mutual_information(fixed, affine):.3f}", False),
                     ("after the deformable, whole hemisphere", f"{mutual_information(fixed, deformable):.3f}", True),
                     ("the consortium's alignment", f"{mutual_information(fixed, reference):.3f}", False)])
    scene_two(film, "Stages 1 and 2, on a coarse level of the pyramid", "321 µm . red: XPCT . green: MRI",
              views_overlay(fixed, affine), views_overlay(fixed, deformable),
              "after the affine", "after the deformable", 5.0)
    plane = fixed.shape[1] // 2
    scene_two(film, "Read the contour, not the colours", "the MRI's own edges drawn over the XPCT",
              rgb(contours_over(fixed[:, plane], affine[:, plane])), rgb(contours_over(fixed[:, plane], deformable[:, plane])),
              "after the affine", "after the deformable", 5.0)

    # 4 . the ladder
    scene_table(film, "The same command, one level down", "the store is read by the region at every level",
                [("level 4   321 µm   474 x 474 x 533", "0.22 GiB", False),
                 ("level 3   161 µm   947 x 947 x 1065", "1.78 GiB", True),
                 ("level 2    80 µm   1893 x 1893 x 2129", "14.2 GiB", False),
                 ("level 1    40 µm   3785 x 3785 x 4257", "113.6 GiB", False),
                 ("level 0    20 µm   7569 x 7569 x 8514", "908.5 GiB", True)])

    # 5 . the tiles, at level 3
    level3_run = OUT / "linc_tiled3_bend100" / "P000"
    if (level3_run / "Transform.h5").exists():
        level3 = read_zarr(DATA / "pair3_bend" / "Fixed.ome.zarr")
        z = level3.shape[0] // 2
        nx, ny, _ = field_header(level3_run / "Transform.h5")[0]
        field = read_field(level3_run / "Transform.h5", (0, 0, z), (nx, ny, z + 1))[0]  # one plane of 23 GiB
        scene_tiles(film, grey(level3[z]), field_colours(field), 256, 31)

    # 6 . the native window
    from evaluate_linc import NATIVE_RUN, native_states
    from stage4_native_linc import written_window

    if (NATIVE_RUN / "Transform.h5").exists():
        image = written_window(DATA / "pair0" / "Fixed.ome.zarr")
        window = sitk.GetArrayFromImage(image)
        z = window.shape[0] // 2
        coarse = image[::4, ::4, ::4]
        start, after = native_states(coarse)
        fixed_coarse = sitk.GetArrayFromImage(coarse)
        before_mi, after_mi = mutual_information(fixed_coarse, start), mutual_information(fixed_coarse, after)
        consortium = normalise(sitk.ReadImage(str(DATA / "dmri_reference.nii.gz")))
        reference = sitk.GetArrayFromImage(sitk.Resample(consortium, coarse, sitk.Transform(), sitk.sitkLinear, 0.0))
        inner = round(4.0 / coarse.GetSpacing()[0])
        middle = np.s_[inner:-inner, inner:-inner, inner:-inner]
        tissue = reference[middle] > 0.02

        def agreement(moving: np.ndarray) -> float:  # correlation with the consortium's map, 4 mm in from the faces
            return float(np.corrcoef(moving[middle][tissue], reference[middle][tissue])[0, 1])

        plane = image[:, :, z : z + 1]
        start_plane, after_plane = native_states(plane)
        reference_plane = sitk.GetArrayFromImage(sitk.Resample(consortium, plane, sitk.Transform(), sitk.sitkLinear, 0.0))[0]
        level = float(np.median(reference_plane[reference_plane > 0.02]))
        frame = canvas("One window of level 0", "15.4 mm at 20.07 µm . 864 MiB read out of 908 GiB")
        place(frame, to_image(grey(window[z])), (90, 200, 820, 790))
        place(frame, to_image(grey(window[z, 256:512, 256:512])), (990, 200, 780, 780))
        caption(frame, "the window", (90, 1005))
        caption(frame, "5 mm of it, at native resolution", (990, 1005))
        film.add(frame, 5.0)
        if args.short:
            scene_two(film, "Refined at 20 µm, from the level-3 result", "125 tiles of 256³ . the MRI map's edges over the XPCT",
                      rgb(iso_contours_over(window[z], start_plane[0], level)),
                      rgb(iso_contours_over(window[z], after_plane[0], level)),
                      f"before   MI {before_mi:.3f}", f"after    MI {after_mi:.3f}", 6.0)
        else:
            scene_two(film, "Refined at 20 µm, from the level-3 result",
                      "125 tiles of 256³ . gold: this chain . cyan: the consortium's alignment of the same MRI map",
                      rgb(iso_contours_over(window[z], start_plane[0], level, reference_plane)),
                      rgb(iso_contours_over(window[z], after_plane[0], level, reference_plane)),
                      f"before   MI {before_mi:.3f}   r with the consortium {agreement(start):.3f}",
                      f"after    MI {after_mi:.3f}   r with the consortium {agreement(after):.3f}", 6.0)

    # 7 . the per-structure check
    dice_table = OUT / "dice_flash.txt"
    if dice_table.exists() and not args.short:
        lines = [line for line in dice_table.read_text().splitlines() if line.strip()]
        frame = canvas("Dice per structure, against the consortium's segmentation",
                       "FLASH T2* of the same hemisphere, brain only, through the same stages . mean of 14 named structures")
        draw = ImageDraw.Draw(frame)
        for i, line in enumerate(lines[:20]):
            draw.text((90, 220 + i * 40), line[:120], font=mono(26), fill=GOLD if line.startswith("mean") else INK)
        film.add(frame, 7.0)

    log = OUT / "run_linc_tiled3_bend100.log"
    if log.exists():
        scene_terminal(film, log)
    film.encode(args.out)


if __name__ == "__main__":
    main()
