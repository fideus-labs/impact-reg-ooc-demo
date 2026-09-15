"""The OCT / light-sheet pair for the short deck: both states on one image, and the native window as a flicker.

    python figures_pair.py
"""

import numpy as np
from PIL import Image, ImageDraw

from figures import ROOT, grey, outline, pair, read_zarr
from figures_compare import BLUE, GOLD, flicker
from figures_linc import bands, save
from video import font

DATA = ROOT / "data" / "dandi"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="plane_g0.4", help="the stage-2 run under out/ to draw")
    parser.add_argument("--stem", default="", help="suffix for the stage-2 figures, e.g. _m_g0.2 -> P3_m_g0.2.png")
    parser.add_argument("--native", action="store_true", help="also draw the native window's flicker")
    parser.add_argument("--native-tag", default="", help="draw plane_native_<tag> and out/native_tiled_<tag> instead")
    parser.add_argument("--native-run", default="", help="the tiled run under out/ to draw, if not native_tiled[_<tag>]")
    args = parser.parse_args()
    # stage 2, the 25 um section: the light-sheet's edges after the affine (blue) and after the deformable (gold)
    fixed = read_zarr(DATA / "plane" / "Fixed.ome.zarr")[0][0]
    affine = read_zarr(DATA / "plane" / "Moving.ome.zarr")[0][0]
    deformable = read_zarr(ROOT / "out" / args.run / "P000" / "Moved.ome.zarr")[0][0]
    canvas = np.dstack([grey(fixed)] * 3) * 0.85
    canvas[outline(affine)] = BLUE
    canvas[outline(deformable)] = GOLD
    # four 12 mm windows where the two lines part and the section has tissue on both sides, (x, y) in voxels
    centres, half, zoom = ((200, 330), (720, 210), (690, 1380), (1330, 480)), 240, 2
    crops = []
    for number, (cx, cy) in enumerate(centres, 1):
        x0, y0 = min(max(cx - half, 0), fixed.shape[1] - 2 * half), min(max(cy - half, 0), fixed.shape[0] - 2 * half)
        box = np.s_[y0 : y0 + 2 * half, x0 : x0 + 2 * half]
        crop = np.dstack([grey(fixed[box])] * 3) * 0.85
        crop[outline(affine)[box]] = BLUE
        crop[outline(deformable)[box]] = GOLD
        image = Image.fromarray((np.clip(crop, 0, 1) * 255).astype(np.uint8)).resize((2 * half * zoom,) * 2, Image.LANCZOS)
        draw = ImageDraw.Draw(image)
        halo = {"fill": (255, 255, 255), "stroke_width": 2 * zoom, "stroke_fill": (0, 0, 0)}
        draw.text((8 * zoom, 4 * zoom), str(number), font=font(26 * zoom), **halo)
        bar, y = round(2.0 / 0.025) * zoom, (2 * half - 14) * zoom
        draw.rectangle((10 * zoom - zoom, y - zoom, 10 * zoom + bar + zoom, y + 4 * zoom), fill=(0, 0, 0))
        draw.rectangle((10 * zoom, y, 10 * zoom + bar, y + 3 * zoom), fill=(255, 255, 255))
        draw.text((10 * zoom + bar + 6 * zoom, y + 2 * zoom), "2 mm", font=font(13 * zoom), anchor="lm", **halo)
        crops.append(np.asarray(image) / 255.0)
        # the same box on the section
        canvas_box = (slice(y0, y0 + 2 * half), slice(x0, x0 + 2 * half))
        for edge in (np.s_[y0 : y0 + 3, x0 : x0 + 2 * half], np.s_[y0 + 2 * half - 3 : y0 + 2 * half, x0 : x0 + 2 * half],
                     np.s_[y0 : y0 + 2 * half, x0 : x0 + 3], np.s_[y0 : y0 + 2 * half, x0 + 2 * half - 3 : x0 + 2 * half]):
            canvas[edge] = 1.0
        del canvas_box
    locator = Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(locator)
    for number, (cx, cy) in enumerate(centres, 1):
        x0, y0 = min(max(cx - half, 0), fixed.shape[1] - 2 * half), min(max(cy - half, 0), fixed.shape[0] - 2 * half)
        at = (x0 + 2 * half + 8, y0 - 4) if x0 + 2 * half + 60 < fixed.shape[1] else (x0 + 12, y0 + 6)
        draw.text(at, str(number), font=font(40), fill=(255, 255, 255), stroke_width=4, stroke_fill=(0, 0, 0))
    save(bands([("the section at 25 um · blue: affine · gold: deformable · boxes: the four windows", [np.asarray(locator) / 255.0])],
               size=28), f"P1_both{args.stem}.png", scale=0.6)
    save(bands([("four 12 mm windows · blue: affine · gold: deformable", crops[:2]), ("", crops[2:])], size=24),
         f"P3_crops{args.stem}.png", scale=0.5)
    if not args.native:
        return

    # stage 3, the native window at 3 um: the OCT in red stays, the light-sheet in green moves
    suffix = f"_{args.native_tag}" if args.native_tag else ""
    run = args.native_run or f"native_tiled{suffix}"
    fixed_volume = read_zarr(DATA / f"plane_native{suffix}" / "Fixed.ome.zarr")[0]
    k = fixed_volume.shape[0] // 2  # the middle plane, as scripts/oct_native_score.py scores it
    fixed_native, start = fixed_volume[k], read_zarr(DATA / f"plane_native{suffix}" / "Moving.ome.zarr")[0][k]
    # the start warped through the field by hand: the same image KonfAI writes, without its zero-filled faces
    import linc
    from scipy.ndimage import map_coordinates
    field = ROOT / "out" / run / "P000" / "Transform.h5"
    size, _, sp = linc.field_header(field)
    u = linc.read_field(field, (0, 0, 0), size)[k]
    yy, xx = np.mgrid[: start.shape[0], : start.shape[1]].astype(np.float32)
    moved = map_coordinates(start, [yy + u[..., 1] / sp[1], xx + u[..., 0] / sp[0]], order=1, mode="nearest")
    before, after = pair(fixed_native, start, gamma=0.55), pair(fixed_native, moved, gamma=0.55)
    flicker(f"P2_flicker{args.stem}.gif", [("red: OCT · green: light-sheet, before the tiles", before),
                               ("red: OCT · green: light-sheet, after the tiles", after)], scale=0.5)
    save(bands([("before the tiles", [before]), ("after the tiles", [after])], size=28), f"P2_before_after{args.stem}.png", scale=0.5)


if __name__ == "__main__":
    main()
