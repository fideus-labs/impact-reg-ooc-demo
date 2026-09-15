"""Every stage-2 candidate of the OCT / light-sheet section side by side, in the same four 12 mm windows.

Two sheets: the candidate's edge in gold against the affine's in blue, and against the kept run's in blue,
so the eye sees both what each run did and where it differs from the one in the deck.

    python figures_candidates.py plane_g0.4 plane_g0.4_it3000_s32768 ...
"""

import sys

import numpy as np
from PIL import Image, ImageDraw

from figures import ROOT, grey, outline, read_zarr
from figures_compare import BLUE, GOLD
from figures_linc import bands, save
from video import font

DATA = ROOT / "data" / "dandi"
CENTRES, HALF = ((200, 330), (720, 210), (690, 1380), (1330, 480)), 240


def crops(fixed: np.ndarray, blue: np.ndarray, gold: np.ndarray, label: str, zoom: int = 1) -> list[np.ndarray]:
    blue_edges, gold_edges = outline(blue), outline(gold)
    out = []
    for number, (cx, cy) in enumerate(CENTRES, 1):
        x0, y0 = min(max(cx - HALF, 0), fixed.shape[1] - 2 * HALF), min(max(cy - HALF, 0), fixed.shape[0] - 2 * HALF)
        box = np.s_[y0 : y0 + 2 * HALF, x0 : x0 + 2 * HALF]
        crop = np.dstack([grey(fixed[box])] * 3) * 0.85
        crop[blue_edges[box]] = BLUE
        crop[gold_edges[box]] = GOLD
        image = Image.fromarray((np.clip(crop, 0, 1) * 255).astype(np.uint8))
        if zoom != 1:
            image = image.resize((2 * HALF * zoom,) * 2, Image.LANCZOS)
        draw = ImageDraw.Draw(image)
        draw.text((8, 4), f"{number}  {label}" if number == 1 else str(number), font=font(22),
                  fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        out.append(np.asarray(image) / 255.0)
    return out


def main() -> None:
    runs = sys.argv[1:]
    fixed = read_zarr(DATA / "plane" / "Fixed.ome.zarr")[0][0]
    affine = read_zarr(DATA / "plane" / "Moving.ome.zarr")[0][0]
    kept = read_zarr(ROOT / "out" / "plane_g0.4" / "P000" / "Moved.ome.zarr")[0][0]
    against_affine, against_kept = [], []
    for run in runs:
        moved = read_zarr(ROOT / "out" / run / "P000" / "Moved.ome.zarr")[0][0]
        label = run.replace("plane_", "")
        against_affine.append((f"{label} · blue: affine · gold: this run", crops(fixed, affine, moved, label)))
        against_kept.append((f"{label} · blue: the deck's run · gold: this run", crops(fixed, kept, moved, label)))
    save(bands(against_affine, size=22), "P4_candidates_vs_affine.png", scale=0.5)
    save(bands(against_kept, size=22), "P4_candidates_vs_kept.png", scale=0.5)


if __name__ == "__main__":
    main()
