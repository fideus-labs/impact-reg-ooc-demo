"""The zoom snapping onto the overview: four 4 mm windows at 20 um, the overview's edges drawn in gold over the
zoom, alternating between the hand placement and the tiled deformable. In the first frame the zoom's own edges
sit beside the gold ones; in the second they lie under them.

    python figures_zoomflicker.py
"""

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_dilation, gaussian_filter, sobel

import zoom_linc
from figures_compare import GOLD, flicker
from figures_linc import bands, grey, save

WINDOWS = ((49.1, 54.9, 67.6), (49.1, 54.9, 85.6), (67.1, 72.9, 67.6), (67.1, 72.9, 85.6))   # windows 1, 2, 7, 8
FIELD = zoom_linc.ROOT / "out" / "zoom1_level1_bend100" / "P000" / "Transform.h5"


def edges(plane: np.ndarray) -> np.ndarray:
    smooth = gaussian_filter(grey(plane), 1.5)
    magnitude = np.hypot(sobel(smooth, 0), sobel(smooth, 1))
    return binary_dilation(magnitude > np.percentile(magnitude, 96), np.ones((2, 2)))


def main() -> None:
    hand, tiled, overview_edges = [], [], []
    for centre in WINDOWS:
        centre, half = np.array(centre), 2.0
        window = zoom_linc.box(zoom_linc.OVERVIEW, "0", centre - half, centre + half)
        overview = sitk.GetArrayFromImage(window)
        by_hand = zoom_linc.placement(correction=False)
        chain = zoom_linc.placement()
        chain.AddTransform(zoom_linc.field(FIELD, window))                                # applied first
        lo, hi = zoom_linc.bounds(by_hand, centre - half, centre + half)
        source = zoom_linc.box(zoom_linc.ZOOM, "2", lo - 1.0, hi + 1.0)
        k = overview.shape[0] // 2
        for state, keep in ((by_hand, hand), (chain, tiled)):
            moved = sitk.GetArrayFromImage(sitk.Resample(source, window, state, sitk.sitkLinear, 0.0))[k]
            keep.append(np.dstack([grey(moved)] * 3))
        overview_edges.append(edges(overview[k]))
    for panels in (hand, tiled):
        for panel, edge in zip(panels, overview_edges):
            panel[edge] = GOLD
    gap = np.ones((hand[0].shape[0], 6, 3)) * 0.06
    row = lambda panels: np.hstack([p for pair in zip(panels, [gap] * 4) for p in pair][:-1])
    flicker("Z3_flicker.gif", [("gold: the overview's edges · the zoom placed by hand", row(hand)),
                               ("gold: the overview's edges · the zoom after the tiles", row(tiled))], scale=1.4)
    save(bands([("the zoom placed by hand · gold: the overview's edges", [row(hand)]),
                ("the zoom after the tiles", [row(tiled)])], size=20), "Z3_edges.png", scale=1.4)


if __name__ == "__main__":
    main()
