"""The masks a tiled deformable needs: where the moving image has tissue, widened by about 1 mm.

A tile holding only a sliver of tissue has almost nothing to fit, and without a mask elastix fits the
background instead: at level 3 that moved the tissue edge by up to 69 mm. The fixed mask keeps the
metric on the tissue and its edge; a tile the mask does not reach gets a zero field. The moving mask
restricts nothing, and is written only so both masks share the pair's format.

    python masks_linc.py data/linc/pair3
    python masks_linc.py data/linc/pair0
"""

import argparse
from pathlib import Path

import numpy as np
import zarr
from konfai.utils.ome_zarr import write_ome_zarr
from scipy.ndimage import maximum_filter1d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pair", type=Path, help="a directory holding Fixed.ome.zarr and Moving.ome.zarr")
    parser.add_argument("--widen-mm", type=float, default=1.0)
    args = parser.parse_args()

    group = zarr.open_group(str(args.pair / "Moving.ome.zarr"), mode="r")
    dataset = dict(group.attrs)["multiscales"][0]["datasets"][0]
    scale = dataset["coordinateTransformations"][0]["scale"][-3:]
    origin = next(t["translation"][-3:] for t in dataset["coordinateTransformations"] if t["type"] == "translation")

    support = (np.asarray(group[dataset["path"]][0]) > 0.02).astype(np.uint8)
    width = 2 * int(round(args.widen_mm / scale[-1])) + 1
    for axis in range(3):  # a cube, one axis at a time
        support = maximum_filter1d(support, width, axis=axis)

    geometry = dict(spacing=tuple(scale[::-1]), origin=tuple(origin[::-1]), chunks=(1, 64, 256, 256))
    write_ome_zarr(args.pair / "FixedMask.ome.zarr", support[None], **geometry)
    write_ome_zarr(args.pair / "MovingMask.ome.zarr", np.ones_like(support)[None], **geometry)
    print(f"{args.pair}: fixed mask covers {support.mean():.3f} (widened by {args.widen_mm} mm)")


if __name__ == "__main__":
    main()
