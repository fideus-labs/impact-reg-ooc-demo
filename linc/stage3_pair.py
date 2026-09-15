"""Build the checkable pair from two native windows: two markers, one known deformation.

The two light-sheet volumes of this block (NeuN and Calretinin) were imaged on the same grid, so
they start aligned. Deforming one of them by a field we choose gives a registration problem whose
answer is known exactly — the residual after the run is then a number, not an impression.

The two markers stain different cells, so the pair is what the semantic loss is for: the images are
not the same picture at different positions, they are different pictures of the same tissue.

    python stage3_pair.py --amplitude 8
"""

import argparse
from pathlib import Path

import numpy as np
import zarr
from konfai.utils.ome_zarr import write_ome_zarr
from scipy.ndimage import map_coordinates


DATA = Path(__file__).parent / "data" / "dandi"


def warp(array: np.ndarray, amplitude: float) -> tuple[np.ndarray, np.ndarray]:
    """Deform by a smooth low-frequency field; return the deformed array and the field, in voxels."""
    z, y, x = np.mgrid[[slice(0, n) for n in array.shape]].astype(np.float32)
    field = np.stack(
        [
            amplitude * np.sin(x / 40.0) * np.cos(y / 45.0),
            amplitude * np.sin(y / 35.0) * np.cos(z / 50.0),
            amplitude * np.cos(x / 30.0) * np.sin(z / 42.0),
        ],
        axis=-1,
    ).astype(np.float32)
    moved = map_coordinates(array, [z + field[..., 0], y + field[..., 1], x + field[..., 2]],
                            order=1, mode="nearest")
    return moved.astype(np.float32), field


def read(path: Path) -> tuple[np.ndarray, list[float], list[float]]:
    group = zarr.open_group(str(path), mode="r")
    multiscale = dict(group.attrs)["multiscales"][0]
    dataset = multiscale["datasets"][0]
    array = np.asarray(group[dataset["path"]])
    array = array[0] if array.ndim == 4 else array
    scale = dataset["coordinateTransformations"][0]["scale"][-3:]
    translation = next(
        (t["translation"][-3:] for t in dataset["coordinateTransformations"] if t["type"] == "translation"),
        [0.0, 0.0, 0.0],
    )
    return array, scale, translation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", type=Path, default=DATA / "cortex" / "Moving.ome.zarr",
                        help="the light-sheet window; the fixed image of the pair")
    parser.add_argument("--moving", type=Path, default=DATA / "cortex" / "Moving.ome.zarr",
                        help="what gets deformed; another marker of the same block also works")
    parser.add_argument("--amplitude", type=float, default=8.0, help="warp amplitude, voxels")
    parser.add_argument("--out", type=Path, default=DATA / "pair_neun")
    args = parser.parse_args()

    fixed, scale, translation = read(args.fixed)
    moving, _, _ = read(args.moving)
    deformed, _ = warp(moving, args.amplitude)

    args.out.mkdir(parents=True, exist_ok=True)
    # write_ome_zarr takes geometry in SimpleITK (x, y, z) order; the file gave it as (z, y, x)
    geometry = dict(spacing=scale[::-1], origin=translation[::-1], chunks=(1, 32, 256, 256))
    for name, array in (("Fixed", fixed), ("Moving", deformed), ("Truth", moving)):
        write_ome_zarr(args.out / f"{name}.ome.zarr", array.astype(np.float32)[None], **geometry)
        print(f"{name}.ome.zarr {array.shape} @ {[round(s, 4) for s in scale]} mm")
    print(f"known deformation: {args.amplitude:.0f} voxels peak = {args.amplitude * scale[1] * 1000:.0f} um")


if __name__ == "__main__":
    main()
