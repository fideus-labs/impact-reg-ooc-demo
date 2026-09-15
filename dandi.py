"""Read from the two public DANDI stores of one human brain block (dandiset 000026, sub-I48, Broca S23).

    OCT       s3://dandiarchive/zarr/a6fa8bd2-...   70 x 12441 x 14074  float32  3.0 um   50 GiB
    SPIM NeuN s3://dandiarchive/zarr/c36e0b26-...  207 x 10309 x 12261  uint16   3.6 um   31 GiB

Nothing is downloaded whole: a read touches the chunks its window intersects, and nothing else.

    python dandi.py overview --level 4          # the coarse pair, for the global step
    python dandi.py window --z 32 --y 5000 --x 6000 --size 2048   # a native-resolution window
"""

import argparse
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import zarr

OUT = Path(__file__).parent / "data" / "dandi"
STORES = {
    "OCT": "a6fa8bd2-c46b-445b-a219-e97dac4f2ee8",
    "SPIM": "c36e0b26-ac42-443a-94f9-1fa2cda794b5",
}


def level(name: str, index: int):
    """One pyramid level of one store, with the voxel size OME-Zarr declares for it."""
    group = zarr.open_group(f"s3://dandiarchive/zarr/{STORES[name]}", mode="r", storage_options={"anon": True})
    scales = {d["path"]: d["coordinateTransformations"][0]["scale"] for d in group.attrs["multiscales"][0]["datasets"]}
    return group[str(index)], scales[str(index)]


def write(array: np.ndarray, spacing_zyx, path: Path) -> None:
    image = sitk.GetImageFromArray(array.astype(np.float32))
    image.SetSpacing(tuple(float(s) / 1000.0 for s in spacing_zyx[::-1]))  # um -> mm, ITK is x,y,z
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path), useCompression=True)
    print(f"  {path.name}: {array.shape} @ {[round(s / 1000, 4) for s in spacing_zyx]} mm")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    coarse = sub.add_parser("overview", help="one coarse level of each store, for the global step")
    coarse.add_argument("--level", type=int, default=4)
    native = sub.add_parser("window", help="a window at native resolution (level 0) of each store")
    native.add_argument("--z", type=int, default=32)
    native.add_argument("--y", type=int, default=5000)
    native.add_argument("--x", type=int, default=6000)
    native.add_argument("--size", type=int, default=2048, help="in-plane window, voxels")
    native.add_argument("--depth", type=int, default=64, help="z extent, voxels")
    args = parser.parse_args()

    for name in STORES:
        start = time.perf_counter()
        if args.command == "overview":
            array, spacing = level(name, args.level)
            data = array[:]
            write(data, spacing, OUT / f"overview_{name}.mha")
        else:
            array, spacing = level(name, 0)
            # The stores differ in voxel size, so the same physical window is a different voxel box in each.
            ratio = 3.0 / spacing[1]
            half = int(args.size * ratio) // 2
            depth = int(args.depth * 3.0 / spacing[0])
            z, y, x = (int(args.z * 3.0 / spacing[0]), int(args.y * ratio), int(args.x * ratio))
            data = array[z : z + depth, y - half : y + half, x - half : x + half]
            write(data, spacing, OUT / f"window_{name}.mha")
        print(f"  {name}: {data.nbytes / 2**20:.0f} MiB read in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
