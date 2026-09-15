"""The two checks: did the run recover the deformation, and did the tiling leave a seam?

The moving image was built by warping a volume with a known field, so the answer is exact: the
run's displacement should be that field, negated, and what is left is the residual in voxels.

The seam check needs no ground truth. A field assembled from independent tiles would step at the
tile boundaries; the blend is a partition of unity, so the change in displacement across a boundary
plane should be no larger than the change anywhere else.

    python score.py out/dandi_window/P000/Transform.h5 --data data/dandi/pair --patch 512 --overlap 61
"""

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import zarr

from stage3_pair import warp


def read_zarr(path: Path) -> np.ndarray:
    group = zarr.open_group(str(path), mode="r")
    array = group[dict(group.attrs)["multiscales"][0]["datasets"][0]["path"]]
    return np.asarray(array[0] if array.ndim == 4 else array)


def residual(field: np.ndarray, truth: np.ndarray, amplitude: float) -> None:
    _, known = warp(truth, amplitude)  # (Z, Y, X, 3) voxels, the field that built the moving image
    tissue = truth > np.percentile(truth, 60)
    before = np.linalg.norm(known, axis=-1)[tissue]
    after = np.linalg.norm(known + field, axis=-1)[tissue]
    print(f"residual displacement over tissue, voxels:  before {before.mean():.2f}   after {after.mean():.2f}")
    print(f"  worst 5 %:                                before {np.percentile(before, 95):.2f}"
          f"   after {np.percentile(after, 95):.2f}")


def seams(field: np.ndarray, patch: int, overlap: int) -> None:
    step = patch - overlap
    for axis, name in ((1, "y"), (2, "x")):
        change = np.abs(np.diff(field, axis=axis)).sum(-1)
        planes = [p - 1 for p in range(step, change.shape[axis], step)]
        if not planes:
            continue
        at_seam = max(float(change.take(p, axis=axis).mean()) for p in planes)
        print(f"seam on {name}: {at_seam / change.mean():.2f} x the change everywhere else "
              f"({len(planes)} boundaries)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("transform", type=Path)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data" / "small")
    parser.add_argument("--amplitude", type=float, default=4.0, help="the known warp, voxels")
    parser.add_argument("--patch", type=int, default=128)
    parser.add_argument("--overlap", type=int, default=24)
    args = parser.parse_args()

    field = sitk.GetArrayFromImage(
        sitk.DisplacementFieldTransform(sitk.ReadTransform(str(args.transform))).GetDisplacementField()
    )
    field = field[..., ::-1]  # ITK writes (x, y, z) components; the key is (z, y, x)

    truth_path = args.data / "Truth.ome.zarr"
    if truth_path.exists():
        truth = read_zarr(truth_path)
        voxel = zarr.open_group(str(truth_path), mode="r").attrs["multiscales"][0]["datasets"][0][
            "coordinateTransformations"][0]["scale"][-1]
        residual(field / voxel, truth, args.amplitude)  # mm -> voxels
    seams(field, args.patch, args.overlap)


if __name__ == "__main__":
    main()
