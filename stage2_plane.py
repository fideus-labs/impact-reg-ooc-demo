"""Build the pair the data actually supports: one plane per modality, on a common grid.

The OCT images a 0.21 mm block face; the light-sheet images the 0.75 mm section that was cut off
it. No depth correspondence between the two is recoverable, so the pair is built in plane: each
modality is projected over the slab it holds, and the plane is stacked into a thin volume (the
engines are 3-D). The stack is deliberately coarse in z, so any residual z displacement lands on an
identical plane and changes nothing.

    python stage2_plane.py --spacing 0.025                    # the whole section, 25 um
    python stage2_plane.py --spacing 0.003 --window 23 11 6.1  # one window at native resolution
"""

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import zarr
from konfai.utils.ome_zarr import write_ome_zarr

from stage2_native import to_3d

DATA = Path(__file__).parent / "data" / "dandi"
DEPTH, Z_SPACING = 8, 0.4  # planes, mm: coarse on purpose, see the module docstring


def mean_slab(path: Path) -> sitk.Image:
    """One plane: the mean over the slab the modality holds, on its own grid."""
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image).mean(0)[None]
    out = sitk.GetImageFromArray(array.astype(np.float32))
    out.SetSpacing(image.GetSpacing())
    out.SetOrigin(image.GetOrigin())
    return out


def window_zarr(store: Path, spacing: float) -> sitk.Image:
    """One plane out of a native-resolution window already written by stage2_native."""
    group = zarr.open_group(str(store), mode="r")
    dataset = dict(group.attrs)["multiscales"][0]["datasets"][0]
    array = np.asarray(group[dataset["path"]])
    array = (array[0] if array.ndim == 4 else array).mean(0)[None]
    translation = next(t["translation"][-3:] for t in dataset["coordinateTransformations"]
                       if t["type"] == "translation")
    out = sitk.GetImageFromArray(array.astype(np.float32))
    out.SetSpacing((spacing, spacing, spacing))
    out.SetOrigin((translation[2], translation[1], 0.0))
    return out


def stack(source: sitk.Image, grid: sitk.Image, transform: sitk.Transform) -> sitk.Image:
    plane = sitk.GetArrayFromImage(sitk.Resample(source, grid, transform, sitk.sitkLinear, 0.0))
    volume = sitk.GetImageFromArray(np.repeat(plane, DEPTH, axis=0))
    volume.SetSpacing((grid.GetSpacing()[0], grid.GetSpacing()[1], Z_SPACING))
    volume.SetOrigin(grid.GetOrigin())
    return volume


def normalise(image: sitk.Image) -> np.ndarray:
    array = sitk.GetArrayFromImage(image)
    return np.clip(array / (np.percentile(array, 99.5) + 1e-6), 0, 1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spacing", type=float, default=0.025, help="in-plane grid, mm")
    parser.add_argument("--window", nargs=3, type=float, metavar=("X", "Y", "SIZE"),
                        help="centre x, centre y and side in mm; reads the native windows instead")
    parser.add_argument("--fixed-window", type=Path, default=DATA / "cortex")
    parser.add_argument("--moving-window", type=Path, default=DATA / "cortex")
    parser.add_argument("--coarse-transform", type=Path,
                        help="the displacement field of the coarse deformable run; the native pair "
                             "starts from it, so each tile only has the fine residual left to solve")
    parser.add_argument("--out", type=Path, default=DATA / "plane")
    args = parser.parse_args()
    affine = to_3d(sitk.ReadTransform(str(DATA / "coarse_affine.tfm")))

    if args.window:
        fixed_source = window_zarr(args.fixed_window / "Fixed.ome.zarr", args.spacing)
        moving_source = window_zarr(args.moving_window / "Moving.ome.zarr", args.spacing)
        affine = sitk.Transform()  # stage2_native already resampled the moving through the affine
        if args.coarse_transform:
            affine = sitk.ReadTransform(str(args.coarse_transform))
            print(f"starting from the coarse field: {args.coarse_transform}")
        origin = fixed_source.GetOrigin()
        size = [fixed_source.GetSize()[0], fixed_source.GetSize()[1], 1]
    else:
        fixed_source = mean_slab(DATA / "overview_OCT.mha")
        moving_source = mean_slab(DATA / "overview_SPIM.mha")
        origin = fixed_source.GetOrigin()
        extent = [n * s for n, s in zip(fixed_source.GetSize()[:2], fixed_source.GetSpacing()[:2])]
        size = [int(np.ceil(extent[0] / args.spacing)), int(np.ceil(extent[1] / args.spacing)), 1]

    grid = sitk.Image(size, sitk.sitkFloat32)
    grid.SetSpacing((args.spacing, args.spacing, args.spacing))
    grid.SetOrigin(origin)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, source, transform in (("Fixed", fixed_source, sitk.Transform()), ("Moving", moving_source, affine)):
        volume = stack(source, grid, transform)
        array = normalise(volume)
        write_ome_zarr(args.out / f"{name}.ome.zarr", array[None], spacing=volume.GetSpacing(),
                       origin=volume.GetOrigin(), chunks=(1, DEPTH, 256, 256))
        print(f"{name}.ome.zarr {array.shape} @ {[round(s, 4) for s in volume.GetSpacing()]} mm")


if __name__ == "__main__":
    main()
