"""Stage 4: the native grid. One window of level 0, at 20.07 um, read straight out of the 908 GiB store.

Stages 1 to 3 solved the global part and the coarse deformation on levels that fit. This one works
where the analysis lives: the store's own grid. The window is a bounded request — zarr fetches the
64^3 chunks the box intersects and no others — and the moving image arrives through the chain the
earlier stages produced, so the tiles have only the fine residual left.

    python stage4_native_linc.py --size 768                 # centred on the tissue's centre of mass
    python stage4_native_linc.py --centre -76 -60 85 --size 768
    python stage4_native_linc.py --reuse-fixed              # rebuild the moving side, no new read
"""

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import zarr
from konfai.utils.ome_zarr import write_ome_zarr

import linc
from stage1_linc import LPS, normalise

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "linc"


def tissue_centre() -> tuple[float, float, float]:
    """The centre of mass of the hemisphere, in mm, from a coarse level."""
    image = sitk.ReadImage(str(DATA / "XPCT_level4.mha"))
    image.SetDirection(LPS)
    array = sitk.GetArrayFromImage(image)
    mask = array > np.percentile(array, 85)
    centre_index = np.argwhere(mask).mean(0)[::-1]  # (z, y, x) -> (x, y, z)
    return image.TransformContinuousIndexToPhysicalPoint(centre_index.tolist())


def written_window(store: Path) -> sitk.Image:
    """A window this script already wrote, back in the store's frame: the zarr kept no direction."""
    group = zarr.open_group(str(store), mode="r")
    dataset = dict(group.attrs)["multiscales"][0]["datasets"][0]
    array = np.asarray(group[dataset["path"]])
    array = array[0] if array.ndim == 4 else array
    scale = dataset["coordinateTransformations"][0]["scale"][-3:]
    origin = next(t["translation"][-3:] for t in dataset["coordinateTransformations"] if t["type"] == "translation")
    image = sitk.GetImageFromArray(array.astype(np.float32))
    image.SetSpacing(tuple(scale[::-1]))
    image.SetOrigin(tuple(origin[::-1]))
    image.SetDirection(LPS)
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--centre", nargs=3, type=float, metavar=("X", "Y", "Z"), help="window centre, mm")
    parser.add_argument("--size", type=int, default=768, help="window side, voxels of level 0")
    parser.add_argument("--reuse-fixed", action="store_true", help="keep the window already read")
    parser.add_argument("--prior-field", type=Path, nargs="*", default=[],
                        help="fields run on the level-4 pair before the coarse deformable, in run order")
    parser.add_argument("--coarse-field", type=Path, default=ROOT / "out" / "linc_coarse_bend100" / "P000" / "Transform.h5")
    parser.add_argument("--fine-field", type=Path,
                        help="the field of the deformable at a finer level, e.g. level 3; applied before the coarse one")
    parser.add_argument("--out", type=Path, default=DATA / "pair0")
    args = parser.parse_args()

    if args.reuse_fixed:
        fixed = written_window(args.out / "Fixed.ome.zarr")
        print(f"fixed window reused: {fixed.GetSize()} @ {fixed.GetSpacing()[0] * 1000:.2f} um")
    else:
        centre = tuple(args.centre) if args.centre else tissue_centre()
        voxel = linc.levels()["0"]
        print(f"window centre {tuple(round(c, 1) for c in centre)} mm, "
              f"{args.size} voxels of {voxel * 1000:.2f} um = {args.size * voxel:.1f} mm on a side")
        fixed = normalise(linc.read_window("0", centre, args.size * voxel))

    moving = normalise(sitk.ReadImage(str(DATA / "dmri_oriented.mha")))
    chain = sitk.CompositeTransform(3)
    # a point of the native grid goes through the finest field first, then the coarser one, then the affine
    chain.AddTransform(sitk.ReadTransform(str(DATA / "affine.tfm")))       # applied last
    # only the box of each field over the window is read, never the whole hemisphere's field
    for field in args.prior_field:
        chain.AddTransform(linc.field_in_store_frame(field, fixed))
    chain.AddTransform(linc.field_in_store_frame(args.coarse_field, fixed))
    if args.fine_field:
        chain.AddTransform(linc.field_in_store_frame(args.fine_field, fixed))      # applied first
        print(f"starting from the finer field: {args.fine_field}")
    moved = sitk.Resample(moving, fixed, chain, sitk.sitkLinear, 0.0)

    args.out.mkdir(parents=True, exist_ok=True)
    written = (("Moving", moved),) if args.reuse_fixed else (("Fixed", fixed), ("Moving", moved))
    for name, image in written:
        array = sitk.GetArrayFromImage(image)
        write_ome_zarr(args.out / f"{name}.ome.zarr", array[None], spacing=image.GetSpacing(),
                       origin=image.GetOrigin(), chunks=(1, 64, 256, 256))
        print(f"{name}.ome.zarr {array.shape} @ {image.GetSpacing()[0] * 1000:.2f} um "
              f"({array.nbytes / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
