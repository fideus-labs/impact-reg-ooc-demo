"""Stage 2: cut one native-resolution window out of the two stores and put it on the fixed grid.

Nothing is downloaded whole. The window is a bounded request: zarr fetches the chunks the box
intersects and no others, so a 6 x 6 mm field of view at 3 um costs about 2 GB out of the 81 GB the
two stores hold. The moving window is read through the stage-1 affine, so what lands on disk is
already globally aligned and only the local deformation is left for the tiled run.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import zarr
from konfai.utils.ome_zarr import write_ome_zarr

DATA = Path(__file__).parent / "data" / "dandi"
STORES = {
    "OCT": "a6fa8bd2-c46b-445b-a219-e97dac4f2ee8",
    "NeuN": "c36e0b26-ac42-443a-94f9-1fa2cda794b5",
    "Calretinin": "cbee4a8c-8bc7-4b25-ba17-c6978b3e2157",
}
SPACING = {"OCT": (0.003, 0.003, 0.003), "NeuN": (0.0036,) * 3, "Calretinin": (0.0036,) * 3}  # mm (z, y, x)


def store(name: str):
    return zarr.open_group(f"s3://dandiarchive/zarr/{STORES[name]}", mode="r", storage_options={"anon": True})["0"]


def as_image(array: np.ndarray, spacing_zyx, origin_zyx=(0.0, 0.0, 0.0)) -> sitk.Image:
    image = sitk.GetImageFromArray(array.astype(np.float32))
    image.SetSpacing(tuple(float(s) for s in spacing_zyx[::-1]))
    image.SetOrigin(tuple(float(o) for o in origin_zyx[::-1]))
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=float, default=21.0, help="window centre in the fixed frame, mm")
    parser.add_argument("--y", type=float, default=19.0)
    parser.add_argument("--size", type=int, default=2048, help="in-plane window, fixed voxels")
    parser.add_argument("--depth", type=int, default=64, help="z extent, fixed voxels")
    parser.add_argument("--out", type=Path, default=DATA / "window")
    parser.add_argument("--moving-store", default="NeuN", choices=["NeuN", "Calretinin"])
    parser.add_argument("--skip-fixed", action="store_true", help="reuse a fixed window already written")
    parser.add_argument("--coarse-field", type=Path, nargs="+",
                        help="displacement field(s) of the coarse deformable run(s), in run order (a second pass "
                             "after the first); applied after the affine, so the window that lands on disk is "
                             "already corrected for them")
    parser.add_argument("--margin", type=float, default=0.3, help="margin around the mapped box, mm")
    args = parser.parse_args()

    affine = sitk.ReadTransform(str(DATA / "coarse_affine.tfm"))  # 2-D, fixed -> moving, in mm
    fixed_spacing, moving_spacing = SPACING["OCT"], SPACING[args.moving_store]

    # The fixed window, straight out of the 50 GiB store.
    y0 = int(args.y / fixed_spacing[1]) - args.size // 2
    x0 = int(args.x / fixed_spacing[2]) - args.size // 2
    fetched = 0
    if args.skip_fixed:
        fixed_array = None
        print("fixed window: reusing the one already written")
    else:
        start = time.perf_counter()
        fixed_array = store("OCT")[: args.depth, y0 : y0 + args.size, x0 : x0 + args.size]
        fetched += fixed_array.nbytes
        print(f"fixed window {fixed_array.shape} read in {time.perf_counter() - start:.0f} s")

    # The same field of view in the moving store, through the coarse affine, with a margin for the
    # local deformation the tiled run still has to absorb.
    corners_mm = [(x * fixed_spacing[2], y * fixed_spacing[1]) for y in (y0, y0 + args.size) for x in (x0, x0 + args.size)]
    mapped = np.array([affine.TransformPoint(c) for c in corners_mm])
    margin = args.margin
    lo = (mapped.min(0) - margin) / np.array(moving_spacing[1:][::-1])
    hi = (mapped.max(0) + margin) / np.array(moving_spacing[1:][::-1])
    mx0, my0 = int(max(lo[0], 0)), int(max(lo[1], 0))
    mx1, my1 = int(hi[0]), int(hi[1])
    depth = int(args.depth * fixed_spacing[0] / moving_spacing[0])
    start = time.perf_counter()
    # the moving box depends on the affine and the margin only, so it is cached: a new coarse field re-resamples it for free
    cache = DATA / "cache" / f"{args.moving_store}_{depth}_{my0}_{my1}_{mx0}_{mx1}.npy"
    if cache.exists():
        moving_array = np.load(cache)
    else:
        moving_array = store(args.moving_store)[:depth, my0:my1, mx0:mx1]
        cache.parent.mkdir(exist_ok=True)
        np.save(cache, moving_array)
    fetched += moving_array.nbytes
    print(f"moving window {moving_array.shape} read in {time.perf_counter() - start:.0f} s")

    if fixed_array is None:
        fixed = sitk.GetImageFromArray(np.zeros((args.depth, args.size, args.size), np.float32))
        fixed.SetSpacing(fixed_spacing[::-1])
        fixed.SetOrigin((x0 * fixed_spacing[2], y0 * fixed_spacing[1], 0.0))
    else:
        fixed = as_image(fixed_array, fixed_spacing, (0.0, y0 * fixed_spacing[1], x0 * fixed_spacing[2]))
    moving = as_image(moving_array, moving_spacing, (0.0, my0 * moving_spacing[1], mx0 * moving_spacing[2]))
    mapping = to_3d(affine)
    if args.coarse_field:
        composite = sitk.CompositeTransform(3)
        composite.AddTransform(mapping)                    # applied last
        for field in args.coarse_field:                    # SimpleITK applies the last one added first:
            composite.AddTransform(sitk.ReadTransform(str(field)))  # a later pass maps into the earlier one's output
        mapping = composite
        print(f"window read through the coarse field(s): {', '.join(map(str, args.coarse_field))}")
    moved = sitk.Resample(moving, fixed, mapping, sitk.sitkLinear, 0.0)

    args.out.mkdir(parents=True, exist_ok=True)
    written = (("Moving", moved),) if args.skip_fixed else (("Fixed", fixed), ("Moving", moved))
    for name, image in written:
        array = sitk.GetArrayFromImage(image)
        array = array / (np.percentile(array, 99.5) + 1e-6)
        write_ome_zarr(
            args.out / f"{name}.ome.zarr", np.clip(array, 0, 1).astype(np.float32)[None],
            spacing=image.GetSpacing(), origin=image.GetOrigin(), chunks=(1, 64, 256, 256),
        )
        print(f"{name}.ome.zarr {array.shape} @ {[round(s, 4) for s in image.GetSpacing()]} mm")
    print(f"fetched {fetched / 2**30:.2f} GiB out of the 81 GiB the two stores hold")


def to_3d(affine: sitk.Transform) -> sitk.Transform:
    """The stage-1 in-plane affine, lifted to 3-D with z left alone."""
    if affine.GetName() == "CompositeTransform":
        affine = sitk.CompositeTransform(affine).GetNthTransform(0)
    a = sitk.AffineTransform(affine)
    matrix, translation, centre = a.GetMatrix(), a.GetTranslation(), a.GetCenter()
    out = sitk.AffineTransform(3)
    out.SetMatrix([matrix[0], matrix[1], 0.0, matrix[2], matrix[3], 0.0, 0.0, 0.0, 1.0])
    out.SetTranslation([translation[0], translation[1], 0.0])
    out.SetCenter([centre[0], centre[1], 0.0])
    return out


if __name__ == "__main__":
    main()
