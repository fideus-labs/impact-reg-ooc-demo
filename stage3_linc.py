"""Stage 3 for the LINC pair: the same refinement, one pyramid level finer, in tiles.

Stages 1 and 2 solved the global part on a level that fits in memory. Stage 3 moves to the next
level down and never holds it: the volume is cut into tiles, each tile is registered on its own, and
the displacement fields are blended on their overlaps.

Nothing about the command changes with the level. Going further down the pyramid — 14 GiB at 80 um,
113 GiB at 40 um, 908 GiB at 20 um — changes the number of tiles and nothing else.

    python stage3_linc.py --level 3
"""

import argparse
from pathlib import Path

import SimpleITK as sitk
from konfai.utils.ome_zarr import write_ome_zarr

import linc
from stage1_linc import LPS, normalise

DATA = Path(__file__).parent / "data" / "linc"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="3")
    parser.add_argument("--prior-field", type=Path, nargs="*", default=[],
                        help="fields run on the level-4 pair before the coarse deformable, in run order")
    parser.add_argument("--out", type=Path, help="the pair's directory; data/linc/pair{level} by default")
    parser.add_argument("--coarse-field", type=Path,
                        default=Path(__file__).parent / "out" / "linc_coarse_bend100" / "P000" / "Transform.h5")
    args = parser.parse_args()

    fixed = sitk.ReadImage(str(DATA / f"XPCT_level{args.level}.mha"))
    fixed.SetDirection(LPS)
    fixed = normalise(fixed)

    moving = normalise(sitk.ReadImage(str(DATA / "dmri_oriented.mha")))
    chain = sitk.CompositeTransform(3)
    chain.AddTransform(sitk.ReadTransform(str(DATA / "affine.tfm")))       # applied last
    for field in args.prior_field:                                          # each later field is applied first
        chain.AddTransform(linc.field_in_store_frame(field))
    if args.coarse_field.exists():
        chain.AddTransform(linc.field_in_store_frame(args.coarse_field))    # applied first
        print(f"starting from the coarse field: {args.coarse_field}")
    moved = sitk.Resample(moving, fixed, chain, sitk.sitkLinear, 0.0)

    out = args.out or DATA / f"pair{args.level}"
    out.mkdir(parents=True, exist_ok=True)
    for name, image in (("Fixed", fixed), ("Moving", moved)):
        array = sitk.GetArrayFromImage(image)
        write_ome_zarr(out / f"{name}.ome.zarr", array[None], spacing=image.GetSpacing(),
                       origin=image.GetOrigin(), chunks=(1, 64, 128, 128))
        print(f"{name}.ome.zarr {array.shape} @ {[round(s, 4) for s in image.GetSpacing()]} mm "
              f"({array.nbytes / 2**20:.0f} MiB)")


if __name__ == "__main__":
    main()
