"""The per-structure check for the LINC pair: carry a segmentation through the registration and compare.

The dandiset ships a SynthSeg-style segmentation of the FLASH T2* volume twice: in the FLASH's own
frame, and in XPCT space as the consortium moved it. Register the FLASH volume to the XPCT with the
same stages, push the native segmentation through that chain with nearest-neighbour sampling, and the
Dice per structure against the consortium's version is a number that does not depend on either
image's contrast.

    python evaluate_linc.py brain                                   # FLASH at 0.24 mm, brain only
    python stage1_linc.py --level 4 --moving data/linc/flash_brain_024.nii.gz --name flash
    python evaluate_linc.py pair --name flash
    python masks_linc.py data/linc/pair_flash
    impact-reg-konfai register Generic_Rigid_BSpline -f data/linc/pair_flash/Fixed.ome.zarr \
        -m data/linc/pair_flash/Moving.ome.zarr --fixed-mask ... --moving-mask ... -o out/linc_flash_mask ...
    python evaluate_linc.py dice --name flash --field out/linc_flash_mask/P000/Transform.h5
"""

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from konfai.utils.ome_zarr import write_ome_zarr

from stage1_linc import LPS, normalise

DATA = Path(__file__).parent / "data" / "linc"
# FreeSurfer colour table, the labels SynthSeg writes for one left hemisphere
LABELS = {
    2: "cerebral white matter", 3: "cerebral cortex", 4: "lateral ventricle", 5: "inferior lateral ventricle",
    7: "cerebellum white matter", 8: "cerebellum cortex", 10: "thalamus", 11: "caudate", 12: "putamen",
    13: "pallidum", 14: "3rd ventricle", 15: "4th ventricle", 16: "brain stem", 17: "hippocampus",
    18: "amygdala", 24: "CSF", 26: "accumbens", 28: "ventral DC",
}


def brain(name: str, voxel_mm: float = 0.24, widen_mm: float = 1.0) -> Path:
    """The moving image for the per-structure check: a smoothed copy at ``voxel_mm``, masked to the brain.

    The dMRI map the consortium ships is already zero outside the brain; the FLASH T2* is not, and the
    container around the hemisphere carries signal of its own. Left in, it won the orientation screen
    for the wrong one of two near-tied orientations. The mask is the consortium's own segmentation,
    widened by ``widen_mm``.
    """
    from scipy.ndimage import maximum_filter1d

    source = DATA / f"{name}_T2starw.nii.gz"
    small_path = DATA / f"{name}_T2starw_{int(voxel_mm * 100):03d}.nii.gz"
    if not small_path.exists():
        image = sitk.ReadImage(str(source), sitk.sitkFloat32)
        size = [int(round(n * s / voxel_mm)) for n, s in zip(image.GetSize(), image.GetSpacing())]
        smoothed = sitk.SmoothingRecursiveGaussian(image, sigma=voxel_mm / 2)
        small = sitk.Resample(smoothed, size, sitk.Transform(), sitk.sitkLinear, image.GetOrigin(), [voxel_mm] * 3,
                              image.GetDirection(), 0.0, sitk.sitkFloat32)
        sitk.WriteImage(small, str(small_path))
    small = sitk.ReadImage(str(small_path))
    inside = sitk.Resample(sitk.ReadImage(str(DATA / "dseg_native.nii.gz")), small, sitk.Transform(),
                           sitk.sitkNearestNeighbor, 0)
    mask = (sitk.GetArrayFromImage(inside) > 0).astype(np.uint8)
    for axis in range(3):
        mask = maximum_filter1d(mask, 2 * int(round(widen_mm / voxel_mm)) + 1, axis=axis)
    out = sitk.GetImageFromArray(sitk.GetArrayFromImage(small) * mask)
    out.CopyInformation(small)
    path = DATA / f"{name}_brain_{int(voxel_mm * 100):03d}.nii.gz"
    sitk.WriteImage(out, str(path), useCompression=True)
    print(f"{path.name}: {small.GetSize()} at {voxel_mm} mm, brain mask covers {mask.mean():.3f}")
    return path


def fixed_grid() -> sitk.Image:
    image = sitk.ReadImage(str(DATA / "XPCT_level4.mha"))
    image.SetDirection(LPS)
    return normalise(image)


def build_pair(name: str) -> None:
    fixed = fixed_grid()
    moving = normalise(sitk.ReadImage(str(DATA / f"{name}_affine.mha")))
    out = DATA / f"pair_{name}"
    out.mkdir(parents=True, exist_ok=True)
    for label, image in (("Fixed", fixed), ("Moving", moving)):
        array = sitk.GetArrayFromImage(image)
        write_ome_zarr(out / f"{label}.ome.zarr", array[None], spacing=image.GetSpacing(),
                       origin=image.GetOrigin(), chunks=(1, 64, 128, 128))
        print(f"{label}.ome.zarr {array.shape} @ {image.GetSpacing()[0]:.4f} mm")


def dice_table(result: np.ndarray, reference: np.ndarray) -> dict[int, float]:
    scores = {}
    for label in sorted(set(np.unique(reference)) - {0}):
        a, b = result == label, reference == label
        if b.sum() < 50:
            continue  # a structure a few voxels wide says nothing at 0.32 mm
        scores[int(label)] = 2 * (a & b).sum() / (a.sum() + b.sum())
    return scores


def carried_segmentation(name: str, moving: Path) -> tuple[sitk.Image, sitk.Image, sitk.Transform]:
    """The native segmentation, the same one turned as stage 1 turned the moving image, and stage 1's affine.

    Stage 1 turned the moving image about that image's own origin. The segmentation covers the same
    hemisphere from another corner, so turning it about its own origin would put it tens of mm away:
    it is turned about the moving image's origin instead.
    """
    rotation = np.load(DATA / f"{name}_orientation.npy")
    native = sitk.ReadImage(str(DATA / "dseg_native.nii.gz"))
    header = sitk.ImageFileReader()
    header.SetFileName(str(moving))
    header.ReadImageInformation()
    pivot = np.array(header.GetOrigin())
    oriented = sitk.Image(native)
    oriented.SetDirection((rotation @ np.array(native.GetDirection()).reshape(3, 3)).flatten().tolist())
    oriented.SetOrigin(tuple(pivot + rotation @ (np.array(native.GetOrigin()) - pivot)))
    return native, oriented, sitk.ReadTransform(str(DATA / f"{name}_affine.tfm"))


def dice(name: str, moving: Path, fields: list[Path]) -> None:
    fixed = fixed_grid()
    native, oriented, affine = carried_segmentation(name, moving)
    reference = sitk.GetArrayFromImage(sitk.Resample(
        sitk.ReadImage(str(DATA / "dseg_spaceXPCT.nii.gz")), fixed, sitk.Transform(), sitk.sitkNearestNeighbor, 0))

    states = [("as it comes", native, sitk.Transform()),
              ("orientation + affine", oriented, affine)]
    if fields:
        from linc import field_in_store_frame  # the run's pair went to OME-Zarr without its direction

        chain = sitk.CompositeTransform(3)
        chain.AddTransform(affine)                           # applied last
        for field in fields:                                 # each later field is applied before the earlier ones
            chain.AddTransform(field_in_store_frame(field))
        states.append(("+ " + " + ".join(f.parent.parent.name.replace("linc_flash", "").strip("_") or "run" for f in fields), oriented, chain))

    tables = []
    for label, image, transform in states:
        moved = sitk.GetArrayFromImage(sitk.Resample(image, fixed, transform, sitk.sitkNearestNeighbor, 0))
        tables.append((label, dice_table(moved, reference)))

    # the named structures of one left hemisphere; the reference also carries right-hemisphere labels,
    # hypointensities and hypothalamic subunits a few voxels wide, which say nothing at 0.32 mm
    structures = [s for s in sorted(set().union(*(t.keys() for _, t in tables))) if s in LABELS]
    header = "structure".ljust(28) + "".join(label[:22].rjust(24) for label, _ in tables)
    print(header)
    for s in structures:
        print(LABELS[s].ljust(28) + "".join(f"{t.get(s, float('nan')):24.3f}" for _, t in tables))
    print(f"mean of {len(structures)} structures".ljust(28)
          + "".join(f"{np.mean([t.get(s, 0.0) for s in structures]):24.3f}" for _, t in tables))


def anatomy(name: str, moving: Path, fields: list[Path]) -> None:
    """The FLASH segmentation, carried by this pipeline and by the consortium, judged on the XPCT alone.

    An Otsu threshold splits the XPCT into tissue and fluid with neither registration's help. Inside one
    region common to all the alignments, the tissue labels should cover the tissue and the ventricle labels
    the fluid; a smaller mask cannot win this by sitting inside the tissue, because Dice counts coverage too.
    """
    import linc
    from scipy.ndimage import binary_dilation

    xpct = sitk.ReadImage(str(DATA / "XPCT_level4.mha"))
    xpct.SetDirection(LPS)
    otsu = sitk.OtsuThresholdImageFilter()
    otsu.SetInsideValue(0)
    otsu.SetOutsideValue(1)
    otsu.Execute(xpct)
    tissue_xpct = sitk.GetArrayFromImage(xpct) >= otsu.GetThreshold()

    _, oriented, affine = carried_segmentation(name, moving)
    states = {"this pipeline, affine": sitk.Resample(oriented, xpct, affine, sitk.sitkNearestNeighbor, 0)}
    if fields:
        chain = sitk.CompositeTransform(3)
        chain.AddTransform(affine)
        for field in fields:
            chain.AddTransform(linc.field_in_store_frame(field))
        states["this pipeline, + " + " + ".join(f.parent.parent.name for f in fields)] = \
            sitk.Resample(oriented, xpct, chain, sitk.sitkNearestNeighbor, 0)
    states["the consortium"] = sitk.Resample(sitk.ReadImage(str(DATA / "dseg_spaceXPCT.nii.gz")), xpct,
                                             sitk.Transform(), sitk.sitkNearestNeighbor, 0)
    states = {label: sitk.GetArrayFromImage(image) for label, image in states.items()}

    ventricles, tissue_labels = (4, 5), (2, 3, 7, 8, 10, 11, 12, 13, 16, 17, 18, 26, 28)
    mm = xpct.GetSpacing()[0]
    widen = lambda mask, millimetres: binary_dilation(mask, iterations=int(round(millimetres / mm)))  # noqa: E731
    brain = widen(np.any([s > 0 for s in states.values()], axis=0), 3)
    around_ventricles = widen(np.any([np.isin(s, ventricles) for s in states.values()], axis=0), 3)
    inside_tissue = widen(np.any([np.isin(s, tissue_labels) for s in states.values()], axis=0), 1)
    xpct_tissue, xpct_fluid = tissue_xpct & brain, ~tissue_xpct & around_ventricles & inside_tissue
    voxel_ml = mm ** 3 / 1000

    def dice(a: np.ndarray, b: np.ndarray) -> float:
        return float(2 * (a & b).sum() / (a.sum() + b.sum()))

    print(f"XPCT alone, Otsu {otsu.GetThreshold():.0f}: {xpct_tissue.sum() * voxel_ml:.1f} ml of tissue, "
          f"{xpct_fluid.sum() * voxel_ml:.1f} ml of ventricular fluid")
    print(f"{'':44s} Dice tissue   Dice ventricles   tissue ml   ventricle ml")
    for label, segmentation in states.items():
        tissue, ventricle = np.isin(segmentation, tissue_labels), np.isin(segmentation, ventricles)
        print(f"  {label[:42]:42s} {dice(tissue, xpct_tissue):9.3f}   {dice(ventricle, xpct_fluid):13.3f}   "
              f"{tissue.sum() * voxel_ml:9.1f}   {ventricle.sum() * voxel_ml:10.1f}")


def coarse(runs: list[Path]) -> None:
    """A whole-hemisphere run on the level-4 dMRI pair, scored against the consortium's alignment of the same map.

    The source MRI goes through the affine and then the run's field, onto the level-4 grid. Mutual
    information is what the engine maximised; the correlation with the consortium's map and the brain
    Dice are what it never saw.
    """
    import linc
    from stage1_linc import mutual_information

    fixed_image = fixed_grid()
    fixed = sitk.GetArrayFromImage(fixed_image)
    source = normalise(sitk.ReadImage(str(DATA / "dmri_oriented.mha")))
    reference = sitk.GetArrayFromImage(sitk.Resample(normalise(sitk.ReadImage(str(DATA / "dmri_reference.nii.gz"))),
                                                     fixed_image, sitk.Transform(), sitk.sitkLinear, 0.0))
    tissue = reference > 0.02
    affine = sitk.ReadTransform(str(DATA / "affine.tfm"))
    chain = sitk.CompositeTransform(3)
    chain.AddTransform(affine)                                              # applied last
    for run in runs:                                                        # each later run is applied first
        chain.AddTransform(linc.field_in_store_frame(run / "Transform.h5"))
    print("level 4, 0.32 mm                 MI      correlation with the consortium   brain Dice")
    for label, transform in (("affine only", affine), (" + ".join(r.parent.name for r in runs), chain)):
        moved = sitk.GetArrayFromImage(sitk.Resample(source, fixed_image, transform, sitk.sitkLinear, 0.0))
        brain = moved > 0.02
        print(f"  {label:28s}  {mutual_information(fixed, moved):.4f}   {np.corrcoef(moved[tissue], reference[tissue])[0, 1]:.4f}"
              f"                          {2 * (brain & tissue).sum() / (brain.sum() + tissue.sum()):.3f}")


ROOT = Path(__file__).parent
NATIVE_RUN = ROOT / "out" / "linc_native_bend100" / "P000"
COARSE_FIELD = ROOT / "out" / "linc_coarse_bend100" / "P000" / "Transform.h5"
FINE_FIELD = ROOT / "out" / "linc_tiled3_bend100" / "P000" / "Transform.h5"
PRIOR_FIELDS: tuple[Path, ...] = ()  # fields run on the level-4 pair before the coarse deformable, in run order


def native_states(grid: sitk.Image, run: Path = NATIVE_RUN, coarse_field: Path = COARSE_FIELD,
                  fine_field: Path = FINE_FIELD, step: int = 4,
                  prior_fields: tuple[Path, ...] = PRIOR_FIELDS) -> tuple[np.ndarray, np.ndarray]:
    """The MRI on ``grid`` at the start of the native run and after it, both resampled from the source.

    The run's own Moved image warps the 15 mm window it was handed, so wherever the field points past
    the window's edge it reads zeros that are not in the MRI. The source has no such edge: carry it
    through the affine, the level-4 field, the level-3 field, and then the native field.
    """
    import linc

    source = normalise(sitk.ReadImage(str(DATA / "dmri_oriented.mha")))
    chain = sitk.CompositeTransform(3)
    chain.AddTransform(sitk.ReadTransform(str(DATA / "affine.tfm")))       # applied last
    for field in prior_fields:
        chain.AddTransform(linc.field_in_store_frame(field, grid))
    chain.AddTransform(linc.field_in_store_frame(coarse_field, grid))
    chain.AddTransform(linc.field_in_store_frame(fine_field, grid))
    start = sitk.GetArrayFromImage(sitk.Resample(source, grid, chain, sitk.sitkLinear, 0.0))
    chain.AddTransform(linc.field_in_store_frame(run / "Transform.h5", grid, step=step))  # applied first
    after = sitk.GetArrayFromImage(sitk.Resample(source, grid, chain, sitk.sitkLinear, 0.0))
    return start, after


def native(run: Path, coarse_field: Path, fine_field: Path, patch: int, overlap: int, step: int = 4) -> None:
    """The native window before and after the tiled run, on every step-th voxel, against the reference; then the seams."""
    from linc import field_header, read_field
    from score import seams
    from stage1_linc import mutual_information
    from stage4_native_linc import written_window

    grid = written_window(DATA / "pair0" / "Fixed.ome.zarr")[::step, ::step, ::step]
    fixed = sitk.GetArrayFromImage(grid)
    start, after = native_states(grid, run, coarse_field, fine_field, step)
    reference = sitk.GetArrayFromImage(sitk.Resample(normalise(sitk.ReadImage(str(DATA / "dmri_reference.nii.gz"))),
                                                     grid, sitk.Transform(), sitk.sitkLinear, 0.0))

    inner = int(round(4.0 / grid.GetSpacing()[0]))  # 4 mm in from every face of the window
    middle = np.s_[inner:-inner, inner:-inner, inner:-inner]
    tissue = reference[middle] > 0.02

    def agreement(moving: np.ndarray) -> float:  # the check the engine never saw: the consortium's alignment of the same map
        return float(np.corrcoef(moving[middle][tissue], reference[middle][tissue])[0, 1])

    print(f"native window, every {step}th voxel of 20.07 um      MI whole   MI 4 mm in   r with the consortium, 4 mm in")
    for label, moving in (("the start of the native run", start), ("after the tiled native run", after),
                          ("the consortium's alignment", reference)):
        print(f"  {label:28s}  {mutual_information(fixed, moving):.4f}     {mutual_information(fixed[middle], moving[middle]):.4f}"
              f"       {agreement(moving):.4f}")

    size = field_header(run / "Transform.h5")[0]
    centre = size[2] // 2  # a slab of eight planes carries every y and x boundary, and costs 8 planes, not the field
    slab = read_field(run / "Transform.h5", (0, 0, centre - 4), (size[0], size[1], centre + 4))
    seams(slab[..., ::-1], patch, overlap)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    zero = sub.add_parser("brain", help="the FLASH at 0.24 mm, masked to the brain")
    zero.add_argument("--name", default="flash")
    one = sub.add_parser("pair", help="write the coarse pair for the deformable run")
    one.add_argument("--name", default="flash")
    two = sub.add_parser("dice", help="Dice per structure against the consortium's segmentation")
    two.add_argument("--name", default="flash")
    two.add_argument("--moving", type=Path, default=DATA / "flash_brain_024.nii.gz", help="the image stage 1 oriented")
    two.add_argument("--field", type=Path, nargs="*", default=[], help="the runs' fields, in the order they were run")
    five = sub.add_parser("anatomy", help="judge this pipeline and the consortium on the XPCT alone")
    five.add_argument("--name", default="flash")
    five.add_argument("--moving", type=Path, default=DATA / "flash_brain_024.nii.gz")
    five.add_argument("--field", type=Path, nargs="*", default=[], help="the runs' fields, in the order they were run")
    four = sub.add_parser("coarse", help="score a whole-hemisphere level-4 run against the consortium")
    four.add_argument("--run", type=Path, nargs="+", default=[Path(__file__).parent / "out" / "linc_coarse_bend100" / "P000"],
                      help="the runs, in the order they were run")
    three = sub.add_parser("native", help="score the tiled run on the native window")
    three.add_argument("--run", type=Path, default=NATIVE_RUN)
    three.add_argument("--coarse-field", type=Path, default=COARSE_FIELD)
    three.add_argument("--fine-field", type=Path, default=FINE_FIELD)
    three.add_argument("--patch", type=int, default=256)
    three.add_argument("--overlap", type=int, default=128)
    args = parser.parse_args()
    if args.command == "brain":
        brain(args.name)
    elif args.command == "pair":
        build_pair(args.name)
    elif args.command == "anatomy":
        anatomy(args.name, args.moving, args.field)
    elif args.command == "coarse":
        coarse(args.run)
    elif args.command == "dice":
        dice(args.name, args.moving, args.field)
    else:
        native(args.run, args.coarse_field, args.fine_field, args.patch, args.overlap)


if __name__ == "__main__":
    main()
