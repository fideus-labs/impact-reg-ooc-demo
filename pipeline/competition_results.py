"""One run of the competition pair as the organisers' result files, in the layout of the APEX 2026 registration
competition: the FA carried into the retardance's space and the retardance carried into the FA's.

    python pipeline/competition_results.py APEX_FIREANTS_SYN_CC subject_v      (pixi run nifti <preset> <subject>)

The organisers' folder is competition/ (their Competition_Results), with this repository as Team C:

    Data/Subject_V/PSOCT.nii.gz                        the competition's inputs, linked from data/input/<subject>/
    Data/Subject_V/DMRI.nii.gz                         (Ret_slide_deck.nii.gz and dti_FA.nii.gz)
    Team_C_method_<n>/Subject_V/DMRI_to_PSOCT.nii.gz   the FA in the retardance's space, on PSOCT.nii.gz's own grid
    Team_C_method_<n>/Subject_V/PSOCT_to_DMRI.nii.gz   the retardance in the FA's space, on DMRI.nii.gz's own grid
    Team_C_method_<n>/README.md                        the method

The method number is the preset's row in the README's table (METHODS below). A run is the field u of
out/<subject>/<preset>/P000/Transform.h5 on the 0.4 mm pair grid, after the stage-1 affine A of
out/<subject>/stage1_affine.tfm: a point x of the retardance's frame corresponds to A(x + u(x)) in the FA's, so
DMRI_to_PSOCT(x) = FA(A(x + u(x))), sampled linearly on the retardance's own grid. PSOCT_to_DMRI needs the inverse,
which the run does not write: the inverse field v is computed by fixed-point iteration (ITK's
InvertDisplacementFieldImageFilter) and PSOCT_to_DMRI(y) = Ret(A⁻¹(y) + v(A⁻¹(y))), the retardance smoothed first as
the pair's Fixed was (anti-aliasing for the 0.4 mm grid). Everything is computed in the frame of the OME-Zarr stores
`pixi run convert` wrote, the frame the affine and the field are in; the stores hold the NIfTI files' own voxels, so
each output is written on its reference's voxel grid with that NIfTI's placement, and overlays the organisers' Data
file exactly. Intensities are the inputs' own, untouched.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_pair import read_ome_zarr  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
INPUTS = {"PSOCT": "Ret_slide_deck", "DMRI": "dti_FA"}  # the organisers' names of the competition's inputs
METHODS = {  # the organisers' method number of each preset: the rows of the README's table, in order
    "APEX_FA_RET_BSPLINE": (1, "elastix: B-spline, Mattes mutual information + bending energy"),
    "APEX_FIREANTS_SYN_CC": (2, "FireANTs: symmetric diffeomorphic SyN, local cross-correlation"),
    "APEX_FIREANTS_TS": (3, "FireANTs: SyN with the IMPACT loss, TotalSegmentator features instead of intensities"),
    "APEX_CONVEXADAM_MIND": (4, "ConvexAdam: coupled convex optimisation refined with Adam, on MIND features"),
}
REPOSITORY = "https://github.com/fideus-labs/impact-reg-ooc-demo"


def subject_folder(subject: str) -> str:
    """subject_v -> Subject_V."""
    stem = subject.removeprefix("subject_")
    return f"Subject_{stem.upper()}"


def link_inputs(inputs: Path, folder: Path) -> None:
    """Data/Subject_X/{PSOCT,DMRI}.nii.gz as relative links to the inputs; a real file there (the organisers' own copy) is kept."""
    folder.mkdir(parents=True, exist_ok=True)
    for name, stem in INPUTS.items():
        link = folder / f"{name}.nii.gz"
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            continue
        link.symlink_to(os.path.relpath(inputs / f"{stem}.nii.gz", folder))


def read_transforms(subject_out: Path, preset: str, subject: str) -> tuple[sitk.AffineTransform, sitk.Image]:
    """The stage-1 affine and the run's displacement field (on the pair grid, mm)."""
    affine_path, field_path = subject_out / "stage1_affine.tfm", subject_out / preset / "P000" / "Transform.h5"
    for path, task in ((affine_path, f"pixi run prepare {subject}"), (field_path, f"pixi run register {preset} {subject}")):
        if not path.exists():
            raise SystemExit(f"{path}: not here ({task} writes it)")
    affine = sitk.ReadTransform(str(affine_path))
    if affine.GetName() == "CompositeTransform":
        affine = sitk.CompositeTransform(affine).GetNthTransform(0)
    affine = sitk.AffineTransform(affine)
    transform = sitk.ReadTransform(str(field_path))
    if transform.GetName() == "CompositeTransform":
        transform = sitk.CompositeTransform(transform).GetNthTransform(0)
    return affine, sitk.DisplacementFieldTransform(transform).GetDisplacementField()


def as_transform(field: sitk.Image) -> sitk.DisplacementFieldTransform:
    return sitk.DisplacementFieldTransform(sitk.Image(field))  # the constructor takes the image over; keep the field


def inverse_residual(field: sitk.Image, inverse: sitk.Image, tissue: np.ndarray) -> str:
    """|x - v(u(x))| in mm on the field's grid inside the tissue: how exact the inverse is, as one line. The worst
    voxels lie where the tissue touches the grid's edge, where u points out of the grid; where the field folds
    (ConvexAdam) it has no inverse and the line says how much of the tissue that is."""
    both = sitk.CompositeTransform([as_transform(inverse), as_transform(field)])  # the last is applied first
    residual = sitk.TransformToDisplacementField(both, sitk.sitkVectorFloat64, field.GetSize(), field.GetOrigin(),
                                                 field.GetSpacing(), field.GetDirection())
    r = np.linalg.norm(sitk.GetArrayFromImage(residual), axis=-1)[tissue]
    voxel = min(field.GetSpacing())
    return (f"inverse residual median {np.median(r):.3f} mm, 99th percentile {np.percentile(r, 99):.3f} mm, "
            f"max {r.max():.2f} mm; {np.mean(r > voxel):.1%} of the tissue beyond one voxel ({voxel:g} mm)")


def input_image(store: Path, nifti: Path) -> tuple[sitk.Image, sitk.Image]:
    """The store's image (the frame of the registration) and the NIfTI's, the same voxels; the pair is checked."""
    image = read_ome_zarr(store)
    reference = sitk.ReadImage(str(nifti), sitk.sitkFloat32)
    if reference.GetSize() != image.GetSize():
        raise SystemExit(f"{store} is {image.GetSize()}, {nifti} is {reference.GetSize()}: rerun `pixi run convert`")
    if not np.allclose(sitk.GetArrayViewFromImage(image), sitk.GetArrayViewFromImage(reference), rtol=1e-5, atol=1e-6):
        raise SystemExit(f"{store} does not hold the voxels of {nifti}: rerun `pixi run convert`")
    return image, reference


def write_as(image: sitk.Image, reference: sitk.Image, path: Path) -> None:
    """`image`, on its reference's voxel grid, placed as the reference NIfTI is (its own header) and written."""
    image.CopyInformation(reference)
    sitk.WriteImage(image, str(path), useCompression=True)
    print(f"-> {path} ({path.stat().st_size / 2**20:.1f} MiB)")


def method_readme(folder: Path, preset: str, number: int, description: str, subject: str, inverse: str) -> None:
    """The method's README, with one line per subject on its inverse (the other subjects' lines are kept)."""
    readme = folder / "README.md"
    lines = [line for line in readme.read_text().splitlines() if line.startswith("- Subject_")] if readme.exists() else []
    lines = sorted(line for line in lines if not line.startswith(f"- {subject}:")) + [f"- {subject}: {inverse}"]
    readme.write_text(f"""# Team C, method {number}: {description}

Preset `{preset}` of [{REPOSITORY}]({REPOSITORY}) (KonfAI and impact-reg-konfai), run on the pair the
repository builds from the competition's files: the retardance smoothed and resampled to 0.4 mm isotropic as the
fixed image, the FA through a stage-1 affine (the best of the 48 signed axis permutations by mutual information,
refined by Mattes MI) as the moving image. The run's field is on that grid; these files carry it to the native grids.

- `Subject_<X>/DMRI_to_PSOCT.nii.gz`: `DMRI.nii.gz` in the PS-OCT space, on the grid and header of
  `PSOCT.nii.gz`, through the affine and the field, linear interpolation. Intensities as in `DMRI.nii.gz`.
- `Subject_<X>/PSOCT_to_DMRI.nii.gz`: `PSOCT.nii.gz` in the DMRI space, on the grid and header of `DMRI.nii.gz`,
  through the inverses (the field's by fixed-point iteration), after a Gaussian smoothing for the 0.4 mm grid,
  linear interpolation. Intensities as in `PSOCT.nii.gz`.

The inverse is checked by composing it with the field on the tissue, |x - v(u(x))|; the worst voxels are on the
grid's edge, and where a field folds it has no inverse:

{chr(10).join(sorted(lines))}

`pipeline/competition_results.py` in the repository writes them: `pixi run nifti {preset} <subject>`.
""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("preset", help="a run under out/<subject>/, one of " + ", ".join(METHODS))
    parser.add_argument("subject")
    parser.add_argument("--method", type=int, help="the method number (default: the preset's row in the README's table)")
    parser.add_argument("--team", default="C", help="the team letter (default: C)")
    parser.add_argument("--competition", type=Path, default=HERE / "competition", help="the organisers' folder (default: competition)")
    parser.add_argument("--inputs", type=Path, default=HERE / "data" / "input", help="the competition's NIfTI files (default: data/input)")
    parser.add_argument("--data", type=Path, default=HERE / "data" / "ome-zarr", help="the converted inputs (default: data/ome-zarr)")
    parser.add_argument("--out", type=Path, default=HERE / "out", help="the runs (default: out)")
    parser.add_argument("--iterations", type=int, default=100, help="at most this many fixed-point iterations for the inverse field (default: 100)")
    args = parser.parse_args()

    if args.method is None and args.preset not in METHODS:
        raise SystemExit(f"{args.preset} has no method number in the table; give one with --method")
    number, description = METHODS.get(args.preset, (args.method, args.preset))
    number = args.method or number
    subject_out = args.out / args.subject
    affine, field = read_transforms(subject_out, args.preset, args.subject)
    tissue = sitk.GetArrayFromImage(read_ome_zarr(subject_out / "pair" / "FixedTissue.ome.zarr")).astype(bool)

    start = time.perf_counter()
    # No boundary condition: with one, the inverse is zero on the grid's edge, where the tissue often reaches.
    inverse = sitk.InvertDisplacementField(field, maximumNumberOfIterations=args.iterations,
                                           maxErrorToleranceThreshold=0.1, meanErrorToleranceThreshold=0.001,
                                           enforceBoundaryCondition=False)
    residual = inverse_residual(field, inverse, tissue)
    magnitude = np.linalg.norm(sitk.GetArrayFromImage(field), axis=-1)[tissue]
    print(f"{args.subject} {args.preset}: field {np.median(magnitude):.2f}/{magnitude.max():.2f} mm in the tissue; "
          f"{residual} ({time.perf_counter() - start:.0f} s)")

    ret, ret_nifti = input_image(args.data / args.subject / "Ret_slide_deck.ome.zarr", args.inputs / args.subject / "Ret_slide_deck.nii.gz")
    fa, fa_nifti = input_image(args.data / args.subject / "dti_FA.ome.zarr", args.inputs / args.subject / "dti_FA.nii.gz")
    folder = args.competition / f"Team_{args.team}_method_{number}" / subject_folder(args.subject)
    folder.mkdir(parents=True, exist_ok=True)
    link_inputs(args.inputs / args.subject, args.competition / "Data" / subject_folder(args.subject))

    start = time.perf_counter()
    forward = sitk.CompositeTransform([affine, as_transform(field)])  # u first, then A
    write_as(sitk.Resample(fa, ret, forward, sitk.sitkLinear, 0.0, sitk.sitkFloat32), ret_nifti, folder / "DMRI_to_PSOCT.nii.gz")
    mm = min(fa.GetSpacing())
    smooth = sitk.SmoothingRecursiveGaussian(ret, [max(mm / 2 - s / 2, 0.01) for s in ret.GetSpacing()])
    backward = sitk.CompositeTransform([as_transform(inverse), affine.GetInverse()])  # A⁻¹ first, then v
    write_as(sitk.Resample(smooth, fa, backward, sitk.sitkLinear, 0.0, sitk.sitkFloat32), fa_nifti, folder / "PSOCT_to_DMRI.nii.gz")
    method_readme(folder.parent, args.preset, number, description, subject_folder(args.subject), residual)
    print(f"{subject_folder(args.subject)} of Team_{args.team}_method_{number} written in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
