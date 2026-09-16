"""Score one subject's runs on their pair grid, everything measured on the images.

    python pipeline/score_pair.py subject_v                            (pixi run score subject_v)
    python pipeline/score_pair.py subject_v APEX_FIREANTS_SYN_CC ...   named runs under out/<subject>/

Rows: the stage-1 affine alone (the pair as `prepare_pair.py` built it), then every run that wrote a field. Tissue
Dice (the retardance tissue against the FA tissue carried by the field); mutual information inside the retardance
tissue; the residual shift, the in-plane shift of the FA (±2 mm, one voxel at a time) that best matches the retardance
in 12 mm windows of the three mid-planes, judged by mutual information and again by normalised gradients, which no run
optimises; the fraction of the tissue folded; and the field's size. These are the measures
[docs/apex_results.md](../docs/apex_results.md) reports.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_pair import mutual_information, read_ome_zarr  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
WINDOW_MM, SEARCH_MM, TISSUE_FRACTION = 12.0, 2.0, 0.8


def planes(fixed: np.ndarray, moved: np.ndarray, tissue: np.ndarray):
    """The three mid-planes, each as (fixed, moved, tissue)."""
    z, y, x = (s // 2 for s in fixed.shape)
    return ((fixed[z], moved[z], tissue[z]), (fixed[:, y], moved[:, y], tissue[:, y]),
            (fixed[:, :, x], moved[:, :, x], tissue[:, :, x]))


def normalised_gradients(plane: np.ndarray):
    gy, gx = np.gradient(plane)
    norm = np.sqrt(gx**2 + gy**2 + 1e-3**2)
    return gx / norm, gy / norm


def residual(fixed: np.ndarray, moved: np.ndarray, tissue: np.ndarray, half: int, search: int, mm: float, judge: str) -> np.ndarray:
    """Per window, the shift (mm) that best matches the moved image to the fixed one, by ``judge``: "mi" (mutual
    information) or "ngf" (the mean squared cosine between the two images' normalised gradients)."""
    shifts = []
    for f, m, mask in planes(fixed, moved, tissue):
        fx, fy = normalised_gradients(f) if judge == "ngf" else (None, None)
        mx, my = normalised_gradients(m) if judge == "ngf" else (None, None)
        for cy in range(half + search, f.shape[0] - half - search, 2 * half):
            for cx in range(half + search, f.shape[1] - half - search, 2 * half):
                if mask[cy - half:cy + half, cx - half:cx + half].mean() < TISSUE_FRACTION:
                    continue
                window = lambda a, dy, dx: a[cy - half + dy:cy + half + dy, cx - half + dx:cx + half + dx]  # noqa: E731
                if judge == "mi":
                    reference = window(f, 0, 0)
                    score = lambda dy, dx: mutual_information(reference, window(m, dy, dx))  # noqa: E731
                else:
                    rx, ry = window(fx, 0, 0), window(fy, 0, 0)
                    score = lambda dy, dx: float(np.mean((rx * window(mx, dy, dx) + ry * window(my, dy, dx)) ** 2))  # noqa: E731
                best = max((score(dy, dx), dx, dy) for dy in range(-search, search + 1) for dx in range(-search, search + 1))
                shifts.append(np.hypot(best[1], best[2]) * mm)
    return np.array(shifts)


def folded(field: sitk.Image, tissue: np.ndarray) -> float:
    """The fraction of the tissue where the Jacobian determinant of x + u(x) is not positive."""
    u = sitk.GetArrayFromImage(field)
    spacing = field.GetSpacing()
    jacobian = np.empty(u.shape[:3] + (3, 3), np.float32)
    for i in range(3):
        gz, gy, gx = np.gradient(u[..., i], spacing[2], spacing[1], spacing[0])
        jacobian[..., i, 0], jacobian[..., i, 1], jacobian[..., i, 2] = gx, gy, gz
    for i in range(3):
        jacobian[..., i, i] += 1
    return float(np.mean(np.linalg.det(jacobian[tissue]) <= 0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("subject")
    parser.add_argument("runs", nargs="*", help="run directories under out/<subject>/ (default: every one with a field)")
    parser.add_argument("--out", type=Path, default=HERE / "out", help="output root (default: out)")
    args = parser.parse_args()

    root = args.out / args.subject
    pair = root / "pair"
    if not (pair / "Fixed.ome.zarr").exists():
        raise SystemExit(f"{pair}: no pair here; run `pixi run prepare {args.subject}` first")
    fixed_img = read_ome_zarr(pair / "Fixed.ome.zarr")
    moving_img = read_ome_zarr(pair / "Moving.ome.zarr")
    moving_tissue = read_ome_zarr(pair / "MovingTissue.ome.zarr")
    fixed = sitk.GetArrayFromImage(fixed_img)
    tissue = sitk.GetArrayFromImage(read_ome_zarr(pair / "FixedTissue.ome.zarr")).astype(bool)
    mm = float(fixed_img.GetSpacing()[0])
    half, search = max(1, round(WINDOW_MM / 2 / mm)), max(1, round(SEARCH_MM / mm))

    def row(name: str, moved: sitk.Image, carried_tissue: sitk.Image, extra: str) -> None:
        volume = sitk.GetArrayFromImage(moved)
        carried = sitk.GetArrayFromImage(carried_tissue).astype(bool)
        dice = 2 * np.sum(carried & tissue) / (carried.sum() + tissue.sum())
        mi_shift = residual(fixed, volume, tissue, half, search, mm, "mi")
        ngf_shift = residual(fixed, volume, tissue, half, search, mm, "ngf")
        print(f"{name:34s} MI {mutual_information(fixed[tissue], volume[tissue]):.3f} | residual by MI median "
              f"{np.median(mi_shift):.2f} mm, {int(np.sum(mi_shift >= 2.0))}/{len(mi_shift)} at 2 mm+ | by gradients median "
              f"{np.median(ngf_shift):.2f} mm, {int(np.sum(ngf_shift >= 2.0))}/{len(ngf_shift)} at 2 mm+ | tissue Dice {dice:.3f}{extra}")

    runs = args.runs or sorted(d.name for d in root.iterdir() if (d / "P000" / "Transform.h5").exists())
    print(f"{args.subject}: {fixed_img.GetSize()} at {mm} mm, {WINDOW_MM:g} mm windows, ±{SEARCH_MM:g} mm search")
    row("the stage-1 affine alone", moving_img, moving_tissue, "")
    for run in runs:
        transform = sitk.ReadTransform(str(root / run / "P000" / "Transform.h5"))
        if transform.GetName() == "CompositeTransform":
            transform = sitk.CompositeTransform(transform).GetNthTransform(0)
        displacement = sitk.DisplacementFieldTransform(transform)
        field = sitk.Resample(displacement.GetDisplacementField(), fixed_img)
        magnitude = np.linalg.norm(sitk.GetArrayFromImage(field), axis=-1)[tissue]
        row(run, sitk.Resample(moving_img, fixed_img, displacement, sitk.sitkLinear, 0.0),
            sitk.Resample(moving_tissue, fixed_img, displacement, sitk.sitkNearestNeighbor, 0),
            f" | folded {folded(field, tissue):.2%} | field {np.median(magnitude):.2f}/{magnitude.max():.2f} mm")


if __name__ == "__main__":
    main()
