"""The competition pair from the OME-Zarr inputs: stage 1 (orientation, then affine) and the 0.4 mm pair.

    python pipeline/prepare_pair.py subject_v          (pixi run prepare subject_v)

Reads data/ome-zarr/<subject>/Ret_slide_deck.ome.zarr (fixed) and dti_FA.ome.zarr (moving), as `pixi run convert`
writes them. The headers do not place the two volumes in a common frame, so every signed axis permutation of the FA
(48, mirrors included) is scored by mutual information after matching the tissue centroids, at 1.6 mm; the three best
are refined by an affine (Mattes MI, 0.8 mm then 0.4 mm) and the best refined one is kept as
out/<subject>/stage1_affine.tfm. Then the pair the registration presets take: the retardance smoothed and resampled to
0.4 mm isotropic (the FA's resolution) as Fixed, the FA through the affine on that grid as Moving, the retardance tissue
widened by 1 mm as FixedMask, and the whole grid as MovingMask, each an OME-Zarr store under out/<subject>/pair/.
FixedTissue and MovingTissue (the unwidened masks) are written beside them for scoring.

The grid is the store's own frame: identity direction, scale and translation as the OME-Zarr metadata gives them.
"""
import argparse
import itertools
import time
from pathlib import Path

import ngff_zarr
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_dilation, binary_fill_holes
from konfai.utils.dataset import image_to_data
from konfai.utils.ome_zarr import write_ome_zarr

HERE = Path(__file__).resolve().parent.parent
TISSUE = 0.02  # normalised intensity above which a voxel is tissue


def read_ome_zarr(path: Path) -> sitk.Image:
    """Level 0 of an OME-Zarr store as a scalar SimpleITK image in the store's frame (identity direction)."""
    image = ngff_zarr.from_ngff_zarr(str(path)).images[0]
    dims = list(image.dims)
    data = np.asarray(image.data)
    for axis in [d for d in dims if d not in ("z", "y", "x")]:
        index = dims.index(axis)
        if data.shape[index] != 1:
            raise ValueError(f"{path}: axis {axis!r} has size {data.shape[index]}, expected a scalar volume")
        data = np.take(data, 0, axis=index)
        dims.pop(index)
    if dims != ["z", "y", "x"]:
        raise ValueError(f"{path}: spatial axes {dims}, expected z, y, x")
    out = sitk.GetImageFromArray(data.astype(np.float32))
    out.SetSpacing([float(image.scale[a]) for a in ("x", "y", "z")])
    out.SetOrigin([float(image.translation[a]) for a in ("x", "y", "z")])
    return out


def write_store(path: Path, array: np.ndarray, grid: sitk.Image) -> None:
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(grid)
    data, attributes = image_to_data(image)
    if path.exists():
        import shutil
        shutil.rmtree(path)
    write_ome_zarr(path, data, spacing=attributes.get_np_array("Spacing"), origin=attributes.get_np_array("Origin"),
                   attributes=dict(attributes))


def mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    """Mutual information of two equally shaped arrays, in nats, on a joint histogram of `bins`² cells."""
    joint, _, _ = np.histogram2d(a.ravel(), b.ravel(), bins=bins)
    p = joint / joint.sum()
    outer = p.sum(axis=1, keepdims=True) @ p.sum(axis=0, keepdims=True)
    nz = p > 0
    return float(np.sum(p[nz] * np.log(p[nz] / outer[nz])))


def normalise(img: sitk.Image) -> sitk.Image:
    a = sitk.GetArrayFromImage(img)
    nz = a[a > 0]
    lo, hi = np.percentile(nz, [1, 99])
    out = sitk.GetImageFromArray(np.clip((a - lo) / (hi - lo), 0, 1).astype(np.float32))
    out.CopyInformation(img)
    return out


def iso(img: sitk.Image, mm: float) -> sitk.Image:
    """The image smoothed for, and resampled onto, an isotropic grid of `mm` over the same extent."""
    smooth = sitk.SmoothingRecursiveGaussian(img, [max(mm / 2 - s / 2, 0.01) for s in img.GetSpacing()])
    size = [int(round(n * s / mm)) for n, s in zip(img.GetSize(), img.GetSpacing())]
    return sitk.Resample(smooth, size, sitk.Transform(), sitk.sitkLinear, img.GetOrigin(), [mm] * 3, img.GetDirection(), 0.0)


def centroid(img: sitk.Image) -> np.ndarray:
    idx = np.argwhere(sitk.GetArrayFromImage(img) > TISSUE)[:, ::-1]  # x, y, z
    return np.array(img.TransformContinuousIndexToPhysicalPoint(idx.mean(0).tolist()))


def stage1(fixed: sitk.Image, moving: sitk.Image, subject: str) -> sitk.Transform:
    cF, cM = centroid(fixed), centroid(moving)
    f16, m16 = iso(fixed, 1.6), iso(moving, 1.6)
    F16 = sitk.GetArrayFromImage(f16)
    mask16 = F16 > TISSUE
    candidates = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for row, (col, sign) in enumerate(zip(perm, signs)):
                M[row, col] = sign
            t = sitk.AffineTransform(3)
            t.SetMatrix(M.ravel().tolist())
            t.SetCenter(cF.tolist())
            t.SetTranslation((cM - cF).tolist())
            warped = sitk.GetArrayFromImage(sitk.Resample(m16, f16, t, sitk.sitkLinear, 0.0))
            candidates.append((mutual_information(F16[mask16], warped[mask16]), np.linalg.det(M), perm, signs, t))
    candidates.sort(key=lambda c: -c[0])
    print(f"{subject}: 48 orientations at 1.6 mm, best five by MI:")
    for mi, det, perm, signs, _ in candidates[:5]:
        print(f"  MI {mi:.3f}  axes {perm} signs {signs} {'mirror' if det < 0 else 'rotation'}")

    fixed04, moving04 = iso(fixed, 0.4), iso(moving, 0.4)
    fixed_mask04 = sitk.Cast(fixed04 > TISSUE, sitk.sitkUInt8)
    f8, m8 = iso(fixed, 0.8), iso(moving, 0.8)
    F8 = sitk.GetArrayFromImage(f8)
    mask8 = F8 > TISSUE

    def refine(t0: sitk.AffineTransform) -> tuple[float, sitk.Transform]:
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(32)
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(0.2, seed=42)
        reg.SetMetricFixedMask(fixed_mask04)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-4, 300, relaxationFactor=0.6)
        reg.SetOptimizerScalesFromPhysicalShift()
        reg.SetShrinkFactorsPerLevel([4, 2, 1])
        reg.SetSmoothingSigmasPerLevel([2, 1, 0])
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
        reg.SetInitialTransform(sitk.AffineTransform(t0), inPlace=False)
        t = reg.Execute(fixed04, moving04)
        warped = sitk.GetArrayFromImage(sitk.Resample(m8, f8, t, sitk.sitkLinear, 0.0))
        return mutual_information(F8[mask8], warped[mask8]), t

    best = None
    for _, _, perm, signs, t0 in candidates[:3]:
        start = time.perf_counter()
        mi, t = refine(t0)
        print(f"  refined axes {perm} signs {signs}: MI at 0.8 mm {mi:.3f} ({time.perf_counter() - start:.0f} s)")
        if best is None or mi > best[0]:
            best = (mi, t, perm, signs)
    mi, t, perm, signs = best
    A = np.array(t.GetParameters()[:9]).reshape(3, 3)
    print(f"kept axes {perm} signs {signs}, MI {mi:.3f}; affine det {np.linalg.det(A):.3f} (volume ratio moving/fixed), "
          f"singular values {np.round(np.linalg.svd(A)[1], 3).tolist()}")
    return t


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("subject")
    parser.add_argument("--data", type=Path, default=HERE / "data" / "ome-zarr", help="the converted inputs (default: data/ome-zarr)")
    parser.add_argument("--out", type=Path, default=HERE / "out", help="output root (default: out)")
    parser.add_argument("--mm", type=float, default=0.4, help="the pair's isotropic spacing in mm (default: 0.4)")
    parser.add_argument("--force", action="store_true", help="redo stage 1 even if its affine is newer than the inputs")
    args = parser.parse_args()

    out = args.out / args.subject
    out.mkdir(parents=True, exist_ok=True)
    ret = read_ome_zarr(args.data / args.subject / "Ret_slide_deck.ome.zarr")
    fa = read_ome_zarr(args.data / args.subject / "dti_FA.ome.zarr")
    fixed, moving = normalise(ret), normalise(fa)

    if args.mm <= 0:
        raise SystemExit(f"--mm must be positive, got {args.mm}")
    affine_path = out / "stage1_affine.tfm"
    inputs_written = max((args.data / args.subject / f"{name}.ome.zarr" / "zarr.json").stat().st_mtime for name in ("Ret_slide_deck", "dti_FA"))
    if affine_path.exists() and affine_path.stat().st_mtime > inputs_written and not args.force:
        print(f"{args.subject}: stage 1 affine is newer than the inputs, reusing {affine_path}")
        affine = sitk.ReadTransform(str(affine_path))
    else:
        affine = stage1(fixed, moving, args.subject)
        sitk.WriteTransform(affine, str(affine_path))
        print(f"-> {affine_path}")

    mm = args.mm
    grid = iso(fixed, mm)
    fixed_a = sitk.GetArrayFromImage(grid)
    moving_a = sitk.GetArrayFromImage(sitk.Resample(moving, grid, affine, sitk.sitkLinear, 0.0))
    fixed_tissue = binary_fill_holes(fixed_a > TISSUE)
    moving_tissue = binary_fill_holes(moving_a > TISSUE)
    fixed_mask = binary_dilation(fixed_tissue, iterations=max(1, int(round(1.0 / mm))))  # widened by 1 mm, at least one voxel
    pair = out / "pair"
    pair.mkdir(exist_ok=True)
    for name, a in (("Fixed", fixed_a), ("Moving", moving_a),
                    ("FixedMask", fixed_mask.astype(np.uint8)), ("MovingMask", np.ones_like(fixed_mask, np.uint8)),
                    ("FixedTissue", fixed_tissue.astype(np.uint8)), ("MovingTissue", moving_tissue.astype(np.uint8))):
        write_store(pair / f"{name}.ome.zarr", a, grid)
    dice = 2 * np.sum(fixed_tissue & moving_tissue) / (fixed_tissue.sum() + moving_tissue.sum())
    print(f"{args.subject}: grid {grid.GetSize()} at {mm} mm; tissue Dice after the affine {dice:.3f}; "
          f"fixed mask covers {fixed_mask.mean():.1%} -> {pair}")


if __name__ == "__main__":
    main()
