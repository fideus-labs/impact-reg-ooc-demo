"""APEX CONNECTS competition, stage 1: dti_FA (moving) onto Ret_slide_deck (fixed), orientation then affine.

The headers do not place the two volumes in a common frame, so every signed axis permutation (48, mirrors included)
is scored by mutual information after matching the tissue centroids, at 1.6 mm; the three best are refined by an
affine (Mattes MI, 0.8 mm then 0.4 mm), and the best refined one is kept.

    python apex/stage1.py subject_m
"""
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from stage1_linc import mutual_information

subject = sys.argv[1]
OUT = HERE / "out" / subject
OUT.mkdir(parents=True, exist_ok=True)


def load(name: str) -> sitk.Image:
    return sitk.ReadImage(str(HERE / "data" / subject / f"{name}.nii.gz"), sitk.sitkFloat32)


def normalise(img: sitk.Image) -> sitk.Image:
    a = sitk.GetArrayFromImage(img); nz = a[a > 0]
    lo, hi = np.percentile(nz, [1, 99])
    out = sitk.GetImageFromArray(np.clip((a - lo) / (hi - lo), 0, 1).astype(np.float32)); out.CopyInformation(img)
    return out


def iso(img: sitk.Image, mm: float) -> sitk.Image:
    smooth = sitk.SmoothingRecursiveGaussian(img, [max(mm / 2 - s / 2, 0.01) for s in img.GetSpacing()])
    size = [int(round(n * s / mm)) for n, s in zip(img.GetSize(), img.GetSpacing())]
    return sitk.Resample(smooth, size, sitk.Transform(), sitk.sitkLinear, img.GetOrigin(), [mm] * 3, img.GetDirection(), 0.0)


def centroid(img: sitk.Image) -> np.ndarray:
    a = sitk.GetArrayFromImage(img); idx = np.argwhere(a > 0.02)[:, ::-1]      # x, y, z
    return np.array(img.TransformContinuousIndexToPhysicalPoint(idx.mean(0).tolist()))


fixed, moving = normalise(load("Ret_slide_deck")), normalise(load("dti_FA"))
cF, cM = centroid(fixed), centroid(moving)
f16, m16 = iso(fixed, 1.6), iso(moving, 1.6)
F16 = sitk.GetArrayFromImage(f16); mask16 = F16 > 0.02

candidates = []
for perm in itertools.permutations(range(3)):
    for signs in itertools.product((1, -1), repeat=3):
        M = np.zeros((3, 3))
        for row, (col, sign) in enumerate(zip(perm, signs)):
            M[row, col] = sign
        t = sitk.AffineTransform(3)
        t.SetMatrix(M.ravel().tolist()); t.SetCenter(cF.tolist()); t.SetTranslation((cM - cF).tolist())
        warped = sitk.GetArrayFromImage(sitk.Resample(m16, f16, t, sitk.sitkLinear, 0.0))
        candidates.append((mutual_information(F16[mask16], warped[mask16]), np.linalg.det(M), perm, signs, t))
candidates.sort(key=lambda c: -c[0])
print(f"{subject}: 48 orientations at 1.6 mm, best five by MI:")
for mi, det, perm, signs, _ in candidates[:5]:
    print(f"  MI {mi:.3f}  axes {perm} signs {signs} {'mirror' if det < 0 else 'rotation'}")

def refine(t0: sitk.AffineTransform) -> tuple[float, sitk.Transform]:
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(32)
    reg.SetMetricSamplingStrategy(reg.RANDOM); reg.SetMetricSamplingPercentage(0.2, seed=42)
    reg.SetMetricFixedMask(sitk.Cast(iso(fixed, 0.4) > 0.02, sitk.sitkUInt8))
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-4, 300, relaxationFactor=0.6)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1]); reg.SetSmoothingSigmasPerLevel([2, 1, 0]); reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    reg.SetInitialTransform(sitk.AffineTransform(t0), inPlace=False)
    t = reg.Execute(iso(fixed, 0.4), iso(moving, 0.4))
    f8, m8 = iso(fixed, 0.8), iso(moving, 0.8)
    F8 = sitk.GetArrayFromImage(f8); mk = F8 > 0.02
    return mutual_information(F8[mk], sitk.GetArrayFromImage(sitk.Resample(m8, f8, t, sitk.sitkLinear, 0.0))[mk]), t

best = None
for mi0, det, perm, signs, t0 in candidates[:3]:
    start = time.perf_counter()
    mi, t = refine(t0)
    print(f"  refined axes {perm} signs {signs}: MI at 0.8 mm {mi:.3f} ({time.perf_counter() - start:.0f} s)")
    if best is None or mi > best[0]:
        best = (mi, t, perm, signs)
mi, t, perm, signs = best
sitk.WriteTransform(t, str(OUT / "stage1_affine.tfm"))
A = np.array(sitk.CompositeTransform(t).GetNthTransform(0).GetParameters()[:9]).reshape(3, 3) if t.GetName() == "CompositeTransform" else np.array(t.GetParameters()[:9]).reshape(3, 3)
print(f"kept axes {perm} signs {signs}, MI {mi:.3f}; affine det {np.linalg.det(A):.3f} (volume ratio moving/fixed), "
      f"singular values {np.round(np.linalg.svd(A)[1], 3).tolist()} -> {OUT / 'stage1_affine.tfm'}")
