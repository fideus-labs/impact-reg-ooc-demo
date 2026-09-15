"""APEX, stage 1 check: the left-right mirror. Both orientations (axes as stored, and mirrored in x) are refined by an
affine exactly as in stage1.py; each writes its transform, its optimiser report and its MI, and a sheet shows the
retardance beside the FA through each affine, three mid-planes on the fixed grid with a 5 mm grid drawn on all.

    python apex/mirror_check.py subject_m
"""
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from stage1_linc import mutual_information

subject = sys.argv[1]
OUT = HERE / "out" / subject


def load(name):
    return sitk.ReadImage(str(HERE / "data" / subject / f"{name}.nii.gz"), sitk.sitkFloat32)


def normalise(img):
    a = sitk.GetArrayFromImage(img); nz = a[a > 0]; lo, hi = np.percentile(nz, [1, 99])
    out = sitk.GetImageFromArray(np.clip((a - lo) / (hi - lo), 0, 1).astype(np.float32)); out.CopyInformation(img); return out


def iso(img, mm):
    smooth = sitk.SmoothingRecursiveGaussian(img, [max(mm / 2 - s / 2, 0.01) for s in img.GetSpacing()])
    size = [int(round(n * s / mm)) for n, s in zip(img.GetSize(), img.GetSpacing())]
    return sitk.Resample(smooth, size, sitk.Transform(), sitk.sitkLinear, img.GetOrigin(), [mm] * 3, img.GetDirection(), 0.0)


def centroid(img):
    a = sitk.GetArrayFromImage(img); idx = np.argwhere(a > 0.02)[:, ::-1]
    return np.array(img.TransformContinuousIndexToPhysicalPoint(idx.mean(0).tolist()))


fixed, moving = normalise(load("Ret_slide_deck")), normalise(load("dti_FA"))
cF, cM = centroid(fixed), centroid(moving)
f4, m4, f8, m8 = iso(fixed, 0.4), iso(moving, 0.4), iso(fixed, 0.8), iso(moving, 0.8)
F8 = sitk.GetArrayFromImage(f8); mk = F8 > 0.02
rows = []
for label, sx in (("as stored", 1), ("mirrored in x", -1)):
    t0 = sitk.AffineTransform(3); t0.SetMatrix([sx, 0, 0, 0, 1, 0, 0, 0, 1]); t0.SetCenter(cF.tolist()); t0.SetTranslation((cM - cF).tolist())
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(32); reg.SetMetricSamplingStrategy(reg.RANDOM); reg.SetMetricSamplingPercentage(0.2, seed=42)
    reg.SetMetricFixedMask(sitk.Cast(f4 > 0.02, sitk.sitkUInt8)); reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-4, 300, relaxationFactor=0.6); reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1]); reg.SetSmoothingSigmasPerLevel([2, 1, 0]); reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    reg.SetInitialTransform(sitk.AffineTransform(t0), inPlace=False)
    t = reg.Execute(f4, m4)
    mi = mutual_information(F8[mk], sitk.GetArrayFromImage(sitk.Resample(m8, f8, t, sitk.sitkLinear, 0.0))[mk])
    A = np.array((sitk.CompositeTransform(t).GetNthTransform(0) if t.GetName() == "CompositeTransform" else t).GetParameters()[:9]).reshape(3, 3)
    tag = "nomirror" if sx == 1 else "mirror"
    sitk.WriteTransform(t, str(OUT / f"stage1_affine_{tag}.tfm"))
    print(f"{subject} {label:14s} MI at 0.8 mm {mi:.3f} | det {np.linalg.det(A):+.3f} | optimiser: {reg.GetOptimizerStopConditionDescription()} after {reg.GetOptimizerIteration()} iterations at the last level")
    warped = sitk.GetArrayFromImage(sitk.Resample(m4, f4, t, sitk.sitkLinear, 0.0))
    rows.append((label, warped))
F4 = sitk.GetArrayFromImage(f4)
z, y, x = (s // 2 for s in F4.shape)
def planes(v): return [v[z], v[:, y], v[:, :, x]]
def panel(p):
    g = np.dstack([np.clip(p, 0, 1)] * 3)
    step = int(round(5 / 0.4)); g[::step] = (0, 0.8, 1); g[:, ::step] = (0, 0.8, 1)       # 5 mm grid
    return g
sheet_rows = []
for label, vol in [("retardance (fixed)", F4)] + rows:
    ps = [panel(p) for p in planes(vol)]
    h = max(p.shape[0] for p in ps)
    ps = [np.pad(p, ((0, h - p.shape[0]), (0, 4), (0, 0)), constant_values=0.1) for p in ps]
    sheet_rows.append(np.hstack(ps))
w = max(r.shape[1] for r in sheet_rows)
sheet = np.vstack([np.pad(r, ((0, 6), (0, w - r.shape[1]), (0, 0)), constant_values=0.1) for r in sheet_rows])
path = f"/tmp/claude-1000/-home-valentin-Documents-lib-ITKIMPACT/66517e9d-e281-42c1-af4e-92537b98fee2/scratchpad/apex_mirror_{subject}.png"
Image.fromarray((sheet * 255).astype(np.uint8)).save(path)
print("sheet rows: retardance, FA as stored, FA mirrored in x; columns: axial, coronal, sagittal mid-planes at 0.4 mm ->", path)
