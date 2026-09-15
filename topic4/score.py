"""Topic 4, MF283: score MRI -> PS-OCT alignments on the PS-OCT grid, all measured on the images.

    python topic4/score.py RUN...        RUN = a folder under topic4/out holding P000/Transform.h5

Rows: the dataset's affine alone, the dataset's nonlinear field + affine, then each run (a field on top of the affine).
Brain-mask Dice; MI inside the OCT brain mask against the smoothed OCT; residual shift, the in-plane shift of the MRI
(±1 mm, 0.1 mm steps) that maximises MI in 6 mm windows of the three mid-planes; the fraction of the brain folded.
"""
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from stage1_linc import mutual_information

SRC = Path.home() / "Downloads/anatomix_input_data/derivatives/sub-MF283"
D = HERE / "data"
fixed_img = sitk.ReadImage(str(D / "Fixed.mha"))
fixed = sitk.GetArrayFromImage(fixed_img)
O = sitk.GetArrayFromImage(sitk.ReadImage(str(D / "FixedMask.mha"))).astype(bool)
moving_img = sitk.ReadImage(str(D / "Moving.mha"))
moving_mask = sitk.ReadImage(str(D / "MovingMask.mha"))
mri = sitk.ReadImage(str(SRC / "anat/sub-MF283_acq-MSME_desc-brain.nii.gz"), sitk.sitkFloat32)
mri_mask = sitk.ReadImage(str(SRC / "anat/sub-MF283_acq-MSME_label-brain_mask.nii.gz")) > 0.5
affine = sitk.ReadTransform(str(SRC / "xfm/sub-MF283_from-dMRI_to-PSOCT_mode-image_xfm.txt"))
their_field = sitk.DisplacementFieldTransform(sitk.Cast(sitk.ReadImage(
    str(SRC / "xfm/sub-MF283_from-dMRI_to-PSOCT_mode-image_desc-nonlinear_xfm.nii.gz")), sitk.sitkVectorFloat64))


def norm(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a[O], [1, 99])
    return np.clip((a - lo) / (hi - lo + 1e-12), 0, 1)


def folded(field: sitk.Image) -> float:
    u = sitk.GetArrayFromImage(field)                       # z, y, x, (ux, uy, uz) in mm
    sp = field.GetSpacing()                                 # x, y, z
    J = np.empty(u.shape[:3] + (3, 3), np.float32)
    for i in range(3):                                      # component i
        gz, gy, gx = np.gradient(u[..., i], sp[2], sp[1], sp[0])
        J[..., i, 0], J[..., i, 1], J[..., i, 2] = gx, gy, gz
    J[..., 0, 0] += 1; J[..., 1, 1] += 1; J[..., 2, 2] += 1
    return float(np.mean(np.linalg.det(J[O]) <= 0))


def residual(vol: np.ndarray) -> np.ndarray:
    half, search = 30, 10
    z, y, x = (s // 2 for s in fixed.shape)
    out = []
    for f, m, mask in ((fixed[z], vol[z], O[z]), (fixed[:, y], vol[:, y], O[:, y]), (fixed[:, :, x], vol[:, :, x], O[:, :, x])):
        for cy in range(half + search, f.shape[0] - half - search, 2 * half):
            for cx in range(half + search, f.shape[1] - half - search, 2 * half):
                if mask[cy - half:cy + half, cx - half:cx + half].mean() < 0.8:
                    continue
                ref = f[cy - half:cy + half, cx - half:cx + half]
                best = max((mutual_information(ref, m[cy - half + dy:cy + half + dy, cx - half + dx:cx + half + dx]), dx, dy)
                           for dy in range(-search, search + 1) for dx in range(-search, search + 1))
                out.append(np.hypot(best[1], best[2]) * 0.1)
    return np.array(out)


def row(name: str, image: sitk.Image, mask: sitk.Image, fold: str) -> None:
    vol = norm(sitk.GetArrayFromImage(image))
    r = sitk.GetArrayFromImage(mask).astype(bool)
    dice = 2 * np.sum(r & O) / (r.sum() + O.sum())
    res = residual(vol)
    print(f"{name:28s} Dice {dice:.3f} | MI {mutual_information(fixed[O], vol[O]):.3f} | residual shift median {np.median(res):.2f} mm, "
          f"90 % under {np.percentile(res, 90):.2f} mm, windows beyond 1 mm {int(np.sum(res >= 1.0))}/{len(res)} | folded {fold}")


row("dataset affine alone", moving_img, moving_mask, "-")
chain = sitk.CompositeTransform(3); chain.AddTransform(affine); chain.AddTransform(their_field)
row("dataset field + affine", sitk.Resample(mri, fixed_img, chain, sitk.sitkLinear, 0.0),
    sitk.Resample(mri_mask, fixed_img, chain, sitk.sitkNearestNeighbor, 0), f"{folded(their_field.GetDisplacementField()):.2%}")
for run in sys.argv[1:]:
    t = sitk.ReadTransform(str(HERE / "out" / run / "P000" / "Transform.h5"))
    if t.GetName() == "CompositeTransform":
        t = sitk.CompositeTransform(t).GetNthTransform(0)
    dft = sitk.DisplacementFieldTransform(t)
    fld = dft.GetDisplacementField()
    row(run, sitk.Resample(moving_img, fixed_img, dft, sitk.sitkLinear, 0.0),
        sitk.Resample(moving_mask, fixed_img, dft, sitk.sitkNearestNeighbor, 0), f"{folded(sitk.Resample(fld, fixed_img)):.2%}")
