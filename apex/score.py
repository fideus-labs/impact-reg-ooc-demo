"""APEX, score stage-2 runs on their 0.4 mm pair grid, all measured on the images.

    python apex/score.py SUBJECT TAG [RUN_DIR...]      RUN_DIR under apex/out/<subject>/ holding P000/Transform.h5

Rows: the stage-1 affine alone (the pair as built), then each run. Tissue Dice (retardance tissue against the FA
tissue carried by the field); MI inside the retardance tissue; residual shift, the in-plane shift of the FA (±2 mm,
0.4 mm steps) that maximises MI in 12 mm windows of the three mid-planes; the fraction of the tissue folded; the
field's size.
"""
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from stage1_linc import mutual_information

subject, tag, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
pair = HERE / "out" / subject / f"pair_{tag}"
fixed_img = sitk.ReadImage(str(pair / "Fixed.mha"))
F = sitk.GetArrayFromImage(fixed_img)
T = sitk.GetArrayFromImage(sitk.ReadImage(str(pair / "FixedTissue.mha"))).astype(bool)
moving_img = sitk.ReadImage(str(pair / "Moving.mha"))
moving_tissue = sitk.ReadImage(str(pair / "MovingTissue.mha"))
HALF, SEARCH = 15, 5


def residual(vol: np.ndarray) -> np.ndarray:
    z, y, x = (s // 2 for s in F.shape)
    out = []
    for f, m, mask in ((F[z], vol[z], T[z]), (F[:, y], vol[:, y], T[:, y]), (F[:, :, x], vol[:, :, x], T[:, :, x])):
        for cy in range(HALF + SEARCH, f.shape[0] - HALF - SEARCH, 2 * HALF):
            for cx in range(HALF + SEARCH, f.shape[1] - HALF - SEARCH, 2 * HALF):
                if mask[cy - HALF:cy + HALF, cx - HALF:cx + HALF].mean() < 0.8:
                    continue
                ref = f[cy - HALF:cy + HALF, cx - HALF:cx + HALF]
                best = max((mutual_information(ref, m[cy - HALF + dy:cy + HALF + dy, cx - HALF + dx:cx + HALF + dx]), dx, dy)
                           for dy in range(-SEARCH, SEARCH + 1) for dx in range(-SEARCH, SEARCH + 1))
                out.append(np.hypot(best[1], best[2]) * 0.4)
    return np.array(out)


def ngf_residual(vol: np.ndarray) -> np.ndarray:
    """The same windows and search, judged by normalized gradient fields: the mean squared cosine between the two
    images' gradients. No run optimises it, so it does not favour mutual-information runs the way the MI judge does."""
    def grads(a):
        gy, gx = np.gradient(a)
        n = np.sqrt(gx ** 2 + gy ** 2 + 1e-3 ** 2)
        return gx / n, gy / n
    z, y, x = (s // 2 for s in F.shape)
    out = []
    for f, m, mask in ((F[z], vol[z], T[z]), (F[:, y], vol[:, y], T[:, y]), (F[:, :, x], vol[:, :, x], T[:, :, x])):
        fx, fy = grads(f); mx, my = grads(m)
        for cy in range(HALF + SEARCH, f.shape[0] - HALF - SEARCH, 2 * HALF):
            for cx in range(HALF + SEARCH, f.shape[1] - HALF - SEARCH, 2 * HALF):
                if mask[cy - HALF:cy + HALF, cx - HALF:cx + HALF].mean() < 0.8:
                    continue
                rx, ry = fx[cy - HALF:cy + HALF, cx - HALF:cx + HALF], fy[cy - HALF:cy + HALF, cx - HALF:cx + HALF]
                best = max((float(np.mean((rx * mx[cy - HALF + dy:cy + HALF + dy, cx - HALF + dx:cx + HALF + dx]
                                           + ry * my[cy - HALF + dy:cy + HALF + dy, cx - HALF + dx:cx + HALF + dx]) ** 2)), dx, dy)
                           for dy in range(-SEARCH, SEARCH + 1) for dx in range(-SEARCH, SEARCH + 1))
                out.append(np.hypot(best[1], best[2]) * 0.4)
    return np.array(out)


def folded(field: sitk.Image) -> float:
    u = sitk.GetArrayFromImage(field); sp = field.GetSpacing()
    J = np.empty(u.shape[:3] + (3, 3), np.float32)
    for i in range(3):
        gz, gy, gx = np.gradient(u[..., i], sp[2], sp[1], sp[0])
        J[..., i, 0], J[..., i, 1], J[..., i, 2] = gx, gy, gz
    for i in range(3):
        J[..., i, i] += 1
    return float(np.mean(np.linalg.det(J[T]) <= 0))


def row(name: str, moved: sitk.Image, tissue: sitk.Image, extra: str) -> None:
    vol = sitk.GetArrayFromImage(moved); r = sitk.GetArrayFromImage(tissue).astype(bool)
    dice = 2 * np.sum(r & T) / (r.sum() + T.sum())
    res, ngf = residual(vol), ngf_residual(vol)
    print(f"{name:26s} MI {mutual_information(F[T], vol[T]):.3f} | residual by MI median {np.median(res):.2f} mm, 2 mm+ {int(np.sum(res >= 2.0))}/{len(res)} "
          f"| by gradients median {np.median(ngf):.2f} mm, 2 mm+ {int(np.sum(ngf >= 2.0))}/{len(ngf)} | tissue Dice {dice:.3f}{extra}")


print(f"{subject} {tag}")
row("stage-1 affine", moving_img, moving_tissue, "")
for run in runs:
    t = sitk.ReadTransform(str(HERE / "out" / subject / run / "P000" / "Transform.h5"))
    if t.GetName() == "CompositeTransform":
        t = sitk.CompositeTransform(t).GetNthTransform(0)
    dft = sitk.DisplacementFieldTransform(t)
    field = sitk.Resample(dft.GetDisplacementField(), fixed_img)
    mag = np.linalg.norm(sitk.GetArrayFromImage(field), axis=-1)[T]
    row(run, sitk.Resample(moving_img, fixed_img, dft, sitk.sitkLinear, 0.0), sitk.Resample(moving_tissue, fixed_img, dft, sitk.sitkNearestNeighbor, 0),
        f" | folded {folded(field):.2%} | field {np.median(mag):.2f}/{mag.max():.2f} mm")
