"""Is the 0.5 mm correction of the consortium's zoom-1 placement real? Rigid at 80 um, then checked at 20 um.

The check is one the registration never sees: level 0 of the overview (20 um) against level 2 of the zoom
(17 um), in 6 mm windows where the overview has the most structure. In each window the residual shift is read
straight from the peak of the cross-correlation, for the manual placement and for the refined one.
"""
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import zarr
from PIL import Image
from scipy import fft

sys.path.insert(0, "/home/valentin/Documents/ImpactReg_OOC_Demo")
from figures_linc import checkerboard

S = Path(__file__).parent
U_OUT = 0.010072
M = np.array([[0.999445745, 0.031534287, -0.011279779, 3817.405521157],
              [-0.031774949, 0.999263261, -0.021834133, 4528.589463744],
              [0.010582873, 0.022180297, 0.999704696, 5999.076104469]])


def group(key):
    return zarr.open_group(f"s3://dandiarchive/zarr/{key}", mode="r", storage_options={"anon": True})


OVERVIEW = group("6f11427f-ef86-42b8-9800-d3e09265c965")
ZOOM = group("d483e140-ccd7-4bcf-8a69-a10b7087381c")


def geometry(g, level):
    attrs = dict(g.attrs)
    multiscales = attrs.get("multiscales") or attrs["ome"]["multiscales"]
    d = next(d for d in multiscales[0]["datasets"] if d["path"] == level)
    t = {c["type"]: c for c in d["coordinateTransformations"]}
    return np.array(t["scale"]["scale"]) / 1000, np.array(t["translation"]["translation"]) / 1000, g[level]


def box(g, level, lo_mm, hi_mm):
    """The store's voxels covering [lo, hi] mm, as an image in the store's own frame (identity direction)."""
    s, t, arr = geometry(g, level)
    i0 = np.clip(np.floor((np.asarray(lo_mm) - t) / s).astype(int), 0, arr.shape)
    i1 = np.clip(np.ceil((np.asarray(hi_mm) - t) / s).astype(int) + 1, 0, arr.shape)
    start = time.perf_counter()
    a = np.asarray(arr[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]])
    image = sitk.GetImageFromArray(np.ascontiguousarray(a.transpose(2, 1, 0)).astype(np.float32))
    image.SetSpacing(s.tolist())
    image.SetOrigin((t + i0 * s).tolist())
    print(f"  level {level} box {a.shape} read in {time.perf_counter() - start:.0f} s")
    return image


def bounds(transform, lo, hi):
    points = np.array([transform.TransformPoint([hi[i] if c[i] else lo[i] for i in range(3)]) for c in np.ndindex(2, 2, 2)])
    return points.min(0), points.max(0)


def residual_shift(a, b, reach=75):
    """The integer shift, within +-reach voxels, that best lines b up with a: the cross-correlation's peak. (x, y, z) voxels."""
    window = np.einsum("i,j,k->ijk", *(np.hanning(n).astype(np.float32) for n in a.shape))
    fa = fft.rfftn((a - a.mean()) * window, workers=-1)
    fb = fft.rfftn((b - b.mean()) * window, workers=-1)
    cc = fft.irfftn(fa * np.conj(fb), s=a.shape, workers=-1)
    shifts = [np.fft.fftfreq(n, 1 / n).astype(int) for n in a.shape]
    keep = [np.flatnonzero(np.abs(v) <= reach) for v in shifts]
    sub = cc[np.ix_(*keep)]
    peak = np.unravel_index(np.argmax(sub), sub.shape)
    return np.array([shifts[axis][keep[axis][peak[axis]]] for axis in range(3)])[::-1]


s0, t0, zoom0 = geometry(ZOOM, "0")
zoom_lo, zoom_hi = t0, t0 + (np.array(zoom0.shape) - 1) * s0
manual = sitk.AffineTransform(M[:, :3].ravel().tolist(), (M[:, 3] * U_OUT).tolist())   # zoom mm -> overview mm
inverse = manual.GetInverse()                                                          # overview mm -> zoom mm

print("== rigid correction at 80 um: overview level 2 against zoom level 4")
fixed = box(OVERVIEW, "2", *bounds(manual, zoom_lo, zoom_hi))
moving = box(ZOOM, "4", zoom_lo, zoom_hi)
border = int(round(2.0 / moving.GetSpacing()[0]))
inside = np.zeros(sitk.GetArrayViewFromImage(moving).shape, np.uint8)
inside[border:-border, border:-border, border:-border] = 1
inside = sitk.GetImageFromArray(inside)
inside.CopyInformation(moving)
mask = sitk.Resample(inside, fixed, inverse, sitk.sitkNearestNeighbor, 0)      # where the zoom has data, 2 mm in

euler = sitk.Euler3DTransform()
euler.SetCenter(fixed.TransformContinuousIndexToPhysicalPoint([(n - 1) / 2 for n in fixed.GetSize()]))
reg = sitk.ImageRegistrationMethod()
reg.SetMetricAsMattesMutualInformation(50)
reg.SetMetricFixedMask(mask)
reg.SetMetricSamplingStrategy(reg.RANDOM)
reg.SetMetricSamplingPercentagePerLevel([0.1, 0.02, 0.005], seed=1)
reg.SetInterpolator(sitk.sitkLinear)
reg.SetOptimizerAsRegularStepGradientDescent(0.5, 1e-4, 200, relaxationFactor=0.7)
reg.SetOptimizerScalesFromPhysicalShift()
reg.SetShrinkFactorsPerLevel([4, 2, 1])
reg.SetSmoothingSigmasPerLevel([2, 1, 0])
reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
reg.SetMovingInitialTransform(inverse)
reg.SetInitialTransform(euler, inPlace=True)
start = time.perf_counter()
reg.Execute(fixed, moving)
chain = sitk.CompositeTransform([inverse, euler])        # the correction first, then the manual placement
sitk.WriteTransform(euler, str(S / "zoom1_rigid_correction.tfm"))
m = sitk.GetArrayFromImage(mask) > 0
f2 = sitk.GetArrayFromImage(fixed)
for label, transform in (("manual", inverse), ("rigid", chain)):
    moved = sitk.GetArrayFromImage(sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0.0))
    print(f"  {label}: correlation {np.corrcoef(moved[m], f2[m])[0, 1]:.3f}")
idx = np.argwhere(m[::8, ::8, ::8])[:, ::-1] * 8
moves = [np.linalg.norm(np.subtract(chain.TransformPoint(p), inverse.TransformPoint(p)))
         for p in (fixed.TransformIndexToPhysicalPoint(i.tolist()) for i in idx)]
print(f"  the correction moves the zoom by median {np.median(moves):.2f} mm, max {np.max(moves):.2f} mm "
      f"(angles {np.round(np.degrees(euler.GetParameters()[:3]), 3).tolist()} deg, translation {np.round(euler.GetParameters()[3:], 3).tolist()} mm), "
      f"{time.perf_counter() - start:.0f} s")

block = 75                                                  # 6 mm at 80 um
scores = []
for k, j, i in np.ndindex(*(np.array(f2.shape) // block)):
    sl = np.s_[k * block:(k + 1) * block, j * block:(j + 1) * block, i * block:(i + 1) * block]
    if m[sl].all():
        scores.append((f2[sl].std(), (i, j, k)))
centres = [np.array(fixed.TransformContinuousIndexToPhysicalPoint([(v + 0.5) * block for v in ijk])) for _, ijk in sorted(scores)[::-1][:3]]

print("== held out: 6 mm windows, overview level 0 (20 um) against zoom level 2 (17 um)")
rows = []
for n, c in enumerate(centres):
    fixed_w = box(OVERVIEW, "0", c - 3, c + 3)
    zoom_w = box(ZOOM, "2", *(v + d for v, d in zip(bounds(inverse, c - 3, c + 3), (-1.5, 1.5))))
    f0 = sitk.GetArrayFromImage(fixed_w)
    voxel = fixed_w.GetSpacing()[0]
    for label, transform in (("manual", inverse), ("rigid", chain)):
        moved = sitk.GetArrayFromImage(sitk.Resample(zoom_w, fixed_w, transform, sitk.sitkLinear, 0.0))
        shift = residual_shift(f0, moved) * voxel
        print(f"  window {n + 1} at {np.round(c, 1).tolist()} mm, {label}: correlation {np.corrcoef(f0.ravel(), moved.ravel())[0, 1]:.3f}, "
              f"residual shift {np.round(shift, 2).tolist()} mm, |{np.linalg.norm(shift):.2f}| mm")
        h = f0.shape[0] // 2
        rows.append(np.hstack([checkerboard(f0[h], moved[h], 25), checkerboard(f0[:, f0.shape[1] // 2], moved[:, f0.shape[1] // 2], 25)]))
Image.fromarray((np.clip(np.vstack(rows), 0, 1) * 255).astype(np.uint8)).save(S / "zoom1_check20.png")
print(f"figure: {S / 'zoom1_check20.png'} (rows: window 1 manual, rigid, window 2 manual, rigid, ...; axial | coronal; 0.5 mm squares)")
