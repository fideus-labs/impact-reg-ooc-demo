"""Which reading of the consortium's Neuroglancer matrix puts zoom 1 on the overview? Checked on the images.

Frame: the store's own physical frame, x y z in mm, identity direction, OME translations kept.
  H_phys: out_mm = M·in_mm + t·u_out          (the matrix acts on physical coordinates)
  H_vox : out_mm = u_out·(M·in_mm/u_in + t)   (the matrix acts on unit counts)
Scored by the correlation of the two X-ray volumes inside the zoom's field of view, at level 3 of the overview.
"""
import numpy as np
import SimpleITK as sitk
import zarr

U_OUT, U_IN = 0.010072, 0.004257
M = np.array([[0.999445745, 0.031534287, -0.011279779, 3817.405521157],
              [-0.031774949, 0.999263261, -0.021834133, 4528.589463744],
              [0.010582873, 0.022180297, 0.999704696, 5999.076104469]])


def image(array_xyz, spacing, origin):
    out = sitk.GetImageFromArray(np.ascontiguousarray(array_xyz.transpose(2, 1, 0)).astype(np.float32))
    out.SetSpacing((spacing,) * 3)
    out.SetOrigin((origin,) * 3)
    return out


zoom_store = zarr.open_group("s3://dandiarchive/zarr/d483e140-ccd7-4bcf-8a69-a10b7087381c", mode="r", storage_options={"anon": True})
zoom = image(np.asarray(zoom_store["5"]), 0.136224, 0.068112)
overview = sitk.ReadImage("/home/valentin/Documents/ImpactReg_OOC_Demo/data/linc/XPCT_level3.mha")
overview = image(sitk.GetArrayFromImage(overview).transpose(2, 1, 0), 0.16056, 0.08028)
z = sitk.GetArrayViewFromImage(zoom)
print(f"zoom level 5 {zoom.GetSize()}, nonzero {np.mean(z > 0):.2f}, p1/p50/p99 of nonzero {np.percentile(z[z > 0], [1, 50, 99]).round()}")

for name, linear, translation in (("H_phys", M[:, :3], M[:, 3] * U_OUT),
                                  ("H_vox", M[:, :3] * U_OUT / U_IN, M[:, 3] * U_OUT)):
    forward = sitk.AffineTransform(linear.ravel().tolist(), translation.tolist())     # zoom mm -> overview mm
    corners = np.array([forward.TransformPoint(zoom.TransformContinuousIndexToPhysicalPoint([c * (n - 1) for c, n in zip(k, zoom.GetSize())]))
                        for k in np.ndindex(2, 2, 2)])
    lo, hi = corners.min(0), corners.max(0)
    grid = sitk.Image([int(n) for n in np.ceil((hi - lo) / 0.16056)], sitk.sitkFloat32)
    grid.SetSpacing((0.16056,) * 3)
    grid.SetOrigin(lo.tolist())
    moved = sitk.GetArrayFromImage(sitk.Resample(zoom, grid, forward.GetInverse(), sitk.sitkLinear, 0.0))
    fixed = sitk.GetArrayFromImage(sitk.Resample(overview, grid, sitk.Transform(), sitk.sitkLinear, 0.0))
    both = (moved > 0) & (fixed > 0)
    r = np.corrcoef(moved[both], fixed[both])[0, 1] if both.sum() > 1000 else float("nan")
    print(f"{name}: zoom spans {np.round(lo, 1).tolist()} .. {np.round(hi, 1).tolist()} mm; overlap {both.mean():.2f}; correlation {r:.3f}")
    # the same, shifted 1 mm along each axis: a real placement is a peak, not a slope
    for axis in range(3):
        shifted = translation.copy(); shifted[axis] += 1.0
        f2 = sitk.AffineTransform(linear.ravel().tolist(), shifted.tolist())
        m2 = sitk.GetArrayFromImage(sitk.Resample(zoom, grid, f2.GetInverse(), sitk.sitkLinear, 0.0))
        b2 = (m2 > 0) & (fixed > 0)
        print(f"   +1 mm along {'xyz'[axis]}: correlation {np.corrcoef(m2[b2], fixed[b2])[0, 1]:.3f}")
