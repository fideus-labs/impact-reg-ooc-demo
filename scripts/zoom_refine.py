"""How far is the consortium's manual placement of zoom 1 from the best rigid and affine fit? At 160 um."""
import numpy as np
import SimpleITK as sitk
import zarr

U_OUT = 0.010072
M = np.array([[0.999445745, 0.031534287, -0.011279779, 3817.405521157],
              [-0.031774949, 0.999263261, -0.021834133, 4528.589463744],
              [0.010582873, 0.022180297, 0.999704696, 5999.076104469]])


def image(array_xyz, spacing, origin):
    out = sitk.GetImageFromArray(np.ascontiguousarray(array_xyz.transpose(2, 1, 0)).astype(np.float32))
    out.SetSpacing((spacing,) * 3)
    out.SetOrigin((origin,) * 3)
    return out


store = zarr.open_group("s3://dandiarchive/zarr/d483e140-ccd7-4bcf-8a69-a10b7087381c", mode="r", storage_options={"anon": True})
zoom = image(np.asarray(store["5"]), 0.136224, 0.068112)
overview = sitk.ReadImage("/home/valentin/Documents/ImpactReg_OOC_Demo/data/linc/XPCT_level3.mha")
overview = image(sitk.GetArrayFromImage(overview).transpose(2, 1, 0), 0.16056, 0.08028)

manual = sitk.AffineTransform(M[:, :3].ravel().tolist(), (M[:, 3] * U_OUT).tolist())   # zoom mm -> overview mm
inverse = manual.GetInverse()                                                          # overview -> zoom
corners = np.array([manual.TransformPoint(zoom.TransformContinuousIndexToPhysicalPoint([c * (n - 1) for c, n in zip(k, zoom.GetSize())]))
                    for k in np.ndindex(2, 2, 2)])
grid = sitk.Image([int(n) for n in np.ceil((corners.max(0) - corners.min(0)) / 0.16056)], sitk.sitkFloat32)
grid.SetSpacing((0.16056,) * 3); grid.SetOrigin(corners.min(0).tolist())
fixed = sitk.Resample(overview, grid, sitk.Transform(), sitk.sitkLinear, 0.0)
# the zoom's own box, 2 mm in from its faces: outside it the zoom has no data
inside = sitk.Resample(sitk.Image(zoom.GetSize(), sitk.sitkUInt8) + 1, grid, inverse, sitk.sitkNearestNeighbor, 0)
mask = sitk.BinaryErode(inside, [int(2 / 0.16056)] * 3)
centre = np.array(fixed.TransformContinuousIndexToPhysicalPoint([(n - 1) / 2 for n in fixed.GetSize()]))


def correlation(transform):
    moved = sitk.GetArrayFromImage(sitk.Resample(zoom, fixed, transform, sitk.sitkLinear, 0.0))
    m = sitk.GetArrayFromImage(mask) > 0
    return np.corrcoef(moved[m], sitk.GetArrayFromImage(fixed)[m])[0, 1]


def points_moved(transform, reference):
    """How far each fixed point of the zoom's box lands from where the manual placement sends it, in mm."""
    idx = np.argwhere(sitk.GetArrayFromImage(mask)[::6, ::6, ::6] > 0)[:, ::-1] * 6
    d = [np.linalg.norm(np.array(transform.TransformPoint(fixed.TransformIndexToPhysicalPoint(i.tolist())))
                        - np.array(reference.TransformPoint(fixed.TransformIndexToPhysicalPoint(i.tolist())))) for i in idx]
    return np.median(d), np.max(d)


print(f"manual placement: correlation {correlation(inverse):.3f}")
for name, start in (("rigid", sitk.Euler3DTransform()), ("affine", sitk.AffineTransform(3))):
    start.SetCenter(centre.tolist())
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(50)
    reg.SetMetricFixedMask(mask)
    reg.SetMetricSamplingStrategy(reg.RANDOM); reg.SetMetricSamplingPercentage(0.2, seed=1)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(0.5, 1e-4, 300, relaxationFactor=0.7)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([2, 1]); reg.SetSmoothingSigmasPerLevel([1, 0])
    reg.SetMovingInitialTransform(inverse)
    reg.SetInitialTransform(start, inPlace=True)
    reg.Execute(fixed, zoom)
    chain = sitk.CompositeTransform([inverse, start])      # the correction is applied first, then the manual placement
    med, mx = points_moved(chain, inverse)
    extra = ""
    if name == "affine":
        a = np.array(start.GetMatrix()).reshape(3, 3)
        extra = f", scales {np.round(np.linalg.svd(a)[1], 4).tolist()}"
    print(f"{name}: correlation {correlation(chain):.3f}; moves the zoom by median {med:.2f} mm, max {mx:.2f} mm{extra}; stop: {reg.GetOptimizerStopConditionDescription()[:60]}")
