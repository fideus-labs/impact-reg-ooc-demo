"""Stage 1: the coarse global alignment, from the pyramid's coarse levels.

The 50 GiB and 31 GiB stores each carry a pyramid. Stage 1 reads one coarse level of each, projects
the slab it holds, and solves a global affine between the two modalities by mutual information — the
one metric that does not care what each modality measures. On an image a few hundred pixels across
this costs seconds, and it never touches the native data.

It also searches the four in-plane mirrorings before optimising. That is not a formality: the
sections are mounted by hand, and this one is mirrored in x between the two microscopes. An affine
started from the identity never crosses a reflection, so without that search every later stage is
matching a section to its own mirror image.

What it writes is the affine that stage 2 starts from, so the native run has only local deformation
left to solve.
"""

from pathlib import Path

import numpy as np
import SimpleITK as sitk

DATA = Path(__file__).parent / "data" / "dandi"
SPACING = 0.05  # mm, the coarse in-plane grid


def projection(name: str) -> sitk.Image:
    """The slab of one modality, projected to one plane and put on the common coarse grid."""
    image = sitk.ReadImage(str(DATA / f"overview_{name}.mha"))
    array = sitk.GetArrayFromImage(image).mean(0)
    array = array / (np.percentile(array, 99.5) + 1e-6)
    # No masking: in the OCT the cortex sits at the same grey as the embedding medium, so any
    # intensity mask cuts the cortex out and the affine then fits white matter to a whole section.
    plane = sitk.GetImageFromArray(np.clip(array, 0, 1).astype(np.float32))
    plane.SetSpacing(image.GetSpacing()[:2])
    size = [int(round(n * s / SPACING)) for n, s in zip(plane.GetSize(), plane.GetSpacing())]
    return sitk.Resample(plane, size, sitk.Transform(), sitk.sitkLinear, plane.GetOrigin(), [SPACING] * 2)


def mirror(image: sitk.Image, axes: tuple[bool, bool]) -> sitk.AffineTransform:
    """The reflection of `image` about its own centre on the chosen axes, as a transform."""
    diagonal = np.diag([-1.0 if axes[0] else 1.0, -1.0 if axes[1] else 1.0])
    extent = (np.array(image.GetSize()) - 1) * np.array(image.GetSpacing())
    transform = sitk.AffineTransform(2)
    transform.SetMatrix(diagonal.flatten().tolist())
    transform.SetTranslation([extent[i] if axes[i] else 0.0 for i in range(2)])
    return transform


def affine_mi(fixed: sitk.Image, moving: sitk.Image) -> sitk.Transform:
    """Global affine by mutual information, from a moments-centred start."""
    initial = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.AffineTransform(2), sitk.CenteredTransformInitializerFilter.MOMENTS
    )
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=48)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(0.2, seed=42)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=400)
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel([8, 4, 2, 1])
    method.SetSmoothingSigmasPerLevel([4, 2, 1, 0])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(initial, inPlace=False)
    return method.Execute(fixed, moving)


def compose(reflection: sitk.AffineTransform, affine: sitk.Transform) -> sitk.AffineTransform:
    """One affine for `reflection` applied after `affine`, so stage 2 can map a box with a single call."""
    if affine.GetName() == "CompositeTransform":
        affine = sitk.CompositeTransform(affine).GetNthTransform(0)
    a = sitk.AffineTransform(affine)
    matrix = np.array(a.GetMatrix()).reshape(2, 2)
    centre, translation = np.array(a.GetCenter()), np.array(a.GetTranslation())
    m = np.array(reflection.GetMatrix()).reshape(2, 2)
    b = np.array(reflection.GetTranslation())

    out = sitk.AffineTransform(2)
    out.SetMatrix((m @ matrix).flatten().tolist())
    out.SetTranslation((m @ (centre - matrix @ centre + translation) + b).tolist())
    return out


def mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    histogram, _, _ = np.histogram2d(a.ravel(), b.ravel(), bins=bins, range=[[0, 1], [0, 1]])
    joint = histogram / histogram.sum()
    marginal = joint.sum(1, keepdims=True) @ joint.sum(0, keepdims=True)
    nonzero = joint > 0
    return float((joint[nonzero] * np.log(joint[nonzero] / marginal[nonzero])).sum())


def main() -> None:
    fixed, moving = projection("OCT"), projection("SPIM")
    target = sitk.GetArrayFromImage(fixed)
    source = sitk.GetArrayFromImage(moving)

    best = None
    for axes, name in (((False, False), "as imaged"), ((True, False), "mirrored in x"),
                       ((False, True), "mirrored in y"), ((True, True), "mirrored in both")):
        flipped = sitk.GetImageFromArray(np.ascontiguousarray(
            source[:, ::-1] if axes[0] and not axes[1] else
            source[::-1] if axes[1] and not axes[0] else
            source[::-1, ::-1] if axes[0] else source))
        flipped.CopyInformation(moving)
        transform = affine_mi(fixed, flipped)
        score = mutual_information(target, sitk.GetArrayFromImage(
            sitk.Resample(flipped, fixed, transform, sitk.sitkLinear, 0.0)))
        print(f"  {name:16s} mutual information after the affine: {score:.4f}")
        if best is None or score > best[0]:
            best = (score, axes, transform, name)

    score, axes, transform, name = best
    total = compose(mirror(moving, axes), transform)
    moved = sitk.Resample(moving, fixed, total, sitk.sitkLinear, 0.0)
    print(f"kept: {name}   MI {mutual_information(target, sitk.GetArrayFromImage(moved)):.4f} "
          f"(identity: {mutual_information(target, sitk.GetArrayFromImage(sitk.Resample(moving, fixed))):.4f})")

    sitk.WriteTransform(total, str(DATA / "coarse_affine.tfm"))
    for label, image in (("OCT", fixed), ("SPIM_moved", moved), ("SPIM", moving)):
        sitk.WriteImage(image, str(DATA / f"coarse_{label}.mha"), useCompression=True)


if __name__ == "__main__":
    main()
