"""Stage 1 for the LINC pair: find the orientation, then the affine, on a coarse pyramid level.

The OME-Zarr store carries axes, a scale and a half-voxel translation. It carries nothing about
anatomy, so which way the hemisphere faces in it is not written down anywhere. The dMRI map does
carry its own frame, in RAS, and the two do not agree: the volumes differ by an axis permutation.

An affine started from the identity never crosses a permutation, so stage 1 screens the 24
axis-aligned orientations by mutual information, keeps the best, and refines it into a full affine.
All of it on a coarse level, in a couple of minutes, without touching the 908 GiB of level 0.

    python stage1_linc.py --level 4
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import SimpleITK as sitk

DATA = Path(__file__).parent / "data" / "linc"
LPS = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)


def rotations() -> list[np.ndarray]:
    """The 24 axis-aligned rotations of a cube: every permutation with an even number of flips."""
    out = []
    for order in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            matrix = np.zeros((3, 3))
            for row, column in enumerate(order):
                matrix[row, column] = signs[row]
            if round(np.linalg.det(matrix)) == 1:
                out.append(matrix)
    return out


def normalise(image: sitk.Image) -> sitk.Image:
    array = sitk.GetArrayFromImage(image)
    array = np.clip(array / (np.percentile(array, 99.5) + 1e-9), 0, 1).astype(np.float32)
    out = sitk.GetImageFromArray(array)
    out.CopyInformation(image)
    return out


def downsample(image: sitk.Image, factor: int) -> sitk.Image:
    size = [max(n // factor, 1) for n in image.GetSize()]
    spacing = [s * factor for s in image.GetSpacing()]
    return sitk.Resample(image, size, sitk.Transform(), sitk.sitkLinear, image.GetOrigin(),
                         spacing, image.GetDirection(), 0.0)


def mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    histogram, _, _ = np.histogram2d(a.ravel(), b.ravel(), bins=bins, range=[[0, 1], [0, 1]])
    joint = histogram / histogram.sum()
    marginal = joint.sum(1, keepdims=True) @ joint.sum(0, keepdims=True)
    nonzero = joint > 0
    return float((joint[nonzero] * np.log(joint[nonzero] / marginal[nonzero])).sum())


def register(fixed: sitk.Image, moving: sitk.Image, kind: str, iterations: int) -> sitk.Transform:
    transform = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform() if kind == "rigid" else sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.MOMENTS,
    )
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=48)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(0.15, seed=42)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=iterations)
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel([4, 2, 1])
    method.SetSmoothingSigmasPerLevel([2, 1, 0])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(transform, inPlace=False)
    return method.Execute(fixed, moving)


def oriented(image: sitk.Image, rotation: np.ndarray) -> sitk.Image:
    out = sitk.Image(image)
    out.SetDirection((rotation @ np.array(image.GetDirection()).reshape(3, 3)).flatten().tolist())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="4")
    parser.add_argument("--moving", type=Path, default=DATA / "dmri.nii.gz")
    parser.add_argument("--screen-factor", type=int, default=4, help="downsampling for the 24-way screen")
    parser.add_argument("--name", default="dmri", help="prefix of what gets written, so two moving images coexist")
    args = parser.parse_args()
    # the dMRI run keeps the names stage 3 and stage 4 already read
    legacy = args.name == "dmri"
    affine_path = DATA / ("affine.tfm" if legacy else f"{args.name}_affine.tfm")
    orientation_path = DATA / ("orientation.npy" if legacy else f"{args.name}_orientation.npy")

    fixed = sitk.ReadImage(str(DATA / f"XPCT_level{args.level}.mha"))
    fixed.SetDirection(LPS)
    fixed = normalise(fixed)
    moving = normalise(sitk.ReadImage(str(args.moving)))

    screen_fixed = downsample(fixed, args.screen_factor)
    target = sitk.GetArrayFromImage(screen_fixed)
    print(f"screening 24 orientations at {screen_fixed.GetSpacing()[0]:.2f} mm, grid {screen_fixed.GetSize()}")

    scores = []
    for index, rotation in enumerate(rotations()):
        candidate = oriented(moving, rotation)
        transform = register(screen_fixed, candidate, "rigid", 120)
        moved = sitk.Resample(candidate, screen_fixed, transform, sitk.sitkLinear, 0.0)
        score = mutual_information(target, sitk.GetArrayFromImage(moved))
        scores.append((score, index, rotation, transform))
        print(f"  orientation {index:2d}: MI {score:.4f}")

    scores.sort(key=lambda row: -row[0])
    best_score, best_index, best_rotation, _ = scores[0]
    print(f"kept orientation {best_index} with MI {best_score:.4f}; refining the affine at "
          f"{fixed.GetSpacing()[0]:.2f} mm")

    candidate = oriented(moving, best_rotation)
    affine = register(fixed, candidate, "affine", 600)
    moved = sitk.Resample(candidate, fixed, affine, sitk.sitkLinear, 0.0)
    print(f"after the affine: MI {mutual_information(sitk.GetArrayFromImage(fixed), sitk.GetArrayFromImage(moved)):.4f}")

    sitk.WriteImage(candidate, str(DATA / f"{args.name}_oriented.mha"), useCompression=True)
    sitk.WriteImage(moved, str(DATA / f"{args.name}_affine.mha"), useCompression=True)
    sitk.WriteTransform(affine, str(affine_path))
    np.save(orientation_path, best_rotation)


if __name__ == "__main__":
    main()
