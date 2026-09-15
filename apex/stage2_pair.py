"""APEX, the stage-2 pair: the retardance on a 0.4 mm isotropic grid (FA's resolution) as the fixed image, the FA
through a stage-1 affine on the same grid as the moving image, and both tissue masks (the fixed one widened by 1 mm).

    python apex/stage2_pair.py subject_m nomirror
    python apex/stage2_pair.py subject_m nomirror_slidenorm apex/out/subject_m/Ret_slidenorm.nii.gz nomirror
"""
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_dilation, binary_fill_holes

HERE = Path(__file__).resolve().parent
subject, tag = sys.argv[1], sys.argv[2]
ret_path = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / "data" / subject / "Ret_slide_deck.nii.gz"  # another retardance source
affine_tag = sys.argv[4] if len(sys.argv) > 4 else tag                                                        # and the stage-1 affine to use
OUT = HERE / "out" / subject / f"pair_{tag}"
OUT.mkdir(parents=True, exist_ok=True)


def normalise(a: np.ndarray) -> np.ndarray:
    nz = a[a > 0]; lo, hi = np.percentile(nz, [1, 99])
    return np.clip((a - lo) / (hi - lo), 0, 1).astype(np.float32)


ret = sitk.ReadImage(str(ret_path), sitk.sitkFloat32)
fa = sitk.ReadImage(str(HERE / "data" / subject / "dti_FA.nii.gz"), sitk.sitkFloat32)
affine = sitk.ReadTransform(str(HERE / "out" / subject / f"stage1_affine_{affine_tag}.tfm"))
mm = 0.4
size = [int(round(n * s / mm)) for n, s in zip(ret.GetSize(), ret.GetSpacing())]
smooth = sitk.SmoothingRecursiveGaussian(ret, [max(mm / 2 - s / 2, 0.01) for s in ret.GetSpacing()])
grid = sitk.Resample(smooth, size, sitk.Transform(), sitk.sitkLinear, ret.GetOrigin(), [mm] * 3, ret.GetDirection(), 0.0)
fixed_a = normalise(sitk.GetArrayFromImage(grid))
moving_a = normalise(sitk.GetArrayFromImage(sitk.Resample(fa, grid, affine, sitk.sitkLinear, 0.0)))
fixed_tissue = binary_fill_holes(fixed_a > 0.02)
moving_tissue = binary_fill_holes(moving_a > 0.02)
fixed_mask = binary_dilation(fixed_tissue, iterations=int(round(1.0 / mm)))
for name, a in (("Fixed", fixed_a), ("Moving", moving_a), ("FixedMask", fixed_mask.astype(np.uint8)), ("MovingMask", np.ones_like(fixed_mask, np.uint8)),
                ("FixedTissue", fixed_tissue.astype(np.uint8)), ("MovingTissue", moving_tissue.astype(np.uint8))):
    img = sitk.GetImageFromArray(a); img.CopyInformation(grid)
    sitk.WriteImage(img, str(OUT / f"{name}.mha"), useCompression=True)
dice = 2 * np.sum(fixed_tissue & moving_tissue) / (fixed_tissue.sum() + moving_tissue.sum())
print(f"{subject} {tag}: grid {grid.GetSize()} at {mm} mm; tissue Dice after the affine {dice:.3f}; fixed mask covers {fixed_mask.mean():.1%} -> {OUT}")
