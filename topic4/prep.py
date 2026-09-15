"""Topic 4, MF283: the MRI -> PS-OCT pair on the PS-OCT grid (100 um), ready for impact-reg.

Fixed: the PS-OCT birefringence map as |v| smoothed at 0.3 mm (the raw signed map carries no usable signal), scaled
to [0, 1] inside the brain. Moving: the MRI brain, placed with the dataset's own affine only, so a deformable run
here replaces the dataset's nonlinear field and can be compared with it. Masks: the two brain masks.

    python topic4/prep.py
"""
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter

SRC = Path.home() / "Downloads/anatomix_input_data/derivatives/sub-MF283"
OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)

oct_img = sitk.ReadImage(str(SRC / "micr/sub-MF283_acq-normal0deg_rec-bir_res-100um_desc-iso_OCT.nii.gz"))
oct_mask = sitk.ReadImage(str(SRC / "micr/sub-MF283_acq-normal0deg_rec-bir_res-100um_label-brain_desc-iso_mask.nii.gz")) > 0.5
affine = sitk.ReadTransform(str(SRC / "xfm/sub-MF283_from-dMRI_to-PSOCT_mode-image_xfm.txt"))  # fixed = OCT grid

v = sitk.GetArrayFromImage(oct_img)
inside = sitk.GetArrayFromImage(oct_mask).astype(bool)
smooth = gaussian_filter(np.abs(v), 3)
lo, hi = np.percentile(smooth[inside], [1, 99])
fixed = sitk.GetImageFromArray(np.clip((smooth - lo) / (hi - lo), 0, 1).astype(np.float32))
fixed.CopyInformation(oct_img)

mri = sitk.ReadImage(str(SRC / "anat/sub-MF283_acq-MSME_desc-brain.nii.gz"), sitk.sitkFloat32)
mri_mask = sitk.ReadImage(str(SRC / "anat/sub-MF283_acq-MSME_label-brain_mask.nii.gz")) > 0.5
moving = sitk.Resample(mri, oct_img, affine, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
m = sitk.GetArrayFromImage(moving); lo, hi = np.percentile(m[m > 0], [1, 99])
moving = sitk.GetImageFromArray(np.clip((m - lo) / (hi - lo), 0, 1).astype(np.float32)); moving.CopyInformation(oct_img)
moving_mask = sitk.Resample(mri_mask, oct_img, affine, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)

for name, image in (("Fixed", fixed), ("Moving", moving), ("FixedMask", sitk.Cast(oct_mask, sitk.sitkUInt8)), ("MovingMask", moving_mask)):
    sitk.WriteImage(image, str(OUT / f"{name}.mha"), useCompression=True)
    a = sitk.GetArrayViewFromImage(image)
    print(f"{name:10s} {image.GetSize()} spacing {image.GetSpacing()} mean {float(a.mean()):.3f} nonzero {float(np.mean(a > 0)):.1%}")
