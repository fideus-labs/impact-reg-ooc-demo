"""Where the level-4 deformable moves the MRI: a 40 mm window over the deep grey nuclei at 0.1 mm, the MRI's
contour over the XPCT after the affine and after the deformable.

    python figures_deep.py
"""

import numpy as np
import SimpleITK as sitk

import linc
from figures_linc import DATA, ROOT, bands, iso_contours_over, save
from stage1_linc import LPS, normalise


def main() -> None:
    consortium_seg = sitk.ReadImage(str(DATA / "dseg_spaceXPCT.nii.gz"))          # only to centre the window
    where = np.argwhere(np.isin(sitk.GetArrayFromImage(consortium_seg), (12, 13)))   # putamen and pallidum
    cx, cy, cz = consortium_seg.TransformContinuousIndexToPhysicalPoint(where.mean(0)[::-1].tolist())
    step, half = 0.1, 200                                                             # 40 mm at 0.1 mm
    xpct3 = sitk.ReadImage(str(DATA / "XPCT_level3.mha"))
    xpct3.SetDirection(LPS)
    source = normalise(sitk.ReadImage(str(DATA / "dmri_oriented.mha")))
    affine = sitk.ReadTransform(str(DATA / "affine.tfm"))
    coarse_field = ROOT / "out" / "linc_coarse_bend100" / "P000" / "Transform.h5"
    rows = []
    for plane, size, origin in (("axial", (2 * half, 2 * half, 1), (cx + half * step, cy + half * step, cz)),
                                ("coronal", (2 * half, 1, 2 * half), (cx + half * step, cy, cz - half * step))):
        slab = sitk.Image(size, sitk.sitkFloat32)
        slab.SetSpacing((step,) * 3)
        slab.SetDirection(LPS)
        slab.SetOrigin(origin)
        chain = sitk.CompositeTransform(3)
        chain.AddTransform(affine)
        chain.AddTransform(linc.field_in_store_frame(coarse_field, slab))

        def on(image: sitk.Image, transform: sitk.Transform, slab: sitk.Image = slab) -> np.ndarray:
            return sitk.GetArrayFromImage(sitk.Resample(image, slab, transform, sitk.sitkLinear, 0.0)).squeeze()

        background, before, after = on(xpct3, sitk.Transform()), on(source, affine), on(source, chain)
        level = float(np.median(after[after > 0.02]))
        rows.append((f"{plane} · 40 mm over the deep nuclei · after the affine | after the deformable",
                     [iso_contours_over(background, before, level), iso_contours_over(background, after, level)]))
    save(bands(rows, size=20), "D10_deep_nuclei.png", scale=0.9)


if __name__ == "__main__":
    main()
