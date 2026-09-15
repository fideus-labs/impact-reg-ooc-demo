"""This pipeline's FLASH segmentation (gold) against the consortium's (cyan), over the XPCT, three whole planes."""
import sys
from pathlib import Path
ROOT = Path("/home/valentin/Documents/ImpactReg_OOC_Demo")
sys.path.insert(0, str(ROOT))
import numpy as np
import SimpleITK as sitk
import linc
from evaluate_linc import DATA, carried_segmentation
from figures_linc import bands, outlines, save
from stage1_linc import LPS

consortium = sitk.ReadImage(str(DATA / "dseg_spaceXPCT.nii.gz"))
inside = np.argwhere(sitk.GetArrayFromImage(consortium) > 0)
corners = np.array([consortium.TransformContinuousIndexToPhysicalPoint(p[::-1].tolist()) for p in (inside.min(0), inside.max(0))])
lo, hi = corners.min(0) - 5, corners.max(0) + 5           # the hemisphere's box, in mm, plus a margin
centre = inside.mean(0)[::-1]
cx, cy, cz = consortium.TransformContinuousIndexToPhysicalPoint(centre.tolist())
step = 0.2
n = np.ceil((hi - lo) / step).astype(int)
xpct = sitk.ReadImage(str(DATA / "XPCT_level3.mha"))
xpct.SetDirection(LPS)
_, oriented, affine = carried_segmentation("flash", DATA / "flash_brain_024.nii.gz")
names = {2: "white matter", 3: "cortex", 4: "ventricle", 12: "putamen", 10: "thalamus", 8: "cerebellum"}
panels = []
for plane, size, origin in (("axial", (n[0], n[1], 1), (hi[0], hi[1], cz)),
                            ("coronal", (n[0], 1, n[2]), (hi[0], cy, lo[2])),
                            ("sagittal", (1, n[1], n[2]), (cx, hi[1], lo[2]))):
    slab = sitk.Image([int(v) for v in size], sitk.sitkFloat32)
    slab.SetSpacing((step,) * 3)
    slab.SetDirection(LPS)
    slab.SetOrigin(tuple(float(v) for v in origin))
    chain = sitk.CompositeTransform(3)
    chain.AddTransform(affine)
    chain.AddTransform(linc.field_in_store_frame(ROOT / "out/linc_flash_bend100/P000/Transform.h5", slab))
    on = lambda image, transform, labels: sitk.GetArrayFromImage(sitk.Resample(image, slab, transform, sitk.sitkNearestNeighbor if labels else sitk.sitkLinear, 0.0)).squeeze()
    background = on(xpct, sitk.Transform(), False)
    panels.append(outlines(background, on(oriented, chain, True), on(consortium, sitk.Transform(), True), names))
    print(plane, panels[-1].shape)
save(bands([("gold: this pipeline (elastix, mask, bending 100)  ·  cyan: the consortium (TIRL + MIND)  ·  over the XPCT", panels)], size=34),
     "L11_ours_vs_consortium.png", scale=0.6)
