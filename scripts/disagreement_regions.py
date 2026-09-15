"""Where this pipeline and the consortium disagree most, and a close look at each of those places.

The field of `out/linc_vs_consortium` moves this pipeline's dMRI map onto the consortium's: its length is
how far apart the two alignments put the same anatomy. The four largest, at least 20 mm apart, are drawn
over the XPCT: FLASH outlines (gold this pipeline, cyan the consortium) and the two MRI maps (green, magenta).
"""
import sys
from pathlib import Path

ROOT = Path("/home/valentin/Documents/ImpactReg_OOC_Demo")
sys.path.insert(0, str(ROOT))
import numpy as np
import SimpleITK as sitk
import zarr
from scipy.ndimage import gaussian_filter

import linc
from evaluate_linc import DATA, LABELS, carried_segmentation
from figures_linc import agreement, bands, outlines, save
from stage1_linc import LPS, normalise

field_path = ROOT / "out/linc_vs_consortium/P000/Transform.h5"
size, origin, spacing = linc.field_header(field_path)
step = 2
length = np.linalg.norm(linc.read_field(field_path, (0, 0, 0), size, step), axis=-1)       # (z, y, x) at 0.64 mm
group = zarr.open_group(str(DATA / "pair_vs_consortium/FixedMask.ome.zarr"), mode="r")
mask = np.asarray(group[dict(group.attrs)["multiscales"][0]["datasets"][0]["path"]][0, ::step, ::step, ::step]) > 0
mask = mask[: length.shape[0], : length.shape[1], : length.shape[2]]
voxel = spacing[0] * step
inside = length[mask]
print(f"distance between the two alignments inside the brain: median {np.median(inside):.2f} mm, "
      f"95 % under {np.percentile(inside, 95):.2f} mm, max {inside.max():.2f} mm")

smooth = gaussian_filter(length, 2.0 / voxel) * mask
consortium_seg = sitk.ReadImage(str(DATA / "dseg_spaceXPCT.nii.gz"))
zz, yy, xx = np.mgrid[0 : smooth.shape[0], 0 : smooth.shape[1], 0 : smooth.shape[2]]
regions = []
work = smooth.copy()
for _ in range(4):
    k, j, i = np.unravel_index(np.argmax(work), work.shape)
    q = np.array([i, j, k], dtype=float) * voxel + origin          # the field's own frame
    p = q * np.array([-1.0, -1.0, 1.0])                            # the store's frame: levels 3 and 4 start at 0
    label = int(consortium_seg.GetPixel(consortium_seg.TransformPhysicalPointToIndex(p.tolist())))
    regions.append((float(smooth[k, j, i]), p, LABELS.get(label, f"label {label}" if label else "outside the labels")))
    work[(zz - k) ** 2 + (yy - j) ** 2 + (xx - i) ** 2 <= (20.0 / voxel) ** 2] = 0
for n, (distance, p, structure) in enumerate(regions, 1):
    print(f"region {n}: {distance:.2f} mm apart (2 mm average), at {np.round(p, 1).tolist()} mm, in {structure}")

xpct = sitk.ReadImage(str(DATA / "XPCT_level3.mha"))
xpct.SetDirection(LPS)
_, oriented, flash_affine = carried_segmentation("flash", DATA / "flash_brain_024.nii.gz")
source = normalise(sitk.ReadImage(str(DATA / "dmri_oriented.mha")))
consortium_mri = normalise(sitk.ReadImage(str(DATA / "dmri_reference.nii.gz")))
dmri_affine = sitk.ReadTransform(str(DATA / "affine.tfm"))
flash_field = ROOT / "out/linc_flash_bend100/P000/Transform.h5"
coarse_field = ROOT / "out/linc_coarse_bend100/P000/Transform.h5"
reference_array = sitk.GetArrayFromImage(consortium_seg)

for n, (distance, (cx, cy, cz), structure) in enumerate(regions, 1):
    rows = []
    for plane, slab_size, slab_origin in (("axial", (400, 400, 1), (cx + 20, cy + 20, cz)),
                                          ("coronal", (400, 1, 400), (cx + 20, cy, cz - 20))):
        slab = sitk.Image(list(slab_size), sitk.sitkFloat32)
        slab.SetSpacing((0.1, 0.1, 0.1))
        slab.SetDirection(LPS)
        slab.SetOrigin((float(slab_origin[0]), float(slab_origin[1]), float(slab_origin[2])))

        def on(image, transform, labels=False):
            interpolator = sitk.sitkNearestNeighbor if labels else sitk.sitkLinear
            return sitk.GetArrayFromImage(sitk.Resample(image, slab, transform, interpolator, 0.0)).squeeze()

        flash_chain = sitk.CompositeTransform(3)
        flash_chain.AddTransform(flash_affine)
        flash_chain.AddTransform(linc.field_in_store_frame(flash_field, slab))
        dmri_chain = sitk.CompositeTransform(3)
        dmri_chain.AddTransform(dmri_affine)
        dmri_chain.AddTransform(linc.field_in_store_frame(coarse_field, slab))
        reference = on(consortium_seg, sitk.Transform(), True)
        present = [int(v) for v in np.unique(reference) if int(v) in LABELS]
        largest = sorted(present, key=lambda v: -(reference == v).sum())[:5]
        names = {v: LABELS[v] for v in largest}
        rows.append((f"{plane} · {distance:.1f} mm apart · ours gold/green, consortium cyan/magenta",
                     [outlines(on(xpct, sitk.Transform()), on(oriented, flash_chain, True), reference, names),
                      agreement(on(source, dmri_chain), on(consortium_mri, sitk.Transform()))]))
    save(bands(rows, size=20), f"L12_region{n}.png")
    print(f"  L12_region{n}.png: {structure}")
