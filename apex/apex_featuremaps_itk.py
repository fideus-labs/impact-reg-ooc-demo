"""IMPACT feature maps on the APEX pair with itk-impact's own ImageToFeaturesMap (0.1.5, the configuration of
ITKIMPACT examples/MakeExampleImages.py): the retardance and the FA after FireANTs (subject_v) through MIND,
TotalSegmentator (M730), SAM 2.1 Tiny and Anatomix. One PCA basis per model, fitted on the tissue of both images.
Plane z = 57 of the 0.4 mm grid, cropped to the tissue."""
import time
from pathlib import Path
import itk, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HF = Path.home() / ".cache/huggingface/hub/models--VBoussot--impact-torchscript-models/snapshots/eed8957e1984adff47a376144ebd5d3e19848958"
PAIR = Path("/home/valentin/Documents/ImpactReg_OOC_Demo/upload/apex/subject_v_VB")
OUT = Path("/tmp/claude-1000/-home-valentin-Documents-lib-ITKIMPACT/66517e9d-e281-42c1-af4e-92537b98fee2/scratchpad/featuremaps")
Z, DEVICE = 57, "cpu"
# name, file, dimension, channels, voxel (mm), patch (0 = whole image), overlap, layers mask
MODELS = [("MIND", "MIND/R1D2_3D.pt", 3, 1, 0.4, 0, 0, [True]),
          ("TotalSegmentator", "TS/M730.pt", 3, 1, 0.4, 0, 0, [False, True]),
          ("SAM 2.1", "SAM2.1/SAM2.1_Tiny.pt", 2, 3, 0.2, 256, 32, [True]),  # tiles a multiple of its window grid
          ("Anatomix", "Anatomix/Anatomix.pt", 3, 1, 0.4, [128, 176, 128], 0, [True])]  # one tile over the whole 123x175x113 volume: no seam


def feature_map(image, rel, dim, channels, voxel, patch, overlap, mask):
    ImageType = itk.Image[itk.F, dim]
    patches = patch if isinstance(patch, list) else [patch] * dim
    config = itk.ImpactModelConfiguration(str(HF / rel), dim, channels, patches, [voxel] * dim, [overlap] * dim, mask, False)
    config.SetPatchCombine("cosinus")
    f = itk.ImageToFeaturesMap[ImageType, itk.BSplineInterpolateImageFunction[ImageType, itk.D, itk.F]].New()
    f.SetModelConfiguration(config)
    f.SetDevice(DEVICE)
    f.AddInput(image)
    f.Update()
    out = f.GetOutput(0)
    return np.array(itk.array_view_from_image(out), copy=True), out


def slice2d(image3d):
    arr = np.ascontiguousarray(itk.array_view_from_image(image3d)[Z])
    im = itk.image_from_array(arr)
    sp, org = image3d.GetSpacing(), image3d.GetOrigin()
    im.SetSpacing([sp[0], sp[1]]); im.SetOrigin([org[0], org[1]])
    d = itk.array_from_matrix(image3d.GetDirection())
    im.SetDirection(itk.matrix_from_array(np.ascontiguousarray(d[:2, :2])))
    return im


def pca_pair(a, b, inside, chroma=0.34):
    """MakeExampleImages.pca_rgb with one basis and one stretch for both maps, fitted on the tissue."""
    flat = np.concatenate([a[inside], b[inside]])
    sample = flat[:: max(1, flat.shape[0] // 200000)]
    mean = sample.mean(0)
    _, _, basis = np.linalg.svd(sample - mean, full_matrices=False)
    comps = [((m.reshape(-1, m.shape[-1]) - mean) @ basis[:3].T).reshape(*m.shape[:2], 3) for m in (a, b)]
    both = np.concatenate([c[inside] for c in comps])
    lo, hi = np.percentile(both, 2, axis=0), np.percentile(both, 98, axis=0)
    rgb = []
    for c in comps:
        n = np.clip((c - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
        lum, ca, cb = n[..., 0] * 0.88 + 0.12, n[..., 1] * 2 - 1, n[..., 2] * 2 - 1
        img = np.clip(np.stack([lum * (1 + chroma * ca), lum * (1 - chroma * 0.5 * (ca - cb)), lum * (1 - chroma * cb)], -1), 0, 1)
        img[~inside] = 0.04
        rgb.append(img)
    return rgb


fixed = itk.imread(str(PAIR / "Fixed.mha"), itk.F)
moved = itk.imread(str(PAIR / "Moved_FireANTs_SyN_CC.mha"), itk.F)
F = itk.array_from_image(fixed)[Z]; M = itk.array_from_image(moved)[Z]
tissue = F > np.percentile(F[F > 0], 5)
rows, cols = np.where(tissue)
box = (slice(max(rows.min() - 2, 0), rows.max() + 3), slice(max(cols.min() - 2, 0), cols.max() + 3))
grey = [np.clip(x / np.percentile(x[x > 0], 99.5), 0, 1) for x in (F, M)]
plane_point = fixed.TransformIndexToPhysicalPoint([0, 0, Z])
panels = []
for name, rel, dim, channels, voxel, patch, overlap, mask in MODELS:
    t0 = time.time()
    planes = []
    for image in (fixed, moved):
        src = slice2d(image) if dim == 2 else image
        feat, feat_img = feature_map(src, rel, dim, channels, voxel, patch, overlap, mask)
        if dim == 3:  # the feature plane at the physical position of plane Z
            k = int(np.clip(feat_img.TransformPhysicalPointToIndex(plane_point)[2], 0, feat.shape[0] - 1))
            plane = feat[k]
        else:
            plane = feat
        if plane.ndim == 2:
            plane = plane[..., None]
        yi = np.clip((np.arange(F.shape[0]) * plane.shape[0] / F.shape[0]).astype(int), 0, plane.shape[0] - 1)
        xi = np.clip((np.arange(F.shape[1]) * plane.shape[1] / F.shape[1]).astype(int), 0, plane.shape[1] - 1)
        planes.append(plane[yi][:, xi])
        spacing = feat_img.GetSpacing()[0]
    panels.append((name, pca_pair(planes[0], planes[1], tissue), spacing, planes[0].shape[-1]))
    print(f"{name}: {planes[0].shape[-1]} channels at {spacing:.2f} mm, {time.time() - t0:.1f} s", flush=True)

GROUND, LABEL = "#0B0F14", "#e6edf2"
h, w = grey[0][box].shape
fig, axes = plt.subplots(2, 5, figsize=(5 * 2.3 * w / h + 0.5, 2 * 2.3 + 0.95), dpi=160, facecolor=GROUND)
for r, row_label in enumerate(("PS-OCT retardance", "DTI FA, registered")):
    axes[r, 0].imshow(grey[r][box], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[r, 0].set_ylabel(row_label, color=LABEL, fontsize=11)
    for c, (name, rgb, spacing, channels) in enumerate(panels, 1):
        axes[r, c].imshow(rgb[r][box], interpolation="nearest")
        if r == 0:
            axes[r, c].set_title(f"{name}\n{channels} ch @ {spacing:.1f} mm", color=LABEL, fontsize=11)
axes[0, 0].set_title("image\n0.4 mm", color=LABEL, fontsize=11)
for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor(GROUND)
    for s in ax.spines.values():
        s.set_visible(False)
fig.subplots_adjust(left=0.035, right=0.995, top=0.86, bottom=0.01, wspace=0.03, hspace=0.04)
fig.savefig(OUT / "apex_featuremaps_itk.png", facecolor=GROUND)
print(OUT / "apex_featuremaps_itk.png")
