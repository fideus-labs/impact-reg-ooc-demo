"""APEX alignment figure in the style of the LINC deck: the FA in grey, moving from panel to panel (after the affine,
elastix, FireANTs, ConvexAdam), under the retardance's own white-matter outline in gold, identical in every panel.
Aligned = the FA's bright tracts lie under the gold lines. Top row: the plane through the tissue centre; bottom row: the
24 mm box where the FireANTs field moves the FA most in the plane. Display: 4x bilinear upsampling (geometry unchanged).
Also a flicker GIF per subject and stage (retardance / FA alternating) on the zoom box, for the slideshow."""
from pathlib import Path
import numpy as np, SimpleITK as sitk, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter, uniform_filter, zoom
from PIL import Image, ImageDraw, ImageFont

U = Path("/home/valentin/Documents/ImpactReg_OOC_Demo/upload/apex")
OUT = Path("/tmp/claude-1000/-home-valentin-Documents-lib-ITKIMPACT/66517e9d-e281-42c1-af4e-92537b98fee2/scratchpad/overlays5")
BACKUP = Path("/home/valentin/Documents/backup/apex_out_2026-09-15")
RUN = {"subject_v_VB": "subject_v/stage2_mirror_APEX_FIREANTS_SYN_CC", "subject_m_VB": "subject_m/stage2_nomirror_slidenorm_APEX_FIREANTS_SYN_CC"}
PANELS = [("affine only", "Moving.mha"), ("elastix MI + bending", "Moved_elastix_MI_bending.mha"),
          ("FireANTs SyN CC", "Moved_FireANTs_SyN_CC.mha"), ("ConvexAdam MIND", "Moved_ConvexAdam_MIND.mha")]
HALF, UP, LEVEL, GOLD = 30, 4, 0.45, "#F0B429"
FONT = ImageFont.truetype("/usr/share/fonts/truetype/crosextra/Carlito-Bold.ttf", 22)


def norm(a):
    v = a[a > 0]
    return np.clip(a / (np.percentile(v, 99.5) if v.size else 1.0), 0, 1)


def up(a):
    return zoom(a, UP, order=1)


for sd in sorted(U.glob("subject_*")):
    fixed = sitk.ReadImage(str(sd / "Fixed.mha"))
    F3 = norm(sitk.GetArrayFromImage(fixed))
    z = int(np.round(np.argwhere(F3 > 0.1).mean(axis=0))[0])
    f = F3[z]
    rows, cols = np.where(f > 0.05)
    r0, r1, c0, c1 = max(rows.min() - 2, 0), min(rows.max() + 3, f.shape[0]), max(cols.min() - 2, 0), min(cols.max() + 3, f.shape[1])
    f = f[r0:r1, c0:c1]; h, w = f.shape
    # the zoom box: where the FireANTs field moves the FA most within the plane (its grid is the fixed grid, checked)
    u = sitk.DisplacementFieldTransform(sitk.ReadTransform(str(BACKUP / RUN[sd.name] / "P000" / "Transform.h5"))).GetDisplacementField()
    assert u.GetSize() == fixed.GetSize() and np.allclose(u.GetOrigin(), fixed.GetOrigin()) and np.allclose(u.GetDirection(), fixed.GetDirection())
    inplane = np.linalg.norm(sitk.GetArrayFromImage(u)[z][r0:r1, c0:c1, :2], axis=-1)
    tissue = (f > 0.05).astype(float)
    mean_move = uniform_filter(inplane * tissue, 2 * HALF) / np.maximum(uniform_filter(tissue, 2 * HALF), 1e-6)
    valid = uniform_filter(tissue, 2 * HALF) >= 0.8
    valid[:HALF, :] = valid[h - HALF:, :] = False; valid[:, :HALF] = valid[:, w - HALF:] = False
    cy, cx = np.unravel_index(np.argmax(np.where(valid, mean_move, -1)), mean_move.shape)
    zr0, zc0 = int(cy - HALF), int(cx - HALF)
    zoom_sel = (slice(zr0 * UP, (zr0 + 2 * HALF) * UP), slice(zc0 * UP, (zc0 + 2 * HALF) * UP))
    f_up = up(gaussian_filter(f, 0.8))  # the retardance's outline, drawn once per panel
    moved = {}
    for label, name in PANELS:
        img = sitk.ReadImage(str(sd / name))
        assert img.GetSize() == fixed.GetSize() and np.allclose(img.GetOrigin(), fixed.GetOrigin()) \
            and np.allclose(img.GetSpacing(), fixed.GetSpacing()) and np.allclose(img.GetDirection(), fixed.GetDirection()), (sd.name, name)
        moved[label] = up(norm(sitk.GetArrayFromImage(img))[z][r0:r1, c0:c1])

    top_h = 3.2
    fig = plt.figure(figsize=(4 * top_h * w / h + 0.3, top_h + top_h * w / h + 0.15), dpi=150, facecolor="black")
    gs = fig.add_gridspec(2, 4, height_ratios=[top_h, top_h * w / h], hspace=0.04, wspace=0.03, left=0.004, right=0.996, top=0.955, bottom=0.004)
    for c, (label, _) in enumerate(PANELS):
        for r, sel in enumerate([(slice(None), slice(None)), zoom_sel]):
            ax = fig.add_subplot(gs[r, c])
            m, fc = moved[label][sel], f_up[sel]
            ax.imshow(m, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
            yy, xx = np.arange(fc.shape[0]), np.arange(fc.shape[1])
            ax.contour(xx, yy, fc, levels=[LEVEL], colors=GOLD, linewidths=1.1 if r == 0 else 1.6)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.add_patch(Rectangle((zc0 * UP - 0.5, zr0 * UP - 0.5), 2 * HALF * UP, 2 * HALF * UP, fill=False, edgecolor="white", linewidth=0.9, linestyle="--"))
                ax.set_title(label, color="white", fontsize=13, pad=4)
    fig.savefig(OUT / f"{sd.name}_edges_slide.png", facecolor="black")
    plt.close(fig)

    # flicker GIFs on the zoom box: retardance / FA alternating, 6x bilinear, a label in the corner
    px = 2 * HALF * 6
    ret = (zoom(f[zr0:zr0 + 2 * HALF, zc0:zc0 + 2 * HALF], 6, order=1) * 255).clip(0, 255).astype(np.uint8)
    for label, key in (("affine", "affine only"), ("fireants", "FireANTs SyN CC")):
        fa = (moved[key][zoom_sel] * 255).clip(0, 255).astype(np.uint8)
        fa = np.array(Image.fromarray(fa).resize((px, px), Image.BILINEAR))
        frames = []
        for arr, text in ((ret, "retardance"), (fa, f"FA · {key}")):
            im = Image.fromarray(arr).convert("RGB")
            d = ImageDraw.Draw(im); d.rectangle([0, 0, px, 34], fill=(0, 0, 0)); d.text((8, 5), text, fill=(240, 180, 41), font=FONT)
            frames.append(im)
        frames[0].save(OUT / f"{sd.name}_flicker_{label}.gif", save_all=True, append_images=frames[1:], duration=700, loop=0)
    print(sd.name, f"plane z = {z}, zoom box rows {zr0}-{zr0 + 2 * HALF} cols {zc0}-{zc0 + 2 * HALF}; edges figure + 2 flicker GIFs")
