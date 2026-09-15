"""Where does the tiled level-3 run lose mutual information: in the tissue, at its edge, or in the background tiles?"""
import os
import sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import zarr
ROOT = Path("/home/valentin/Documents/ImpactReg_OOC_Demo")
sys.path.insert(0, str(ROOT))
from stage1_linc import LPS, mutual_information, normalise
from linc import field_header, read_field

S = 2  # every 2nd voxel of level 3
def arr(p):
    g = zarr.open_group(str(p), mode="r"); a = g[dict(g.attrs)["multiscales"][0]["datasets"][0]["path"]]
    return np.asarray(a[0, ::S, ::S, ::S])

def tile_table(run, t):
    # per tile, on the stride the run used (256 with 12 % overlap)
    n, stride = 256 // S, 225 // S
    path = run / "Transform.h5"
    size = field_header(path)[0]
    mag = np.stack([np.linalg.norm(read_field(path, (0, 0, z), (size[0], size[1], z + 1))[0, ::8, ::8], axis=-1) for z in range(0, size[2], 8)])
    rows = []
    for z in range(0, f.shape[0] - n + 1, stride):
        for y in range(0, f.shape[1] - n + 1, stride):
            for x in range(0, f.shape[2] - n + 1, stride):
                box = np.s_[z:z + n, y:y + n, x:x + n]
                tissue = support[box].mean()
                fb = mag[z * S // 8:(z + n) * S // 8, y * S // 8:(y + n) * S // 8, x * S // 8:(x + n) * S // 8]
                rows.append((tissue, mutual_information(f[box], m[box]), mutual_information(f[box], t[box]), float(np.percentile(fb, 50)), float(fb.max())))
    rows = np.array(rows)
    print(f"\n{len(rows)} tiles   tissue fraction | MI start -> tiled | displacement median / max (mm)")
    for lo, hi in ((0, 0.001), (0.001, 0.1), (0.1, 0.4), (0.4, 1.01)):
        sel = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
        if sel.any():
            r = rows[sel]
            print(f"  tissue {lo:.3f}-{hi:.2f}: {sel.sum():2d} tiles   MI {r[:, 1].mean():.3f} -> {r[:, 2].mean():.3f}   "
                  f"worse in {(r[:, 2] < r[:, 1]).sum():2d}   displacement {np.median(r[:, 3]):.2f} / {r[:, 4].max():.2f}")


RUNS = [ROOT / "out" / name / "P000" for name in (sys.argv[1:] or ["linc_tiled3"])]
PAIR = ROOT / "data" / "linc" / os.environ.get("PAIR", "pair3")  # the start the runs were made from
f, m = arr(PAIR / "Fixed.ome.zarr"), arr(PAIR / "Moving.ome.zarr")
grid = sitk.GetImageFromArray(f)
grid.SetSpacing((0.16056 * S,) * 3); grid.SetDirection(LPS)
ref = sitk.GetArrayFromImage(sitk.Resample(normalise(sitk.ReadImage(str(ROOT / "data/linc/dmri_reference.nii.gz"))), grid, sitk.Transform(), sitk.sitkLinear, 0.0))
support = ref > 0.02
print(f"grid {f.shape}")
for name, v in (("start", m), ("consortium", ref)):
    print(f"  {name:22s} MI whole {mutual_information(f, v):.4f}   MI in consortium support {mutual_information(f[support], v[support]):.4f}   MRI support {(v > 0.02).mean():.3f}   r with consortium {np.corrcoef(v[support], ref[support])[0, 1]:.4f}")
for run in RUNS:
    t = arr(run / "Moved.ome.zarr")
    print(f"  {run.parent.name:22s} MI whole {mutual_information(f, t):.4f}   MI in consortium support {mutual_information(f[support], t[support]):.4f}   MRI support {(t > 0.02).mean():.3f}   r with consortium {np.corrcoef(t[support], ref[support])[0, 1]:.4f}")
    tile_table(run, t)

