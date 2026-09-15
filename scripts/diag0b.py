"""Native run: are the tiles that got worse the thin ones at the window's far faces?"""
import sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk
ROOT = Path("/home/valentin/Documents/ImpactReg_OOC_Demo")
sys.path.insert(0, str(ROOT))
import linc
from stage1_linc import mutual_information, normalise
from stage4_native_linc import written_window

step = 4
grid = written_window(ROOT / "data/linc/pair0/Fixed.ome.zarr")[::step, ::step, ::step]
fixed = sitk.GetArrayFromImage(grid)
source = normalise(sitk.ReadImage(str(ROOT / "data/linc/dmri_oriented.mha")))
chain = sitk.CompositeTransform(3)
chain.AddTransform(sitk.ReadTransform(str(ROOT / "data/linc/affine.tfm")))
chain.AddTransform(linc.field_in_store_frame(ROOT / "out/linc_coarse/P000/Transform.h5", grid))
chain.AddTransform(linc.field_in_store_frame(ROOT / "out/linc_tiled3_mask/P000/Transform.h5", grid))
start = sitk.GetArrayFromImage(sitk.Resample(source, grid, chain, sitk.sitkLinear, 0.0))
native = linc.field_in_store_frame(ROOT / "out/linc_native/P000/Transform.h5", grid, step=step)
chain.AddTransform(native)
after = sitk.GetArrayFromImage(sitk.Resample(source, grid, chain, sitk.sitkLinear, 0.0))
u = np.linalg.norm(sitk.GetArrayFromImage(sitk.DisplacementFieldTransform(native).GetDisplacementField()), axis=-1)

def starts(n, p=256, stride=225):
    out, i = [], 0
    while True:
        a = stride * i; out.append((a, min(a + p, n)))
        if a + p >= n: return out
        i += 1
groups = {"full 256^3": [], "thin on some axis": []}
for z0, z1 in starts(768):
    for y0, y1 in starts(768):
        for x0, x1 in starts(768):
            b = np.s_[z0 // step:z1 // step, y0 // step:y1 // step, x0 // step:x1 // step]
            thin = min(z1 - z0, y1 - y0, x1 - x0) < 256
            groups["thin on some axis" if thin else "full 256^3"].append(
                (mutual_information(fixed[b], start[b]), mutual_information(fixed[b], after[b]), np.median(u[b]), u[b].max()))
for name, rows in groups.items():
    r = np.array(rows)
    print(f"{name:18s} {len(r):2d} tiles   MI {r[:,0].mean():.3f} -> {r[:,1].mean():.3f}   worse in {(r[:,1] < r[:,0]).sum():2d}   |u| median {np.median(r[:,2]):.2f}  max {r[:,3].max():.2f} mm")
