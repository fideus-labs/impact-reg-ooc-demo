"""Is the native run's lower MI a worse alignment, or a scale effect? MI at several block sizes, the field, per tile."""
import sys
from pathlib import Path
import numpy as np
import zarr
ROOT = Path("/home/valentin/Documents/ImpactReg_OOC_Demo")
sys.path.insert(0, str(ROOT))
from stage1_linc import mutual_information
from linc import field_header, read_field

def arr(p):
    g = zarr.open_group(str(p), mode="r"); return np.asarray(g[dict(g.attrs)["multiscales"][0]["datasets"][0]["path"]][0])
def blocks(v, k):
    n = (np.array(v.shape) // k) * k
    v = v[:n[0], :n[1], :n[2]]
    return v.reshape(n[0] // k, k, n[1] // k, k, n[2] // k, k).mean((1, 3, 5))

run = ROOT / "out/linc_native/P000"
f, s, t = arr(ROOT / "data/linc/pair0/Fixed.ome.zarr"), arr(ROOT / "data/linc/pair0/Moving.ome.zarr"), arr(run / "Moved.ome.zarr")
print("MI start -> after the native run, block-averaged:")
for k in (4, 8, 16):
    fb, sb, tb = blocks(f, k), blocks(s, k), blocks(t, k)
    print(f"  {k * 20.07:5.0f} um blocks ({fb.shape[0]}^3): {mutual_information(fb, sb):.4f} -> {mutual_information(fb, tb):.4f}")
print(f"  moving intensity mean/std: start {s.mean():.3f}/{s.std():.3f}   after {t.mean():.3f}/{t.std():.3f}")

size = field_header(run / "Transform.h5")[0]
mag = np.stack([np.linalg.norm(read_field(run / "Transform.h5", (0, 0, z), (size[0], size[1], z + 1))[0, ::8, ::8], axis=-1)
                for z in range(0, size[2], 16)])
print(f"native field |u| (mm): median {np.median(mag):.3f}  p90 {np.percentile(mag, 90):.3f}  p99 {np.percentile(mag, 99):.3f}  max {mag.max():.3f}")

def starts(n, p=256, stride=225):
    out, i = [], 0
    while True:
        a = stride * i; out.append((a, min(a + p, n)))
        if a + p >= n: return out
        i += 1
rows = []
for z0, z1 in starts(768):
    for y0, y1 in starts(768):
        for x0, x1 in starts(768):
            b = np.s_[z0:z1:4, y0:y1:4, x0:x1:4]
            rows.append((mutual_information(f[b], s[b]), mutual_information(f[b], t[b])))
rows = np.array(rows)
print(f"{len(rows)} tiles, MI every 4th voxel: mean {rows[:,0].mean():.3f} -> {rows[:,1].mean():.3f}, worse in {(rows[:,1] < rows[:,0]).sum()}")
