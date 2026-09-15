"""The native field, moved into the store's frame, must reproduce the run's own Moved image inside the window."""
import sys
sys.path.insert(0, "/home/valentin/Documents/ImpactReg_OOC_Demo")
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import zarr
import linc
from stage4_native_linc import written_window

ROOT = Path("/home/valentin/Documents/ImpactReg_OOC_Demo")
start = written_window(ROOT / "data/linc/pair0/Moving.ome.zarr")      # the run's moving image, in the store's frame
grid = written_window(ROOT / "data/linc/pair0/Fixed.ome.zarr")[200:568:4, 200:568:4, 200:568:4]  # interior, every 4th voxel
field = linc.field_in_store_frame(ROOT / "out/linc_native/P000/Transform.h5", grid, step=4)
mine = sitk.GetArrayFromImage(sitk.Resample(start, grid, field, sitk.sitkLinear, 0.0))
g = zarr.open_group(str(ROOT / "out/linc_native/P000/Moved.ome.zarr"), mode="r")
theirs = np.asarray(g[dict(g.attrs)["multiscales"][0]["datasets"][0]["path"]][0, 200:568:4, 200:568:4, 200:568:4])
before = np.asarray(g.store and zarr.open_group(str(ROOT / "data/linc/pair0/Moving.ome.zarr"), mode="r")["0" if "0" in zarr.open_group(str(ROOT / "data/linc/pair0/Moving.ome.zarr"), mode="r") else dict(zarr.open_group(str(ROOT / "data/linc/pair0/Moving.ome.zarr"), mode="r").attrs)["multiscales"][0]["datasets"][0]["path"]][0, 200:568:4, 200:568:4, 200:568:4])
corr = np.corrcoef(mine.ravel(), theirs.ravel())[0, 1]
print(f"interior: mean |mine - run's Moved| {np.abs(mine - theirs).mean():.4f}, correlation {corr:.4f}   "
      f"(start against run's Moved: {np.abs(before - theirs).mean():.4f}, {np.corrcoef(before.ravel(), theirs.ravel())[0, 1]:.4f})")
assert corr > 0.99, corr
print("ok")
