"""Does the evaluation read FireANTs' field the way it reads elastix's? Each run's field, converted, must reproduce its own Moved image."""
import sys
from pathlib import Path
ROOT = Path("/home/valentin/Documents/ImpactReg_OOC_Demo")
sys.path.insert(0, str(ROOT))
import numpy as np
import SimpleITK as sitk
import zarr
import linc
from stage1_linc import mutual_information
from stage4_native_linc import written_window

pair = ROOT / "data/linc/pair_flash"
fixed, moving = written_window(pair / "Fixed.ome.zarr"), written_window(pair / "Moving.ome.zarr")
f, start = sitk.GetArrayFromImage(fixed), sitk.GetArrayFromImage(moving)
g = zarr.open_group(str(pair / "FixedMask.ome.zarr"), mode="r")
mask = np.asarray(g[dict(g.attrs)["multiscales"][0]["datasets"][0]["path"]][0]) > 0
print(f"start: MI {mutual_information(f, start):.4f}")
for name in ("linc_flash_mask", "linc_flash_fireants"):
    run = ROOT / "out" / name / "P000"
    field = linc.field_in_store_frame(run / "Transform.h5")
    mine = sitk.GetArrayFromImage(sitk.Resample(moving, fixed, field, sitk.sitkLinear, 0.0))
    g = zarr.open_group(str(run / "Moved.ome.zarr"), mode="r")
    theirs = np.asarray(g[dict(g.attrs)["multiscales"][0]["datasets"][0]["path"]][0])
    print(f"{name:22s} run's Moved: MI {mutual_information(f, theirs):.4f}   field as read: MI {mutual_information(f, mine):.4f}   "
          f"correlation of the two inside the mask {np.corrcoef(mine[mask], theirs[mask])[0, 1]:.4f}")
