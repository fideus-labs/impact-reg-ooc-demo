"""A native field read every 4th voxel must move points as the full read does, to well under the MRI's voxel."""
import sys
sys.path.insert(0, "/home/valentin/Documents/ImpactReg_OOC_Demo")
from pathlib import Path
import numpy as np
import linc
from stage4_native_linc import written_window

path = Path("/home/valentin/Documents/ImpactReg_OOC_Demo/out/linc_native/P000/Transform.h5")
window = written_window(Path("/home/valentin/Documents/ImpactReg_OOC_Demo/data/linc/pair0/Fixed.ome.zarr"))
sub = window[320:448, 320:448, 320:448]  # 2.6 mm inside the window, across tile boundaries at 225/450
full, stepped = linc.field_in_store_frame(path, sub, margin_mm=0.2), linc.field_in_store_frame(path, sub, margin_mm=0.2, step=4)
rng = np.random.default_rng(1)
size = np.array(sub.GetSize())
points = [sub.TransformContinuousIndexToPhysicalPoint((rng.random(3) * (size - 1)).tolist()) for _ in range(5000)]
err = np.linalg.norm(np.array([full.TransformPoint(p) for p in points]) - np.array([stepped.TransformPoint(p) for p in points]), axis=1)
print(f"step 4 against step 1: max {err.max() * 1000:.2f} um, p99 {np.percentile(err, 99) * 1000:.2f} um")
assert err.max() < 0.02, err.max()
print("ok")
