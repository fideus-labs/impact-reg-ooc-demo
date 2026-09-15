"""How far each run's field moves the image inside the pair's fixed mask: median, 95th percentile and max, in mm.

    python scripts/field_size.py pair_flash linc_flash_bend100 linc_flash_mind
"""
import sys
from pathlib import Path

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import linc

group = zarr.open_group(str(ROOT / "data" / "linc" / sys.argv[1] / "FixedMask.ome.zarr"), mode="r")
mask = np.asarray(group[dict(group.attrs)["multiscales"][0]["datasets"][0]["path"]][0, ::2, ::2, ::2]) > 0
for run in sys.argv[2:]:
    path = ROOT / "out" / run / "P000" / "Transform.h5"
    if not path.exists():
        print(f"{run:26s} no field")
        continue
    length = np.linalg.norm(linc.read_field(path, (0, 0, 0), linc.field_header(path)[0], 2), axis=-1)
    inside = length[mask[: length.shape[0], : length.shape[1], : length.shape[2]]]
    print(f"{run:26s} median {np.median(inside):.2f} mm, 95 % {np.percentile(inside, 95):.2f} mm, max {inside.max():.2f} mm")
