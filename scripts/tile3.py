"""One tissue-rich 256^3 tile of the level-3 pair, to test the deformable's configuration on its own.

    python tile3.py make tissue 320 384 352
    python tile3.py score tissue 320 384 352 out/tile3_tissue_baseline ...
"""
import sys
from pathlib import Path

import numpy as np
import zarr
from konfai.utils.ome_zarr import write_ome_zarr
from scipy.ndimage import binary_dilation, binary_erosion

ROOT = Path("/home/valentin/Documents/ImpactReg_OOC_Demo")
sys.path.insert(0, str(ROOT))
from stage1_linc import mutual_information  # noqa: E402

NAME, Z, Y, X = sys.argv[2], *map(int, sys.argv[3:6])  # e.g. tissue 320 384 352, edge 225 675 225
TILE, N = ROOT / "data" / "linc" / f"tile3_{NAME}", 256
SPACING = 0.16056


def arr(path: Path) -> np.ndarray:
    group = zarr.open_group(str(path), mode="r")
    return group[dict(group.attrs)["multiscales"][0]["datasets"][0]["path"]]


def crop(path: Path) -> np.ndarray:
    return np.asarray(arr(path)[0, Z : Z + N, Y : Y + N, X : X + N])


def scores(fixed: np.ndarray, moving: np.ndarray, mask: np.ndarray) -> str:
    return f"MI tile {mutual_information(fixed, moving):.4f}   MI in tissue {mutual_information(fixed[mask], moving[mask]):.4f}"


def make() -> None:
    TILE.mkdir(parents=True, exist_ok=True)
    fixed, moving = crop(ROOT / "data/linc/pair3/Fixed.ome.zarr"), crop(ROOT / "data/linc/pair3/Moving.ome.zarr")
    mask = binary_erosion(moving > 0.02, np.ones((3, 3, 3)), iterations=3)
    origin = (X * SPACING, Y * SPACING, Z * SPACING)
    for name, volume in (("Fixed", fixed), ("Moving", moving)):
        write_ome_zarr(TILE / f"{name}.ome.zarr", volume[None], spacing=(SPACING,) * 3, origin=origin, chunks=(1, 64, 128, 128))
    np.save(TILE / "mask.npy", mask)
    # the registration masks: the MRI's support widened by ~1 mm, and a moving mask that restricts nothing
    support = binary_dilation(moving > 0.02, np.ones((3, 3, 3)), iterations=6)
    write_ome_zarr(TILE / "FixedMask.ome.zarr", support.astype(np.uint8)[None], spacing=(SPACING,) * 3, origin=origin, chunks=(1, 64, 128, 128))
    write_ome_zarr(TILE / "MovingMask.ome.zarr", np.ones_like(support, np.uint8)[None], spacing=(SPACING,) * 3, origin=origin, chunks=(1, 64, 128, 128))
    print(f"tile z{Z} y{Y} x{X} {N}^3, mask covers {mask.mean():.3f}")
    print(f"  start (stages 1-2)           {scores(fixed, moving, mask)}")
    print(f"  what the tiled level-3 run gave here  {scores(fixed, crop(ROOT / 'out/linc_tiled3/P000/Moved.ome.zarr'), mask)}")


def score(runs: list[str]) -> None:
    fixed, mask = np.asarray(arr(TILE / "Fixed.ome.zarr")[0]), np.load(TILE / "mask.npy")
    print(f"  {'start':28s} {scores(fixed, np.asarray(arr(TILE / 'Moving.ome.zarr')[0]), mask)}")
    start = np.asarray(arr(TILE / "Moving.ome.zarr")[0])
    print(f"  {'':28s} MRI support {(start > 0.02).mean():.3f}")
    from linc import field_header, read_field

    for run in runs:
        moved = Path(run) / "P000" / "Moved.ome.zarr"
        if not moved.exists():
            print(f"  {Path(run).name:28s} no output")
            continue
        volume = np.asarray(arr(moved)[0])
        field = Path(run) / "P000" / "Transform.h5"
        largest = np.linalg.norm(read_field(field, (0, 0, 0), field_header(field)[0]), axis=-1)
        print(f"  {Path(run).name:28s} {scores(fixed, volume, mask)}   MRI support {(volume > 0.02).mean():.3f}   "
              f"displacement median {np.median(largest):.2f} max {largest.max():.2f} mm")


if __name__ == "__main__":
    make() if sys.argv[1] == "make" else score(sys.argv[6:])
