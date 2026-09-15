"""The BRAIN CONNECTS / LINC pair: one human hemisphere, X-ray phase contrast against diffusion MRI.

    XPCT   synchrotron X-ray phase-contrast tomography, 7569 x 7569 x 8514 at 20.07 um
           908 GiB at level 0, eight pyramid levels, OME-Zarr on DANDI's public S3
    dMRI   the b-ratio map derived from ex vivo diffusion MRI, 0.4 mm, 24 MiB

Both are volumes of the same left hemisphere, so unlike a block face against a cleared section the
depth correspondence is real. The contrasts are not: one is X-ray phase, the other the anisotropy of
water diffusion. That is the pair the consortium names as hard.

The dandiset also ships that same map already resampled into XPCT space, which is the consortium's
own alignment. This module reads it too, as the reference to measure against.

    python linc.py fetch --level 4        # one pyramid level of the 908 GiB store, plus the maps
"""

import argparse
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import zarr

DATA = Path(__file__).parent / "data" / "linc"
XPCT_ZARR = "6f11427f-ef86-42b8-9800-d3e09265c965"   # sub-Hb1 lefthemi overview, 908 GiB at level 0
ASSETS = {                                            # dandiset 001278, the small derived maps
    "dmri": "6b28deff-4898-42a0-81fe-094f3e730b62",              # b-ratio mean, native space
    "dmri_reference": "22a8055c-5b6a-4c8e-a878-b77684df27a3",    # the same, in XPCT space
    "fa": "83c2d8f0-1168-43be-a862-2a8757f07a87",                # fractional anisotropy, native
    # the per-structure check: the FLASH T2* of the same hemisphere, and its segmentation twice
    "flash_T2starw": "941cb3b1-aee2-47f5-8f39-84776d07ce37",     # 2.8 GiB, native space
    "dseg_native": "eb56ca92-a0d2-422b-9376-107103ec3dc8",       # on the FLASH
    "dseg_spaceXPCT": "6f126f57-b9a6-4704-a0d3-78854b152124",    # the same, moved into XPCT space by the consortium
}
# The store declares axes, scale and a half-voxel translation, and nothing about anatomy. NIfTI
# carries RAS and SimpleITK reads it as LPS, so the convention has to be adopted by hand here; this
# is the one that puts the consortium's own XPCT-space map on top of the volume.
LPS = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)


def store():
    return zarr.open_group(f"s3://dandiarchive/zarr/{XPCT_ZARR}", mode="r", storage_options={"anon": True})


def levels() -> dict[str, float]:
    """Each pyramid level of the store and its voxel size in mm."""
    group = store()
    datasets = dict(group.attrs)["multiscales"][0]["datasets"]
    return {d["path"]: d["coordinateTransformations"][0]["scale"][0] / 1000.0 for d in datasets}


def as_itk(array: np.ndarray, voxel_mm: float) -> sitk.Image:
    """The store's (x, y, z) array as an ITK image in the adopted LPS frame."""
    image = sitk.GetImageFromArray(np.ascontiguousarray(np.transpose(array, (2, 1, 0)).astype(np.float32)))
    image.SetSpacing((voxel_mm,) * 3)
    image.SetDirection(LPS)
    return image


def field_header(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Size, origin and spacing of a displacement field ITK wrote, in (x, y, z), without reading its buffer."""
    import h5py

    with h5py.File(path, "r") as handle:
        fixed = np.asarray(handle["TransformGroup/0/TransformFixedParameters"])
    assert np.allclose(fixed[9:18], np.eye(3).ravel()), "expected a field written on a pair without direction"
    return fixed[:3].round().astype(int), fixed[3:6], fixed[6:9]


def read_field(path: Path, lo, hi, step: int = 1) -> np.ndarray:
    """The (z, y, x, 3) block [lo, hi) of a displacement field, every ``step``-th voxel, read plane by plane.

    ITK stores the buffer flat, x fastest, three components per voxel. A level-3 field is 23 GB in
    float64; loading it whole, with the copies a resample makes, is what ran the machine out of memory.
    """
    import h5py

    size = field_header(path)[0]
    plane = size[0] * size[1] * 3
    planes = range(lo[2], hi[2], step)
    block = np.empty((len(planes), len(range(lo[1], hi[1], step)), len(range(lo[0], hi[0], step)), 3))
    with h5py.File(path, "r") as handle:
        flat = handle["TransformGroup/0/TransformParameters"]
        for k, z in enumerate(planes):
            plane_z = flat[z * plane : (z + 1) * plane].reshape(size[1], size[0], 3)
            block[k] = plane_z[lo[1] : hi[1] : step, lo[0] : hi[0] : step]
    return block


def field_in_store_frame(
    path: Path, region: sitk.Image | None = None, margin_mm: float = 10.0, step: int = 1
) -> sitk.Transform:
    """A displacement field from a run on an OME-Zarr pair, moved into the adopted LPS frame.

    The pair went to disk without a direction — OME-Zarr has no field for one — so the engine
    registered it in a frame where x and y are not negated about the pair's origin. The field is right
    there and zero everywhere in the store's frame; mirroring its grid about that origin, and its
    vectors, puts it back. The level-3 and level-4 pairs start at 0, where this is a plain mirror; a
    window of level 0 does not.

    With a region, only the box of the field over it is read, widened by a margin that has to cover
    whatever displacement a finer field applies before this one. A ``step`` above 1 keeps every step-th
    voxel: a B-spline field is smooth on the scale of its control grid, so a native field read at 80 um
    moves points as the 20 um one does, for a sixty-fourth of the memory.
    """
    mirror = np.array(LPS).reshape(3, 3)
    size, origin, spacing = field_header(path)
    lo, hi = np.zeros(3, int), size
    if region is not None:
        corners = np.array([region.TransformIndexToPhysicalPoint([c * (n - 1) for c, n in zip(corner, region.GetSize())])
                            for corner in np.ndindex(2, 2, 2)])
        index = ((corners - origin) @ mirror.T) / spacing
        pad = int(np.ceil(margin_mm / spacing.min()))
        lo = np.clip(np.floor(index.min(0)).astype(int) - pad, 0, size)
        hi = np.clip(np.ceil(index.max(0)).astype(int) + pad + 1, 0, size)
    block = read_field(path, lo, hi, step)
    block *= np.diag(mirror)  # the mirror is diagonal: this is vectors @ mirror.T, in place
    print(f"field {path.parent.parent.name}: box {tuple(int(n) for n in hi - lo)} of {tuple(int(n) for n in size)}, "
          f"{block.nbytes / 2**20:.0f} MiB, largest displacement {np.linalg.norm(block, axis=-1).max():.2f} mm")
    out = sitk.GetImageFromArray(block, isVector=True)
    out.SetSpacing(tuple(spacing * step))
    out.SetDirection(tuple(mirror.ravel()))
    out.SetOrigin(tuple(origin + mirror @ (lo * spacing)))
    return sitk.DisplacementFieldTransform(out)


def read_level(level: str) -> sitk.Image:
    voxel = levels()[level]
    start = time.perf_counter()
    array = np.asarray(store()[level])
    print(f"XPCT level {level}: {array.shape} @ {voxel:.4f} mm, "
          f"{array.nbytes / 2**20:.0f} MiB read in {time.perf_counter() - start:.0f} s")
    return as_itk(array, voxel)


def read_window(level: str, centre_mm, size_mm: float) -> sitk.Image:
    """One box of the store at any level: only the chunks it intersects are fetched."""
    voxel = levels()[level]
    shape = store()[level].shape
    half = int(round(size_mm / voxel / 2))
    # the adopted LPS frame negates x and y, so a physical centre maps back to these indices
    centre = [int(round(-centre_mm[0] / voxel)), int(round(-centre_mm[1] / voxel)),
              int(round(centre_mm[2] / voxel))]
    box = [slice(max(c - half, 0), min(c + half, n)) for c, n in zip(centre, shape)]
    start = time.perf_counter()
    array = np.asarray(store()[level][tuple(box)])
    print(f"window {array.shape} at level {level} read in {time.perf_counter() - start:.0f} s "
          f"({array.nbytes / 2**20:.0f} MiB)")
    image = as_itk(array, voxel)
    image.SetOrigin((-box[0].start * voxel, -box[1].start * voxel, box[2].start * voxel))
    return image


def fetch(level: str) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(read_level(level), str(DATA / f"XPCT_level{level}.mha"), useCompression=True)
    import urllib.request

    for name, asset in ASSETS.items():
        path = DATA / f"{name}.nii.gz"
        if path.exists():
            continue
        urllib.request.urlretrieve(f"https://api.dandiarchive.org/api/assets/{asset}/download/", path)
        print(f"{name}: {path.stat().st_size / 2**20:.0f} MiB")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("fetch", help="one pyramid level of the store, plus the derived maps")
    one.add_argument("--level", default="4")
    args = parser.parse_args()
    if args.command == "fetch":
        fetch(args.level)


if __name__ == "__main__":
    main()
