"""A registration's transform as an OME-Zarr store with the OME-Zarr RFC-5 `displacements` transformation.

    python pipeline/transform_to_ome_zarr.py out/<subject>/<preset>/P000/Transform.h5 out/<subject>/pair/Fixed.ome.zarr
    python pipeline/transform_to_ome_zarr.py --stream TRANSFORM.h5 STORE   a displacement field too large for memory
                                                                          (the in-memory path peaks near 6x the file)
    python pipeline/transform_to_ome_zarr.py --selftest                    round trips of an anisotropic, off-origin
                                                                          field: axis-aligned, flipped, and streamed

The presets write their displacement field as an ITK transform file (Transform.h5). This writes the same field, in
mm on the fixed grid, as Transform.ome.zarr beside it through KonfAI's writer with displacement_field=True: the
component axis is typed `displacement` and a `displacements` coordinate transformation maps `physical` onto itself
through the array. That is NGFF 0.6rc0 (OME-Zarr RFC-5) in Zarr v3, the form docs/apex_results.md describes. A
transform that is not a displacement field (an affine, a B-spline) is sampled on the fixed image's grid first. The
store is read back and checked against the field.
"""
import itertools
import json
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import SimpleITK as sitk
import zarr
from konfai.utils.dataset import Attribute, image_to_data
from konfai.utils.ITK import read_displacement_field
from konfai.utils.ome_zarr import create_ome_zarr_store, write_ome_zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_pair import read_ome_zarr  # noqa: E402

IDENTITY = sitk.Transform(3, sitk.sitkIdentity)


def field_of(transform: Path, fixed: Path | None) -> sitk.Image:
    t = sitk.ReadTransform(str(transform))
    if t.GetName() == "DisplacementFieldTransform":
        return sitk.DisplacementFieldTransform(t).GetDisplacementField()
    if fixed is None:
        raise SystemExit(f"{transform} is a {t.GetName()}: give the fixed image so it can be sampled on its grid")
    ref = read_ome_zarr(fixed) if fixed.name.endswith(".zarr") else sitk.ReadImage(str(fixed))
    return sitk.TransformToDisplacementField(t, sitk.sitkVectorFloat64, ref.GetSize(), ref.GetOrigin(), ref.GetSpacing(), ref.GetDirection())


def axis_aligned(field: sitk.Image) -> sitk.Image:
    """The same field on an identity-direction grid.

    OME-Zarr RFC-5 places a field's array in space by scale and translation only, so KonfAI declares the
    `displacements` transformation only for an axis-aligned grid; a pair read from `.mha` files can carry a direction
    such as diag(-1, -1, 1). A grid whose direction is a signed permutation holds the same voxels re-indexed:
    nearest-neighbour resampling onto the aligned grid moves no value, which resampling back checks bit for bit. The
    vectors are physical and stay as they are. A pair written by `prepare_pair.py` is already axis-aligned.
    """
    D = np.array(field.GetDirection()).reshape(3, 3)
    if np.allclose(D, np.eye(3)):
        return field
    if not (np.allclose(np.abs(D).max(axis=0), 1) and np.allclose(np.abs(D).sum(axis=0), 1)):
        raise SystemExit(f"direction {D.ravel().tolist()} is not a signed permutation: aligning the grid would interpolate the field")
    size, spacing = np.array(field.GetSize()), np.array(field.GetSpacing())
    corners = [field.TransformIndexToPhysicalPoint([int(c * (n - 1)) for c, n in zip(corner, size)]) for corner in itertools.product((0, 1), repeat=3)]
    physical_axis = np.abs(D).argmax(axis=0)  # the physical axis each index axis runs along
    new_size, new_spacing = np.empty(3, dtype=int), np.empty(3)
    new_size[physical_axis], new_spacing[physical_axis] = size, spacing
    aligned = sitk.Resample(field, new_size.tolist(), IDENTITY, sitk.sitkNearestNeighbor, np.min(corners, axis=0).tolist(),
                            new_spacing.tolist(), np.eye(3).ravel().tolist(), 0.0, field.GetPixelID())
    back = sitk.Resample(aligned, field, IDENTITY, sitk.sitkNearestNeighbor)
    assert np.array_equal(sitk.GetArrayViewFromImage(back), sitk.GetArrayViewFromImage(field)), "re-indexing moved values"
    return aligned


def declared_version(path: Path) -> str:
    """The NGFF version the store declares; it must declare the `displacements` transformation."""
    ome = json.loads((path / "zarr.json").read_text())["attributes"]["ome"]
    kinds = [t.get("type") for t in ome["multiscales"][0].get("coordinateTransformations", [])]
    if "displacements" not in kinds:
        raise SystemExit(f"{path}: no OME-Zarr RFC-5 displacements transformation declared ({kinds})")
    return ome["version"]


def store(field: sitk.Image, path: Path) -> tuple[str, float]:
    """Write the field, read it back; return the NGFF version written and the largest component difference (mm)."""
    field = axis_aligned(field)
    data, attributes = image_to_data(field)
    if path.exists():
        shutil.rmtree(path)
    write_ome_zarr(path, data, spacing=attributes.get_np_array("Spacing"), origin=attributes.get_np_array("Origin"),
                   attributes=dict(attributes), displacement_field=True)
    version = declared_version(path)
    back = read_displacement_field(path)
    for get in ("GetSize", "GetSpacing", "GetOrigin", "GetDirection"):
        assert np.allclose(getattr(back, get)(), getattr(field, get)()), (get, getattr(back, get)(), getattr(field, get)())
    return version, float(np.abs(sitk.GetArrayViewFromImage(back) - sitk.GetArrayViewFromImage(field)).max())


def stream(transform: Path, path: Path, checked_slices: int = 3) -> tuple[str, float]:
    """An ITK DisplacementFieldTransform .h5 written slab by slab, one z-chunk of the store at a time.

    The .h5 holds the field flat: every voxel's (dx, dy, dz), x running fastest, after the fixed parameters size,
    origin, spacing, direction. Slabs follow the store's own z-chunks so no write straddles one. The check reads whole
    z-planes back from the store (components in ITK order) against the file; the full round trip through KonfAI's
    reader is what the selftest runs on a small field.
    """
    with h5py.File(transform, "r") as f:
        g = f["TransformGroup/0"]
        kind = g["TransformType"][0]
        kind = kind.decode() if isinstance(kind, bytes) else str(kind)
        if not (kind.startswith("DisplacementFieldTransform") and kind.endswith("_3_3")):
            raise SystemExit(f"{transform}: {kind} is not a 3-D displacement field; the in-memory path samples other transforms")
        fixed = np.array(g["TransformFixedParameters"])
        nx, ny, nz = fixed[0:3].astype(int)
        attributes = Attribute()
        attributes["Origin"], attributes["Spacing"], attributes["Direction"] = fixed[3:6], fixed[6:9], fixed[9:18]
        if not np.allclose(fixed[9:18], np.eye(3).ravel()):
            raise SystemExit(f"{transform}: the streaming path writes axis-aligned grids only")
        flat = g["TransformParameters"]
        assert flat.shape == (nx * ny * nz * 3,), (flat.shape, nx, ny, nz)
        if path.exists():
            shutil.rmtree(path)
        writer = create_ome_zarr_store(path, (3, nz, ny, nx), flat.dtype, spacing=attributes.get_np_array("Spacing"),
                                       origin=attributes.get_np_array("Origin"), attributes=dict(attributes), displacement_field=True)
        plane, step = nx * ny * 3, writer.chunks[1]
        for z0 in range(0, nz, step):
            z1 = min(nz, z0 + step)
            writer[:, z0:z1] = np.moveaxis(flat[z0 * plane:z1 * plane].reshape(z1 - z0, ny, nx, 3), -1, 0)
        version = declared_version(path)
        worst = 0.0
        for z in np.random.default_rng(0).choice(nz, size=min(checked_slices, nz), replace=False):
            expected = np.moveaxis(flat[z * plane:(z + 1) * plane].reshape(ny, nx, 3), -1, 0)
            worst = max(worst, float(np.abs(writer[:, int(z)] - expected).max()))
    return version, worst


def selftest() -> None:
    # Anisotropic spacing, an origin off zero and a field that differs per voxel and per component: an axis or
    # component swap anywhere in the round trip changes the numbers. A pair is isotropic, where such a swap would go
    # unseen; a flipped direction is the second case, the streaming path the third.
    rng = np.random.default_rng(0)
    for direction in ((1, 0, 0, 0, 1, 0, 0, 0, 1), (-1, 0, 0, 0, -1, 0, 0, 0, 1), "stream"):
        field = sitk.GetImageFromArray(rng.normal(size=(6, 5, 4, 3)), isVector=True)  # numpy z, y, x, component
        field.SetSpacing((0.3, 0.7, 1.9))
        field.SetOrigin((-2.0, 5.0, 11.0))
        field.SetDirection(direction if direction != "stream" else (1, 0, 0, 0, 1, 0, 0, 0, 1))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selftest.ome.zarr"
            if direction == "stream":
                h5 = Path(tmp) / "field.h5"
                sitk.WriteTransform(sitk.DisplacementFieldTransform(sitk.Image(field)), str(h5))
                assert stream(h5, path, checked_slices=6)[1] == 0.0
                back = read_displacement_field(path)
                assert np.array_equal(sitk.GetArrayViewFromImage(back), sitk.GetArrayViewFromImage(field))
                for get in ("GetSpacing", "GetOrigin", "GetDirection"):
                    assert np.allclose(getattr(back, get)(), getattr(field, get)()), get
            else:
                assert store(field, path)[1] == 0.0
                back = read_displacement_field(path)
            probe = field.TransformIndexToPhysicalPoint((1, 3, 2))  # the same physical point must carry the same vector
            assert np.allclose(back.GetPixel(back.TransformPhysicalPointToIndex(probe)), field.GetPixel(1, 3, 2))
            ms = json.loads((path / "zarr.json").read_text())["attributes"]["ome"]["multiscales"][0]
            physical = [a["name"] for a in next(c for c in ms["coordinateSystems"] if c["name"] == "physical")["axes"]]
            index = back.TransformPhysicalPointToIndex(probe)
            raw = zarr.open_array(str(path / ms["datasets"][0]["path"]), mode="r")[:, index[2], index[1], index[0]]  # c, z, y, x on disk
            expected = np.array(field.GetPixel(1, 3, 2))[[{"x": 0, "y": 1, "z": 2}[a] for a in physical]]
            assert np.allclose(raw, expected), (raw, expected, physical)
        print(f"selftest {direction}: exact round trip, OME-Zarr RFC-5 displacements declared, components on disk in the physical axis order {' '.join(physical)}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--selftest"]:
        selftest()
    elif len(args) == 3 and args[0] == "--stream":
        out = Path(args[2])
        version, worst = stream(Path(args[1]), out)
        print(f"{out}: NGFF {version} with OME-Zarr RFC-5 displacements, streamed; max |diff| on 3 whole z-planes {worst:.1e} mm")
    elif len(args) in (1, 2) and not args[0].startswith("--"):
        transform = Path(args[0])
        out = transform.with_name("Transform.ome.zarr")
        version, worst = store(field_of(transform, Path(args[1]) if len(args) == 2 else None), out)
        mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
        print(f"{out}: NGFF {version} with OME-Zarr RFC-5 displacements, {mb:.0f} MB; read back, max |diff| {worst:.1e} mm")
    else:
        raise SystemExit(__doc__)
