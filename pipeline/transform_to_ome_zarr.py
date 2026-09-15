"""A registration's transform as an OME-Zarr store with the NGFF RFC-5 `displacements` transformation.

    python pipeline/transform_to_ome_zarr.py out/<subject>/<preset>/P000/Transform.h5 out/<subject>/pair/Fixed.ome.zarr

The presets write their displacement field as an ITK transform file (Transform.h5). This writes the same field, in
mm on the fixed grid, as Transform.ome.zarr beside it through KonfAI's writer with displacement_field=True: the
component axis is typed `displacement` and a `displacements` coordinate transformation maps `physical` onto itself
through the array. That is NGFF 0.6rc0 (RFC-5) in Zarr v3, the form docs/apex_results.md describes. A transform that
is not a displacement field (an affine, a B-spline) is sampled on the fixed image's grid first. The store is read back
and checked against the field.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from konfai.utils.dataset import image_to_data
from konfai.utils.ITK import read_displacement_field
from konfai.utils.ome_zarr import write_ome_zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_pair import read_ome_zarr  # noqa: E402


def field_of(transform: Path, fixed: Path | None) -> sitk.Image:
    t = sitk.ReadTransform(str(transform))
    if t.GetName() == "DisplacementFieldTransform":
        return sitk.DisplacementFieldTransform(t).GetDisplacementField()
    if fixed is None:
        raise SystemExit(f"{transform} is a {t.GetName()}: give the fixed image so it can be sampled on its grid")
    ref = read_ome_zarr(fixed) if fixed.name.endswith(".zarr") else sitk.ReadImage(str(fixed))
    return sitk.TransformToDisplacementField(t, sitk.sitkVectorFloat64, ref.GetSize(), ref.GetOrigin(), ref.GetSpacing(), ref.GetDirection())


def store(field: sitk.Image, path: Path) -> tuple[str, float]:
    """Write the field, read it back; return the NGFF version written and the largest component difference (mm)."""
    if not np.allclose(np.array(field.GetDirection()).reshape(3, 3), np.eye(3)):
        raise SystemExit(f"{path}: the field's grid is not axis-aligned; RFC-5 places an array by scale and translation only")
    data, attributes = image_to_data(field)
    if path.exists():
        shutil.rmtree(path)
    write_ome_zarr(path, data, spacing=attributes.get_np_array("Spacing"), origin=attributes.get_np_array("Origin"),
                   attributes=dict(attributes), displacement_field=True)
    ome = json.loads((path / "zarr.json").read_text())["attributes"]["ome"]
    kinds = [t.get("type") for t in ome["multiscales"][0].get("coordinateTransformations", [])]
    if "displacements" not in kinds:
        raise SystemExit(f"{path}: no RFC-5 displacements transformation declared ({kinds})")
    back = read_displacement_field(path)
    for get in ("GetSize", "GetSpacing", "GetOrigin", "GetDirection"):
        assert np.allclose(getattr(back, get)(), getattr(field, get)()), (get, getattr(back, get)(), getattr(field, get)())
    return ome["version"], float(np.abs(sitk.GetArrayViewFromImage(back) - sitk.GetArrayViewFromImage(field)).max())


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit(__doc__)
    transform = Path(sys.argv[1])
    fixed = Path(sys.argv[2]) if len(sys.argv) == 3 else None
    out = transform.with_name("Transform.ome.zarr")
    version, worst = store(field_of(transform, fixed), out)
    mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"{out}: NGFF {version} with RFC-5 displacements, {mb:.0f} MB; read back, max |diff| {worst:.1e} mm")
