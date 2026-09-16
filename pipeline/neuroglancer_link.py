"""One subject's stores on the bucket in Neuroglancer: the retardance under the FA after the affine and after each engine.

    python pipeline/neuroglancer_link.py subject_v           the link                     (pixi run neuroglancer subject_v)
    python pipeline/neuroglancer_link.py subject_v --json    the viewer state the link encodes

Six panels on one linked view. Each holds the PS-OCT retardance at its own resolution
(data/ome-zarr/<subject>/Ret_slide_deck.ome.zarr) in grey. The first holds it alone, with the 0.4 mm fixed image the
engines saw hidden behind it (out/<subject>/pair/Fixed.ome.zarr: click its tab). The others add the FA in orange, the
brighter the higher: after the stage-1 affine (out/<subject>/pair/Moving.ome.zarr), then after each preset of the
README's table (out/<subject>/<preset>/P000/Moved.ome.zarr). The grid's centre and every store's spacing are read
from the bucket. The displacement fields are not shown: Neuroglancer reads neither float64 arrays nor NGFF 0.6rc0.
"""
import argparse
import json
import urllib.error
import urllib.request
from urllib.parse import quote

BUCKET = "https://impact-reg-ooc-demo.s3.filebase.io"
VIEWER = "https://neuroglancer-demo.appspot.com/"
PRESETS = [("FA elastix", "APEX_FA_RET_BSPLINE"), ("FA FireANTs CC", "APEX_FIREANTS_SYN_CC"),
           ("FA FireANTs IMPACT", "APEX_FIREANTS_TS"), ("FA ConvexAdam", "APEX_CONVEXADAM_MIND")]


def grey(window: tuple[float, float]) -> str:
    """Grey levels over the store's own display window."""
    return f"#uicontrol invlerp normalized(range=[{window[0]:g}, {window[1]:g}])\nvoid main() {{ emitGrayscale(normalized()); }}"


def orange(window: tuple[float, float]) -> str:
    """Orange, as opaque as the value is high over the window: the FA's tracts, over the retardance."""
    return (f"#uicontrol invlerp normalized(range=[{window[0]:g}, {window[1]:g}])\n"
            "void main() { float v = normalized(); emitRGBA(vec4(1.0, 0.6, 0.1, v)); }")


NORMALISED = (0.0, 0.8)  # the pair's stores are normalised to [0, 1] by pipeline/prepare_pair.py and carry no window


def fetch(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def multiscale(bucket: str, key: str) -> tuple[dict, list[int], int, tuple[float, float] | None]:
    """The store's first multiscale, its level-0 array's shape, its Zarr version (NGFF 0.5 in Zarr v3 or 0.4 in v2)
    and the display window of its omero metadata, when it has one."""
    try:
        attrs = fetch(f"{bucket}/{key}/zarr.json")
        ome, zarr = attrs["attributes"]["ome"], 3
        meta = ome["multiscales"][0]
        shape = attrs["consolidated_metadata"]["metadata"][meta["datasets"][0]["path"]]["shape"]
    except urllib.error.HTTPError:
        ome, zarr = fetch(f"{bucket}/{key}/.zattrs"), 2
        meta = ome["multiscales"][0]
        shape = fetch(f"{bucket}/{key}/{meta['datasets'][0]['path']}/.zarray")["shape"]
    window = None
    if "omero" in ome:
        w = ome["omero"]["channels"][0]["window"]
        window = (float(w["start"]), float(w["end"]))
    return meta, shape, zarr, window


def dimensions(meta: dict) -> dict:
    """The store's axes as Neuroglancer dimensions: the spatial ones in metres from their mm scale, a channel axis as
    a local dimension (its name with a prime, as Neuroglancer reads an OME channel axis). The stores carry no unit, and
    Neuroglancer compares scales as numbers, so the same dimensions are given as the source's input and as its output:
    then voxel maps to voxel, and the scale bar reads in mm."""
    scale = next(t["scale"] for t in meta["datasets"][0]["coordinateTransformations"] if t["type"] == "scale")
    out = {}
    for axis, s in zip(meta["axes"], scale):
        if axis["type"] == "channel":
            out[axis["name"] + "'"] = [1, ""]
        else:
            if axis.get("unit", "millimeter") != "millimeter":
                raise ValueError(f"axis {axis['name']} in {axis['unit']}, expected millimetres")
            out[axis["name"]] = [s * 1e-3, "m"]
    return out


def state(subject: str, bucket: str) -> dict:
    raw_key = f"data/ome-zarr/{subject}/Ret_slide_deck.ome.zarr"
    fixed_key = f"out/{subject}/pair/Fixed.ome.zarr"
    raw, _, raw_zarr, raw_window = multiscale(bucket, raw_key)
    fixed, shape, pair_zarr, _ = multiscale(bucket, fixed_key)
    fixed_dims = dimensions(fixed)
    fa_keys = [("FA affine", f"out/{subject}/pair/Moving.ome.zarr")]
    fa_keys += [(name, f"out/{subject}/{preset}/P000/Moved.ome.zarr") for name, preset in PRESETS]

    def layer(name: str, key: str, zarr: int, dims: dict, shader: str, visible: bool = True) -> dict:
        out = {"type": "image", "name": name, "shader": shader,
               "source": {"url": f"zarr{zarr}://{bucket}/{key}",
                          "transform": {"inputDimensions": dims, "outputDimensions": dims}}}
        if not visible:
            out["visible"] = False
        return out

    layers = [layer("retardance", raw_key, raw_zarr, dimensions(raw), grey(raw_window or NORMALISED)),
              layer("fixed 0.4 mm", fixed_key, pair_zarr, fixed_dims, grey(NORMALISED), visible=False)]
    layers += [layer(name, key, pair_zarr, fixed_dims, orange(NORMALISED)) for name, key in fa_keys]

    def panel(*names: str) -> dict:
        return {"type": "viewer", "layers": list(names), "layout": "xy"}

    spatial = [a["name"] for a in fixed["axes"] if a["type"] != "channel"]
    size = dict(zip(spatial, [n for a, n in zip(fixed["axes"], shape) if a["type"] != "channel"]))
    return {
        "dimensions": {a: fixed_dims[a] for a in ("x", "y", "z")},
        "position": [size[a] / 2 for a in ("x", "y", "z")],
        "crossSectionScale": 0.5,
        "crossSectionBackgroundColor": "#000000",
        "layers": layers,
        "layout": {"type": "column", "children": [
            {"type": "row", "children": [panel("retardance", "fixed 0.4 mm"),
                                         panel("retardance", "FA affine"), panel("retardance", "FA elastix")]},
            {"type": "row", "children": [panel("retardance", "FA FireANTs CC"),
                                         panel("retardance", "FA FireANTs IMPACT"), panel("retardance", "FA ConvexAdam")]},
        ]},
    }


def link(viewer: str, state_: dict) -> str:
    return viewer + "#!" + quote(json.dumps(state_, separators=(",", ":")), safe=":/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("subject", nargs="?", default="subject_v")
    parser.add_argument("--json", action="store_true", help="print the viewer state instead of the link")
    parser.add_argument("--bucket", default=BUCKET, help=f"the stores' HTTPS root (default: {BUCKET})")
    parser.add_argument("--viewer", default=VIEWER, help=f"the Neuroglancer instance (default: {VIEWER})")
    args = parser.parse_args()
    s = state(args.subject, args.bucket.rstrip("/"))
    print(json.dumps(s, indent=2) if args.json else link(args.viewer, s))


if __name__ == "__main__":
    main()
