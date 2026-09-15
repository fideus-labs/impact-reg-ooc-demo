"""Zoom 1 of the hemisphere (4.257 um, 602 GiB) on the overview (20.07 um, 908 GiB): X-ray against X-ray.

The consortium placed every zoom by hand, one affine per acquisition, in Neuroglancer. Its matrix acts on
physical coordinates, its translation in units of 10.072 um; read the other way, as voxel counts, the zoom
comes out 2.4 times too large and does not correlate with the overview at all. This script starts from that
hand placement and refines it: a rigid correction on levels that fit in memory, then a deformable in tiles on
a pair the stores never have to hold whole.

Everything here stays in the stores' own frame: x, y, z in mm, identity direction, OME translations kept.
Every box read from S3 is kept under data/linc/zoom1/cache, so a second run reads nothing.

    python zoom_linc.py rigid                                  # the correction, overview level 2 against zoom level 4
    python zoom_linc.py pair --level 1 --size 1024 1024 896    # the zoom's field of view at 40 um, for the tiles
    python zoom_linc.py check --field out/zoom1_level1_bend100/P000/Transform.h5   # 20 um windows no run has seen
"""

import argparse
import functools
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import zarr
from scipy import fft

import linc

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "linc" / "zoom1"
OVERVIEW = "6f11427f-ef86-42b8-9800-d3e09265c965"
ZOOM = "d483e140-ccd7-4bcf-8a69-a10b7087381c"
# the consortium's placement of zoom 1, from the dandiset's Neuroglancer scene: zoom mm -> overview, in 10.072 um
PLACEMENT = np.array([[0.999445745, 0.031534287, -0.011279779, 3817.405521157],
                      [-0.031774949, 0.999263261, -0.021834133, 4528.589463744],
                      [0.010582873, 0.022180297, 0.999704696, 5999.076104469]])
SCENE_UNIT = 0.010072


@functools.cache
def group(key: str):
    return zarr.open_group(f"s3://dandiarchive/zarr/{key}", mode="r", storage_options={"anon": True})


def geometry(key: str, level: str):
    """Voxel size and translation of one level in mm, and its array."""
    attrs = dict(group(key).attrs)
    multiscales = attrs.get("multiscales") or attrs["ome"]["multiscales"]
    dataset = next(d for d in multiscales[0]["datasets"] if d["path"] == level)
    transforms = {t["type"]: t for t in dataset["coordinateTransformations"]}
    return (np.array(transforms["scale"]["scale"]) / 1000, np.array(transforms["translation"]["translation"]) / 1000,
            group(key)[level])


def box(key: str, level: str, lo_mm, hi_mm) -> sitk.Image:
    """The voxels of one level covering [lo, hi] mm: only the chunks the box intersects are fetched, once."""
    spacing, translation, array = geometry(key, level)
    i0 = np.clip(np.floor((np.asarray(lo_mm) - translation) / spacing).astype(int), 0, array.shape)
    i1 = np.clip(np.ceil((np.asarray(hi_mm) - translation) / spacing).astype(int) + 1, 0, array.shape)
    cache = DATA / "cache" / f"{key[:8]}_level{level}_{'_'.join(map(str, i0))}_{'_'.join(map(str, i1))}.npy"
    start = time.perf_counter()
    if cache.exists():
        block = np.load(cache)
    else:
        block = np.asarray(array[i0[0] : i1[0], i0[1] : i1[1], i0[2] : i1[2]])
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, block)
        print(f"  level {level}: box {block.shape} read from S3 in {time.perf_counter() - start:.0f} s "
              f"({block.nbytes / 2**30:.2f} GiB)")
    image = sitk.GetImageFromArray(np.ascontiguousarray(block.transpose(2, 1, 0)).astype(np.float32))
    image.SetSpacing(spacing.tolist())
    image.SetOrigin((translation + i0 * spacing).tolist())
    return image


def bounds(transform: sitk.Transform, lo, hi) -> tuple[np.ndarray, np.ndarray]:
    corners = [transform.TransformPoint([hi[i] if c[i] else lo[i] for i in range(3)]) for c in np.ndindex(2, 2, 2)]
    return np.min(corners, 0), np.max(corners, 0)


def zoom_extent() -> tuple[np.ndarray, np.ndarray]:
    spacing, translation, array = geometry(ZOOM, "0")
    return translation, translation + (np.array(array.shape) - 1) * spacing


def placement(correction: bool = True) -> sitk.CompositeTransform:
    """Overview mm -> zoom mm: the hand placement, after the rigid correction when one has been computed."""
    manual = sitk.AffineTransform(PLACEMENT[:, :3].ravel().tolist(), (PLACEMENT[:, 3] * SCENE_UNIT).tolist())
    chain = sitk.CompositeTransform([manual.GetInverse()])
    if correction and (DATA / "rigid.tfm").exists():
        chain.AddTransform(sitk.ReadTransform(str(DATA / "rigid.tfm")))           # applied first
    return chain


def field_of_view() -> tuple[np.ndarray, np.ndarray]:
    """The box zoom 1 covers on the overview, in mm, by the hand placement."""
    return bounds(placement(correction=False).GetInverse(), *zoom_extent())


def field(path: Path, region: sitk.Image, margin_mm: float = 3.0) -> sitk.Transform:
    """A displacement field run on a pair of this script, over a region only: the pair's frame is the stores'."""
    size, origin, spacing = linc.field_header(path)
    lo_mm = np.array(region.GetOrigin())
    hi_mm = np.array(region.TransformIndexToPhysicalPoint([n - 1 for n in region.GetSize()]))
    lo = np.clip(np.floor((lo_mm - margin_mm - origin) / spacing).astype(int), 0, size)
    hi = np.clip(np.ceil((hi_mm + margin_mm - origin) / spacing).astype(int) + 1, 0, size)
    out = sitk.GetImageFromArray(linc.read_field(path, lo, hi), isVector=True)
    out.SetSpacing(spacing.tolist())
    out.SetOrigin((origin + lo * spacing).tolist())
    return sitk.DisplacementFieldTransform(out)


def zoom_mask(zoom: sitk.Image, grid: sitk.Image, transform: sitk.Transform, inset_mm: float) -> sitk.Image:
    """Where the zoom has data on a grid of the overview, ``inset_mm`` in from its faces."""
    inset = round(inset_mm / zoom.GetSpacing()[0])
    inside = np.zeros(sitk.GetArrayViewFromImage(zoom).shape, np.uint8)
    inside[inset:-inset, inset:-inset, inset:-inset] = 1
    inside = sitk.GetImageFromArray(inside)
    inside.CopyInformation(zoom)
    return sitk.Resample(inside, grid, transform, sitk.sitkNearestNeighbor, 0)


def rigid(_args) -> None:
    """The rigid correction, on overview level 2 (80 um) against zoom level 4 (68 um), 2 mm inside the zoom."""
    DATA.mkdir(parents=True, exist_ok=True)
    manual = placement(correction=False)
    fixed = box(OVERVIEW, "2", *field_of_view())
    moving = box(ZOOM, "4", *zoom_extent())
    mask = zoom_mask(moving, fixed, manual, inset_mm=2.0)

    euler = sitk.Euler3DTransform()
    euler.SetCenter(fixed.TransformContinuousIndexToPhysicalPoint([(n - 1) / 2 for n in fixed.GetSize()]))
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(50)
    reg.SetMetricFixedMask(mask)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentagePerLevel([0.1, 0.02, 0.005], seed=1)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(0.5, 1e-4, 200, relaxationFactor=0.7)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    reg.SetMovingInitialTransform(manual)
    reg.SetInitialTransform(euler, inPlace=True)
    reg.Execute(fixed, moving)
    sitk.WriteTransform(euler, str(DATA / "rigid.tfm"))
    print(f"rigid correction: angles {np.round(np.degrees(euler.GetParameters()[:3]), 3).tolist()} deg, "
          f"translation {np.round(euler.GetParameters()[3:], 3).tolist()} mm -> {DATA / 'rigid.tfm'}")


def scaled(image: sitk.Image) -> np.ndarray:
    array = sitk.GetArrayFromImage(image)
    lo, hi = np.percentile(array[::4, ::4, ::4], [0.5, 99.5])
    return np.clip((array - lo) / (hi - lo), 0, 1).astype(np.float32)


def pair(args) -> None:
    """Overview level L over the zoom's field of view (or a box of it), and the zoom through the placement."""
    from konfai.utils.ome_zarr import write_ome_zarr

    spacing = geometry(OVERVIEW, args.level)[0]
    transform = placement()
    fov_lo, fov_hi = field_of_view()
    centre = np.array(args.centre) if args.centre else (fov_lo + fov_hi) / 2
    half = np.array(args.size) * spacing / 2
    fixed = box(OVERVIEW, args.level, centre - half, centre + half - spacing)
    assert all(n >= m for n, m in zip(fixed.GetSize(), args.size)), f"the box {fixed.GetSize()} leaves the overview"
    fixed = sitk.RegionOfInterest(fixed, list(args.size), [0, 0, 0])

    # the coarsest zoom level no coarser than the overview's, read only where the fixed box lands
    zoom_level = max((str(k) for k in range(8)), key=lambda k: geometry(ZOOM, k)[0][0] * (geometry(ZOOM, k)[0][0] <= spacing[0] + 1e-9))
    zoom_lo, zoom_hi = zoom_extent()
    lo, hi = bounds(transform, fixed.GetOrigin(), fixed.TransformIndexToPhysicalPoint([n - 1 for n in fixed.GetSize()]))
    zoom = box(ZOOM, zoom_level, np.maximum(lo - 1, zoom_lo), np.minimum(hi + 1, zoom_hi))
    moved = sitk.Resample(zoom, fixed, transform, sitk.sitkLinear, 0.0)
    del zoom
    mask = zoom_mask(box(ZOOM, "5", zoom_lo, zoom_hi), fixed, transform, inset_mm=1.0)   # the faces, not the voxels

    out = args.out or ROOT / "data" / "linc" / f"zoom1_pair{args.level}"
    out.mkdir(parents=True, exist_ok=True)
    kw = {"spacing": fixed.GetSpacing(), "origin": fixed.GetOrigin(), "chunks": (1, 64, 256, 256)}
    for name, image in (("Fixed", fixed), ("Moving", moved)):
        array = scaled(image)
        write_ome_zarr(out / f"{name}.ome.zarr", array[None], **kw)
        print(f"{name}.ome.zarr {array.shape} @ {spacing[0] * 1000:.2f} um ({array.nbytes / 2**30:.2f} GiB), zoom level {zoom_level}")
    support = sitk.GetArrayFromImage(mask)
    write_ome_zarr(out / "FixedMask.ome.zarr", support[None], **kw)
    write_ome_zarr(out / "MovingMask.ome.zarr", np.ones_like(support)[None], **kw)
    print(f"fixed mask covers {support.mean():.3f} of the box (the zoom, 1 mm in from its faces)")


def residual_shift(a: np.ndarray, b: np.ndarray, reach: int = 50) -> np.ndarray:
    """The shift, within +-reach voxels, that best lines b up with a: the cross-correlation's peak, (x, y, z) voxels."""
    window = np.einsum("i,j,k->ijk", *(np.hanning(n).astype(np.float32) for n in a.shape))
    cc = fft.irfftn(fft.rfftn((a - a.mean()) * window, workers=-1) * np.conj(fft.rfftn((b - b.mean()) * window, workers=-1)),
                    s=a.shape, workers=-1)
    shifts = [np.fft.fftfreq(n, 1 / n).astype(int) for n in a.shape]
    keep = [np.flatnonzero(np.abs(v) <= reach) for v in shifts]
    sub = cc[np.ix_(*keep)]
    peak = np.unravel_index(np.argmax(sub), sub.shape)
    return np.array([shifts[axis][keep[axis][peak[axis]]] for axis in range(3)])[::-1]


def check(args) -> None:
    """Level 0 of the overview against level 2 of the zoom, in 4 mm windows: correlation and residual shift.

    By default the windows sit on a 2 x 2 x 2 lattice 9 mm either side of the zoom's centre, so a correction
    that helps one side and hurts the other shows.
    """
    if args.centre:
        centres = np.array(args.centre).reshape(-1, 3)
    else:
        middle = np.mean(field_of_view(), axis=0)
        centres = np.array([middle + 9.0 * (2 * np.array(c) - 1) for c in np.ndindex(2, 2, 2)])
    from figures_linc import bands, checkerboard, save

    half = args.window / 2
    labels = ["hand placement", "+ rigid"] + (["+ deformable"] if args.field else [])
    scores = {label: [] for label in labels}
    rows = []
    for n, centre in enumerate(centres):
        window = box(OVERVIEW, "0", centre - half, centre + half)
        a = sitk.GetArrayFromImage(window)
        states = [placement(correction=False), placement()]
        if args.field:
            chain = placement()
            chain.AddTransform(field(args.field, window))                     # applied first
            states.append(chain)
        source = box(ZOOM, "2", *(v + d for v, d in zip(bounds(states[0], centre - half, centre + half), (-1.0, 1.0))))
        panels = []
        for label, transform in zip(labels, states):
            b = sitk.GetArrayFromImage(sitk.Resample(source, window, transform, sitk.sitkLinear, 0.0))
            panels.append(checkerboard(a[a.shape[0] // 2], b[b.shape[0] // 2], 25))       # 0.5 mm squares
            shift = residual_shift(a, b) * window.GetSpacing()[0]
            r = np.corrcoef(a.ravel(), b.ravel())[0, 1]
            scores[label].append((r, np.linalg.norm(shift)))
            print(f"window {n + 1} at {np.round(centre, 1).tolist()} mm, {label}: correlation {r:.3f}, "
                  f"residual shift {np.round(shift, 2).tolist()} mm (|{np.linalg.norm(shift):.2f}|)", flush=True)
        rows.append((f"window {n + 1} · " + " | ".join(labels), panels))
    save(bands(rows, size=20), "Z1_windows.png")
    for label, values in scores.items():
        r, d = np.array(values).T
        print(f"{label}: correlation median {np.median(r):.3f}; residual shift median {np.median(d):.2f} mm, "
              f"max {d.max():.2f} mm, over {len(values)} windows")


def detail(args) -> None:
    """One plane at the zoom's own voxel, 4.257 um: the overview (20 um) and the zoom, for each placement."""
    from figures_linc import bands, checkerboard, grey, save

    centre, half = np.array(args.centre), np.array([args.side / 2, args.side / 2, 0.0])
    voxel = geometry(ZOOM, "0")[0][0]
    plane = sitk.Image([round(args.side / voxel), round(args.side / voxel), 1], sitk.sitkFloat32)
    plane.SetSpacing([voxel] * 3)
    plane.SetOrigin((centre - half).tolist())
    states = [("hand placement", placement(correction=False)), ("+ rigid", placement())]
    if args.field:
        chain = placement()
        chain.AddTransform(field(args.field, plane))                              # applied first
        states.append(("+ deformable", chain))
    overview = box(OVERVIEW, "0", centre - half - 0.2, centre + half + 0.2)
    zoom = box(ZOOM, "0", *(v + d for v, d in zip(bounds(states[0][1], centre - half, centre + half), (-args.margin, args.margin))))
    a = sitk.GetArrayFromImage(sitk.Resample(overview, plane, sitk.Transform(), sitk.sitkLinear, 0.0))[0]
    rows = []
    for label, transform in states:
        b = sitk.GetArrayFromImage(sitk.Resample(zoom, plane, transform, sitk.sitkLinear, 0.0))[0]
        rows.append((f"{label} · overview 20 um | zoom 4.3 um | squares 0.25 mm",
                     [np.dstack([grey(a)] * 3), np.dstack([grey(b)] * 3), checkerboard(a, b, round(0.25 / voxel))]))
    save(bands(rows, size=20), args.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("rigid").set_defaults(run=rigid)
    one = sub.add_parser("pair")
    one.add_argument("--level", default="1")
    one.add_argument("--size", nargs=3, type=int, default=[1024, 1024, 896], metavar=("X", "Y", "Z"),
                     help="voxels; a tiled run wants stride * k + patch on each axis")
    one.add_argument("--centre", nargs=3, type=float, metavar=("X", "Y", "Z"), help="mm; the zoom's centre by default")
    one.add_argument("--out", type=Path)
    one.set_defaults(run=pair)
    two = sub.add_parser("check")
    two.add_argument("--centre", nargs="+", type=float, help="window centres, mm, three numbers each")
    two.add_argument("--window", type=float, default=4.0, help="window side, mm")
    two.add_argument("--field", type=Path)
    two.set_defaults(run=check)
    three = sub.add_parser("detail")
    three.add_argument("--centre", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"), help="mm")
    three.add_argument("--side", type=float, default=1.5, help="mm")
    three.add_argument("--margin", type=float, default=1.0, help="mm of zoom read around the hand placement")
    three.add_argument("--field", type=Path)
    three.add_argument("--name", default="Z2_detail.png")
    three.set_defaults(run=detail)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
