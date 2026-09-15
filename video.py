"""The video: the mechanism, then the real run.

Part 1 is built from the run's own images: the two modalities, the mirror the coarse stage finds,
the zoom to native resolution, the tiles lighting up with the displacement assembling behind them,
and the before/after at cellular scale. Part 2 replays the terminal log of the real run.

    python video.py --out video/out_of_core.mp4
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import zarr
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
W, H = 1920, 1080
FPS = 30
BG = (10, 14, 19)
INK, MUTED, ACCENT, GOLD = (231, 237, 243), (144, 161, 177), (79, 195, 206), (240, 180, 41)
FONTS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf", "/usr/share/fonts/TTF/DejaVuSans%s.ttf")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for pattern in FONTS:
        path = Path(pattern % ("-Bold" if bold else ""))
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size)


def mono(size: int) -> ImageFont.FreeTypeFont:
    for pattern in FONTS:
        path = Path(pattern.replace("DejaVuSans", "DejaVuSansMono") % "")
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size)


def read_zarr(path: Path) -> np.ndarray:
    group = zarr.open_group(str(path), mode="r")
    array = group[dict(group.attrs)["multiscales"][0]["datasets"][0]["path"]]
    return np.asarray(array[0] if array.ndim == 4 else array)


def grey(plane: np.ndarray, high: float = 99.6, gamma: float = 1.0) -> np.ndarray:
    lo, hi = np.percentile(plane, [1, high])
    stretched = np.clip((plane - lo) / (hi - lo + 1e-9), 0, 1)
    return stretched**gamma


def to_image(plane: np.ndarray, colour=(1, 1, 1)) -> Image.Image:
    return Image.fromarray((np.clip(plane[..., None] * np.array(colour), 0, 1) * 255).astype(np.uint8))


def overlay(a: np.ndarray, b: np.ndarray) -> Image.Image:
    rgb = np.dstack([np.clip(a, 0, 1), np.clip(b, 0, 1), np.zeros_like(a)])
    return Image.fromarray((rgb * 255).astype(np.uint8))


class Film:
    def __init__(self, directory: Path):
        self.directory = directory
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True)
        self.count = 0

    def add(self, frame: Image.Image, seconds: float = 1 / FPS) -> None:
        for _ in range(max(1, round(seconds * FPS))):
            frame.save(self.directory / f"{self.count:05d}.png")
            self.count += 1

    def encode(self, out: Path) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                        "-i", str(self.directory / "%05d.png"), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "20", str(out)], check=True)
        print(f"{out}  {self.count / FPS:.0f} s")


def canvas(title: str = "", subtitle: str = "") -> Image.Image:
    frame = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(frame)
    if title:
        draw.text((90, 70), title, font=font(46, bold=True), fill=INK)
    if subtitle:
        draw.text((90, 134), subtitle, font=font(27), fill=MUTED)
    return frame


def place(frame: Image.Image, image: Image.Image, box) -> tuple[int, int, float]:
    x, y, w, h = box
    factor = min(w / image.width, h / image.height)
    resized = image.resize((max(1, int(image.width * factor)), max(1, int(image.height * factor))), Image.LANCZOS)
    ox, oy = x + (w - resized.width) // 2, y + (h - resized.height) // 2
    frame.paste(resized, (ox, oy))
    return ox, oy, factor


def caption(frame: Image.Image, text: str, at, colour=MUTED, size: int = 24, code: bool = False) -> None:
    ImageDraw.Draw(frame).text(at, text, font=(mono(size) if code else font(size)), fill=colour)


# ------------------------------------------------------------------------------------ scenes


def scene_title(film: Film, fixed: Image.Image, moving: Image.Image) -> None:
    frame = canvas("Out-of-core registration", "one human brain block, two microscopes, 81 GiB, one laptop GPU")
    place(frame, fixed, (90, 215, 840, 700))
    place(frame, moving, (990, 215, 840, 700))
    caption(frame, "OCT   70 x 12441 x 14074   3.0 µm   50 GiB", (90, 950), GOLD, code=True)
    caption(frame, "light-sheet, NeuN   207 x 10309 x 12261   3.6 µm   31 GiB", (990, 950), GOLD, code=True)
    caption(frame, "DANDI 000026  .  sub-I48  .  BrocaAreaS23  .  public OME-Zarr on S3", (90, 1000))
    film.add(frame, 5.0)


def scene_mirror(film: Film, target: np.ndarray, source: np.ndarray, moved: np.ndarray) -> None:
    steps = [
        (overlay(target, source), "as the two stores hand them over", "mutual information 0.16", 3.0),
        (overlay(target, moved), "stage 1: the coarse level, a global affine, and the mirror it had to find",
         "mutual information 0.46", 4.0),
    ]
    for image, subtitle, number, hold in steps:
        frame = canvas("The two modalities, overlaid", subtitle)
        place(frame, image, (90, 200, 1740, 790))
        caption(frame, number, (90, 1005), GOLD, code=True)
        film.add(frame, hold)

    frame = canvas("Nothing in the metadata says which way the section faces",
                   "NGFF 0.4 carries axes, scale and translation. No orientation.")
    draw = ImageDraw.Draw(frame)
    rows = [("as imaged", "0.2954", MUTED), ("mirrored in x", "0.4605", GOLD),
            ("mirrored in y", "0.2880", MUTED), ("mirrored in both", "0.2862", MUTED)]
    y = 330
    for label, value, colour in rows:
        draw.text((200, y), label, font=mono(38), fill=colour)
        draw.text((900, y), value, font=mono(38), fill=colour)
        if colour is GOLD:
            draw.text((1120, y), "kept", font=font(30), fill=GOLD)
        y += 90
    caption(frame, "mutual information after the affine, from each of the four starts", (200, 250))
    film.add(frame, 5.0)


def scene_zoom(film: Film, section: np.ndarray, box_px, window: np.ndarray, cells: np.ndarray) -> None:
    image = Image.fromarray((section * 255).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle(box_px, outline=GOLD, width=4)
    frame = canvas("One window, at native resolution", "6.1 x 6.1 mm of the 37 x 42 mm section, 3 µm voxels")
    place(frame, image, (90, 200, 820, 790))
    place(frame, to_image(window), (960, 200, 390, 390))
    place(frame, to_image(cells), (1400, 200, 390, 390))
    caption(frame, "the window", (960, 610))
    caption(frame, "1.5 x 1.5 mm of it: single neurons", (1400, 610))
    caption(frame, "1.67 GiB read out of the 81 GiB the two stores hold", (960, 700), GOLD, code=True)
    caption(frame, "no level 0 volume was ever downloaded whole", (960, 740))
    film.add(frame, 6.0)


def scene_tiles(film: Film, window: np.ndarray, field: np.ndarray, patch: int, overlap: int) -> None:
    base = to_image(window).convert("RGB")
    step = patch - overlap
    tiles = [(x, y) for y in range(0, max(window.shape[0] - overlap, 1), step)
             for x in range(0, max(window.shape[1] - overlap, 1), step)]
    built = Image.new("RGB", base.size, (8, 11, 15))
    field_image = Image.fromarray((np.clip(field, 0, 1) * 255).astype(np.uint8))

    for index, (x, y) in enumerate(tiles):
        box = (x, y, min(x + patch, base.width), min(y + patch, base.height))
        built.paste(field_image.crop(box), (x, y))
        left = base.copy()
        draw = ImageDraw.Draw(left, "RGBA")
        for px, py in tiles[: index + 1]:
            draw.rectangle([px, py, px + patch, py + patch], outline=(240, 180, 41, 90), width=3)
        draw.rectangle([x, y, x + patch, y + patch], fill=(240, 180, 41, 40), outline=GOLD, width=9)

        frame = canvas("One patch in, one patch out",
                       f"patch {index + 1} of {len(tiles)}   .   {patch} x {patch} voxels, {overlap} of overlap")
        place(frame, left, (90, 200, 810, 790))
        place(frame, built, (1010, 200, 810, 790))
        caption(frame, "read one region of the store", (90, 1010))
        caption(frame, "blend it into the displacement field", (1010, 1010))
        film.add(frame, 0.5 if index < 3 else 0.25)
    film.add(frame, 1.5)


def scene_two(film: Film, title: str, subtitle: str, before: Image.Image, after: Image.Image,
              left: str, right: str, hold: float = 5.0) -> None:
    frame = canvas(title, subtitle)
    place(frame, before, (90, 210, 850, 700))
    place(frame, after, (980, 210, 850, 700))
    caption(frame, left, (90, 950), MUTED)
    caption(frame, right, (980, 950), GOLD)
    film.add(frame, hold)


def scene_numbers(film: Film) -> None:
    frame = canvas("What it cost", "NVIDIA RTX PRO 5000, 24 GB, one card")
    draw = ImageDraw.Draw(frame)
    rows = [
        ("read", "1.67 GiB out of 81 GiB", "only the chunks the window meets"),
        ("stage 1 + 2", "the whole section at 25 µm, 1 min", "mutual information 0.16 → 0.84"),
        ("stage 3", "25 tiles at 3 µm, 1 min 22 s", "1.9 GB on the card"),
        ("the check", "21 µm deformation → 5.6 µm residual", "no seam: 1.08x and 1.13x"),
    ]
    y = 280
    for label, value, note in rows:
        draw.rounded_rectangle([90, y, 1830, y + 150], radius=10, fill=(17, 23, 31))
        draw.text((130, y + 28), label, font=font(26, bold=True), fill=ACCENT)
        draw.text((130, y + 74), value, font=mono(34), fill=INK)
        draw.text((1180, y + 80), note, font=font(24), fill=MUTED)
        y += 175
    film.add(frame, 6.0)


def scene_slots(film: Film) -> None:
    frame = canvas("Three slots, nothing else moves", "")
    draw = ImageDraw.Draw(frame)
    rows = [
        ("the model", "elastix    FireANTs    ConvexAdam", "same two inputs, same two outputs"),
        ("the loss", "intensity    MIND    TotalSegmentator    SAM",
         "a feature map from a pretrained network is what makes a pair multimodal"),
        ("the patching", "patch_size    overlap    patch_combine",
         "three lines: the difference between holding the volume and never holding it"),
    ]
    y = 300
    for name, value, note in rows:
        draw.rounded_rectangle([90, y, 1830, y + 195], radius=10, fill=(17, 23, 31))
        draw.text((130, y + 32), name, font=font(28, bold=True), fill=ACCENT)
        draw.text((130, y + 84), value, font=mono(36), fill=INK)
        draw.text((130, y + 142), note, font=font(24), fill=MUTED)
        y += 230
    film.add(frame, 6.0)


def scene_terminal(film: Film, log: Path, seconds: float = 14.0) -> None:
    seen, lines = set(), []
    for raw in log.read_text(errors="replace").replace("\r", "\n").splitlines():
        line = raw.rstrip()[:128]
        key = line.split(":")[0]
        if not line.strip() or key in seen:
            continue  # a progress bar rewrites one line thousands of times; keep the first of each
        seen.add(key)
        lines.append(line)
    lines = lines[:26]
    visible = 24
    for cut in range(1, len(lines) + 1):
        frame = Image.new("RGB", (W, H), (8, 11, 15))
        draw = ImageDraw.Draw(frame)
        draw.rectangle([0, 0, W, 56], fill=(19, 25, 33))
        draw.text((26, 15), "the real run", font=font(24), fill=MUTED)
        for row, line in enumerate(lines[max(0, cut - visible):cut]):
            colour = GOLD if line.startswith("$") else INK if "[KonfAI]" in line else MUTED
            draw.text((26, 84 + row * 38), line, font=mono(21), fill=colour)
        film.add(frame, seconds / len(lines))
    film.add(frame, 2.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=Path, default=ROOT / "data" / "dandi" / "cortex")
    parser.add_argument("--native", type=Path, default=ROOT / "data" / "dandi" / "plane_native")
    parser.add_argument("--native-run", type=Path, default=ROOT / "out" / "native_tiled" / "P000")
    parser.add_argument("--log", type=Path, default=ROOT / "out" / "run_native_tiled.log")
    parser.add_argument("--patch", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=61)
    parser.add_argument("--out", type=Path, default=ROOT / "video" / "out_of_core.mp4")
    args = parser.parse_args()
    data = ROOT / "data" / "dandi"
    film = Film(ROOT / "video" / "frames")

    reference = sitk.ReadImage(str(data / "coarse_OCT.mha"))
    section = grey(sitk.GetArrayFromImage(reference))
    raw = grey(sitk.GetArrayFromImage(sitk.Resample(sitk.ReadImage(str(data / "coarse_SPIM.mha")), reference)))
    moved_section = grey(sitk.GetArrayFromImage(sitk.ReadImage(str(data / "coarse_SPIM_moved.mha"))))

    scene_title(film, to_image(section), to_image(moved_section))
    scene_mirror(film, section, raw, moved_section)

    fixed = read_zarr(args.window / "Fixed.ome.zarr")
    light = read_zarr(args.window / "Moving.ome.zarr")
    z = fixed.shape[0] // 2
    group = zarr.open_group(str(args.window / "Fixed.ome.zarr"), mode="r")
    dataset = dict(group.attrs)["multiscales"][0]["datasets"][0]
    scale = dataset["coordinateTransformations"][0]["scale"][-3:]
    origin = next(t["translation"][-3:] for t in dataset["coordinateTransformations"] if t["type"] == "translation")
    x0, y0 = origin[2] / 0.05, origin[1] / 0.05
    box_px = [x0, y0, x0 + fixed.shape[2] * scale[2] / 0.05, y0 + fixed.shape[1] * scale[1] / 0.05]
    scene_zoom(film, section, box_px, grey(light[z], gamma=0.55), grey(light[z, 768:1280, 768:1280], gamma=0.55))

    # the coarse deformable, on the whole section
    section_fixed = read_zarr(ROOT / "data" / "dandi" / "plane" / "Fixed.ome.zarr")[0]
    section_affine = read_zarr(ROOT / "data" / "dandi" / "plane" / "Moving.ome.zarr")[0]
    section_deformed = read_zarr(ROOT / "out" / "plane_g0.4" / "P000" / "Moved.ome.zarr")[0]
    scene_two(film, "Stage 2: the deformable, locked to the plane",
              "elastix, mutual information, a B-spline with no freedom in depth . 25 µm . 1 min",
              overlay(grey(section_fixed), grey(section_affine)),
              overlay(grey(section_fixed), grey(section_deformed)),
              "after the affine        MI 0.46", "after the deformable    MI 0.84", 6.0)

    # the tiles, on the native window
    native_fixed = read_zarr(args.native / "Fixed.ome.zarr")[0]
    native_start = read_zarr(args.native / "Moving.ome.zarr")[0]
    native_moved = read_zarr(args.native_run / "Moved.ome.zarr")[0]
    transform = sitk.DisplacementFieldTransform(sitk.ReadTransform(str(args.native_run / "Transform.h5")))
    plane = sitk.GetArrayFromImage(transform.GetDisplacementField())[0]
    import colorsys

    angle = (np.arctan2(plane[..., 1], plane[..., 0]) + np.pi) / (2 * np.pi)
    magnitude = np.linalg.norm(plane[..., :2], axis=-1)
    magnitude = magnitude / (np.percentile(magnitude, 99) + 1e-9)
    lut = np.array([colorsys.hsv_to_rgb(h / 255.0, 0.85, 1.0) for h in range(256)])
    field = lut[(angle * 255).astype(np.uint8)] * np.clip(magnitude, 0, 1)[..., None]
    scene_tiles(film, grey(native_fixed, gamma=0.8), field, args.patch, args.overlap)

    scene_two(film, "Stage 3: the native window, tiled", "6.1 x 6.1 mm at 3 µm, 25 tiles, blended",
              overlay(grey(native_fixed), grey(native_start, gamma=0.55)),
              overlay(grey(native_fixed), grey(native_moved, gamma=0.55)),
              "before the tiled run", "after the tiled run", 6.0)

    # why the coarse stage comes first
    naive = ROOT / "out" / "native_tiled_affine" / "P000" / "Moved.ome.zarr"
    if naive.exists():
        scene_two(film, "Why the coarse stage comes first",
                  "the same tiles, the same command, a different starting point",
                  overlay(grey(native_fixed), grey(read_zarr(naive)[0], gamma=0.55)),
                  overlay(grey(native_fixed), grey(native_moved, gamma=0.55)),
                  "tiles alone: each one solves a millimetre on its own",
                  "after the coarse stage: they agree", 6.0)

    scene_numbers(film)
    scene_slots(film)
    if args.log.exists():
        scene_terminal(film, args.log)
    film.encode(args.out)


if __name__ == "__main__":
    main()
