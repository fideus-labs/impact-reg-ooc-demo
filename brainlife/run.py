#!/usr/bin/env python3
"""The brainlife.io App: config.json in, the displacement field and the moved image out, as OME-Zarr.

brainlife stages the inputs, writes config.json in the repository root and runs ./main, which runs this
file inside the container (or, without singularity, in the repository's pixi environment). This file:

1. reads config.json: the four OME-Zarr stores (fixed, moving and the optional masks) and the parameters;
2. picks the pixi environment the preset needs (the elastix presets run on torch 2.8, the LibTorch the
   elastix-IMPACT binary links; FireANTs and ConvexAdam on torch 2.12; see pixi.toml);
3. runs `impact-reg-konfai register` with the repository's own presets (KONFAI_IMPACTREG_REPO=presets/),
   tiled when a patch size is given;
4. writes the field as OME-Zarr RFC-5 (pipeline/transform_to_ome_zarr.py) and lays the outputs out the way
   the brainlife datatypes expect: transform/data.ome.zarr (+ Transform.h5 for 3D Slicer) and
   moved/data.ome.zarr, both neuro/ome-zarr; and product.json, the run's summary brainlife displays.

Standard library only: it runs before any environment is chosen. Usable by hand too:

    cp config.json.example config.json   # edit the paths
    python3 brainlife/run.py             # or ./main
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRESETS = ROOT / "presets"
# Written by the Dockerfile: `pixi shell-hook` of each environment, ending in `exec "$@"`. Absent outside
# the container, where `pixi run` serves the same purpose.
ENV_HOOKS = Path(os.environ.get("IMPACT_REG_OOC_ENV_HOOKS", "/opt/impact-reg-ooc-demo/env"))
# The feature models the presets pull from Hugging Face, fetched at image build so a compute node
# without internet access still runs. Seeded into the user's cache on first use (singularity mounts
# $HOME from the host, so the image's own $HOME cache is hidden).
BAKED_HF_CACHE = Path(os.environ.get("IMPACT_REG_OOC_HF_CACHE", "/opt/impact-reg-ooc-demo/hf-cache"))

OUT = ROOT / "out"  # impact-reg-konfai's output: out/P000/{Transform.h5,Moved.ome.zarr}
WORK = ROOT / "work"  # its intermediates (volume-sized; keep them off a tmpfs TMPDIR)
TRANSFORM_DIR = ROOT / "transform"  # brainlife output `transform`, neuro/ome-zarr
MOVED_DIR = ROOT / "moved"  # brainlife output `moved`, neuro/ome-zarr
STORE = "data.ome.zarr"  # the store root the neuro/ome-zarr datatype names
STREAM_ABOVE_BYTES = 2 * 1024**3  # a Transform.h5 above this is written slab by slab


def log(message: str) -> None:
    print(f"[brainlife/run.py] {message}", flush=True)


def fail(message: str, code: int = 1) -> None:
    log(f"error: {message}")
    write_product({"brainlife": [{"type": "error", "msg": message}]})
    sys.exit(code)


def write_product(product: dict) -> None:
    (ROOT / "product.json").write_text(json.dumps(product, indent=2) + "\n")


# ----------------------------------------------------------------------------------------------- config


def load_config() -> dict:
    path = ROOT / "config.json"
    if not path.is_file():
        fail("config.json not found (brainlife writes it at runtime; locally, copy config.json.example)")
    return json.loads(path.read_text())


def store_path(config: dict, key: str, required: bool) -> Path | None:
    """An input store: brainlife maps the datatype's `data.ome.zarr` directory to this key. An optional
    input the user did not supply is absent from config.json (or null / empty)."""
    value = config.get(key)
    if value in (None, "", "None"):
        if required:
            fail(f"config.json: '{key}' is required")
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not (path / "zarr.json").is_file() and not (path / ".zattrs").is_file() and not (path / ".zgroup").is_file():
        fail(f"config.json: '{key}' = {path} is not an OME-Zarr store root (no zarr.json / .zgroup)")
    return path


def parse_patch_size(value) -> list[int] | None:
    """'256 256 256', '256,256,256', '[256, 256, 256]', 256 or '' (no tiling)."""
    if value in (None, "", "None", 0, "0", False):
        return None
    if isinstance(value, (int, float)):
        return [int(value)] * 3
    if isinstance(value, list):
        sizes = [int(v) for v in value]
    else:
        sizes = [int(v) for v in re.findall(r"\d+", str(value))]
    if len(sizes) == 1:
        sizes *= 3
    if len(sizes) != 3 or min(sizes) <= 0:
        fail(f"config.json: patch_size must be three positive integers (z y x) or one for a cube, got {value!r}")
    return sizes


def parse_overlap(value) -> str | None:
    """'12%' (of the patch) or '16' (voxels); KonfAI reads either."""
    if value in (None, "", "None"):
        return None
    text = str(value).strip()
    if not re.fullmatch(r"\d+(\.\d+)?%?", text):
        fail(f"config.json: overlap must be a percentage like '12%' or a voxel count like '16', got {value!r}")
    return text


def as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def extra_overrides(value) -> list[str]:
    """Advanced `--set NAME=VALUE` overrides, one per line or ';'-separated (a value may hold spaces)."""
    if not value:
        return []
    items = [item.strip() for item in re.split(r"[;\n]", str(value)) if item.strip()]
    for item in items:
        if "=" not in item:
            fail(f"config.json: extra_set entry {item!r} is not NAME=VALUE")
    return items


# ---------------------------------------------------------------------------------------------- presets


def available_presets() -> list[str]:
    names = []
    for folder in sorted(PRESETS.iterdir()):
        app_json = folder / "app.json"
        if app_json.is_file():
            try:
                if json.loads(app_json.read_text()).get("task") == "registration":
                    names.append(folder.name)
            except (OSError, json.JSONDecodeError):
                pass
    return names


def preset_engine(preset: str) -> str:
    """'elastix', 'fireants' or 'convexadam', from the model classpath of the preset's Prediction.yml."""
    text = (PRESETS / preset / "Prediction.yml").read_text()
    match = re.search(r"classpath:\s*impact_reg_konfai\.models\.(\w+)", text)
    if not match:
        fail(f"presets/{preset}/Prediction.yml: no impact_reg_konfai.models.* classpath, cannot pick an environment")
    return match.group(1)


def preset_is_tiled(preset: str) -> bool:
    """`patch_size: None` is a whole-volume preset; a list (inline or on the following lines) tiles."""
    text = (PRESETS / preset / "Prediction.yml").read_text()
    match = re.search(r"^\s*patch_size:[ \t]*(.*)$", text, re.M)
    if not match:
        return False
    value = match.group(1).strip()
    if value in ("None", "null", "~"):
        return False
    return True  # '[256, 256, 256]' inline, or empty with '- 256' items on the next lines


def environment_for(engine: str) -> str:
    return "elastix" if engine == "elastix" else "default"


def in_environment(env: str, command: list[str]) -> list[str]:
    hook = ENV_HOOKS / f"{env}.sh"
    if hook.is_file():
        return ["bash", str(hook), *command]
    if shutil.which("pixi") is None:
        fail(f"neither the container's environment hook {hook} nor pixi is available")
    return ["pixi", "run", "--manifest-path", str(ROOT / "pixi.toml"), "-e", env, *command]


def run(command: list[str], env: dict[str, str]) -> None:
    log("+ " + " ".join(shell_quote(part) for part in command))
    result = subprocess.run(command, cwd=ROOT, env=env)
    if result.returncode != 0:
        fail(f"command failed with exit code {result.returncode}: {command[0]} ... {command[-1]}", result.returncode)


def shell_quote(part: str) -> str:
    return part if re.fullmatch(r"[\w./:=+%,@-]+", part) else "'" + part.replace("'", "'\\''") + "'"


# ----------------------------------------------------------------------------------------------- caches


def seed_hf_cache() -> None:
    """Copy the image's Hugging Face cache into the user's, entry by entry, where the user's lacks it."""
    source = BAKED_HF_CACHE / "hub"
    if not source.is_dir():
        return
    target = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    try:
        target.mkdir(parents=True, exist_ok=True)
        for entry in source.iterdir():
            if not (target / entry.name).exists():
                log(f"seeding {target / entry.name} from the image")
                shutil.copytree(entry, target / entry.name, symlinks=True)
    except OSError as error:
        # $HOME may be read-only on a compute node: point HF at a writable copy in the work directory.
        log(f"could not seed {target} ({error}); using {WORK / 'hf'} as HF_HOME")
        os.environ["HF_HOME"] = str(WORK / "hf")
        shutil.copytree(BAKED_HF_CACHE, WORK / "hf", symlinks=True, dirs_exist_ok=True)


# ----------------------------------------------------------------------------------------------- outputs


def move_store(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(source), str(target))


def store_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def store_summary(path: Path) -> dict:
    """Shape, dtype and scale of the finest level, from the store's own metadata (Zarr v3 or v2)."""
    summary: dict = {"path": str(path.relative_to(ROOT)), "size_bytes": store_size_bytes(path)}
    try:
        meta = json.loads((path / "zarr.json").read_text())
        ome = meta.get("attributes", {}).get("ome", meta.get("attributes", {}))
        multiscales = ome["multiscales"][0]
        summary["ngff_version"] = ome.get("version", multiscales.get("version"))
        summary["axes"] = [axis["name"] for axis in multiscales.get("axes", [])]
        kinds = [t.get("type") for t in multiscales.get("coordinateTransformations", [])]
        if kinds:
            summary["coordinateTransformations"] = kinds
        first = path / multiscales["datasets"][0]["path"]
        array = json.loads((first / "zarr.json").read_text()) if (first / "zarr.json").is_file() else None
        if array is None and (first / "image" / "zarr.json").is_file():  # RFC-5 field: scale0/image
            array = json.loads((first / "image" / "zarr.json").read_text())
        if array:
            summary["shape"] = array.get("shape")
            summary["dtype"] = array.get("data_type")
        for transform in multiscales["datasets"][0].get("coordinateTransformations", []):
            if transform.get("type") == "scale":
                summary["scale"] = transform["scale"]
    except (OSError, KeyError, IndexError, json.JSONDecodeError, TypeError):
        pass
    return summary


# -------------------------------------------------------------------------------------------------- main


def main() -> None:
    started = time.time()
    config = load_config()

    fixed = store_path(config, "fixed", required=True)
    moving = store_path(config, "moving", required=True)
    fixed_mask = store_path(config, "fixed_mask", required=False)
    moving_mask = store_path(config, "moving_mask", required=False)

    presets = available_presets()
    preset = str(config.get("preset") or "APEX_FIREANTS_SYN_CC").strip()
    if preset not in presets:
        fail(f"config.json: preset {preset!r} is not one of the repository's presets: {', '.join(presets)}")
    engine = preset_engine(preset)
    env = environment_for(engine)

    patch_size = parse_patch_size(config.get("patch_size"))
    overlap = parse_overlap(config.get("overlap"))
    fields_only = as_bool(config.get("fields_only", False))
    tta = int(config.get("tta") or 0)
    device = str(config.get("device") or "gpu").strip().lower()
    if device not in ("gpu", "cpu"):
        fail(f"config.json: device must be 'gpu' or 'cpu', got {device!r}")
    if device == "gpu" and os.environ.get("CUDA_VISIBLE_DEVICES", "unset") == "":
        log("CUDA_VISIBLE_DEVICES is empty: no GPU was allocated to this job; falling back to the CPU")
        device = "cpu"
    cpu_workers = int(config.get("cpu_workers") or os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)
    extra = extra_overrides(config.get("extra_set"))

    tiled = patch_size is not None or preset_is_tiled(preset)
    if tiled and (fixed_mask is None or moving_mask is None):
        log("warning: a tiled run without masks lets a tile that holds only background fit the background")
    overrides: list[str] = []
    if patch_size is not None:
        overrides.append(f"Predictor.Dataset.Patch.patch_size={json.dumps(patch_size)}")
        if overlap is None:
            overlap = "12%"
    if overlap is not None:
        if not tiled:
            fail("config.json: overlap needs a patch_size (or a preset that tiles)")
        overrides.append(f"Predictor.Dataset.Patch.overlap={overlap}")
    if patch_size is not None and not preset_is_tiled(preset):
        # The blend of the overlaps (README, step 3); a preset that tiles already declares its own.
        overrides.append("Predictor.outputs_dataset.DisplacementField.OutputDataset.patch_combine=Cosinus")
    overrides += extra  # last, so the user's word is final

    log(f"preset {preset} ({engine}, pixi environment '{env}'), device {device}"
        + (f" x{cpu_workers} workers" if device == "cpu" else "")
        + (f", tiles {patch_size} overlap {overlap}" if patch_size else ""))

    for path in (OUT, WORK, TRANSFORM_DIR, MOVED_DIR):
        if path.exists():
            shutil.rmtree(path)
    WORK.mkdir()
    seed_hf_cache()

    run_env = dict(os.environ)
    run_env["KONFAI_IMPACTREG_REPO"] = str(PRESETS)  # the repository's presets, not the Hugging Face hub
    run_env.setdefault("KONFAI_APPS_INSTALL_REQUIREMENTS", "0")  # the presets' requirements are in the stack already
    run_env.setdefault("TMPDIR", str(WORK))
    if device == "cpu":
        run_env.setdefault("OMP_NUM_THREADS", str(cpu_workers))

    # 1. The registration.
    command = ["impact-reg-konfai", "register", preset, "-f", str(fixed), "-m", str(moving)]
    if fixed_mask is not None:
        command += ["--fixed-mask", str(fixed_mask)]
    if moving_mask is not None:
        command += ["--moving-mask", str(moving_mask)]
    command += ["-o", str(OUT), "--tmp-dir", str(WORK)]
    command += ["--gpu", "0"] if device == "gpu" else ["--cpu", str(cpu_workers)]
    if tta:
        command += ["--tta", str(tta)]
    if fields_only:
        command += ["--fields-only"]
    for override in overrides:
        command += ["--set", override]
    run(in_environment(env, command), run_env)

    case = OUT / "P000"
    transform_h5 = case / "Transform.h5"
    if not transform_h5.is_file():
        candidates = sorted(case.glob("Transform*")) if case.is_dir() else []
        fail(f"{transform_h5} was not written (found: {[c.name for c in candidates]})")

    # 2. The field as OME-Zarr RFC-5, beside Transform.h5; slab by slab when it is large.
    # `python` is the environment's own, whichever hook or `pixi run` resolves it.
    convert = ["python", str(ROOT / "pipeline" / "transform_to_ome_zarr.py")]
    transform_store = case / "Transform.ome.zarr"
    if transform_h5.stat().st_size > STREAM_ABOVE_BYTES:
        convert += ["--stream", str(transform_h5), str(transform_store)]
    else:
        convert += [str(transform_h5), str(fixed)]
    run(in_environment(env, convert), run_env)
    if not (transform_store / "zarr.json").is_file():
        fail(f"{transform_store} was not written")

    # 3. The brainlife layout.
    move_store(transform_store, TRANSFORM_DIR / STORE)
    shutil.copy2(transform_h5, TRANSFORM_DIR / "Transform.h5")
    moved_store = None
    if not fields_only:
        found = sorted(p for p in case.glob("Moved*") if p.is_dir() and ((p / "zarr.json").is_file() or (p / ".zgroup").is_file()))
        if not found:
            fail(f"the moved image was not written under {case} (found: {sorted(p.name for p in case.iterdir())})")
        moved_store = MOVED_DIR / STORE
        move_store(found[0], moved_store)
    shutil.rmtree(WORK, ignore_errors=True)

    # 4. product.json: what brainlife shows on the task page.
    elapsed = time.time() - started
    product = {
        "preset": preset,
        "engine": engine,
        "device": device,
        "tiled": tiled,
        "patch_size": patch_size,
        "overlap": overlap,
        "config_overrides": overrides,
        "fields_only": fields_only,
        "elapsed_seconds": round(elapsed, 1),
        "transform": store_summary(TRANSFORM_DIR / STORE),
        "transform_itk": {"path": "transform/Transform.h5", "size_bytes": transform_h5.stat().st_size},
        "moved": store_summary(moved_store) if moved_store else None,
        "brainlife": [
            {"type": "success", "msg": f"{preset} ({engine}) registered the moving store onto the fixed grid in "
                                       f"{elapsed / 60:.1f} min on the {device.upper()}"
                                       + (f", in {'x'.join(map(str, patch_size))} tiles" if patch_size else "") + "."},
            {"type": "info", "msg": "transform/data.ome.zarr: the displacement field, OME-Zarr RFC-5 `displacements` "
                                    "transformation, mm on the fixed grid; transform/Transform.h5: the same field as an ITK "
                                    "transform (opens in 3D Slicer)."},
        ],
    }
    if moved_store:
        product["brainlife"].append({"type": "info", "msg": "moved/data.ome.zarr: the moving image resampled through the field onto the fixed grid."})
    else:
        product["brainlife"].append({"type": "info", "msg": "fields_only: the moved image was not derived."})
    write_product(product)
    log(f"done in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
