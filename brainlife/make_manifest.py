#!/usr/bin/env python3
"""Write brainlife/app.json: the App's registration form (inputs, outputs, parameters), in the shape of a
brainlife warehouse App record, with the preset menu read from presets/*/app.json.

brainlife registers an App through its web form (https://brainlife.io/docs/apps/register/), not from a file
in the repository: this file is the copy-paste source for that form and the record of what was entered, and
it keeps the preset menu in step with the presets. Run it whenever a preset is added or renamed:

    python3 brainlife/make_manifest.py && git diff brainlife/app.json

The brainlife datatype IDs are those of the public warehouse (https://brainlife.io/api/warehouse/datatype).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRESETS = ROOT / "presets"
MANIFEST = ROOT / "brainlife" / "app.json"

OME_ZARR = {"name": "neuro/ome-zarr", "_id": "6a99e5d7d90f157078ed790c", "file_id": "zarr", "filename": "data.ome.zarr"}

DEFAULT_PRESET = "APEX_FIREANTS_SYN_CC"
ENGINE_LABEL = {"fireants": "FireANTs", "convexadam": "ConvexAdam", "elastix": "elastix"}


def presets() -> list[dict]:
    rows = []
    for folder in sorted(PRESETS.iterdir()):
        app_json = folder / "app.json"
        if not app_json.is_file():
            continue
        meta = json.loads(app_json.read_text())
        if meta.get("task") != "registration":
            continue
        prediction = (folder / "Prediction.yml").read_text()
        engine = next((e for e in ENGINE_LABEL if f"impact_reg_konfai.models.{e}" in prediction), "?")
        rows.append({
            "value": folder.name,
            "label": f"{folder.name} ({ENGINE_LABEL.get(engine, engine)})",
            "desc": meta.get("short_description") or meta.get("description", ""),
        })
    return rows


def manifest() -> dict:
    options = presets()
    assert any(o["value"] == DEFAULT_PRESET for o in options), DEFAULT_PRESET
    return {
        "name": "Out-of-core multimodal registration (KonfAI + IMPACT)",
        "github": "fideus-labs/impact-reg-ooc-demo",
        "github_branch": "main",
        "desc": (
            "Deformable registration of a pre-aligned pair of OME-Zarr volumes, in tiles when the volumes "
            "do not fit in GPU memory: FireANTs SyN, ConvexAdam or elastix B-spline, driven by intensity, MIND "
            "or IMPACT (TotalSegmentator) features. Writes the displacement field as OME-Zarr RFC-5 (and as an "
            "ITK .h5 transform for 3D Slicer) and the moving image resampled onto the fixed grid. "
            "KonfAI + impact-reg-konfai; presets from the APEX BRAIN CONNECTS and LINC hemisphere work."
        ),
        "desc_override": "Out-of-core deformable registration of OME-Zarr volumes (FireANTs / ConvexAdam / elastix, IMPACT features)",
        "tags": ["registration", "ome-zarr", "microscopy", "multimodal", "gpu", "out-of-core"],
        "requires_gpu": True,
        "retry": 0,
        "inputs": [
            {"id": "fixed", "datatype": OME_ZARR["name"], "datatype_id": OME_ZARR["_id"], "datatype_tags": [],
             "desc": "The fixed image: the grid the field and the moved image are written on. A scalar OME-Zarr volume (z, y, x; a size-1 c/t axis is squeezed).",
             "optional": False, "multi": False, "advanced": False},
            {"id": "moving", "datatype": OME_ZARR["name"], "datatype_id": OME_ZARR["_id"], "datatype_tags": [],
             "desc": "The moving image, already affine-aligned to the fixed image (the presets hold the affine at identity).",
             "optional": False, "multi": False, "advanced": False},
            {"id": "fixed_mask", "datatype": OME_ZARR["name"], "datatype_id": OME_ZARR["_id"], "datatype_tags": ["mask"],
             "desc": "Tissue mask on the fixed grid (0/1). Optional for a whole-volume run; needed for a tiled one, where a tile of background otherwise fits the background.",
             "optional": True, "multi": False, "advanced": False},
            {"id": "moving_mask", "datatype": OME_ZARR["name"], "datatype_id": OME_ZARR["_id"], "datatype_tags": ["mask"],
             "desc": "Tissue mask on the moving grid (0/1). Optional; see fixed_mask.",
             "optional": True, "multi": False, "advanced": False},
        ],
        "outputs": [
            {"id": "transform", "datatype": OME_ZARR["name"], "datatype_id": OME_ZARR["_id"],
             "datatype_tags": ["displacement_field"], "archive": True, "output_on_root": False,
             "desc": "The displacement field on the fixed grid, mm, as an OME-Zarr RFC-5 `displacements` coordinate transformation (data.ome.zarr) and as an ITK transform (Transform.h5, opens in 3D Slicer)."},
            {"id": "moved", "datatype": OME_ZARR["name"], "datatype_id": OME_ZARR["_id"],
             "datatype_tags": ["registered"], "archive": True, "output_on_root": False,
             "desc": "The moving image resampled through the field onto the fixed grid (not written when fields_only is set)."},
        ],
        "config": {
            "fixed": {"type": "input", "input_id": "fixed", "file_id": OME_ZARR["file_id"]},
            "moving": {"type": "input", "input_id": "moving", "file_id": OME_ZARR["file_id"]},
            "fixed_mask": {"type": "input", "input_id": "fixed_mask", "file_id": OME_ZARR["file_id"]},
            "moving_mask": {"type": "input", "input_id": "moving_mask", "file_id": OME_ZARR["file_id"]},
            "preset": {
                "id": "preset", "type": "enum", "default": DEFAULT_PRESET, "placeholder": "",
                "desc": "The registration preset (presets/<name> in the repository): the engine, the metric and its features, the resolution it was tuned for, and whether it tiles.",
                "options": options,
            },
            "device": {
                "id": "device", "type": "enum", "default": "gpu", "placeholder": "",
                "desc": "Where the preset runs. The GPU is the intended device (a 256³ elastix tile peaks near 16.5 GiB, a 192³ FireANTs tile near 18 GiB); the CPU is a fallback for small pairs.",
                "options": [
                    {"value": "gpu", "label": "GPU", "desc": "CUDA device 0 of the allocation"},
                    {"value": "cpu", "label": "CPU", "desc": "cpu_workers threads (default: the job's cores)"},
                ],
            },
            "patch_size": {
                "id": "patch_size", "type": "string", "default": "", "placeholder": "256 256 256",
                "desc": "Tile size in voxels (z y x, or one number for a cube). Empty: the preset's own setting (the whole volume for the APEX presets, tiles for the *_TILED ones). Tiles need both masks.",
                "advanced": True,
            },
            "overlap": {
                "id": "overlap", "type": "string", "default": "", "placeholder": "12%",
                "desc": "Overlap between neighbouring tiles, as a percentage of the tile ('12%') or in voxels ('16'). Empty: the preset's own, or 12% when patch_size is set.",
                "advanced": True,
            },
            "tta": {
                "id": "tta", "type": "number", "default": 0, "placeholder": "0",
                "desc": "Test-time augmentation passes (0: none).",
                "advanced": True,
            },
            "fields_only": {
                "id": "fields_only", "type": "boolean", "default": False, "placeholder": "",
                "desc": "Write the displacement field only; skip resampling the moving image (saves the time and disk of a second volume; brainlife then reports the moved output as missing).",
                "advanced": True,
            },
            "cpu_workers": {
                "id": "cpu_workers", "type": "number", "default": 0, "placeholder": "0",
                "desc": "Threads for a CPU run (0: the cores the scheduler allotted).",
                "advanced": True,
            },
            "extra_set": {
                "id": "extra_set", "type": "string", "default": "", "placeholder": "Predictor.Dataset.Patch.overlap=16%; Predictor.outputs_dataset.DisplacementField.OutputDataset.patch_combine=Cosinus",
                "desc": "Further `impact-reg-konfai register --set NAME=VALUE` overrides of the preset's Prediction.yml, ';'-separated (a bare NAME targets the model's parameters; a dotted path any existing key).",
                "advanced": True,
            },
        },
        "citation": {
            "app": "https://github.com/fideus-labs/impact-reg-ooc-demo",
            "konfai": "https://github.com/fideus-labs/KonfAI",
            "impact_reg_konfai": "https://pypi.org/project/impact-reg-konfai/",
            "impact": "Boussot et al., IMPACT: a generic semantic loss for multimodal medical image registration, https://github.com/vboussot/ImpactLoss",
            "brainlife": "Hayashi et al. 2023, brainlife.io: a decentralized and open-source cloud platform to support neuroscience research, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10274934/",
        },
    }


if __name__ == "__main__":
    MANIFEST.write_text(json.dumps(manifest(), indent=2, ensure_ascii=False) + "\n")
    print(f"{MANIFEST.relative_to(ROOT)}: {len(manifest()['config']['preset']['options'])} presets")
