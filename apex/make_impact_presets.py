"""APEX: IMPACT presets built from the ImpactLoss example parameter maps (lib/ImpactLoss/ParameterMaps), verbatim
except for what the macaque pair needs: the feature voxel sizes scaled 4x down (6/3/1.5/1 mm -> 1.6/0.8/0.4/0.4 mm),
the final grid 8 -> 2 mm, ImpactGPU -1 (this elastix-IMPACT build is CPU-only, so mixed precision off) and
DefaultPixelValue 0. Each Prediction.yml carries the matching `resolutions` block (it stages the models), and the
map elastix would receive is rebuilt through the engine's own code and diffed against the written file.

    python apex/make_impact_presets.py [PRESET ...]   (all presets by default)
"""
import difflib, inspect, json, re, shutil, sys
from pathlib import Path

import yaml
from impact_reg_konfai.models import elastix as E
from impact_reg_konfai.models.elastix_engine import ElastixEngine

EXAMPLES = Path("/home/valentin/Documents/lib/ImpactLoss/ParameterMaps")
PRESETS = Path(__file__).resolve().parent.parent / "presets"
VOXELS = [1.6, 0.8, 0.4, 0.4]
# SAM's 29-pixel patch cannot be centred in the pair at 1.6 mm (31x44x28 and 38x41x28 voxels): no sample is valid, the loss
# never gets a derivative and elastix stops with "tensor does not have a device". Its first level starts at 0.8 mm instead.
VOXELS_BY_PRESET = {"APEX_IMPACT_SAM_JACOBIAN": [0.8, 0.8, 0.4, 0.4]}
HUB = "VBoussot/impact-torchscript-models:"

SPECS = {  # name: example map, model, dimension, channels, layers mask, subset, distance, mode
    "APEX_IMPACT_MIND_STATIC": ("ParameterMap_Static.txt", "MIND/R1D2_3D.pt", 3, 1, "1", 12, "L2", "Static"),
    "APEX_IMPACT_MIND_JACOBIAN": ("ParameterMap_Jacobian.txt", "MIND/R1D2_3D.pt", 3, 1, "1", 12, "L2", "Jacobian"),
    "APEX_IMPACT_TS_JACOBIAN": ("ParameterMap_Recommended.txt", "TS/M730.pt", 3, 1, "01", 64, "L2", "Jacobian"),
    "APEX_IMPACT_SAM_JACOBIAN": ("ParameterMap_SAM_2_Layers_Jacobian.txt", "SAM2.1/SAM2.1_Tiny.pt", 2, 3, "01", 1000, "L2", "Jacobian"),
    "APEX_IMPACT_ANATOMIX_STATIC": ("ParameterMap_Static.txt", "Anatomix/Anatomix.pt", 3, 1, "1", 16, "L2", "Static"),
}


def rescale_levels(text: str, model: str, voxels: list) -> str:
    text = re.sub(r'^\(ImpactModelsPath(\d) "[^"]*"\)', lambda m: f'(ImpactModelsPath{m.group(1)} "{model}")', text, flags=re.M)
    def voxel(m):
        n = len(m.group(2).split())
        return f"(ImpactVoxelSize{m.group(1)} " + " ".join([f"{voxels[int(m.group(1))]:g}"] * n) + ")"
    text = re.sub(r"^\(ImpactVoxelSize(\d) ([^)]*)\)", voxel, text, flags=re.M)
    return text


def engine_map(d: Path, cfg: dict) -> str:
    fields = inspect.signature(E.ModelSpec).parameters
    specs = {i: E.ResolutionSpec(max_iterations=r["max_iterations"], models={j: E.ModelSpec(**{k: v for k, v in m.items() if k in fields}) for j, m in r["models"].items()})
             for i, r in cfg["resolutions"].items()}
    eng = ElastixEngine.__new__(ElastixEngine)
    eng._max_iterations, eng._final_grid_spacing, eng._subset_features = cfg["max_iterations"], cfg["final_grid_spacing"], cfg["subset_features"]
    eng._spatial_samples, eng._parameter_overrides, eng._resolutions, eng._mode = cfg["spatial_samples"], list(cfg["parameter_overrides"]), specs, cfg["mode"]
    per_token, exact = eng._parameter_map_overrides()
    text = E.generate_impact_parameter_map((d / cfg["parameter_maps"][0]).read_text(), specs, E.load_models_registry(), cfg["mode"])
    return ElastixEngine._apply_map_overrides(text, per_token, exact, -1)


canon = lambda s: re.sub(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])", lambda m: repr(float(m.group())), s)
for name, (example, model, dim, channels, mask, subset, distance, mode) in SPECS.items():
    if sys.argv[1:] and name not in sys.argv[1:]:
        continue
    voxels = VOXELS_BY_PRESET.get(name, VOXELS)
    d = PRESETS / name
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(PRESETS / "MF283_ANATOMIX", d)
    for old in d.glob("ParameterMap_*.txt"):
        old.unlink()
    text = (EXAMPLES / example).read_text()
    text = rescale_levels(text, model, voxels)
    for a, b in (("(FinalGridSpacingInPhysicalUnits 8)", "(FinalGridSpacingInPhysicalUnits 2)"), ("(ImpactGPU 0)", "(ImpactGPU -1)"),
                 ("(DefaultPixelValue -1024)", "(DefaultPixelValue 0)"), ("(MaximumNumberOfIterations 500)", "(MaximumNumberOfIterations 500 500 500 500)"),
                 ('(CheckNumberOfSamples "true")', '(CheckNumberOfSamples "false")\n(RequiredRatioOfValidSamples 0.05)'),
                 ('(ITKTransformOutputFileNameExtension "itk.txt")', '(ITKTransformOutputFileNameExtension "itk.txt")\n(WriteITKCompositeTransform "true")')):
        assert text.count(a) == 1, (example, a)
        text = text.replace(a, b)
    text = text.replace("(ImpactFeaturesMapUpdateInterval -1)", '(ImpactUseMixedPrecision "false")\n(ImpactFeaturesMapUpdateInterval -1)')
    # what the engine derives, written down so the file says what runs:
    # the patch a model needs is its field of view from the registry (MIND R1D2: 5, TS layers "01": 5, SAM: 29),
    # a 2-D model still takes three voxel sizes (the image dimension), and the subset is the model's feature count.
    registry = E.load_models_registry()
    fov = 0 if mode == "Static" else {"MIND/R1D2_3D.pt": 5, "TS/M730.pt": 5, "SAM2.1/SAM2.1_Tiny.pt": 29}[model]  # Static: whole image
    text = re.sub(r"^\(ImpactPatchSize(\d) [^)]*\)", lambda m: f"(ImpactPatchSize{m.group(1)} " + " ".join([str(fov)] * dim) + ")", text, flags=re.M)
    text = re.sub(r"^\(ImpactVoxelSize(\d) [^)]*\)", lambda m: f"(ImpactVoxelSize{m.group(1)} " + " ".join([f"{voxels[int(m.group(1))]:g}"] * 3) + ")", text, flags=re.M)
    text = re.sub(r"^\(ImpactSubsetFeatures(\d) [^)]*\)", lambda m: f"(ImpactSubsetFeatures{m.group(1)} {subset})", text, flags=re.M)
    header = (f"// APEX CONNECTS: DTI FA (moving) onto PS-OCT retardance (fixed), 0.4 mm pair after the stage-1 affine.\n"
              f"// From ImpactLoss/ParameterMaps/{example}: {model}, {mode}; feature voxel sizes {'/'.join(f'{v:g}' for v in voxels)} mm, "
              f"final grid 2 mm, CPU (ImpactGPU -1, mixed precision off), DefaultPixelValue 0, samples outside the moving image tolerated (RequiredRatioOfValidSamples 0.05).\n"
              f"// The engine derives the patch size from each model's field of view (MIND R1D2 5, TS M730 layers 01: 5, SAM 29), not the example's 7 or 11.\n")
    mapname = f"ParameterMap_{name}.txt"
    (d / mapname).write_text(header + text)
    y = d / "Prediction.yml"; s = y.read_text()
    s = re.sub(r"      parameter_maps:\n      - [^\n]*\n", f"      parameter_maps:\n      - {mapname}\n", s)
    start, end = s.index("      resolutions:\n"), s.index("      mode: ")
    block = "      resolutions:\n"
    for i, v in enumerate(voxels):
        vs = "".join(f"              - {v}\n" for _ in range(3))
        block += (f"        '{i}':\n          max_iterations: 500\n          models:\n            '0':\n              ref: {HUB}{model}\n"
                  f"              voxel_size:\n{vs}              layers_mask: '{mask}'\n              layers_weight:\n              - 1.0\n"
                  f"              subset_features: {subset}\n              pca: 0\n              distance: {distance}\n")
    s = s[:start] + block + re.sub(r"      mode: \w+", f"      mode: {mode}", s[end:], count=1)
    y.write_text(s)
    note = f"From ImpactLoss/ParameterMaps/{example}: IMPACT alone with {model} ({mode}, {distance}), feature voxel sizes {'/'.join(f'{v:g}' for v in voxels)} mm, 500 it, 2000 samples, grid 2 mm, CPU."
    j = json.loads((d / "app.json").read_text()); j.update(display_name=f"APEX CONNECTS: FA onto retardance, {name}", short_description=note, description=note)
    (d / "app.json").write_text(json.dumps(j, indent=4))
    cfg = yaml.safe_load(y.read_text())["Predictor"]["Model"]["RegistrationNet"]
    written = sorted(canon(l.strip()) for l in (d / mapname).read_text().splitlines() if l.strip().startswith("("))
    effective = sorted(canon(l.strip()) for l in engine_map(d, cfg).splitlines() if l.strip().startswith("("))
    diff = [l for l in difflib.unified_diff(written, effective, lineterm="", n=0) if not l.startswith(("---", "+++", "@@"))]
    print(f"{name}: {example} | file vs elastix: {diff or 'identical'}")
