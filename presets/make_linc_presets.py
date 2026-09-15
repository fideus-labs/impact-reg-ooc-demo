"""LINC: the documented elastix runs as presets. Each run was the hub preset Generic_Rigid_BSpline with --set
overrides; here the settings are written into the preset's own files: a parameter map holding the metrics, weights,
grid, iterations and samples, and a Prediction.yml that points to it, leaves the map alone (max_iterations,
final_grid_spacing and spatial_samples at 0, no parameter_overrides) and carries the tiling.

The check uses konfai-apps' and the engine's own code: it applies the documented --set to the hub preset the way
`konfai-apps infer` does, rebuilds the map elastix received from it, rebuilds the map elastix receives from the new
preset, and diffs every entry; then it diffs every other Prediction.yml key.

FireANTs and ConvexAdam on level 3 in 192³ tiles (LINC_FIREANTS_L3_TILED, LINC_CONVEXADAM_L3_TILED) are the APEX
presets with the tiling written into Prediction.yml by that same function; they have no parameter map to check.

    python presets/make_linc_presets.py
"""
import json, re, shutil, tempfile, types
from pathlib import Path

from ruamel.yaml import YAML
from impact_reg_konfai.models.elastix_engine import ElastixEngine
from konfai_apps.app_repository import LocalAppRepository

HUB = max(Path.home().glob(".cache/huggingface/hub/models--VBoussot--ImpactReg/snapshots/*/Generic_Rigid_BSpline/Prediction.yml"),
          key=lambda p: p.stat().st_mtime).parent
PRESETS = Path(__file__).resolve().parent
NET, PATCH = "Predictor.Model.RegistrationNet.", "Predictor.Dataset.Patch."
COSINUS = "Predictor.outputs_dataset.DisplacementField.OutputDataset.patch_combine=Cosinus"
BEND = 'Registration="MultiMetricMultiResolutionRegistration", Metric="AdvancedMattesMutualInformation" "TransformBendingEnergyPenalty", Metric0Weight=1.0, Metric1Weight=100'
ENGINE_KEYS = ("parameter_maps", "max_iterations", "final_grid_spacing", "spatial_samples", "parameter_overrides")

RUNS = {  # preset: (what it runs, the --set of the documented command)
    "LINC_BSPLINE_L4": ("the whole hemisphere at level 4 (0.32 mm): stage 2 and the FLASH check", [
        NET + "parameter_maps=[Parameters_BSpline.txt]", NET + "final_grid_spacing=3.0", NET + "max_iterations=600",
        NET + "spatial_samples=8192", NET + f"parameter_overrides=[{BEND}, RequiredRatioOfValidSamples=0.05]"]),
    "LINC_BSPLINE_L3_TILED": ("level 3 (0.16 mm) in 256³ tiles: stage 3", [
        NET + "parameter_maps=[Parameters_BSpline.txt]", NET + "final_grid_spacing=2.0", NET + "max_iterations=200",
        NET + "spatial_samples=4096", NET + f'parameter_overrides=[{BEND}, CheckNumberOfSamples="false"]',
        PATCH + "patch_size=[256, 256, 256]", PATCH + "overlap=12%", COSINUS]),
    "LINC_BSPLINE_L0_TILED": ("a window of level 0 (20 µm) in 256³ tiles: stage 4", [
        NET + "parameter_maps=[Parameters_BSpline.txt]", NET + "final_grid_spacing=1.0", NET + "max_iterations=200",
        NET + "spatial_samples=4096", NET + f'parameter_overrides=[{BEND}, CheckNumberOfSamples="false"]',
        PATCH + "patch_size=[256, 256, 256]", PATCH + "overlap=128", COSINUS]),
    "LINC_BSPLINE_ZOOM1_TILED": ("zoom 1 on the overview at level 1 (40 µm) in 256³ tiles: stage 5", [
        NET + "parameter_maps=[Parameters_BSpline.txt]", NET + "final_grid_spacing=2.0", NET + "max_iterations=200",
        NET + "spatial_samples=4096", NET + f'parameter_overrides=[{BEND}, CheckNumberOfSamples="false"]',
        PATCH + "patch_size=[256,256,256]", PATCH + "overlap=128", COSINUS]),
    "SECTION_BSPLINE_NATIVE_TILED": ("the section's 3 µm window in [8, 512, 512] tiles: section stage 3", [
        NET + "parameter_maps=[Parameters_BSpline.txt]", NET + "final_grid_spacing=0.3",
        NET + f'parameter_overrides=[{BEND}, FinalGridSpacingInPhysicalUnits=0.3 0.3 1000, RequiredRatioOfValidSamples=0.05, CheckNumberOfSamples="false"]',
        PATCH + "patch_size=[8, 512, 512]", PATCH + "overlap=12%", COSINUS]),
}

yaml = YAML()
repo = types.SimpleNamespace(_model_param_block=LocalAppRepository._model_param_block)
canon = lambda s: re.sub(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])", lambda m: repr(float(m.group())), s)


def received(bundle: Path, net: dict) -> list[str]:
    """The maps elastix is given for this RegistrationNet block, as the engine stages them."""
    eng = ElastixEngine.__new__(ElastixEngine)
    eng._max_iterations, eng._final_grid_spacing, eng._subset_features = net["max_iterations"], net["final_grid_spacing"], 0
    eng._spatial_samples, eng._parameter_overrides, eng._resolutions = net["spatial_samples"], list(net["parameter_overrides"]), {}
    per_token, exact = eng._parameter_map_overrides()
    return [ElastixEngine._apply_map_overrides((bundle / Path(p).name).read_text(), per_token, exact, 0) for p in net["parameter_maps"]]


def entries(maps: list[str]) -> list[str]:
    return sorted(canon(line.strip()) for text in maps for line in text.splitlines() if line.strip().startswith("("))


for name, (what, overrides) in RUNS.items():
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "run"
        shutil.copytree(HUB, run)
        LocalAppRepository._apply_config_overrides(repo, str(run / "Prediction.yml"), overrides)  # as `konfai-apps infer --set`
        documented = yaml.load(run / "Prediction.yml")
        net = documented["Predictor"]["Model"]["RegistrationNet"]
        (text,) = received(run, net)
        expected, documented_plain = entries([text]), json.loads(json.dumps(documented))

    d = PRESETS / name
    shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    mapname = f"ParameterMap_{name}.txt"
    text = text.replace("// appended by impact_reg_konfai parameter_overrides", "// the metric weights and the sample checks of this run")
    header = (f"// LINC, {what}.\n// Generic_Rigid_BSpline's Parameters_BSpline.txt with this run's settings written in: Mattes MI + bending energy "
              f"(weight 100); make_linc_presets.py checks that elastix receives the same map as from the documented --set command.\n\n")
    (d / mapname).write_text(header + text + "\n")
    net["parameter_maps"], net["max_iterations"], net["final_grid_spacing"], net["spatial_samples"], net["parameter_overrides"] = [mapname], 0, 0.0, 0, []
    yaml.dump(documented, d / "Prediction.yml")
    app = json.loads((HUB / "app.json").read_text())
    app.update(display_name=f"LINC: {what}", short_description=f"elastix B-spline, Mattes MI + bending energy (weight 100): {what}.",
               description=f"elastix B-spline, Mattes MI + bending energy (weight 100), every setting in {mapname}: {what}.")
    (d / "app.json").write_text(json.dumps(app, indent=4))
    shutil.copy(HUB / "requirements.txt", d / "requirements.txt")

    preset = yaml.load(d / "Prediction.yml")
    got = entries(received(d, preset["Predictor"]["Model"]["RegistrationNet"]))
    plain = json.loads(json.dumps(preset))
    for cfg in (plain, documented_plain):
        for key in ENGINE_KEYS:
            cfg["Predictor"]["Model"]["RegistrationNet"].pop(key)
    map_diff = sorted(set(got) ^ set(expected))
    print(f"{name}: map entries {'identical' if got == expected else map_diff}, other Prediction.yml keys {'identical' if plain == documented_plain else 'DIFFER'}")
    assert got == expected and plain == documented_plain, name


TILED = {"LINC_FIREANTS_L3_TILED": ("APEX_FIREANTS_SYN_CC", "FireANTs SyN, cross-correlation"),
         "LINC_CONVEXADAM_L3_TILED": ("APEX_CONVEXADAM_MIND", "ConvexAdam on MIND features")}
for name, (base, engine) in TILED.items():
    d = PRESETS / name
    shutil.rmtree(d, ignore_errors=True); d.mkdir()
    for f in ("app.json", "Prediction.yml", "requirements.txt", "NOTICE"):
        if (PRESETS / base / f).exists():
            shutil.copy(PRESETS / base / f, d / f)
    LocalAppRepository._apply_config_overrides(repo, str(d / "Prediction.yml"), [PATCH + "patch_size=[192, 192, 192]", PATCH + "overlap=12%", COSINUS])
    what = f"level 3 (0.16 mm) in 192³ tiles, with {base}'s settings"
    app = json.loads((d / "app.json").read_text())
    app.update(display_name=f"LINC: {engine}, {what}", short_description=f"{engine}: {what}.",
               description=f"{engine}: {what}. The tiling is the Patch block of Prediction.yml.")
    (d / "app.json").write_text(json.dumps(app, indent=4))
    print(f"{name}: {base} with 192³ tiles, 12 % overlap, Cosinus")
