# The brainlife.io App

What is here, and how it gets from this repository onto [brainlife.io](https://brainlife.io).

| file | role |
| --- | --- |
| [`../main`](../main) | the entrypoint brainlife runs: `#SBATCH` directives, then the container around `run.py` |
| [`run.py`](run.py) | reads `config.json`, picks the pixi environment for the preset, runs `impact-reg-konfai register`, writes the outputs and `product.json`; standard library only |
| [`app.json`](app.json) | the registration form's content: inputs, outputs, parameters (the preset menu is generated) |
| [`make_manifest.py`](make_manifest.py) | writes `app.json` from `presets/*/app.json`; rerun when a preset changes |
| [`../config.json.example`](../config.json.example) | the `config.json` brainlife writes, for a run by hand |
| [`../Dockerfile`](../Dockerfile) | the image `main` runs: both pixi environments, elastix-IMPACT, the feature models |
| [`../.github/workflows/container.yml`](../.github/workflows/container.yml) | builds and publishes the image to `ghcr.io/fideus-labs/impact-reg-ooc-demo` |

## How a run goes

1. brainlife clones the repository at the registered branch into a scratch directory on the resource, stages the
   input datasets beside it, and writes `config.json`: each input's `data.ome.zarr` path under the key of the form
   (`fixed`, `moving`, `fixed_mask`, `moving_mask`) and every parameter (`preset`, `device`, ...). An optional input
   the user left out is simply absent.
2. `./main` runs. It pins the image, forwards the scheduler's `CUDA_VISIBLE_DEVICES` and core count through
   singularity's clean environment (`-e`), adds `--nv` when the node has a driver, and runs
   `bash /opt/impact-reg-ooc-demo/env/default.sh python3 brainlife/run.py` in the container.
3. `run.py` finds the engine in the preset's `Prediction.yml` (`impact_reg_konfai.models.<engine>`) and picks the
   environment: `elastix` for elastix presets (torch 2.8, the LibTorch the binary links), `default` otherwise. It
   seeds the user's Hugging Face cache from the image's, so the feature models load without network, sets
   `KONFAI_IMPACTREG_REPO=presets/` and `KONFAI_APPS_INSTALL_REQUIREMENTS=0`, and runs

       impact-reg-konfai register <preset> -f <fixed> -m <moving> [--fixed-mask ..] [--moving-mask ..] \
           -o out --tmp-dir work (--gpu 0 | --cpu N) [--tta N] [--fields-only] [--set ..]...

   A `patch_size` in the form becomes `--set Predictor.Dataset.Patch.patch_size=[z, y, x]`, an `overlap`
   `--set Predictor.Dataset.Patch.overlap=..` (12% when a patch size is set and no overlap), and a preset that did not
   tile gets `patch_combine=Cosinus` for the blend; `extra_set` entries come last.
4. `pipeline/transform_to_ome_zarr.py` writes `out/P000/Transform.ome.zarr` from `Transform.h5` (`--stream` above
   2 GiB). `run.py` moves the stores into place, `transform/data.ome.zarr` (+ `Transform.h5`) and
   `moved/data.ome.zarr`, and writes `product.json`. `main` refuses to finish if the transform store is missing.

The intermediates (`work/`, `out/`) are volume-sized and live beside the outputs, on the resource's scratch, not on a
tmpfs `TMPDIR`. `#SBATCH --mem=48G --cpus-per-task=8 --gres=gpu:1 --time=04:00:00` fits the APEX pair and a level-3
LINC window on a 24 GB card; a resource admin can override them per resource.

## Publish the image

The compute node pulls `docker://ghcr.io/fideus-labs/impact-reg-ooc-demo:<tag>` (the `IMAGE` line of `main`) and
converts it to a Singularity image on first use. The workflow builds and pushes it on a `v*` tag, or by hand from the
Actions page (`workflow_dispatch`, with the tag to publish). It is about 20 GB: two torch stacks with their CUDA
runtimes. After the first push, make the package public on GitHub (Packages → the image → Package settings →
Change visibility), or singularity on the resource cannot pull it.

Locally, for a check before a push:

```bash
docker build -t ghcr.io/fideus-labs/impact-reg-ooc-demo:dev .
IMPACT_REG_OOC_IMAGE=docker-daemon://ghcr.io/fideus-labs/impact-reg-ooc-demo:dev ./main
```

When the stack changes (`pixi.lock`, a new elastix-IMPACT release, a new feature model), publish a new tag and bump the
`IMAGE` line in `main`, then a new App version on brainlife (a new `github_branch`/tag; brainlife pins Apps to a
branch or tag so a published run stays reproducible).

## Register the App on brainlife

brainlife registers an App through its web form ([docs](https://brainlife.io/docs/apps/register/)); there is no
manifest file to upload. [`app.json`](app.json) holds every value to enter. From brainlife's **Apps** page, **+**:

1. **Detail.** Name: *Out-of-core multimodal registration (KonfAI + IMPACT)*. GitHub repository:
   `fideus-labs/impact-reg-ooc-demo`, branch `main` (or a release tag). The description and the topics come from the
   GitHub repository's About box, so set those on GitHub (topics: `registration`, `ome-zarr`, `microscopy`,
   `multimodal`, `gpu`). Tick *requires GPU*.
2. **Inputs.** Four, all datatype `neuro/ome-zarr` (datatype id `6a99e5d7d90f157078ed790c`): `fixed`, `moving`
   (required), `fixed_mask`, `moving_mask` (optional). For each, map the datatype's file `zarr` (`data.ome.zarr`) to the
   config key of the same name as the input id: this is the `{"type": "input", "input_id": .., "file_id": "zarr"}`
   block in `app.json`'s `config`.
3. **Outputs.** `transform` and `moved`, both `neuro/ome-zarr`, with datatype tags `displacement_field` and
   `registered`; the App writes them to `transform/` and `moved/`, the directories named by the output ids, which is
   brainlife's default file mapping.
4. **Configuration parameters.** As in `app.json`'s `config`: `preset` (enum, options from the presets, default
   `APEX_FIREANTS_SYN_CC`), `device` (enum `gpu` | `cpu`), and the advanced ones: `patch_size`, `overlap`, `extra_set`
   (strings), `tta`, `cpu_workers` (numbers), `fields_only` (boolean). The form's configuration editor works on this
   JSON directly, so `app.json`'s `config` block can go in as it is.
5. **Resource.** Submit; then an administrator of a GPU-capable resource enables the App on it (brainlife runs an App
   only where it is enabled). For a first run, ask on brainlife's Slack for the App to be enabled on a shared GPU
   resource; the image must be public by then.

Then a test run: two `neuro/ome-zarr` datasets in a project (the competition pair from `pixi run prepare` uploads as
they are: the store root is `data.ome.zarr`), preset `APEX_FIREANTS_SYN_CC`, device `gpu`. Success is
`transform/data.ome.zarr` with an RFC-5 `displacements` transformation and `moved/data.ome.zarr` on the fixed grid,
and the task page shows `product.json`'s line: preset, engine, minutes, tiles.

Publishing (a DOI for the App) is one more click on brainlife once a run has succeeded. The reference to cite for the
platform is Hayashi et al. 2023, [brainlife.io: a decentralized and open-source cloud platform to support neuroscience
research](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10274934/).

## Not in this App

- **Preparing the pair** (`pipeline/prepare_pair.py`: orientation search, affine, the 0.4 mm grid, the tissue mask) is
  a separate step and would be a separate App, one that reads `neuro/microscopy/nifti` or `neuro/ome-zarr` and writes
  the pair's four stores; this App expects a pre-aligned pair, as the presets do.
- **Scoring** (`pipeline/score_pair.py`) likewise: an App on `fixed`, `moved` and the masks.
- brainlife has a `neuro/transform/h5` datatype, but it expects ANTs' `warp.h5`, `inverse-warp.h5` and `affine.txt`;
  this App's field is a `neuro/ome-zarr` dataset tagged `displacement_field` instead, with `Transform.h5` beside it.
