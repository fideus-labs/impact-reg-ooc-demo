# Out-of-core multimodal registration with KonfAI and IMPACT

One command registers two volumes that do not fit in memory, with the engine of your choice (elastix, FireANTs,
ConvexAdam) and the loss of your choice, including **IMPACT**: registration on the feature maps of a segmentation model.
Shown at the APEX / BRAIN CONNECTS hackathon (UT Austin, September 2026) on a 908 GiB synchrotron X-ray hemisphere and on
the competition's DTI / PS-OCT pair.

```bash
pip install impact-reg-konfai "konfai[omezarr,monitoring,s3]" itk-impact
impact-reg-konfai register FireANTs_SyN -f Fixed.ome.zarr -m Moving.ome.zarr -o out --gpu 0
```

## How it works, in four steps

**1. The volume is cut into patches.** KonfAI never loads the volume. It reads one region at a time from the OME-Zarr store
(on disk or on S3): the patch and a halo around it, the overlap with its neighbours. Two settings, `patch_size` and
`overlap`; leave the patch size out and the whole volume goes through in one pass.

![Step 1: the volume, cut into patches](figures/step1_patches.png)

**2. The backend registers one patch.** A fixed patch and a moving patch go in, one patch of displacement comes out.
Behind that contract, elastix, FireANTs or ConvexAdam: same patches in, same field out, so you change the engine by
changing the preset name. What differs is memory; with tiles, that is a matter of tile size.

![Step 2: the backend](figures/step2_backend.png)

**3. The overlaps are blended.** A patch is right in its centre and worst at its edges. Its weight fades to zero before
its edge while the neighbour's rises, and at every voxel the weights sum to one: a partition of unity. On a known
deformation, the field's step at the tile planes is the same as anywhere else: no seam.

![Step 3: blend](figures/step3_blend.png)

**4. The field is written slab by slab**, as the tiles complete: an OME-Zarr store with an NGFF RFC-5 `displacements`
transformation (or an ITK transform file that 3D Slicer opens), and the moved image in the moving image's own format.
Inputs can be ITK images, DICOM, HDF5 or OME-Zarr.

![Step 4: write the field](figures/step4_write.png)

**The loss.** For every backend you choose what is compared: intensities (mutual information, cross-correlation), or,
with IMPACT, the feature maps a pretrained segmentation model computes. Two modalities share no intensity; the
features do. Below, a PS-OCT retardance and a registered DTI FA through MIND, TotalSegmentator, SAM 2.1 and Anatomix,
with one colour basis per model: the same colour is the same features.

![The loss: IMPACT feature maps](figures/loss_impact.png)

## Try it

```bash
# one pass, whole volume (fits in memory)
impact-reg-konfai register Generic_Rigid_BSpline -f Fixed.ome.zarr -m Moving.ome.zarr \
  --fixed-mask FixedMask.ome.zarr --moving-mask MovingMask.ome.zarr -o out --gpu 0

# the same, in tiles of 256³ with a 128-voxel overlap: the volume can be any size
impact-reg-konfai register Generic_Rigid_BSpline -f Fixed.ome.zarr -m Moving.ome.zarr \
  --fixed-mask FixedMask.ome.zarr --moving-mask MovingMask.ome.zarr -o out --gpu 0 \
  --set "Predictor.Dataset.Patch.patch_size=[256, 256, 256]" \
  --set "Predictor.Dataset.Patch.overlap=128" \
  --set "Predictor.outputs_dataset.DisplacementField.OutputDataset.patch_combine=Cosinus"

# another engine, another loss: a preset name
impact-reg-konfai register FireANTs_SyN …            # GPU SyN
impact-reg-konfai register ConvexAdam_Composite …    # ConvexAdam on MIND features
```

The output is `out/P000/DVF.ome.zarr`, a displacement field any NGFF reader can apply:

```python
import zarr
field = zarr.open_group("out/P000/DVF.ome.zarr", mode="r")["scale0"]["image"]   # (3, z, y, x), mm
```

## Run it with pixi

[pixi](https://pixi.sh) installs the whole stack (Python, PyTorch, KonfAI, impact-reg-konfai, ITKIMPACT, FireANTs,
ngff-zarr) from `pixi.toml` and runs the competition pair end to end. Put the competition's NIfTI files under
`data/input/<subject>/` (`Ret_slide_deck.nii.gz`, `dti_FA.nii.gz`, and the other slide decks alongside), then:

```bash
pixi run convert                          # every NIfTI -> data/ome-zarr/<subject>/<name>.ome.zarr, via the ngff-zarr CLI
pixi run prepare subject_v                # stage 1 (orientation, affine) and the 0.4 mm pair, as OME-Zarr stores
pixi run register FireANTs_SyN subject_v  # one preset on the pair -> out/subject_v/FireANTs_SyN/P000/
pixi run register-all                     # elastix, FireANTs and ConvexAdam on both subjects
```

Each task runs what it depends on: `register` prepares the pair, `prepare` converts the inputs. `register` takes a
preset name, a subject and the device (`--gpu 0` by default, `'--cpu 8'` for eight CPU workers);
`register-fireants`, `register-impact`, `register-convexadam` and `register-elastix` are the same with the preset
filled in. `pixi task list` shows them all. The presets come from the
[VBoussot/ImpactReg](https://huggingface.co/VBoussot/ImpactReg) Hugging Face repository on first use, and the
elastix binary from its GitHub release. A run leaves three things under `out/<subject>/<preset>/P000/`:
`Transform.ome.zarr`, the displacement field as an NGFF 0.6rc0 store with the RFC-5 `displacements` transformation
(the form [docs/apex_results.md](docs/apex_results.md) describes; the register task writes it from the preset's own
`Transform.h5`, which stays beside it for 3D Slicer), and `Moved.ome.zarr`, the FA moved onto the retardance grid.

There are two environments, because the backends disagree on PyTorch: ITKIMPACT pins torch 2.12 and the elastix-IMPACT
binary is built against LibTorch 2.8. `default` runs FireANTs and ConvexAdam; `elastix` runs elastix, which
`register-elastix` selects by itself (`pixi run -e elastix register Generic_Rigid_BSpline subject_v` is the long form).
Both torches are CUDA 12 builds from the PyTorch index; the CUDA 13 build PyPI serves for torch 2.12 cannot load the
elastix binary.

The pair is built by `pipeline/prepare_pair.py` the way the competition runs were: the 48 signed axis permutations of
the FA scored by mutual information at 1.6 mm, the best three refined by an affine, the retardance resampled to 0.4 mm
as the fixed image and the FA through the affine as the moving one, with the retardance tissue widened by 1 mm as the
fixed mask. The stores are in the frame the OME-Zarr metadata gives (identity direction), so a field from here is in
that frame rather than in the `.mha` files' LPS frame of [docs/apex_results.md](docs/apex_results.md).

## On the hackathon data

**The LINC hemisphere** ([DANDI 001278](https://dandiarchive.org/dandiset/001278)): 908 GiB of synchrotron X-ray at
20 µm, and a 24 MiB diffusion-MRI map to bring into it. The first stages run in memory on a coarse pyramid level; every
level down is eight times larger and runs with the same command, in tiles, on one 24 GB GPU.

![The pyramid levels against a 24 GB card](figures/linc_ladder.png)

The MRI map's residual shift goes from about 1 mm after the affine to zero after the deformable (left). A 4.3 µm zoom
scan placed on the 20 µm overview by hand (right, top) lands under the overview's own edges after 294 tiles (right,
bottom): median residual 0.45 mm → under 0.02 mm, one voxel.

<p><img src="figures/linc_level4.png" width="34%"> <img src="figures/linc_zoom_edges.png" width="62%"></p>

**The competition pair**: DTI FA onto PS-OCT retardance, on the FA's 0.4 mm grid, through the same pipeline with intensity
losses or IMPACT features. Gold: the retardance's white-matter outline, the same in every panel; grey: the FA, moving. After
the affine alone the tracts sit beside the lines; after each registration, under them. The zoom is the 24 mm box with the
largest correction.

![subject_v: the FA under the retardance's outline](figures/apex_subject_v.png)

| engine | residual (subject_v / subject_m) | tissue folded | largest displacement |
|---|---|---|---|
| elastix, MI + bending energy | 0.40 / 0.57 mm | 0 % | 5.0 / 3.8 mm |
| FireANTs, SyN + CC | 0.40 / 0.40 mm | 0 % | 2.7 / 2.1 mm |
| FireANTs, SyN + IMPACT (TotalSegmentator features) | 0.40 / 0.57 mm | 0 % | 2.6 / 2.5 mm |
| ConvexAdam, MIND | 0.57 / 0.57 mm | 1.6 % | 13.5 / 14.5 mm |

Residual: the median in-plane shift of the FA, in 12 mm windows, that best matches the retardance; 1.26 / 1.13 mm after
the affine alone. Our measures, the competition sets no criterion. Details, all numbers and the file formats are in
[docs/apex_results.md](docs/apex_results.md).

## In this repository

- [`talk/`](talk/) — the 6-minute talk (PowerPoint, with speaker notes).
- [`docs/linc_demo.md`](docs/linc_demo.md) — the full LINC write-up: every stage, its command, its cost and its checks.
- [`docs/apex_results.md`](docs/apex_results.md) — the competition pair: grids, methods, measures, file formats.
- [`linc/`](linc/), [`apex/`](apex/) — the scripts those two documents run (they keep the absolute paths of the workstation they ran on).
- [`pixi.toml`](pixi.toml), [`pipeline/`](pipeline/) — the environment and the tasks that run the competition pair from the NIfTI files, above.

The input data and the registration outputs are not in this repository: the datasets are not ours to redistribute. The
LINC hemisphere is public on DANDI and the scripts read it from there.

Built on [KonfAI](https://github.com/fideus-labs/KonfAI), [impact-reg-konfai](https://pypi.org/project/impact-reg-konfai/),
[ITKIMPACT](https://github.com/InsightSoftwareConsortium/ITKIMPACT) and [ImpactLoss](https://github.com/vboussot/ImpactLoss).
Fideus Labs.
