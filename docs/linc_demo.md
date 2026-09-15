# Out-of-core registration with impact-reg

Two examples, one mechanism.

**The headline pair — BRAIN CONNECTS / LINC, [dandiset 001278](https://dandiarchive.org/dandiset/001278),
subject `sub-Hb1`.** One left human hemisphere, imaged twice:

| volume | shape at level 0 | voxel | size |
|---|---|---|---|
| XPCT, synchrotron X-ray phase contrast | 7569 × 7569 × 8514 uint16 | 20.07 µm | **908 GiB** |
| b-ratio map, from ex vivo diffusion MRI | 176 × 438 × 264 | 0.4 mm | 24 MiB |

The contrasts come from unrelated physics — X-ray phase against the anisotropy of water diffusion —
which is the difficulty the field names, and the fixed volume does not fit anywhere. The dandiset
also ships the consortium's own alignment of that map into XPCT space, so the pipeline below has a
reference to be measured against.

## The mechanism

**Registration runs like a deep model.** The engine is handed one patch of the fixed image and one
of the moving image, and returns one patch of displacement. The patches are read straight from the
store, the outputs are blended on their overlaps, and the result is written slab by slab.

```
   the store on disk ──▶ read one region ──▶ [ model ] ──▶ patch of displacement
                                                                   │
                          the result on disk ◀── write slab ◀── blend the overlaps
```

| slot | what goes in it |
|---|---|
| **the model** | `elastix`, `FireANTs` or `ConvexAdam` — same two inputs, same two outputs |
| **the loss** | intensity, or a **feature map** from a pretrained network (MIND, TotalSegmentator, SAM) |
| **the patching** | `patch_size`, `overlap`, `patch_combine` |

## Running the LINC pair

```bash
python linc.py fetch --level 4      # one pyramid level of the 908 GiB store, plus the MRI maps
python stage1_linc.py --level 4     # 24 orientations, then the affine
```

The store carries axes, a scale and a half-voxel translation, and nothing about anatomy. The MRI
carries its own frame, in RAS, and the two differ by an axis permutation — which an affine started
from the identity never crosses. Stage 1 screens the 24 axis-aligned orientations at 1.28 mm, keeps
the winner and refines it at 0.32 mm.

```bash
# stage 2: the coarse deformable, whole hemisphere, masked and regularised, 2 min 06 s
python masks_linc.py data/linc/pair
impact-reg-konfai register Generic_Rigid_BSpline \
  -f data/linc/pair/Fixed.ome.zarr -m data/linc/pair/Moving.ome.zarr \
  --fixed-mask data/linc/pair/FixedMask.ome.zarr --moving-mask data/linc/pair/MovingMask.ome.zarr \
  -o out/linc_coarse_bend100 --gpu 0 \
  --set "Predictor.Model.RegistrationNet.parameter_maps=[Parameters_BSpline.txt]" \
  --set "Predictor.Model.RegistrationNet.final_grid_spacing=3.0" \
  --set "Predictor.Model.RegistrationNet.max_iterations=600" \
  --set "Predictor.Model.RegistrationNet.spatial_samples=8192" \
  --set 'Predictor.Model.RegistrationNet.parameter_overrides=[Registration="MultiMetricMultiResolutionRegistration", Metric="AdvancedMattesMutualInformation" "TransformBendingEnergyPenalty", Metric0Weight=1.0, Metric1Weight=100, RequiredRatioOfValidSamples=0.05]'

python linc.py fetch --level 3      # the next level down, 1.78 GiB
python stage3_linc.py --level 3 --out data/linc/pair3_bend    # the pair, through the stage-2 field
python masks_linc.py data/linc/pair3_bend

# stage 3: the same command, in tiles, masked and regularised, 8 min 27 s
impact-reg-konfai register Generic_Rigid_BSpline \
  -f data/linc/pair3_bend/Fixed.ome.zarr -m data/linc/pair3_bend/Moving.ome.zarr \
  --fixed-mask data/linc/pair3_bend/FixedMask.ome.zarr --moving-mask data/linc/pair3_bend/MovingMask.ome.zarr \
  -o out/linc_tiled3_bend100 --gpu 0 \
  --set "Predictor.Model.RegistrationNet.parameter_maps=[Parameters_BSpline.txt]" \
  --set "Predictor.Model.RegistrationNet.final_grid_spacing=2.0" \
  --set "Predictor.Model.RegistrationNet.max_iterations=200" \
  --set "Predictor.Model.RegistrationNet.spatial_samples=4096" \
  --set 'Predictor.Model.RegistrationNet.parameter_overrides=[Registration="MultiMetricMultiResolutionRegistration", Metric="AdvancedMattesMutualInformation" "TransformBendingEnergyPenalty", Metric0Weight=1.0, Metric1Weight=100, CheckNumberOfSamples="false"]' \
  --set "Predictor.Dataset.Patch.patch_size=[256, 256, 256]" \
  --set "Predictor.Dataset.Patch.overlap=12%" \
  --set "Predictor.outputs_dataset.DisplacementField.OutputDataset.patch_combine=Cosinus"
```

On the level-4 grid at 0.32 mm. Mutual information is what the engine maximised; the correlation
with the consortium's alignment of the same MRI map, and the brain Dice against it, are what it never saw:

| state | MI | correlation with the consortium's map | brain Dice |
|---|---|---|---|
| as the store and the NIfTI hand them over | 0.006 | — | — |
| best of the 24 orientations, rigid | 0.247 | — | — |
| after the affine | 0.301 | 0.764 | 0.933 |
| deformable, no mask, no regularisation | 0.352 | 0.818 | 0.956 |
| deformable, masked | 0.358 | 0.796 | 0.946 |
| **deformable, masked, bending penalty 100** | **0.335** | **0.888** | **0.955** |

The regularised run gives up a little mutual information and moves much closer to the consortium's
alignment: without the penalty, a 3 mm B-spline spends its freedom fitting intensities that the two
physics do not share. The penalty's weight was chosen on the per-structure check further down, among
four settings; this table played no part in the choice, and confirms it.

Read so far: **228 MiB of the 908 GiB** level 0, plus 1.78 GiB for level 3.

### A tile needs a mask

125 tiles of 256³ cover the hemisphere's bounding box, and 81 of them hold no tissue. The first
tiled run went without a mask. The tiles holding only a sliver of tissue then fitted the background:
their fields reached 69 mm, the MRI spread into the background (its support grew by a quarter), and
mutual information over the volume fell below its start. The same run with the mask:

| level 3, 0.16 mm | MI, whole volume | MI, inside the consortium's tissue | MRI support | largest displacement |
|---|---|---|---|---|
| start, after stages 1 and 2 | 0.348 | 0.386 | 0.113 | — |
| tiled, no mask | 0.328 | 0.481 | 0.139 | 69.5 mm |
| **tiled, masked** | **0.367** | **0.582** | **0.108** | **5.5 mm** |
| the alignment shipped with the dandiset | 0.339 | 0.315 | 0.111 | — |

The mask costs nothing where the tile is full of tissue. Tested on one tile alone, mutual information
inside the tissue went from 0.482 to 0.766 both with and without it. On a tile holding 0.5 % tissue it
decides the outcome: 0.000 and a 26 mm field without the mask, 0.341 and 1.8 mm with it. Two lines
of the command above go with it:

- **both masks as OME-Zarr.** Give only the fixed one and impact-reg fills the missing moving mask as
  `.mha`, which the zarr pair's dataset does not list.
- **`CheckNumberOfSamples="false"`.** A masked sliver at the edge of its tile can push most samples
  out of the tile on one large early step, and elastix stops the whole run on it: at 0.05 it still did,
  at 3.5 % valid samples. With the bending penalty holding the deformation, the check is switched off.

A tile whose fixed mask is empty gets a zero field instead of a registration, which is also why the
masked run is three times faster.

Mutual information alone is not a verdict on a pair like this: the consortium's own alignment scores
0.315 inside its own tissue. Stage 4 checks the chain against the consortium's map directly.

### And a bending penalty

The table above was run from an unregularised stage 2. From the regularised one, the same tiles, with
the mask and with or without the penalty chosen at stage 2, scored against the consortium's map:

| level 3, 0.16 mm | MI, whole volume | correlation with the consortium's map | largest displacement |
|---|---|---|---|
| start, after the regularised stage 2 | 0.332 | 0.888 | — |
| tiled, masked | 0.359 | 0.871 | 6.9 mm |
| **tiled, masked, bending penalty 100** | **0.340** | **0.929** | **2.5 mm** |

It is stage 2's lesson again. Masked but free, the tiles raise mutual information and drift away from
the consortium's alignment. With the penalty they move toward it, past their start.

## Stage 4 — native, tiled, one window of the 908 GiB store

```bash
python stage4_native_linc.py --size 768 \
  --coarse-field out/linc_coarse_bend100/P000/Transform.h5 \
  --fine-field out/linc_tiled3_bend100/P000/Transform.h5    # a 15.4 mm window of level 0, through every field so far
python masks_linc.py data/linc/pair0

# 125 tiles of 256³, masked and regularised, 14 min 04 s
impact-reg-konfai register Generic_Rigid_BSpline \
  -f data/linc/pair0/Fixed.ome.zarr -m data/linc/pair0/Moving.ome.zarr \
  --fixed-mask data/linc/pair0/FixedMask.ome.zarr --moving-mask data/linc/pair0/MovingMask.ome.zarr \
  -o out/linc_native_bend100 --gpu 0 \
  --set "Predictor.Model.RegistrationNet.parameter_maps=[Parameters_BSpline.txt]" \
  --set "Predictor.Model.RegistrationNet.final_grid_spacing=1.0" \
  --set "Predictor.Model.RegistrationNet.max_iterations=200" \
  --set "Predictor.Model.RegistrationNet.spatial_samples=4096" \
  --set 'Predictor.Model.RegistrationNet.parameter_overrides=[Registration="MultiMetricMultiResolutionRegistration", Metric="AdvancedMattesMutualInformation" "TransformBendingEnergyPenalty", Metric0Weight=1.0, Metric1Weight=100, CheckNumberOfSamples="false"]' \
  --set "Predictor.Dataset.Patch.patch_size=[256, 256, 256]" \
  --set "Predictor.Dataset.Patch.overlap=128" \
  --set "Predictor.outputs_dataset.DisplacementField.OutputDataset.patch_combine=Cosinus"

python evaluate_linc.py native
```

The window is read by the region: 864 MiB of the 908 GiB. The moving image reaches it through the
affine, the level-4 field and the level-3 field, each read only over the window's box (257 MiB of the
23 GB level-3 field). Mutual information on every 4th voxel:

Through the regularised stages 2 and 3, with the same tiles, masked, and with or without the bending
penalty, on every 4th voxel. The correlation is taken 4 mm in from the window's faces:

| native window, 20 µm | MI | correlation with the consortium's map | seams, y and x |
|---|---|---|---|
| start, after the regularised stages 2 and 3 | 0.772 | 0.929 | — |
| tiled, masked | 0.828 | 0.909 | 1.63 and 0.95 |
| **tiled, masked, bending penalty 100** | **0.809** | **0.955** | **1.02 and 0.72** |

The same lesson a third time. Masked but free, the native tiles raise mutual information, move away
from the consortium's alignment and leave a seam on y: 1.63 times the change elsewhere. With the
penalty they move toward it, past their start, and the boundaries carry no more change than the rest
of the field.

The runs below came before the regularisation, and they are where the grid and the scoring were
worked out:

| native window, 20 µm | whole window | 4 mm in from the faces |
|---|---|---|
| start, after stages 1 to 3 | 0.730 | 0.390 |
| tiled, 12 % overlap | 0.699 | 0.446 |
| tiled, overlap 128 | 0.779 | 0.533 |
| the alignment shipped with the dandiset | 0.715 | 0.495 |

**The tile grid has to fit the window.** Tiles are laid from the first voxel, and the last one is
clipped at the far face. At 12 % overlap on a 768 window, that last tile is 93 voxels thick on every
axis, and 37 of the 64 tiles were thin somewhere. Those tiles did not improve, and half of them got
worse. The 27 full tiles went from 0.429 to 0.521. An overlap of 128 makes the stride 128, so the last
tile ends exactly on the face. The seams then carry no more change than the rest of the field: 0.92 and
1.02 of it.

**Score the chain, not the window's Moved image.** The run's Moved image warps the 15 mm window it
was handed, so wherever the field points past a face it reads zeros that are not in the MRI.
`evaluate_linc.py native` resamples the source MRI through all four transforms instead.

Mutual information is what the engine maximised, so it is checked against something it did not see.
The consortium ships the same MRI map already aligned to the XPCT. Two alignments of one map can be
correlated directly:

| native window, correlation with the consortium's map | whole window | 4 mm in from the faces |
|---|---|---|
| start, after stages 1 to 3 | 0.785 | 0.760 |
| tiled, 12 % overlap | 0.779 | 0.653 |
| tiled, overlap 128 | 0.785 | 0.853 |

The run with full tiles moves the interior toward the consortium's alignment. The run with thin tiles
moved it away, while its mutual information still rose. Checked the same way after stage 2, the
correlation over the whole hemisphere is 0.819 and the brain Dice 0.957. In `figures/L6_native_contours.png`
the gold line is this chain and the cyan line is the consortium's alignment, both at one level of the
same map.

## Structure by structure: the FLASH T2* against the consortium's alignment

Mutual information and correlations are proxies. The dandiset also ships a segmentation of this
hemisphere's FLASH T2* volume twice: in the FLASH's own frame, and moved into XPCT space by the
consortium. The FLASH goes through the same stages; its segmentation is carried along with
nearest-neighbour sampling and compared with the consortium's, structure by structure.

The reference is therefore the consortium's registration, not anatomy drawn on the XPCT, and it is
not linear. The best affine through the centres of the 14 structures, fitted on the answer itself,
leaves 0.8 mm at the median and 1.7 mm at worst, and scores a mean Dice of 0.762. A deformable can beat
an affine against it.

```bash
python evaluate_linc.py brain                                    # FLASH at 0.24 mm, brain only
python stage1_linc.py --level 4 --moving data/linc/flash_brain_024.nii.gz --name flash
python evaluate_linc.py pair --name flash
python masks_linc.py data/linc/pair_flash
impact-reg-konfai register Generic_Rigid_BSpline \
  -f data/linc/pair_flash/Fixed.ome.zarr -m data/linc/pair_flash/Moving.ome.zarr \
  --fixed-mask data/linc/pair_flash/FixedMask.ome.zarr --moving-mask data/linc/pair_flash/MovingMask.ome.zarr \
  -o out/linc_flash_bend100 --gpu 0 \
  --set "Predictor.Model.RegistrationNet.parameter_maps=[Parameters_BSpline.txt]" \
  --set "Predictor.Model.RegistrationNet.final_grid_spacing=3.0" \
  --set "Predictor.Model.RegistrationNet.max_iterations=600" \
  --set "Predictor.Model.RegistrationNet.spatial_samples=8192" \
  --set 'Predictor.Model.RegistrationNet.parameter_overrides=[Registration="MultiMetricMultiResolutionRegistration", Metric="AdvancedMattesMutualInformation" "TransformBendingEnergyPenalty", Metric0Weight=1.0, Metric1Weight=100, RequiredRatioOfValidSamples=0.05]'
python evaluate_linc.py dice --name flash --field out/linc_flash_bend100/P000/Transform.h5
```

| Dice at 0.32 mm | orientation + affine | + deformable, masked | + deformable, masked, bending 100 |
|---|---|---|---|
| cerebral white matter | 0.765 | 0.823 | 0.866 |
| cerebral cortex | 0.735 | 0.769 | 0.798 |
| lateral ventricle | 0.837 | 0.895 | 0.911 |
| inferior lateral ventricle | 0.685 | 0.665 | 0.826 |
| cerebellum white matter | 0.622 | 0.664 | 0.787 |
| cerebellum cortex | 0.802 | 0.777 | 0.858 |
| thalamus | 0.754 | 0.723 | 0.885 |
| caudate | 0.748 | 0.794 | 0.845 |
| putamen | 0.733 | 0.100 | 0.880 |
| pallidum | 0.604 | 0.108 | 0.869 |
| 3rd ventricle | 0.369 | 0.334 | 0.633 |
| hippocampus | 0.708 | 0.472 | 0.830 |
| amygdala | 0.586 | 0.572 | 0.889 |
| accumbens | 0.286 | 0.269 | 0.798 |
| **mean of the 14** | 0.660 | 0.569 | **0.834** |

Without regularisation, the 3 mm B-spline improved the large structures and crushed the putamen and
the pallidum, whose contrast in X-ray phase barely differs from the white matter around them. With a
bending-energy penalty the deformation stays smooth enough to leave them in place, and every structure
improves on the affine.

The weight of that penalty was chosen on this table: 10 and 100 were tried, and a coarser 6 mm grid,
and 100 scored best. That makes 0.834 an optimistic number for this check. The choice was then
confirmed where it played no part: on the dMRI, the same setting raises the correlation with the
consortium's map from 0.818 to 0.888 (stage 2 above).

A masked elastix affine before the deformable was tried too. It lifts this table's affine to 0.783,
but on the dMRI it lowers the correlation from 0.764 to 0.724: the consortium's two alignments
disagree on scale, the FLASH shrunk by 4 % in volume and the dMRI not at all, and no single transform
can follow both. The regularised deformable reaches 0.834 without it, so it was left out.

Two things had to be right before any of this meant anything:

- **the FLASH masked to the brain.** The dMRI map the consortium ships is already zero outside the
  brain. The FLASH is not, and the container around the hemisphere carries signal of its own.
  Unmasked, the orientation screen kept the wrong one of two near-tied candidates (0.4563 against
  0.4548), the Dice after the affine was near zero, and the deformable later reached 30 mm. The mask is
  the consortium's own segmentation, widened by 1 mm.
- **the segmentation turned about the image's pivot.** Stage 1 orients the FLASH about the FLASH's
  own origin. The segmentation covers the same hemisphere from another corner, so it has to be turned
  about that same point, not its own.

The mean covers the 14 named left-hemisphere structures. The reference also carries right-hemisphere
labels, hypointensities and hypothalamic subunits a few voxels wide, which say nothing at 0.32 mm.

## The ladder

| level | voxel | shape | size |
|---|---|---|---|
| 4 | 321 µm | 474 × 474 × 533 | 0.22 GiB |
| 3 | 161 µm | 947 × 947 × 1065 | 1.78 GiB |
| 2 | 80 µm | 1893 × 1893 × 2129 | 14.2 GiB |
| 1 | 40 µm | 3785 × 3785 × 4257 | 113.6 GiB |
| 0 | 20 µm | 7569 × 7569 × 8514 | 908.5 GiB |

Going down the ladder changes the number of tiles and the wall clock. It does not change the
command, and it does not change the memory.

## The loss slot, tested

Same engine, same recipe: elastix B-spline on the level-4 pairs, masked, grid 3 mm, bending 100. Only
the loss changes, through a local preset, `presets/LINC_MIND`: the IMPACT metric with MIND R1D2 at 1.28,
0.96, 0.64 and 0.32 mm. The setting is chosen on the FLASH Dice among three (mutual information had
four), then checked on the dMRI:

```bash
scripts/mind_try.sh flash mind_mi     # VARIANT: mind | mind_bend10 | mind_mi ; PAIR: flash | dmri
```

| loss | FLASH, Dice of 14 structures | dMRI, correlation with the consortium | dMRI, brain Dice | run |
|---|---|---|---|---|
| affine only | 0.660 | 0.764 | 0.933 | — |
| **mutual information** | **0.834** | **0.888** | 0.955 | 2 min |
| MIND + mutual information | 0.815 | 0.886 | **0.960** | 10–12 min, 53 GB, CPU |
| MIND alone | 0.728 | — | — | 11 min |
| MIND alone, bending 10 / 1 / none | 0.673 / 0.683 / 0.692 | — | — | 11 min each |

- **MIND with mutual information comes within 0.02 on the FLASH and ties on the dMRI.** It does better
  on the cortex (0.814 against 0.798), the cerebellar cortex (0.878 against 0.858) and the hippocampus,
  and worse on the ventricles and the accumbens.
- **MIND alone moves the brain less.** It improves every structure on the affine, with a median
  displacement of 0.54 mm against 1.78 mm for mutual information. The penalty is not what holds it: with
  the weight at 10, 1 or none it moves 0.54, 0.60 and 0.71 mm, and scores 0.673, 0.683 and 0.692.
- **The elastix-IMPACT build installed here is the CPU one.** Run with `--cpu 1` and
  `ImpactUseMixedPrecision="false"`, since half precision has no 3D pooling on the CPU. The feature
  maps at 0.32 mm take 53 GB.
- **An earlier ConvexAdam + MIND run**, unmasked and unregularised, moved the tissue 2.8 to 5.3 mm away
  from the reference (tissue Dice 0.63 to 0.77, against 0.93 for the affine).
- **The generic MIND model is the only feature model tried.** The clinical models shipped with
  impact-reg are trained on CT and MR and are out of domain on X-ray phase contrast.

## What the files do not carry

Two facts that neither format writes down, and that silently changed results here:

- **Orientation.** The XPCT store declares axes and a scale; the MRI carries RAS. They differ by an
  axis permutation, which is why stage 1 screens 24 orientations instead of starting from identity.
- **Direction, once a pair is written to OME-Zarr.** The pairs go to disk with a spacing and an
  origin, and nothing else, so the engine registers them in a frame where x and y are not negated.
  Their moved images are right, since they share the grid. A displacement field carried to another
  grid is not: until `linc.field_in_store_frame` mirrors it back, it is zero everywhere in the store's
  frame. That bug was in this demo, and fixing it moved two numbers:

| start of the finer stage | field ignored | field in the right frame | consortium |
|---|---|---|---|
| level 3, 0.16 mm | 0.299 | **0.349** | 0.340 |
| native window, 20 µm | 0.522 | **0.616** | 0.715 |

## Reading the result

A red/green overlay lies on a pair like this: the contrasts are unrelated, so a correct alignment
puts red beside green rather than yellow on yellow. `figures/L3_contours.png` draws one image's
edges over the other instead, which is not ambiguous.

`figures/L7_qc_level4.png`, `L8_qc_level3.png` and `L9_qc_native.png` show each level before and after
its deformable, on two planes, in three columns:

- **XPCT | MRI checkerboard.** Two physics, so read edges: the brain's outline and the ventricles run on
  across the squares when the two are aligned.
- **this MRI over the consortium's.** The same map aligned twice, green for this pipeline and magenta
  for the consortium: grey where they agree, colour where they split.
- **checkerboard of the two MRI alignments.** Same contrast on both sides, so any disagreement is a step
  at the squares' edges.

`figures/L10_deep_nuclei.png` zooms on the putamen, pallidum, thalamus and caudate, in a 40 mm window
at 0.1 mm, after the affine and after the level-4 deformable. That deformable moves the MRI by
1.65 mm at the median, which is invisible on a whole-brain view and is the width of the pallidum here.
The outlines are the carried FLASH segmentation (gold) against the consortium's (cyan).

## Stage 5 — zoom 1 on the overview, X-ray against X-ray

The same hemisphere was also scanned at 4.257 µm (zoom 1, 602 GiB) and 2.201 µm (zoom 2, 1124 GiB). The
consortium placed each zoom on the overview by hand, one affine per acquisition, in its Neuroglancer
scene. That placement is the start here.

```bash
python zoom_linc.py rigid                                  # a rigid correction at 80 µm, 17 s
python zoom_linc.py pair --level 1 --size 1024 1024 896    # 40 µm over the zoom, 3.5 GiB an image
impact-reg-konfai register Generic_Rigid_BSpline \
  -f data/linc/zoom1_pair1/Fixed.ome.zarr -m data/linc/zoom1_pair1/Moving.ome.zarr \
  --fixed-mask data/linc/zoom1_pair1/FixedMask.ome.zarr --moving-mask data/linc/zoom1_pair1/MovingMask.ome.zarr \
  -o out/zoom1_level1_bend100 --gpu 0 \
  --set "Predictor.Model.RegistrationNet.parameter_maps=[Parameters_BSpline.txt]" \
  --set "Predictor.Dataset.Patch.patch_size=[256,256,256]" --set "Predictor.Dataset.Patch.overlap=128"   # + stage 4's bending overrides, grid 2.0
python zoom_linc.py check --field out/zoom1_level1_bend100/P000/Transform.h5
python zoom_linc.py detail --centre 49.1 54.9 67.6 --field out/zoom1_level1_bend100/P000/Transform.h5
```

- **The scene's matrix acts on physical coordinates**, its translation in units of 10.072 µm. Read as
  voxel counts instead, the zoom comes out 2.4 times too large and correlates with the overview at −0.04;
  read physically, at 0.39, with a peak that falls 1 mm to either side.
- **The check is one no run sees:** level 0 of the overview (20 µm) against level 2 of the zoom (17 µm),
  in eight windows of 4 mm on a lattice 9 mm either side of the zoom's centre. In each window the
  residual shift is read from the peak of the cross-correlation.

| placement of zoom 1 | correlation, median | residual shift, median | residual shift, max |
|---|---|---|---|
| by hand, from the scene | 0.423 | 0.45 mm | 0.77 mm |
| + rigid correction at 80 µm | 0.431 | 0.33 mm | 0.63 mm |
| **+ tiled deformable at 40 µm, 294 tiles, 35 min** | **0.723** | **< 0.02 mm** | **0.04 mm** |

The residual shift is read on the overview's 20 µm grid, so under 0.02 mm means within one of its voxels.
The rigid correction improves five windows and worsens three: the hand placement's error is not one
rigid motion. After the tiles, the correlation rises in seven windows and stays at 0.30 in the eighth,
whose shift still goes from 0.68 to 0.04 mm.

What the field holds, and where it is not checked:

- **Mostly a scale.** An affine fitted to the field more than 5 mm inside the zoom scales x by 0.8 %,
  y by 1.3 % and z by 1.8 %, and leaves 0.05 mm at the median. The hand placement carries no scale, so
  its error changes sign across the zoom. An affine instead of the rigid correction would take most of it.
- **Largest at the faces.** More than 10 mm in, the field never reaches 1 mm; within 5 mm of the zoom's
  faces, 10 to 15 % of it does, up to 6 mm. The eight windows sit 7 to 11 mm in, so the faces are not
  checked here.

`figures/Z1_windows.png` shows the eight windows as checkerboards at 20 µm. `figures/Z2_detail.png` shows
1.5 mm of window 1 at the zoom's own voxel. There, the rigid correction brings in a tissue edge that the
overview does not have, and the tiled deformable puts the edge where the overview has it.

The tiled run's assembly outgrew a 60 GB memory cap once, after all 294 tiles; it completed under 90 GB
with `--fields-only`, which skips the moved image this stage never reads.

Reading is the cost of this stage: the pair takes 4.0 GiB from the two stores, 27 minutes at the 2.2 MB/s of
this connection. `zoom_linc.box` keeps every box it reads, so a second run reads nothing.

---

## The second pair, at cellular scale

The same three stages, on the same consortium's earlier dataset: one Broca's area block from a human
brain, imaged by serial-section OCT and by light-sheet at 3 µm.

## The data

[DANDI 000026](https://dandiarchive.org/dandiset/000026), a human brain: one hemisphere imaged by
MRI, then blocked and imaged at cellular resolution. Subject `I48`, block `BrocaAreaS23`. Public
OME-Zarr stores on S3, all read here by bounded request:

| volume | shape at level 0 | voxel | size |
|---|---|---|---|
| OCT, serial section | 70 × 12441 × 14074 float32 | 3.0 µm | **50 GiB** |
| Light-sheet, NeuN | 207 × 10309 × 12261 uint16 | 3.6 µm | **31 GiB** |
| Light-sheet, Calretinin | 207 × 10309 × 12261 uint16 | 3.6 µm | **31 GiB** |

## Stage 1 — coarse affine, on a coarse level of the pyramid

```bash
python dandi.py overview --level 3      # one coarse level of each store
python stage1_coarse.py                 # the global affine, seconds
```

At 50 µm the whole 37 × 42 mm section is a few hundred pixels across, so a global affine by mutual
information costs seconds and never touches the native data.

It searches the four in-plane mirrorings first, and that is not a formality: **this section is
mirrored in x between the two microscopes**, and an affine started from the identity never crosses a
reflection. Nothing in the metadata says so. NGFF 0.4 carries axes, scale and translation and no
orientation at all; the OCT store smuggles a NIfTI-2 header into an extra array, and its sform is a
plain diagonal. The mirror has to be found, not read.

    as imaged         mutual information after the affine: 0.2954
    mirrored in x                                          0.4605   <- kept
    mirrored in y                                          0.2880
    mirrored in both                                       0.2862

## Stage 2 — deformable, locked to the plane

```bash
python stage2_plane.py --spacing 0.025        # one plane per modality, common grid, 25 µm

KONFAI_IMPACTREG_REPO=$PWD/presets impact-reg-konfai register OCT_LIGHTSHEET_MIND \
  -f data/dandi/plane/Fixed.ome.zarr -m data/dandi/plane/Moving.ome.zarr \
  --fixed-mask data/dandi/plane/FixedMask.ome.zarr --moving-mask data/dandi/plane/MovingMask.ome.zarr \
  -o out/plane_mind2dR2D2_mi_b0.1 --cpu 1
```

`presets/OCT_LIGHTSHEET_MIND` is elastix with the IMPACT metric: 2-D MIND features (R2D2) at 0.2, 0.1, 0.05
and 0.025 mm, weight 1, plus mutual information, weight 1, plus a bending penalty, weight 0.1; 800 iterations
per level, 16384 samples, a B-spline with 0.4 mm control points in the plane and 1000 mm in z. It takes 6 min on
the CPU (the elastix-IMPACT build here bundles a CPU LibTorch), and the preset reproduces the sweep's run to
the last digit of every score. The masks come from
`scripts/oct_masks.py`.

| the section at 25 µm | MI | light-sheet edge to OCT edge, median | tissue folded |
|---|---|---|---|
| affine | 0.458 | 819 µm | — |
| mutual information alone (`Generic_Rigid_BSpline`, grid 0.4 mm, 800 it) | 0.836 | 700 µm | 1.29 % |
| **MIND R2D2 + MI + bending 0.1** | 0.791 | **552 µm** | **0.10 %** |

MI alone scores higher on MI, the quantity it maximises. The two measures neither loss optimises agree against
it: its edges sit 150 µm further from the OCT's, and it folds thirteen times as much tissue. The edge distance
never reaches zero, because the two contrasts do not share every edge. The whole loss sweep is in
"Stage 2 of the section, the loss" below.

The last override is the one that matters. 1000 mm of control-point spacing in depth leaves the
B-spline no freedom there, which is right for this pair: the OCT images a 0.21 mm block face and the
light-sheet the 0.75 mm section cut off it, so the two share a plane, not a depth. A run left free
in z spends its freedom on noise — measured, it moved voxels 45 µm in z against 43 µm in plane and
the alignment got worse.

## Stage 3 — native, tiled

```bash
python stage2_native.py --x 23 --y 11 --size 2048 --depth 64 --moving-store NeuN --margin 4.0 --skip-fixed \
  --coarse-field out/plane_mind2dR2D2_mi_b0.1/P000/Transform.h5 --out data/dandi/cortex_r2d2
python stage2_plane.py --spacing 0.003 --window 23 11 6.1 \
  --moving-window data/dandi/cortex_r2d2 --out data/dandi/plane_native_r2d2
python scripts/oct_native_masks.py plane_native_r2d2          # the tissue, from the OCT

impact-reg-konfai register Generic_Rigid_BSpline \
  -f data/dandi/plane_native_r2d2/Fixed.ome.zarr -m data/dandi/plane_native_r2d2/Moving.ome.zarr \
  --fixed-mask data/dandi/plane_native_r2d2/FixedMask.ome.zarr \
  --moving-mask data/dandi/plane_native_r2d2/MovingMask.ome.zarr \
  -o out/native_tiled_r2d2_mi_b100_m --gpu 0 \
  --set "Predictor.Model.RegistrationNet.parameter_maps=[Parameters_BSpline.txt]" \
  --set "Predictor.Model.RegistrationNet.final_grid_spacing=0.3" \
  --set "Predictor.Model.RegistrationNet.parameter_overrides=[Registration=\"MultiMetricMultiResolutionRegistration\", Metric=\"AdvancedMattesMutualInformation\" \"TransformBendingEnergyPenalty\", Metric0Weight=1.0, Metric1Weight=100, FinalGridSpacingInPhysicalUnits=0.3 0.3 1000, RequiredRatioOfValidSamples=0.05, CheckNumberOfSamples=\"false\"]" \
  --set "Predictor.Dataset.Patch.patch_size=[8, 512, 512]" \
  --set "Predictor.Dataset.Patch.overlap=12%" \
  --set "Predictor.outputs_dataset.DisplacementField.OutputDataset.patch_combine=Cosinus"
```

`scripts/oct_native.sh RUN TAG` does the first two lines and `scripts/oct_native_tiles.sh TAG CONFIG...` the run.
The three `Patch` lines are the whole difference between a run that must hold the volume and a run that never
does. Reading the window costs **1.67 GiB out of 81 GiB**, 65 s: zarr fetches the chunks the box intersects and
no others, and the light-sheet box is cached under `data/dandi/cache/`, so a new stage-2 field only re-resamples
it. 25 tiles at 3 µm take about 4 min.

**The mask is not optional.** The window holds mounting medium in two sulci. A tile that holds some fits its
noise. Scored by `scripts/oct_native_score.py` (the start warped through each field by hand, MI on the middle
plane 100 voxels in from the faces, and the in-plane Jacobian):

| the window at 3 µm | MI at 3 µm | at 12 µm | at 48 µm | median displacement | folded |
|---|---|---|---|---|---|
| from stage 2 (MIND + MI) | 0.216 | 0.278 | 0.386 | — | — |
| tiles, no mask, no penalty | 0.213 | 0.275 | 0.392 | 86 µm | 3.62 % |
| tiles, masked, no penalty | 0.241 | 0.310 | 0.424 | 72 µm | 1.79 % |
| tiles, masked, bending 1 / 10 / 30 | 0.230 / 0.227 / 0.225 | 0.297 / 0.293 / 0.291 | 0.409 / 0.407 / 0.402 | 54 / 40 / 37 µm | 1.38 / 0.73 / 0.24 % |
| **tiles, masked, bending 100 (kept)** | **0.223** | **0.288** | **0.401** | **30 µm** | **0.18 %** |
| tiles, masked, MIND + MI + bending 1 (CPU, 35 min) | 0.226 | 0.292 | 0.397 | 46 µm | 0.12 % |

Without the mask the window got worse and broke into tile-shaped blocks
(`figures/P5_native_eye_r2d2.png`, `figures/P2_before_after_r2d2.png`). That was also true of the first
version of this stage, from the MI-only stage 2 (`out/native_tiled`: MI 0.239 → 0.225 at 3 µm,
0.413 → 0.400 at 48 µm). Its flicker is kept as `figures/P2_flicker_old_unmasked.gif`. With the mask, the MI
rises at every scale; the penalty trades a little of it for a field that does not fold, and at 100 the grey
matter keeps its shape in the three 1.5 mm crops while the light-sheet's tissue edge still lands on the OCT's.

The 25 µm section and the native window do not look at the same tissue. `stage2_plane.py` averages each
modality over its whole slab, and the window is the top 0.19 mm of each stack, so a field that is right on
average can be tens of micrometres off at the surface. That residual is what the tiles are for.

**The order matters.** Started from the affine alone, each tile is asked to fix a millimetre it cannot see
past its own 1.5 mm, neighbours answer differently, and the blend averages two disagreements into a block —
`figures/8_why_coarse.png` is that comparison. After stage 2 only a local residual is left.

## Does the tiling distort the result

```bash
python stage3_pair.py --amplitude 8     # the light-sheet against itself, deformed by a known field
python score.py out/pair_neun/P000/Transform.h5 --data data/dandi/pair_neun \
  --amplitude 8 --patch 512 --overlap 61
```

| measure | before | after |
|---|---|---|
| displacement over tissue | 20.8 µm | **5.6 µm** |
| worst 5 % | 30.0 µm | 16.9 µm |
| step at a tile boundary, y | — | 1.08 × |
| step at a tile boundary, x | — | 1.13 × |

A seam would show as a step in the assembled field at the tile planes. There is none: the change
there is the change everywhere else.

## What it cost

| run | grid | tiles | time | device memory |
|---|---|---|---|---|
| stage 2, the whole section, mutual information | 25 µm | 1 | 50 s | 2.1 GB |
| stage 2, the whole section, MIND + MI (kept) | 25 µm | 1 | 6 min | CPU |
| stage 3, the native window, masked, bending 100 | 3 µm | 25 | about 4 min | GPU |
| the known-deformation check, ConvexAdam | 3 µm | 25 | 3 min 29 s | 15.1 GB |
| the same, FireANTs, 512² tiles | 3 µm | 25 | out of memory | > 24 GB |

NVIDIA RTX PRO 5000, 24 GB. Engines differ in what they hold: FireANTs keeps its own pyramid on the
card and does not fit a 512² tile where elastix sits at 1.9 GB. Halving the tile is one line.

## Honest limits

At 3 µm the OCT shows speckle and the light-sheet shows individual neurons, and there is no ground truth for
this block. The masked tiles raise the mutual information at every scale, but modestly (0.386 → 0.401 at
48 µm), and mutual information between speckle and neurons is a weak judge. What carries the stage is the
eye, in the three 1.5 mm crops and the flicker: the light-sheet's tissue edge moves onto the OCT's, the grey
matter keeps its shape, and nothing folds. The known-deformation check above shows the tiling itself is
faithful; it is the light-sheet against itself, so it says nothing about the two modalities.

Two markers are also not a dense correspondence: NeuN and Calretinin were imaged on the same grid,
but they stain different cells, so at 3 µm the same run that brings 21 µm down to 5.6 µm on NeuN
against itself leaves 25 µm across the two markers.

## Swapping the model

```bash
impact-reg-konfai register FireANTs_SyN         ...   # GPU SyN, Riemannian Adam
impact-reg-konfai register ConvexAdam_Composite ...   # coupled-convex + Adam, MIND features
impact-reg-konfai register MR_CT_TS             ...   # elastix, TotalSegmentator features
```

Several presets in one command are ensembled: their displacement fields are averaged, and their
spread is an uncertainty map.

## Also runs in 3D Slicer

The same presets drive the [SlicerImpactReg](https://github.com/vboussot/SlicerImpactReg)
extension: pick fixed, moving and preset, press Run.

## Install

```bash
python -m pip install "impact-reg-konfai" "konfai[omezarr,monitoring,s3]" itk-impact
```

Add `fireants` for the FireANTs presets. A GPU is not required, only faster.

## Files

| file | what it does |
|---|---|
| `linc.py` | reads a pyramid level, or a window, of the 908 GiB XPCT store, plus the MRI maps |
| `stage1_linc.py` | the 24-orientation screen, then the affine |
| `stage3_linc.py` | builds the next level down, starting from the coarse field |
| `masks_linc.py` | the masks a tiled deformable needs: the tissue, widened by 1 mm |
| `stage4_native_linc.py` | one window of level 0, the moving image carried through every field so far |
| `evaluate_linc.py` | the native window's score and seams, and the per-structure Dice |
| `zoom_linc.py` | zoom 1 on the overview: the rigid correction, the tiled pair, the 20 µm check, the 4.3 µm detail |
| `scripts/` | the run scripts, sweeps and diagnostics behind the tables: `*_try.sh` per stage, `diag*.py`, `disagreement_regions.py` (figures L12) |
| `presets/LINC_MIND` | a local impact-reg preset: elastix B-spline with the IMPACT metric and MIND, used by `scripts/mind_try.sh` |
| `presets/OCT_LIGHTSHEET_MIND` | the section's stage 2: MIND R2D2 + mutual information + bending 0.1, locked to the plane |
| `presets/SECTION_MIND`, `presets/SECTION_MIND3D`, `presets/NATIVE_MIND` | the bases the loss sweeps copy and edit (2-D and 3-D MIND on the section, 2-D MIND on the 3 µm tiles) |
| `scripts/oct_loss.sh`, `scripts/oct_score.py`, `figures_candidates.py` | the section's loss sweep, its score (folds, edges, MI) and the four-window sheets |
| `scripts/oct_native.sh`, `scripts/oct_native_tiles.sh`, `scripts/oct_native_score.py` | stage 3 through a chosen stage-2 run: the native window (light-sheet box cached), the tiles with a given loss, and their score |
| `figures_linc.py` | the figures of the LINC pair |
| `video_linc.py` | the LINC video |
| `dandi.py` | reads a coarse level, or a native window, from the public stores |
| `stage1_coarse.py` | the global affine, mirror search included |
| `stage2_plane.py` | builds the in-plane pair, at any grid, optionally through a coarse field |
| `stage2_native.py` | cuts one native window out of each store, through the affine and a coarse field |
| `stage3_pair.py` | builds the checkable pair: one known deformation |
| `score.py` | the residual against the known field, and the seam check |
| `figures.py` | the figures under `figures/` |
| `video.py` | the video under `video/` |
| `deck/index.html` | the deck: the short story, 13 plates, the LINC hemisphere then the OCT / light-sheet pair |
| `deck/full.html` | the long deck, 27 plates, with the checks against the consortium and the technical choices |
| `figures_demo.py`, `figures_motion.py`, `figures_landmarks.py`, `figures_morph.py`, `figures_zoomflicker.py`, `figures_pair.py` | the short deck's figures: the displacement maps, the four landmark crops with their locator, the affine-to-deformable morph, the zoom snapping onto the overview |
| `figures_deep.py`, `figures_compare.py`, `figures_arrows.py`, `figures_grid.py` | other views of the same move, not in the deck: the deep-nuclei contours, both contours on one image and a two-frame flicker, arrows, a warped grid |
| `video_linc.py --short` | the 63 s cut of the video for the short deck |

### Stage 2 of the section, measured on the images (2026-09-14)

The edge overlays and the "edges apart" numbers above do not measure alignment across these two modalities. At
the best result only 16 % of the light-sheet's edges have an OCT edge within 50 µm, and 61 % of them are the
tissue's boundary with the mounting medium, which the OCT barely shows. Shifting the light-sheet by 100 µm moves
the median edge distance from 461 to 500 µm only.

`scripts/oct_residual_shift.py` measures the residual on the images themselves: in 22 windows of 6 mm inside the
tissue, the in-plane shift of the light-sheet, within ±1 mm in 50 µm steps, that maximises mutual information
with the OCT. An aligned window peaks at zero.

| run | median | 90 % under | max | windows at zero |
|---|---|---|---|---|
| affine | 1008 µm | 1247 µm | 1414 µm | 0 % |
| mutual information alone, grid 0.4 mm | 0 µm | 336 µm | 1281 µm | 68 % |
| MIND + MI + bending 0.1, grid 0.4 mm | 50 µm | 1151 µm | 1312 µm | 45 % |
| **the same, second pass on a 0.2 mm grid** | **0 µm** | **71 µm** | **500 µm** | **77 %** |

By this measure the single MIND pass is worse than mutual information alone: four windows stay about 1 mm off.
The second pass is the best result, with every window but two under 100 µm. "Zero" means under the 50 µm step.
The judge is mutual information, which the second pass also optimises, so this favours it somewhat; the affine
row shows the measure does detect misalignment. The window around the opened sulcus (window 3) holds too little
tissue for a 6 mm box and is not in the table.

### Stage 2 of the section, the loss (2026-09-14)

The schedule stays fixed (four levels, grid 3.2 → 0.4 mm, locked to the plane); only the loss changes.
`scripts/oct_loss.sh` runs each configuration, `scripts/oct_score.py` scores it on three counts, in this
order: how much tissue the field folds (in-plane Jacobian at or below zero, 0.5 mm in from the edge), the
median distance from the light-sheet's edges to the OCT's, and mutual information. `figures_candidates.py`
draws each run in the same four 12 mm windows as the deck.

| loss (all with the tissue mask unless noted) | MI | edges apart, median | tissue folded |
|---|---|---|---|
| affine | 0.458 | 819 µm | — |
| MI, no mask, 800 it (the first deck run) | 0.836 | 700 µm | 1.29 % |
| MI, no mask, 3000 it, 32768 samples | 0.943 | 728 µm | 6.26 % |
| MI + bending 0.01 / 0.05 / 0.2 | 0.855 / 0.797 / 0.748 | 752 / 770 / 767 µm | 3.07 / 1.06 / 0.24 % |
| normalised correlation on the inverted light-sheet + bending 0.05 | 0.647 | 656 µm | 8.60 % |
| MIND 3-D (0.4 mm in z) + MI + bending 0.05 | 0.808 | 600 µm | 0.47 % |
| MIND 2-D R1D2 + MI + bending 0.05 | 0.795 | 594 µm | 0.20 % |
| MIND 2-D R1D2 alone + bending 0.05 | 0.503 | 775 µm | 0.00 % |
| MIND 2-D R1D2 + MI × 0.3 + bending 0.05 | 0.643 | 722 µm | 0.04 % |
| MIND 2-D R1D2 + MI + bending 0.02 | 0.801 | 633 µm | 0.59 % |
| MIND 2-D R1D2 + MI + bending 0.05, 2000 it | 0.820 | 555 µm | 0.77 % |
| MIND 2-D R1D1 + MI + bending 0.05 | 0.790 | 615 µm | 0.40 % |
| MIND 2-D R1D2 + MI × 2 + bending 0.05 | 0.822 | 564 µm | 0.86 % |
| MIND 2-D R2D2 + MI + bending 0.05 | 0.818 | 548 µm | 0.66 % |
| **MIND 2-D R2D2 + MI + bending 0.1 (kept)** | 0.791 | 552 µm | **0.10 %** |
| MIND 2-D R2D2 + MI × 2 + bending 0.1 | 0.833 | 550 µm | 0.50 % |

What the sweep says:

- **MIND moves the edges, MI moves the tissue.** MIND alone barely leaves the affine; lowering the MI weight
  loses most of the gain. Together they bring the edges 150 µm closer than MI alone.
- **The penalty is what stops the folds, and 0.1 is enough.** Below it the field folds small blobs deep in
  the white matter, where neither image has contrast to hold it; the MI-only run folds at the same places.
- **Longer MI runs buy MI with folds.** 3000 iterations raise the MI to 0.94 and fold 6 % of the tissue.
- Correlation on the inverted light-sheet folds the most and is discarded.

A 2-D model under elastix-IMPACT's Static mode still takes three voxel sizes per model (the image
dimension), hence `voxel_size: [v, v, 0.4]` in the presets; with two it fails to parse.

