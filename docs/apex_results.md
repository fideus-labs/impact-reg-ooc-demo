# Displacement fields as NGFF RFC-5: APEX CONNECTS pair and LINC out-of-core demo

The two folders hold the same kind of result:
- **`apex/`**: the competition's DTI FA registered onto its PS-OCT retardance slide deck, for `subject_v` and `subject_m`, with three methods.
- **`linc/`**: the out-of-core LINC demo, one result per stage that `linc/README.md` describes.

Each result is:
- a displacement field, stored as an OME-Zarr (NGFF RFC-5);
- the warped moving image;
- for APEX, a set of overlays.

```
apex/subject_v_VB/   Fixed.mha  Moving.mha  overlays.png  checkerboard.png
                     Moved_<method>.mha          x 3
                     DVF_<method>.ome.zarr.zip   x 3
apex/subject_m_VB/   (same)
linc/                README.md
                     <stage>.ome.zarr.zip        displacement field   x 7
                     <stage>_Moved.ome.zarr.zip  warped moving image  x 7
```

25.3 GB in total. The largest file is `linc/linc_stage5_zoom1.ome.zarr.zip` (8.8 GB).

## apex/: FA onto retardance

### Grids

| image | native grid | registration grid |
|---|---|---|
| `dti_FA` (moving) | 0.405 × 0.405 × 0.4 mm | through the stage-1 affine onto the grid below |
| `Ret_slide_deck` (fixed) | 0.2 × 0.11 × 0.11 mm (subject_v), 0.11 × 0.2 × 0.11 mm (subject_m) | smoothed, resampled to 0.4 mm isotropic |

The registration grid is 123 × 175 × 113 voxels for subject_v and 150 × 164 × 113 for subject_m (x × y × z), at 0.4 mm, the FA's resolution.

### Stage 1

1. The 48 axis permutations and flips of the FA are scored by mutual information at 1.6 mm.
2. The best three are refined by an affine (Mattes MI, 0.8 then 0.4 mm), and the best refined one is kept.

subject_v needs a mirror (negative affine determinant); subject_m does not. subject_m's retardance is normalised slide by slide.

### Stage 2

Every method registers the 0.4 mm pair, with the retardance tissue mask widened by 1 mm as the fixed mask. No run is tiled.

| method | setup |
|---|---|
| `elastix_MI_bending` | B-spline, Mattes MI + bending energy (weight 100), 4 resolutions, final grid 2 mm |
| `FireANTs_SyN_CC` | symmetric diffeomorphic SyN, local CC (kernel 5), scales 4/2/1, gradient smoothing σ 1.5, warp smoothing σ 1.0 |
| `ConvexAdam_MIND` | ConvexAdam on MIND R2D2 features at 0.8 mm, L1, grid spacing 2, 150 iterations |

### Files in each subject folder

| file | content |
|---|---|
| `Fixed.mha` | retardance on the 0.4 mm grid |
| `Moving.mha` | FA after the stage-1 affine, same grid |
| `Moved_<method>.mha` | FA after the method, same grid |
| `DVF_<method>.ome.zarr.zip` | the method's displacement field |
| `overlays.png` | the retardance in grey with two white-matter contours: the retardance's own (cyan) and the FA's (yellow), after the stage-1 affine alone and after each method; aligned where the two lines coincide. Top row: the grid plane through the tissue centre. Bottom row: the 24 mm box where the FireANTs field moves the FA most in that plane. The two modalities do not outline exactly the same structures: judge the tract edges. |
| `checkerboard.png` | squares of the retardance (grey) alternating with squares of the FA (light amber), after the stage-1 affine alone and after each method; aligned where structures run on across the square edges. Each image is scaled between its 5th and 99.5th percentile inside the retardance tissue, for display only. Top row: 3.2 mm squares on the plane through the tissue centre. Bottom row: 2.4 mm squares in the same 24 mm box as `overlays.png`. |

### Results

All measures are taken on the images of the 0.4 mm grid; the competition gives no evaluation criterion.

| subject | method | MI | residual, MI / gradients | tissue folded | field median / max |
|---|---|---|---|---|---|
| subject_v | stage-1 affine only | 0.489 | 1.26 / 1.26 mm | | |
| subject_v | elastix_MI_bending | 0.702 | 0.40 / 0.40 mm | 0 % | 1.24 / 5.04 mm |
| subject_v | FireANTs_SyN_CC | 0.726 | 0.40 / 0.40 mm | 0 % | 0.92 / 2.73 mm |
| subject_v | ConvexAdam_MIND | 0.399 | 0.57 / 0.40 mm | 1.6 % | 1.90 / 13.49 mm |
| subject_m | stage-1 affine only | 0.257 | 1.13 / 1.13 mm | | |
| subject_m | elastix_MI_bending | 0.355 | 0.57 / 0.40 mm | 0 % | 0.98 / 3.83 mm |
| subject_m | FireANTs_SyN_CC | 0.389 | 0.40 / 0.40 mm | 0 % | 0.57 / 2.11 mm |
| subject_m | ConvexAdam_MIND | 0.295 | 0.57 / 0.40 mm | 1.6 % | 1.71 / 14.54 mm |

- **MI:** mutual information between retardance and warped FA, inside the retardance tissue.
- **Residual:** median over 12 mm windows on the three mid-planes (windows at least 80 % tissue). Each window's residual is the in-plane shift of the warped FA (±2 mm, 0.4 mm steps) that best matches the retardance. "MI" judges the match by mutual information; "gradients" by normalised gradient fields, which no method optimises. Medians move in 0.4 mm steps.
- **Tissue folded:** fraction of tissue voxels where the Jacobian determinant of x + u(x) is ≤ 0.
- **Field:** displacement magnitude inside the tissue.

### Field convention

`u(x)` is in mm, in the physical space of the `.mha` files. The point x of the fixed grid corresponds to x + u(x) in `Moving.mha`, so `Moved(x) = Moving(x + u(x))`.

## linc/: the out-of-core demo

`linc/README.md` is the demo's own README: the mechanism, the commands and the measurements. Its commands write to `out/<run>`; the matching zips here are:

| zip stem | section of `linc/README.md` | run folder | grid (x × y × z) | spacing |
|---|---|---|---|---|
| `section_stage2` | Stage 2 — deformable, locked to the plane | `out/plane_mind2dR2D2_mi_b0.1` | 1689 × 1493 × 8 | 0.025 × 0.025 × 0.4 mm |
| `section_stage3_native_tiled` | Stage 3 — native, tiled | `out/native_tiled_r2d2_mi_b100_m` | 2048 × 2048 × 8 | 0.003 × 0.003 × 0.4 mm |
| `linc_stage2_coarse` | stage 2: the coarse deformable, whole hemisphere | `out/linc_coarse_bend100` | 474 × 474 × 533 | 0.32112 mm |
| `linc_stage3_tiled` | stage 3: the same command, in tiles | `out/linc_tiled3_bend100` | 947 × 947 × 1065 | 0.16056 mm |
| `linc_stage4_native_tiled` | Stage 4 — native, tiled, one window of the 908 GiB store | `out/linc_native_bend100` | 768 × 768 × 768 | 0.02007 mm |
| `linc_flash` | Structure by structure: the FLASH T2* | `out/linc_flash_bend100` | 474 × 474 × 533 | 0.32112 mm |
| `linc_stage5_zoom1` | Stage 5 — zoom 1 on the overview | `out/zoom1_level1_bend100` | 1024 × 1024 × 896 | 0.04014 mm |

- **Fields:** same convention as APEX.
- **Stage-5 warped image:** the run wrote only its field, so the image was computed afterwards from that field with impact-reg's own moved-image step. The same step reproduces the stage-2 warped image exactly.

## Formats

| file | layout |
|---|---|
| `DVF_*.ome.zarr.zip`, `linc/<stage>.ome.zarr.zip` | NGFF 0.6rc0, Zarr v3. The component axis is typed `displacement`; a `displacements` coordinate transformation maps `physical` onto itself through `scale0/image`. Array (3, z, y, x), float64, components ordered z, y, x. |
| `linc/<stage>_Moved.ome.zarr.zip` | NGFF 0.4, Zarr v2. Array `scale0/image` (1, z, y, x), float32. |
| `*.mha` | MetaImage, float32 |

Each zip is uncompressed: the chunks inside are already compressed. Its root is the store root.

## Opening

Directly from the zip with zarr-python 3:

```python
import zarr
store = zarr.storage.ZipStore("apex/subject_v_VB/DVF_FireANTs_SyN_CC.ome.zarr.zip", mode="r")
field = zarr.open_group(store, mode="r")["scale0"]["image"]   # (3, z, y, x), mm, components z, y, x
```

Or unzip into a store directory (`unzip DVF_FireANTs_SyN_CC.ome.zarr.zip -d DVF_FireANTs_SyN_CC.ome.zarr`) and read it with any NGFF reader. With KonfAI, as a SimpleITK transform:

```python
import SimpleITK as sitk
from konfai.utils.ITK import read_displacement_field
field = read_displacement_field("DVF_FireANTs_SyN_CC.ome.zarr")   # vector image, components in ITK order (dx, dy, dz)
transform = sitk.DisplacementFieldTransform(field)
```
