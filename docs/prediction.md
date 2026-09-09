# Prediction

`predict_raster.py` runs a trained model over a whole GeoTIFF with a sliding
window and writes the result both as a class raster and as polygons.

```bash
python predict_raster.py \
    --exp-path experiments/models/20260818_unet-resnet152_naip_2024/version_0 \
    --image experiments/data/raster/2024NAIP_CojoTerrace_RGBIR.tif \
    --bands 1,2,3,4
```

## Options

| option | default | meaning |
| --- | --- | --- |
| `--exp-path` | *required* | run folder containing `checkpoints/` (the `version_N` directory) |
| `--image` | *required* | GeoTIFF to classify |
| `--bands` | all bands | 1-based band indices to feed the model, comma separated |
| `--checkpoint` | `last.ckpt` | checkpoint file inside `<exp-path>/checkpoints` |
| `--patch-size` | `128` | sliding window size, must match the training patch size |
| `--stride` | `64` | window step; smaller = more overlap, slower, smoother |

`--bands` has to select exactly as many bands as the model was trained on, in
the same order. A model trained on `band: [1, 2, 3, 4]` of a NAIP scene expects
`--bands 1,2,3,4`; if the target raster stores the near infrared band somewhere
else, reorder the indices accordingly (e.g. `--bands 3,2,1,4`). Getting this
wrong produces a plausible looking but wrong map, so it is worth checking with
`gdalinfo` first.

The class names, the number of classes and the backbone are read from the
checkpoint, so they never have to be repeated on the command line.

## How it works

The raster is read into memory once, then a `patch_size` window slides over it
with the given `stride`. Every window is normalized exactly like a training
patch (scaled to [0, 1], then ImageNet mean/std with the RGB statistics cycling
over the extra bands), the model's softmax output is accumulated into a
per-class score map weighted by a Gaussian window, and the accumulated scores
are divided by the accumulated weights. Overlapping windows therefore blend
smoothly instead of leaving seams, and the final class of a pixel is the argmax
over the blended scores. The last row and column of windows are aligned to the
right/bottom edge of the raster so the whole scene is covered.

Runtime is dominated by the number of windows, which is
`(width / stride) * (height / stride)`. Halving the stride quadruples the work.

## Output

Both files land in `<exp-path>/predictions/` and are named after the input
raster:

```
experiments/models/<run>/version_0/predictions/
├── <image>_prediction.tif   # single band uint8 raster, pixel value = class id
└── <image>_prediction.shp   # the same map as polygons (+ .dbf/.shx/.prj/.cpg)
```

* The raster keeps the CRS and the geotransform of the input, is LZW
  compressed, and stores the class id (0, 1, 2, ...) per pixel.
* The shapefile has one polygon per connected region of equal class, with a
  `class_id` (integer) and a `class` (name from the checkpoint) attribute. Both
  load directly into QGIS; style them by `class`.

## A verified example

The 10-epoch check run of [training](training.md) applied to the scene it was
trained on and to another year:

```bash
python predict_raster.py \
    --exp-path experiments/models/test_naip2024_r34/version_0 \
    --image experiments/data/raster/2024NAIP_CojoTerrace_RGBIR.tif --bands 1,2,3,4

python predict_raster.py \
    --exp-path experiments/models/test_naip2024_r34/version_0 \
    --image experiments/data/raster/2022NAIP_CojoTerrace.tif --bands 1,2,3,4
```

```
Using checkpoint: experiments/models/test_naip2024_r34/version_0/checkpoints/last.ckpt
Using bands: [1, 2, 3, 4]
Processing patches: 100%|██████████| 92/92 [02:47<00:00,  1.82s/it]
Predicted raster saved to .../predictions/2024NAIP_CojoTerrace_RGBIR_prediction.tif
Shapefile saved to .../predictions/2024NAIP_CojoTerrace_RGBIR_prediction.shp
```

For the 5019 x 5919 px scene that is a 1.3 MB prediction raster and 8239
polygons, covering 15.6% of the scene with iceplant. Comparing the prediction
against the hand labels inside the held-out validation cells gives a pixel
accuracy of 0.853, matching the `val/acc_epoch` of the training run — a useful
sanity check that training and inference preprocess the imagery the same way.

## Checking a prediction

```bash
python -c "
import numpy as np, rasterio
with rasterio.open('experiments/models/test_naip2024_r34/version_0/predictions/2024NAIP_CojoTerrace_RGBIR_prediction.tif') as s:
    a = s.read(1)
u, c = np.unique(a, return_counts=True)
print({int(k): round(float(v) / a.size, 4) for k, v in zip(u, c)})
"
```

```
{0: 0.5472, 1: 0.1555, 2: 0.2973}
```

A class that covers implausibly much or nothing at all usually means the band
order of `--bands` does not match what the model was trained on, or that the
imagery was not stretched to 8 bit the same way as the training imagery.

## Troubleshooting

| symptom | cause |
| --- | --- |
| `FileNotFoundError: .../checkpoints/last.ckpt` | `--exp-path` points at the run folder instead of its `version_N` subfolder, or training ended before the first validation epoch |
| `RuntimeError: ... expected input ... channels` | `--bands` selects a different number of bands than the model was trained on |
| `Raster side of N px is smaller than the patch size` | the raster is smaller than `--patch-size` |
| everything predicted as one class | wrong band order, or imagery scaled differently from the training data |
