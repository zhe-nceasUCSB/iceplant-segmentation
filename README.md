# Iceplant Segmentation

Semantic segmentation pipeline for mapping iceplant (*Carpobrotus edulis*, an
invasive coastal succulent) from aerial and satellite imagery — NAIP,
WorldView-2 and comparable sources. Given a GeoTIFF scene and hand-drawn
polygon labels, it trains a segmentation model (U-Net or ViT backbones) and
turns new imagery into a per-pixel class map, delivered both as a raster and as
GIS-ready polygons.

The default three-class setup is:

| class id | name | content |
| --- | --- | --- |
| 0 | `ovg` | other vegetation (trees, shrubs, grass) — also the fallback for unlabeled pixels |
| 1 | `iceplant` | iceplant, pure or mixed stands |
| 2 | `others` | everything else: bare ground, rock, roads, water, buildings |

Classes are configurable; the mapping from label attribute to class id lives in
the dataset config.

## How it fits together

```
GeoTIFF scene  ┐
label polygons ├─► config/*.yaml ─► train.py ─► experiments/models/<run>/version_N/
grid polygons  ┘                                       │  checkpoints + TensorBoard logs
                                                       ▼
                            new GeoTIFF ─► predict_raster.py ─► class raster + polygons
```

Detailed documentation lives in [`docs/`](docs):

* [Environment setup](docs/environment.md)
* [Preparing a dataset](docs/data-preparation.md)
* [Training](docs/training.md)
* [Prediction](docs/prediction.md)

## 1. Environment

Full instructions, including installing conda itself and the CUDA variants, are
in [docs/environment.md](docs/environment.md).

```bash
# install conda if you do not have it yet
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
"$HOME/miniconda3/bin/conda" init bash
exec bash

# create the environment and install the pinned dependencies
conda create -y -n iceplant python=3.12 gdal -c conda-forge
conda activate iceplant
pip install -r requirements.txt

# check the installation
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

GDAL comes from conda-forge so that the GDAL command line tools are available
next to the Python packages; everything else is installed with pip from
`requirements.txt` (PyTorch 2.3.1, Lightning 2.5.2, timm, rasterio, geopandas).
The reference platform is Linux x86_64 with an NVIDIA GPU; CPU-only machines
work but are much slower.

## 2. Data

A dataset is a GeoTIFF scene, a label shapefile, a grid shapefile and a small
YAML config that ties them together:

* **Raster** — 8-bit GeoTIFF in a projected CRS, any number of bands; the
  config picks which ones to use (e.g. R, G, B, NIR).
* **Labels** — polygons with a text attribute `class` (`Iceplant`,
  `Other Vegetation`, `Others`, ...). Pixels no polygon covers become class 0.
* **Grid** — rectangular cells tiling the scene, with an integer `Index` (cell
  id) and `Labeled` (`1` = fully labeled, usable). The grid decides which parts
  of the scene are used and which cells are held out for validation.
* **Config** — `config/*.yaml`, listing the three files plus the band
  selection, class mapping, patch size, stride and the validation cell ids.

```yaml
# config/naip_2024.yaml
name: naip-rgbir-2024
band: [1, 2, 3, 4]
data:
  - [
      "experiments/data/raster/2024NAIP_CojoTerrace_RGBIR.tif",
      "experiments/data/raster/2024NAIP_CojoTerrace_RGBIR_labels.shp",
      "experiments/data/raster/2024NAIP_CojoTerrace_RGBIR_grid.shp",
    ]
class_mapping: { Iceplant: 1, Mixed Iceplant: 1, Other Vegetation: 0, Tree: 0, Others: 2 }
patch_size: 128
stride: 32
val_grid_ids: [49, 56]
```

Imagery and labels are not tracked in this repository; put them under
`experiments/`, which is git-ignored. Field-by-field requirements, how to build
the grid, and how patches are cut out of it are in
[docs/data-preparation.md](docs/data-preparation.md).

## 3. Training

```bash
python train.py \
    --exp-path 20260818_unet-resnet152_naip_2024 \
    --dataset config/naip_2024.yaml \
    --class "ovg,iceplant,others" \
    --backbone unet-resnet152 \
    --epochs 100 \
    --batch-size 64
```

`--dataset` takes several configs (comma separated) to train one model across
sensors and years, and `--ckpt` initializes from an existing checkpoint to
fine-tune on a new year of imagery. Results are written to
`experiments/models/<exp-path>/version_N/`: `checkpoints/best.ckpt`,
`checkpoints/last.ckpt` and a TensorBoard log with loss and accuracy curves,
confusion matrices and sample predictions.

```bash
tensorboard --logdir experiments/models
```

Note that validation, checkpointing and early stopping only happen every
`--val-every` epochs (10 by default). Options, backbone choices, multi-dataset
runs, fine-tuning and troubleshooting are covered in
[docs/training.md](docs/training.md).

## 4. Prediction

```bash
python predict_raster.py \
    --exp-path experiments/models/20260818_unet-resnet152_naip_2024/version_0 \
    --image experiments/data/raster/2024NAIP_CojoTerrace_RGBIR.tif \
    --bands 1,2,3,4
```

The model slides over the raster in overlapping windows blended with a Gaussian
weight, and writes to `<exp-path>/predictions/`:

* `<image>_prediction.tif` — single-band raster of class ids, same CRS and
  geotransform as the input.
* `<image>_prediction.shp` — the same map as polygons with `class_id` and
  `class` attributes, ready for QGIS.

`--bands` must select the same number of bands, in the same order, as the model
was trained on. Window size, stride, checkpoint selection and output checks are
described in [docs/prediction.md](docs/prediction.md).

`run.sh` collects these commands as a worked example: train on one year,
predict on another, train across sensors, fine-tune from a checkpoint.

## Repository layout

| path | contents |
| --- | --- |
| `config/` | dataset configs (raster + labels + grid, bands, classes, splits) |
| `train.py` | training entry point |
| `predict_raster.py` | inference entry point |
| `raster_dataset.py` | patch extraction from raster + label/grid shapefiles |
| `model.py` | Lightning module: model, Dice loss, metrics, visualization |
| `ssegm/models/` | architectures: U-Net with pluggable backbones, ViT segmentation head |
| `run.sh` | example experiment scripts |
| `docs/` | detailed documentation |
| `experiments/` | imagery, labels and training runs (not tracked by git) |
