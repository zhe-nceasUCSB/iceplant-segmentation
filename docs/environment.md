# Environment setup

The pipeline is a plain Python project: a conda environment provides Python and
the geospatial libraries, and `pip` installs the pinned Python packages listed
in `requirements.txt`.

Reference platform: Linux x86_64 with an NVIDIA GPU (CUDA 12.x driver).
Everything except the CUDA build of PyTorch also works on a CPU-only machine,
it is only slower.

## 1. Install conda

Skip this step if `conda` is already available (`conda --version` prints a
version number).

Miniconda is the smallest distribution that provides `conda`:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
"$HOME/miniconda3/bin/conda" init bash
exec bash
```

The installer writes an activation snippet into `~/.bashrc`; `exec bash` reloads
the shell so that `conda` is on the `PATH`. Check the installation with:

```bash
conda --version
```

## 2. Create the environment

```bash
conda create -y -n iceplant python=3.12 gdal -c conda-forge
conda activate iceplant
```

`gdal` comes from conda-forge rather than pip because the GDAL command line
tools (`gdalinfo`, `gdal_translate`, `ogr2ogr`, ...) are the easiest way to
inspect and convert the rasters and shapefiles used by the pipeline, and the
conda build ships the shared libraries those tools need.

## 3. Install the Python packages

```bash
pip install -r requirements.txt
```

This pulls in, among others:

| package | pinned version | used for |
| --- | --- | --- |
| `torch`, `torchvision` | 2.3.1 / 0.18.1 | model training and inference |
| `lightning` | 2.5.2 | training loop, checkpointing, logging |
| `timm` | 1.0.17 | pretrained ResNet/ViT backbones |
| `albumentations` | 2.0.8 | data augmentation |
| `rasterio` | 1.4.3 | reading/writing GeoTIFF rasters |
| `geopandas` | 1.1.1 | reading label/grid shapefiles, writing predictions |
| `torchmetrics` | 1.4.0 | accuracy and confusion matrices |
| `tensorboard` | 2.17.0 | training curves and sample predictions |

On Linux, the `torch` wheel published on PyPI is built against CUDA 12.1 and
needs no extra step. If your driver requires a different CUDA runtime, install
PyTorch first from the matching index and then the rest of the requirements:

```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 4. Verify the installation

```bash
python -c "
import torch, rasterio, geopandas, timm, lightning
print('torch      ', torch.__version__, '| cuda available:', torch.cuda.is_available())
print('lightning  ', lightning.__version__)
print('timm       ', timm.__version__)
print('rasterio   ', rasterio.__version__)
print('geopandas  ', geopandas.__version__)
"
```

Expected output on the reference platform:

```
torch       2.3.1 | cuda available: True
lightning   2.5.2
timm        1.0.17
rasterio    1.4.3
geopandas   1.1.1
```

`cuda available: False` means training will fall back to the CPU. That is fine
for a smoke test but far too slow for a real run.

Then check that the pipeline itself imports and that a dataset config can be
loaded end to end (this needs the data of
[data preparation](data-preparation.md) in place):

```bash
python train.py \
    --exp-path smoke_test \
    --dataset config/naip_2024.yaml \
    --class "ovg,iceplant,others" \
    --backbone unet-resnet18 \
    --batch-size 4 \
    --workers 0 \
    --debug
```

`--debug` runs a single training batch and a single validation batch and then
exits, so it fails fast on a broken environment, an unreadable raster or a
mismatched shapefile. The first run downloads the pretrained backbone weights
from the Hugging Face hub into `~/.cache/huggingface`, so it needs network
access; later runs work offline.

Remove the throwaway experiment afterwards:

```bash
rm -rf experiments/models/smoke_test
```

## Notes

* Training expects patches of 128x128 px and, with the default batch size of
  64, roughly 12 GB of GPU memory for a `unet-resnet152` model. Lower
  `--batch-size` if you run out of memory.
* `--workers` controls the number of dataloader processes. Use something close
  to the number of physical CPU cores; `--workers 0` loads data in the main
  process, which is handy while debugging.
* Every raster listed in a dataset config is read into RAM once at startup
  (only the bands selected in the config). A 5000x6000 px 4-band scene is about
  120 MB, so a run combining several scenes should have a few GB of RAM
  available.
