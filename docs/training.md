# Training

`train.py` builds a training and a validation set from one or more dataset
configs, trains a segmentation model on them and writes checkpoints and
TensorBoard logs under `experiments/models/`.

```bash
python train.py \
    --exp-path 20260818_unet-resnet152_naip_2024 \
    --dataset config/naip_2024.yaml \
    --class "ovg,iceplant,others" \
    --backbone unet-resnet152 \
    --epochs 100 \
    --batch-size 64 \
    --workers 8
```

## Options

| option | default | meaning |
| --- | --- | --- |
| `--exp-path` | `exp` | run name; artifacts go to `experiments/models/<exp-path>/version_N/` |
| `--dataset` | *required* | dataset config YAML(s), comma separated |
| `--class` | *required* | class names in class-id order, comma separated |
| `--backbone` | `unet-resnet152` | `<architecture>-<backbone>`, see below |
| `--epochs` | `50` | number of training epochs |
| `--batch-size` | `64` | patches per step |
| `--workers` | `8` | dataloader processes |
| `--val-every` | `10` | run validation, checkpointing and early stopping every N epochs |
| `--ckpt` | – | checkpoint to initialize the weights from (fine-tuning) |
| `--predict` | – | comma separated rasters to predict on once training finishes |
| `--accelerator` | `auto` | `auto`, `gpu` or `cpu` |
| `--seed` | `7829347` | random seed; the run is deterministic |
| `--debug` | off | run a single train and val batch, then exit |

### `--class`

The names are only used for labeling plots, confusion matrices and the `class`
attribute of the predicted shapefile, but their **order must match the class ids
of `class_mapping`** in the dataset config. With the reference configs
(`Iceplant: 1`, `Other Vegetation: 0`, `Others: 2`) the correct value is:

```
--class "ovg,iceplant,others"
```

The number of names also sets the number of output channels of the model.

### `--backbone`

`<architecture>-<backbone name>`, where the architecture is either `unet` or
`vit`:

* `unet-resnet18`, `unet-resnet34`, `unet-resnet50`, `unet-resnet152`, ... —
  U-Net with a `timm` ResNet encoder, pretrained on ImageNet. These accept any
  number of input channels; `timm` adapts the first convolution for 4-band
  imagery.
* `unet-unet_encoder` — the plain U-Net encoder from the original paper, no
  pretrained weights.
* `unet-densenet121`, `unet-densenet161`, `unet-densenet169`,
  `unet-densenet201` — torchvision encoders, 3 input channels only.
* `vit-vit_base_patch16_224.mae` — ViT encoder with a light convolutional
  decoder. The patch is resized to the ViT input size internally, so the output
  is coarser than the U-Net's.

`unet-resnet152` is the model the project has been using for its results.

### Combining datasets

```bash
python train.py \
    --exp-path mixed_wv2_naip \
    --dataset config/wv2_202308.yaml,config/naip_2022.yaml,config/naip_2024.yaml \
    --class "ovg,iceplant,others" \
    --epochs 200
```

Each config is loaded separately and its train/val patches are concatenated, so
the configs have to agree on the number of bands, the patch size and the class
ids. Every config keeps its own `val_grid_ids`, so each scene contributes its
own held-out cells.

### Fine-tuning

`--ckpt` loads the weights of an existing run before training starts, which is
how a model trained on earlier imagery is adapted to a new year:

```bash
python train.py \
    --exp-path ft_naip_2024 \
    --dataset config/naip_2024.yaml \
    --class "ovg,iceplant,others" \
    --backbone unet-resnet152 \
    --epochs 200 \
    --ckpt experiments/models/mixed_wv2_naip/version_0/checkpoints/last.ckpt
```

The checkpoint must have been trained with the same backbone, class count and
number of input channels. Training restarts from epoch 0 with a fresh
optimizer; only the weights are reused.

## What a run produces

```
experiments/models/<exp-path>/version_0/
├── checkpoints/
│   ├── best.ckpt         # lowest train/loss
│   └── last.ckpt         # end of the last completed validation epoch
├── events.out.tfevents.* # TensorBoard log
└── hparams.yaml          # every command line option, plus sample counts
```

Re-running the same `--exp-path` creates `version_1`, `version_2`, ... instead
of overwriting.

**Checkpoints are only written when validation runs**, i.e. every `--val-every`
epochs (10 by default). A run of fewer than `--val-every` epochs finishes
without saving anything — lower `--val-every` for short runs.

Early stopping watches `train/loss` with a patience of 10 validation rounds and
also only applies at those epochs.

## Monitoring

```bash
tensorboard --logdir experiments/models
```

Logged per run:

* `train/loss` (per step), `train/acc_epoch`, `train/lr`
* `val/loss`, `val/acc_epoch`
* `train/cm_epoch`, `val/cm_epoch` — row-normalized confusion matrices
* `train/vis`, `val/vis` — 10 random patches as image / ground truth /
  prediction triples

The Dice loss is computed over all classes at once, so it starts near 0.65 for
three classes and drops as the classes separate.

## A verified example run

A short run on the reference 2024 NAIP dataset, used to check the pipeline end
to end:

```bash
python train.py \
    --exp-path test_naip2024_r34 \
    --dataset config/naip_2024.yaml \
    --class "ovg,iceplant,others" \
    --backbone unet-resnet34 \
    --epochs 10 \
    --batch-size 16 \
    --workers 4
```

```
Loaded dataset from config/naip_2024.yaml
  Number of training samples: 2890
  Number of validation samples: 75
Total number of training samples: 2890
Total number of validation samples: 75
Training a new model from scratch
...
Experiment artifacts saved in: experiments/models/test_naip2024_r34/version_0
```

After 10 epochs (181 steps each): `train/acc_epoch` 0.70 → 0.85,
`val/acc_epoch` 0.85, `val/loss` 0.15, and `checkpoints/best.ckpt` plus
`checkpoints/last.ckpt` on disk. A real run uses `unet-resnet152` and 100–200
epochs.

## Predicting straight after training

```bash
python train.py ... \
    --predict experiments/data/raster/2022NAIP_CojoTerrace.tif,experiments/data/raster/2024NAIP_CojoTerrace_RGBIR.tif
```

Each raster listed there is passed to `predict_raster.py` with the checkpoint of
the run that just finished, using the first *n* bands where *n* is the number of
input channels of the model. Use `predict_raster.py` directly when the bands you
need are not the first ones.

## Troubleshooting

| symptom | cause |
| --- | --- |
| `Number of training samples: 0` | grid/raster CRS mismatch, no `Labeled == 1` cell, or all cells listed in `val_grid_ids` |
| CUDA out of memory | lower `--batch-size`, or use a smaller backbone |
| No `checkpoints/` directory | the run ended before the first validation epoch, lower `--val-every` |
| `KeyError: 'class'` | the label shapefile has no `class` attribute |
| `... backbone only supports 3 input channels` | a torchvision encoder was combined with 4-band imagery, use a `resnet` backbone |
