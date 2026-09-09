import os
import sys
import argparse
import subprocess
import numpy as np
import torch
import albumentations as A
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from raster_dataset import RasterDataset
from model import SegmModel


torch.set_float32_matmul_precision("high")


def parse_options():
    parser = argparse.ArgumentParser("Semantic segmentation of iceplant")

    parser.add_argument(
        "--exp-path",
        type=str,
        default="exp",
        help="Experiment name; results are stored under experiments/models/<exp-path>",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        help="Path to an existing checkpoint to initialize the model from",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset config YAML file(s), comma separated",
    )

    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for training"
    )

    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs for training"
    )

    parser.add_argument(
        "--workers", type=int, default=8, help="Number of workers for data loading"
    )

    parser.add_argument(
        "--val-every",
        type=int,
        default=10,
        help="Run validation (and write checkpoints) every N epochs",
    )

    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        help="Lightning accelerator: auto, gpu or cpu",
    )

    parser.add_argument(
        "--predict",
        type=str,
        default=None,
        help="Comma separated rasters to predict on once training is finished",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run a single train/val batch to check the pipeline",
    )

    parser.add_argument(
        "--seed", type=int, default=7829347, help="Seed for reproducibility"
    )

    parser.add_argument(
        "--class",
        type=str,
        required=True,
        help="Comma separated class names ordered by class id, e.g. --class ovg,iceplant,others",
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="unet-resnet152",
        help="Model as <architecture>-<backbone>, e.g. unet-resnet152",
    )

    args = parser.parse_args()
    args.exp_path = os.path.join("experiments/models", args.exp_path)
    # Parse class names as list
    args.class_names = [c.strip() for c in args.__dict__.pop("class").split(",")]
    return args


def normalize_imgnet(image, **kwargs):
    """Scale an HWC uint8 image to [0, 1] and normalize it with ImageNet statistics.

    Bands beyond the third one reuse the RGB statistics in a cyclic fashion, so a
    4-band RGBIR patch is normalized with the R, G, B, R statistics.
    """
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])

    # Get number of channels and extend mean/std if needed
    num_channels = image.shape[2] if len(image.shape) == 3 else 1
    if num_channels != 3:
        # Repeat the RGB values as needed (e.g., for 4 channels: RGBR)
        imagenet_mean = np.tile(imagenet_mean, (num_channels + 2) // 3)[:num_channels]
        imagenet_std = np.tile(imagenet_std, (num_channels + 2) // 3)[:num_channels]

    image = image.astype(np.float32) / 255.0
    normalized_image = (image - imagenet_mean.reshape(1, 1, -1)) / imagenet_std.reshape(
        1, 1, -1
    )
    normalized_image = normalized_image.astype(np.float32)
    return normalized_image


train_transform = A.Compose(
    [
        A.RandomGridShuffle(),
        A.GridDistortion(),
        A.AdditiveNoise(),
        A.GaussianBlur(),
        A.RandomBrightnessContrast(),
        A.D4(),
        A.Lambda(name="normalize_imgnet", image=normalize_imgnet, p=1.0),
        A.pytorch.ToTensorV2(),
    ]
)

val_transform = A.Compose(
    [
        A.Lambda(name="normalize_imgnet", image=normalize_imgnet, p=1.0),
        A.pytorch.ToTensorV2(),
    ]
)


def main(config):
    callbacks = [
        EarlyStopping(monitor="train/loss", patience=10, mode="min"),
        ModelCheckpoint(
            filename="best",
            monitor="train/loss",
            save_top_k=1,
            mode="min",
            save_last=True,
        ),
    ]

    logger = TensorBoardLogger(save_dir=config.exp_path, name="")
    logger.log_hyperparams(config.__dict__)

    trainer = L.Trainer(
        accelerator=config.accelerator,
        logger=logger,
        max_epochs=config.epochs,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=config.val_every,
        callbacks=callbacks,
        deterministic=True,
        fast_dev_run=config.debug,
    )

    # Parse dataset paths as list
    dataset_configs = [d.strip() for d in config.dataset.split(",")]

    # Create lists to hold all datasets
    train_datasets = []
    val_datasets = []

    # Process each dataset config file
    for dataset_config in dataset_configs:
        train_datasets.append(
            RasterDataset(dataset_config, split="train", transforms=train_transform)
        )
        val_datasets.append(
            RasterDataset(dataset_config, split="val", transforms=val_transform)
        )

        print(f"Loaded dataset from {dataset_config}")
        print(f"  Number of training samples: {len(train_datasets[-1])}")
        print(f"  Number of validation samples: {len(val_datasets[-1])}")

    # Combine all train/val datasets
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)

    input_shape = val_dataset[0]["image"].shape
    in_channels = input_shape[0] if len(input_shape) == 3 else 1

    logger.log_hyperparams(
        {
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "input_shape": input_shape,
            "in_channels": in_channels,
        }
    )

    print(f"Total number of training samples: {len(train_dataset)}")
    print(f"Total number of validation samples: {len(val_dataset)}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.batch_size, num_workers=config.workers
    )

    if config.ckpt:
        print(f"Loading model from checkpoint: {config.ckpt}")
        model = SegmModel.load_from_checkpoint(
            config.ckpt,
            class_names=config.class_names,
            backbone=config.backbone,
            in_channels=in_channels,
        )
    else:
        print("Training a new model from scratch")
        model = SegmModel(
            class_names=config.class_names,
            backbone=config.backbone,
            in_channels=in_channels,
        )

    trainer.fit(model, train_loader, val_loader)

    exp_location = trainer.logger.log_dir
    print(f"Experiment artifacts saved in: {exp_location}")

    # Optionally predict on a list of rasters with the model just trained
    if config.predict and exp_location:
        bands = ",".join(str(b) for b in range(1, in_channels + 1))
        for raster in [r.strip() for r in config.predict.split(",") if r.strip()]:
            subprocess.run(
                [
                    sys.executable,
                    "predict_raster.py",
                    "--exp-path",
                    exp_location,
                    "--image",
                    raster,
                    "--bands",
                    bands,
                ],
                check=True,
            )


if __name__ == "__main__":
    args = parse_options()
    L.seed_everything(args.seed)
    main(args)
