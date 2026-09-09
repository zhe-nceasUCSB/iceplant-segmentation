#!/bin/bash
# Example experiment scripts. Run them from the repository root with the
# `iceplant` environment activated.
set -e

DATE=$(date +%Y%m%d)
CLASS="ovg,iceplant,others"
BACKBONE="unet-resnet152"
EPOCHS=100
BSIZE=64

# 1. Train on a single year of NAIP imagery.
python train.py \
    --exp-path "${DATE}_${BACKBONE}_naip_2024" \
    --dataset config/naip_2024.yaml \
    --class "${CLASS}" \
    --backbone "${BACKBONE}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BSIZE}"

# 2. Predict on the imagery of a different year with the model just trained.
python predict_raster.py \
    --exp-path "experiments/models/${DATE}_${BACKBONE}_naip_2024/version_0" \
    --image experiments/data/raster/2022NAIP_CojoTerrace.tif \
    --bands 1,2,3,4

# 3. Train on several sensors/dates at once.
python train.py \
    --exp-path "${DATE}_${BACKBONE}_mixed_wv2_naip" \
    --dataset config/wv2_202308.yaml,config/naip_2022.yaml,config/naip_2024.yaml \
    --class "${CLASS}" \
    --backbone "${BACKBONE}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BSIZE}"

# 4. Fine-tune an existing checkpoint on a new year of imagery.
python train.py \
    --exp-path "${DATE}_${BACKBONE}_ft_naip_2024" \
    --dataset config/naip_2024.yaml \
    --class "${CLASS}" \
    --backbone "${BACKBONE}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BSIZE}" \
    --ckpt "experiments/models/${DATE}_${BACKBONE}_mixed_wv2_naip/version_0/checkpoints/last.ckpt"
