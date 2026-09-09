import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import torchmetrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from ssegm.models.vit import ViTSeg
from ssegm.models.unet import Unet


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.softmax(inputs, dim=1)
        # flatten label and prediction tensors

        num_classes = inputs.shape[1]
        inputs = inputs.flatten()

        targets_onehot = F.one_hot(targets.long(), num_classes)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2)
        targets = targets_onehot.flatten()

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


def vis(model, dataset, device, class_names):
    # randomly select 10 samples and visualize images/masks/predictions
    # in a 10x3 grid
    indices = np.random.choice(len(dataset), 10)

    samples = [dataset[i] for i in indices]

    images = torch.stack([sample["image"] for sample in samples]).to(device)
    masks = torch.stack([sample["mask"] for sample in samples]).to(device)
    outputs = model(images).argmax(1)

    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    outputs = outputs.cpu().numpy()

    max_class_id = len(class_names) - 1
    cmap = plt.get_cmap("jet", max_class_id + 1)
    norm = plt.Normalize(vmin=0, vmax=max_class_id)

    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])

    fig, axes = plt.subplots(10, 3, figsize=(10, 30), dpi=200)
    for i in range(10):
        img = images[i].transpose(1, 2, 0)[:, :, :3]
        img = img * imagenet_std + imagenet_mean
        img = np.clip(img, 0, 1)

        axes[i, 0].imshow(img)
        axes[i, 1].imshow(masks[i], cmap=cmap, norm=norm)
        axes[i, 2].imshow(outputs[i], cmap=cmap, norm=norm)
        axes[i, 0].axis("off")
        axes[i, 1].axis("off")
        axes[i, 2].axis("off")

    axes[0, 0].set_title("Image")
    axes[0, 1].set_title("Ground Truth")
    axes[0, 2].set_title("Prediction")

    # Create legend with class names
    legend_handles = []
    for class_id, class_name in enumerate(class_names):
        color = cmap(norm(class_id))
        legend_handles.append(plt.Line2D([0], [0], color=color, lw=4, label=class_name))

    axes[0, 1].legend(handles=legend_handles, loc="upper right")

    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    fig.tight_layout()
    return fig


class SegmModel(L.LightningModule):
    def __init__(self, class_names, backbone, in_channels=3, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.class_names = class_names
        class_num = len(class_names)

        model_arch = backbone.split('-')[0]
        model_backbone = backbone.replace(f"{model_arch}-", "")

        if model_arch == 'unet':
            self.model = Unet(backbone_name=model_backbone, classes=class_num, in_channels=in_channels)
        elif model_arch == 'vit':
            self.model = ViTSeg(model_name=model_backbone, num_classes=class_num, in_channels=in_channels)
        else:
            raise ValueError(
                f"Unknown architecture '{model_arch}' in backbone '{backbone}', "
                "expected the form 'unet-<backbone>' or 'vit-<backbone>'"
            )

        self.criterion = DiceLoss()

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=class_num)
        self.train_cm = torchmetrics.ConfusionMatrix(
            task="multiclass", num_classes=class_num, normalize="true"
        )

        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=class_num)
        self.val_cm = torchmetrics.ConfusionMatrix(
            task="multiclass", num_classes=class_num, normalize="true"
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"]
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)

        self.train_acc(y_hat, y)
        self.train_cm(y_hat, y)

        self.log("train/loss", loss, prog_bar=True, batch_size=x.size(0))
        return loss

    def on_train_epoch_end(self):
        self.log("train/acc_epoch", self.train_acc)
        fig, _ = self.train_cm.plot()
        self.logger.experiment.add_figure("train/cm_epoch", fig, self.global_step - 1)

        fig = vis(self.model, self.trainer.train_dataloader.dataset, self.device, self.class_names)
        self.logger.experiment.add_figure("train/vis", fig, self.global_step - 1)

        # log learning rate
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"])

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"]
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)

        self.val_acc(y_hat, y)
        self.val_cm(y_hat, y)

        self.log("val/loss", loss, batch_size=x.size(0))
        return loss

    def on_validation_epoch_end(self):
        self.log("val/acc_epoch", self.val_acc)
        fig, _ = self.val_cm.plot()
        self.logger.experiment.add_figure("val/cm_epoch", fig, self.global_step - 1)

        fig = vis(self.model, self.trainer.val_dataloaders.dataset, self.device, self.class_names)
        self.logger.experiment.add_figure("val/vis", fig, self.global_step - 1)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
        lr_scheduler = {
            "scheduler": optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97),
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]
