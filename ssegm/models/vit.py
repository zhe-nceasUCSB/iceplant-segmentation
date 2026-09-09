import torch.nn as nn
import torch.nn.functional as F
import timm


class ViTSeg(nn.Module):
    """
    Vision Transformer for Semantic Segmentation.
    """

    def __init__(
        self,
        model_name="vit_base_patch16_224.mae",
        pretrained=True,
        num_classes=21,
        in_channels=3,
    ):
        """
        Args:
            model_name (str): Name of the ViT model from timm.
            pretrained (bool): Whether to use pretrained weights.
            num_classes (int): Number of output segmentation classes.
            in_channels (int): Number of input channels.
        """
        super().__init__()
        self.num_classes = num_classes

        # Create ViT backbone from timm
        # We don't need the classification head.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # remove classifier head
            in_chans=in_channels,  # set input channels
        )

        self.internal_size = self.backbone.patch_embed.img_size[0]
        self.embed_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_embed.patch_size[0]

        # A simple decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(self.embed_dim, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )

        self.segmentation_head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        H, W = x.shape[2:]

        # Resize input to internal size
        x_resized = F.interpolate(
            x,
            size=(self.internal_size, self.internal_size),
            mode="bilinear",
            align_corners=False,
        )

        # Get features from backbone
        # Output shape: (B, N, C) where N = num_patches + 1 (for cls token)
        features = self.backbone.forward_features(x_resized)

        # Remove class token. For most ViTs, it's the first token.
        features = features[:, 1:, :]

        # Reshape to 2D feature map
        # (B, N-1, C) -> (B, C, H_patch, W_patch)
        h_patch = self.internal_size // self.patch_size
        w_patch = self.internal_size // self.patch_size
        features = features.permute(0, 2, 1).reshape(
            -1, self.embed_dim, h_patch, w_patch
        )

        # Decode features
        decoded_features = self.decoder(features)

        # Get segmentation logits
        logits = self.segmentation_head(decoded_features)

        # Upsample to original image size
        out = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)

        return out
