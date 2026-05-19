from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from models.plain_conv import PlainConv


DINO_IMAGE_SIZES = {
    "dino2": 252,
    "dino3": 256,
}
DINO_FEATURE_TYPE = "cls"


class DinoVisionEncoder(nn.Module):
    def __init__(
        self,
        model_path: str,
        out_dim: int = 256,
        image_size: int = 256,
        encoder_name: str = "dino",
    ):
        super().__init__()
        if not model_path:
            raise ValueError(f"DINO_MODEL_PATH is required when vision_encoder='{encoder_name}'")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"DINO_MODEL_PATH does not exist: {model_path}")
        if not (Path(model_path) / "config.json").exists():
            raise FileNotFoundError(
                f"DINO_MODEL_PATH is incomplete; missing config.json in {model_path}"
            )

        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                f"vision_encoder='{encoder_name}' requires HuggingFace transformers."
            ) from exc

        self.model_path = str(model_path)
        self.image_size = int(image_size)
        self.encoder_name = str(encoder_name)
        self.feature_type = DINO_FEATURE_TYPE
        self.backbone = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.backbone.eval()
        self.backbone.requires_grad_(False)

        hidden_size = int(getattr(self.backbone.config, "hidden_size", 384))
        self.proj = nn.Linear(hidden_size, out_dim)
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] != 3:
            raise ValueError(
                f"DINO expects single-camera RGB with 3 channels, got {image.shape[1]}"
            )
        image = image.to(dtype=torch.float32)
        if image.shape[-2:] != (self.image_size, self.image_size):
            image = F.interpolate(
                image,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        image = (image - self.image_mean) / self.image_std
        with torch.no_grad():
            outputs = self.backbone(pixel_values=image)
            feature = getattr(outputs, "pooler_output", None)
            if feature is None:
                feature = outputs.last_hidden_state[:, 0]
        return self.proj(feature)


def make_dino_data_aug(vision_encoder: str) -> transforms.Compose:
    image_size = DINO_IMAGE_SIZES.get(vision_encoder)
    if image_size is None:
        raise ValueError(f"unsupported DINO vision_encoder={vision_encoder!r}")
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.9, 1.0),
                ratio=(1.0, 1.0),
                antialias=True,
            ),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.02,
            ),
        ]
    )


def make_vision_encoder(
    *,
    vision_encoder: Literal["resnet", "dino2", "dino3"],
    in_channels: int,
    out_dim: int,
    dino_model_path: str = "",
) -> nn.Module:
    if vision_encoder == "resnet":
        return PlainConv(
            in_channels=in_channels,
            out_dim=out_dim,
            pool_feature_map=True,
        )
    if vision_encoder in {"dino2", "dino3"}:
        if in_channels != 3:
            raise ValueError(
                f"vision_encoder='{vision_encoder}' requires single-camera RGB only; got {in_channels} visual channels"
            )
        return DinoVisionEncoder(
            model_path=dino_model_path,
            out_dim=out_dim,
            image_size=DINO_IMAGE_SIZES[vision_encoder],
            encoder_name=vision_encoder,
        )
    raise ValueError(f"unsupported vision_encoder={vision_encoder!r}")
