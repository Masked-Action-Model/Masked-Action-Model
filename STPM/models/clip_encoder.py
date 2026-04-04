from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

class FrozenCLIPEncoder(nn.Module):
    def __init__(self, ckpt: str, device: torch.device):
        super().__init__()
        self.device = device
        resolved_ckpt = self._resolve_ckpt_path(ckpt)
        self.model = CLIPModel.from_pretrained(resolved_ckpt).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(resolved_ckpt, use_fast=False)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _resolve_ckpt_path(ckpt: str) -> str:
        ckpt_path = Path(ckpt).expanduser()
        if ckpt_path.exists():
            return str(ckpt_path)

        repo_root = Path(__file__).resolve().parents[2]
        repo_relative_path = repo_root / ckpt_path
        if repo_relative_path.exists():
            return str(repo_relative_path)

        # Fall back to the original identifier for Hugging Face model ids.
        return ckpt

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        texts: list[str], length B
        returns: (B, 512) CLIP text embeddings
        """
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            text_outputs = self.model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
            text_embeds = self.model.text_projection(text_outputs.pooler_output)
        return text_embeds

    def encode_image(self, images: List[Image.Image], do_rescale=False) -> torch.Tensor:
        """
        images: list of PIL Images, length B
        returns: (B, 512) CLIP image embeddings
        """
        inputs = self.processor(images=images, return_tensors="pt", do_rescale=do_rescale).to(self.device)
        with torch.no_grad():
            vision_outputs = self.model.vision_model(pixel_values=inputs["pixel_values"])
            image_embeds = self.model.visual_projection(vision_outputs.pooler_output)
        return image_embeds
