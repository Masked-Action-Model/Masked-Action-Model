#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 前面加上了dp_transformer，定义dit组成部分， 
# 后面加上了dp的visual_encoder,在class DiTDPModel里组装，（在其他codebase里不用管）
# 搜dit找到policy修改的部分
"""DiTDP (DiT Diffusion Policy) - Diffusion Policy with Transformer-based noise network.

This module replaces the UNet noise network in the original Diffusion Policy with
a Diffusion Transformer (DiT) architecture based on the DiTNoiseNet design.
"""

import copy
from collections import deque
from collections.abc import Callable

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor

# from lerobot.policies.ditdp.configuration_ditdp import DiTDPConfig
# from lerobot.policies.pretrained import PreTrainedPolicy
# from lerobot.policies.utils import (
#     get_device_from_parameters,
#     get_dtype_from_parameters,
#     get_output_shape,
#     populate_queues,
# )
# from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


# =============================================================================
# DiT Components (ported from dit_transformer.py)
# =============================================================================

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return nn.GELU(approximate="tanh")
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


def _with_pos_embed(tensor, pos=None):
    return tensor if pos is None else tensor + pos


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        pe = self.pe[: x.shape[0]]
        pe = pe.repeat((1, x.shape[1], 1))
        return pe.detach().clone()


class _TimeNetwork(nn.Module):
    def __init__(self, time_dim, out_dim, learnable_w=False):
        assert time_dim % 2 == 0, "time_dim must be even!"
        half_dim = int(time_dim // 2)
        super().__init__()

        w = np.log(10000) / (half_dim - 1)
        w = torch.exp(torch.arange(half_dim) * -w).float()
        self.register_parameter("w", nn.Parameter(w, requires_grad=learnable_w))

        self.out_net = nn.Sequential(
            nn.Linear(time_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        assert len(x.shape) == 1, "assumes 1d input timestep array"
        x = x[:, None] * self.w[None]
        x = torch.cat((torch.cos(x), torch.sin(x)), dim=1)
        return self.out_net(x)


class _SelfAttnEncoder(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, pos):
        q = k = _with_pos_embed(src, pos)
        src2, _ = self.self_attn(q, k, value=src, need_weights=False)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class _ShiftScaleMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)
        self.shift = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)
        return x * self.scale(c)[None] + self.shift(c)[None]

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.scale.weight)
        nn.init.xavier_uniform_(self.shift.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.bias)


class _ZeroScaleMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)
        return x * self.scale(c)[None]

    def reset_parameters(self):
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)


class _DiTDecoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.attn_mod1 = _ShiftScaleMod(d_model)
        self.attn_mod2 = _ZeroScaleMod(d_model)
        self.mlp_mod1 = _ShiftScaleMod(d_model)
        self.mlp_mod2 = _ZeroScaleMod(d_model)

    def forward(self, x, t, cond):
        cond = torch.mean(cond, axis=0)
        cond = cond + t

        x2 = self.attn_mod1(self.norm1(x), cond)
        x2, _ = self.self_attn(x2, x2, x2, need_weights=False)
        x = self.attn_mod2(self.dropout1(x2), cond) + x

        x2 = self.mlp_mod1(self.norm2(x), cond)
        x2 = self.linear2(self.dropout2(self.activation(self.linear1(x2))))
        x2 = self.mlp_mod2(self.dropout3(x2), cond)
        return x + x2

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for s in (self.attn_mod1, self.attn_mod2, self.mlp_mod1, self.mlp_mod2):
            s.reset_parameters()


class _FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, out_size, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, t, cond):
        cond = torch.mean(cond, axis=0)
        cond = cond + t
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=1)
        x = x * scale[None] + shift[None]
        x = self.linear(x)
        return x.transpose(0, 1)

    def reset_parameters(self):
        for p in self.parameters():
            nn.init.zeros_(p)


class _TransformerEncoder(nn.Module):
    def __init__(self, base_module, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(base_module) for _ in range(num_layers)])
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, src, pos):
        x, outputs = src, []
        for layer in self.layers:
            x = layer(x, pos)
            outputs.append(x)
        return outputs


class _TransformerDecoder(_TransformerEncoder):
    def forward(self, src, t, all_conds):
        x = src
        for layer, cond in zip(self.layers, all_conds):
            x = layer(x, t, cond)
        return x


# =============================================================================
# DiT Noise Network
# =============================================================================

class DiTNoiseNet(nn.Module):
    """DiT-based noise prediction network for diffusion policy.
    
    This replaces the UNet in the original Diffusion Policy with a
    Transformer Encoder-Decoder architecture.
    """

    def __init__(
        self,
        ac_dim: int,
        ac_chunk: int,
        obs_dim: int,
        time_dim: int = 256,
        hidden_dim: int = 512,
        num_blocks: int = 6,
        dropout: float = 0.1,
        dim_feedforward: int = 2048,
        nhead: int = 8,
        activation: str = "gelu",
        use_mask: bool = False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.ac_dim = ac_dim
        self.use_mask = use_mask

        # Observation projection
        self.obs_proj = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Positional encodings
        self.enc_pos = _PositionalEncoding(hidden_dim)
        self.register_parameter(
            "dec_pos",
            nn.Parameter(torch.empty(ac_chunk, 1, hidden_dim), requires_grad=True),
        )
        nn.init.xavier_uniform_(self.dec_pos.data)

        # Time embedding network
        self.time_net = _TimeNetwork(time_dim, hidden_dim)

        # Action projection (with optional mask channel)
        # If use_mask=True, input is (action + mask) concatenated, so input_dim = ac_dim * 2
        ac_input_dim = ac_dim * 2 if use_mask else ac_dim
        self.ac_proj = nn.Sequential(
            nn.Linear(ac_input_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Transformer encoder
        encoder_module = _SelfAttnEncoder(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.encoder = _TransformerEncoder(encoder_module, num_blocks)

        # Transformer decoder (DiT-style)
        decoder_module = _DiTDecoder(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.decoder = _TransformerDecoder(decoder_module, num_blocks)

        # Output layer
        self.eps_out = _FinalLayer(hidden_dim, ac_dim)

    def forward(
        self, 
        noise_actions: Tensor, 
        timesteps: Tensor, 
        obs_enc: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            noise_actions: (B, T, action_dim) noisy action sequence
            timesteps: (B,) diffusion timesteps
            obs_enc: (B, S, obs_dim) observation encoding
            mask: (B, T, action_dim) optional mask, same shape as noise_actions.
                  If use_mask=True, this must be provided.
            
        Returns:
            (B, T, action_dim) predicted noise or denoised actions
        """
        enc_cache = self.forward_enc(obs_enc)
        return self.forward_dec(noise_actions, timesteps, enc_cache, mask=mask)

    def forward_enc(self, obs_enc: Tensor) -> list[Tensor]:
        """Encode observations.
        
        Args:
            obs_enc: (B, S, obs_dim) observation encoding
            
        Returns:
            List of encoder outputs for each layer
        """
        obs_enc = self.obs_proj(obs_enc)          # (B, S, H)
        obs_enc = obs_enc.transpose(0, 1)         # (S, B, H)
        pos = self.enc_pos(obs_enc)               # (S, B, H)
        enc_cache = self.encoder(obs_enc, pos)    # list[L] of (S, B, H)
        return enc_cache

    def forward_dec(
        self, 
        noise_actions: Tensor, 
        time: Tensor, 
        enc_cache: list[Tensor],
        mask: Tensor | None = None,
    ) -> Tensor:
        """Decode noisy actions conditioned on observations.
        
        Args:
            noise_actions: (B, T, action_dim) noisy actions
            time: (B,) diffusion timesteps
            enc_cache: List of encoder outputs
            mask: (B, T, action_dim) optional mask for conditioning
            
        Returns:
            (B, T, action_dim) predicted noise
        """
        time_enc = self.time_net(time)            # (B, H)

        # Concatenate mask with noise_actions if use_mask is enabled
        if self.use_mask:
            if mask is None:
                raise ValueError("mask must be provided when use_mask=True")
            # Concatenate along the last dimension: (B, T, action_dim * 2)
            ac_input = torch.cat([noise_actions, mask], dim=-1)
        else:
            ac_input = noise_actions

        ac_tokens = self.ac_proj(ac_input)        # (B, T, H)
        ac_tokens = ac_tokens.transpose(0, 1)     # (T, B, H)
        dec_in = ac_tokens + self.dec_pos         # (T, B, H)

        dec_out = self.decoder(dec_in, time_enc, enc_cache)
        return self.eps_out(dec_out, time_enc, enc_cache[-1])


# # =============================================================================
# # Vision Encoder (reused from Diffusion Policy)
# # =============================================================================

# class SpatialSoftmax(nn.Module):
#     """Spatial Soft Argmax operation for visual feature extraction."""

#     def __init__(self, input_shape, num_kp=None):
#         super().__init__()

#         assert len(input_shape) == 3
#         self._in_c, self._in_h, self._in_w = input_shape

#         if num_kp is not None:
#             self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
#             self._out_c = num_kp
#         else:
#             self.nets = None
#             self._out_c = self._in_c

#         pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
#         pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
#         pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
#         self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

#     def forward(self, features: Tensor) -> Tensor:
#         if self.nets is not None:
#             features = self.nets(features)

#         features = features.reshape(-1, self._in_h * self._in_w)
#         attention = F.softmax(features, dim=-1)
#         expected_xy = attention @ self.pos_grid
#         feature_keypoints = expected_xy.view(-1, self._out_c, 2)

#         return feature_keypoints


# class DiTDPRgbEncoder(nn.Module):
#     """RGB image encoder for DiTDP."""

#     def __init__(self, config: DiTDPConfig):
#         super().__init__()
#         # Set up optional preprocessing
#         if config.crop_shape is not None:
#             self.do_crop = True
#             self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
#             if config.crop_is_random:
#                 self.maybe_random_crop = torchvision.transforms.RandomCrop(config.crop_shape)
#             else:
#                 self.maybe_random_crop = self.center_crop
#         else:
#             self.do_crop = False

#         # Set up backbone
#         backbone_model = getattr(torchvision.models, config.vision_backbone)(
#             weights=config.pretrained_backbone_weights
#         )
#         self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
#         if config.use_group_norm:
#             if config.pretrained_backbone_weights:
#                 raise ValueError(
#                     "You can't replace BatchNorm in a pretrained model without ruining the weights!"
#                 )
#             self.backbone = _replace_submodules(
#                 root_module=self.backbone,
#                 predicate=lambda x: isinstance(x, nn.BatchNorm2d),
#                 func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
#             )

#         # Set up pooling and final layers
#         images_shape = next(iter(config.image_features.values())).shape
#         dummy_shape_h_w = config.crop_shape if config.crop_shape is not None else images_shape[1:]
#         dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
#         feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]

#         self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
#         self.feature_dim = config.spatial_softmax_num_keypoints * 2
#         self.out = nn.Linear(config.spatial_softmax_num_keypoints * 2, self.feature_dim)
#         self.relu = nn.ReLU()

#     def forward(self, x: Tensor) -> Tensor:
#         if self.do_crop:
#             if self.training:
#                 x = self.maybe_random_crop(x)
#             else:
#                 x = self.center_crop(x)
#         x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
#         x = self.relu(self.out(x))
#         return x


# def _replace_submodules(
#     root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]
# ) -> nn.Module:
#     """Replace submodules matching a predicate with new modules."""
#     if predicate(root_module):
#         return func(root_module)

#     replace_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
#     for *parents, k in replace_list:
#         parent_module = root_module
#         if len(parents) > 0:
#             parent_module = root_module.get_submodule(".".join(parents))
#         if isinstance(parent_module, nn.Sequential):
#             src_module = parent_module[int(k)]
#         else:
#             src_module = getattr(parent_module, k)
#         tgt_module = func(src_module)
#         if isinstance(parent_module, nn.Sequential):
#             parent_module[int(k)] = tgt_module
#         else:
#             setattr(parent_module, k, tgt_module)
#     assert not any(predicate(m) for _, m in root_module.named_modules(remove_duplicate=True))
#     return root_module


# # =============================================================================
# # Noise Scheduler Factory
# # =============================================================================

# def _make_noise_scheduler(name: str, **kwargs) -> DDPMScheduler | DDIMScheduler:
#     """Factory for noise scheduler instances."""
#     if name == "DDPM":
#         return DDPMScheduler(**kwargs)
#     elif name == "DDIM":
#         return DDIMScheduler(**kwargs)
#     else:
#         raise ValueError(f"Unsupported noise scheduler type {name}")


# # =============================================================================
# # DiTDP Model
# # =============================================================================


# class DiTDPModel(nn.Module):  
#     """Core DiTDP model combining vision encoder, DiT noise network, and diffusion scheduler."""

#     def __init__(self, config: DiTDPConfig):
#         super().__init__()
#         self.config = config

#         # Build observation encoders
#         obs_dim = self.config.robot_state_feature.shape[0]
#         if self.config.image_features:
#             num_images = len(self.config.image_features)
#             if self.config.use_separate_rgb_encoder_per_camera:
#                 encoders = [DiTDPRgbEncoder(config) for _ in range(num_images)]
#                 self.rgb_encoder = nn.ModuleList(encoders)
#                 obs_dim += encoders[0].feature_dim * num_images
#             else:
#                 self.rgb_encoder = DiTDPRgbEncoder(config)
#                 obs_dim += self.rgb_encoder.feature_dim * num_images
#         if self.config.env_state_feature:
#             obs_dim += self.config.env_state_feature.shape[0]

#         # DiT noise network
#         self.dit = DiTNoiseNet(
#             ac_dim=config.action_feature.shape[0],
#             ac_chunk=config.horizon,
#             obs_dim=obs_dim,
#             time_dim=config.time_dim,
#             hidden_dim=config.hidden_dim,
#             num_blocks=config.num_blocks,
#             dropout=config.dropout,
#             dim_feedforward=config.dim_feedforward,
#             nhead=config.nhead,
#             activation=config.activation,
#             use_mask=config.use_mask,
#         )

#         # Noise scheduler
#         self.noise_scheduler = _make_noise_scheduler(
#             config.noise_scheduler_type,
#             num_train_timesteps=config.num_train_timesteps,
#             beta_start=config.beta_start,
#             beta_end=config.beta_end,
#             beta_schedule=config.beta_schedule,
#             clip_sample=config.clip_sample,
#             clip_sample_range=config.clip_sample_range,
#             prediction_type=config.prediction_type,
#         )

#         if config.num_inference_steps is None:
#             self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
#         else:
#             self.num_inference_steps = config.num_inference_steps

#     def conditional_sample(
#         self,
#         batch_size: int,
#         obs_enc: Tensor,
#         generator: torch.Generator | None = None,
#         noise: Tensor | None = None,
#         mask: Tensor | None = None,
#     ) -> Tensor:
#         """Sample actions from noise using the reverse diffusion process.
        
#         Args:
#             batch_size: Number of samples to generate
#             obs_enc: (B, n_obs_steps, obs_dim) observation encoding
#             generator: Optional random generator
#             noise: Optional initial noise
#             mask: (B, horizon, action_dim) optional external mask for conditioning
#         """
#         device = get_device_from_parameters(self)
#         dtype = get_dtype_from_parameters(self)

#         # Sample prior
#         sample = (
#             noise
#             if noise is not None
#             else torch.randn(
#                 size=(batch_size, self.config.horizon, self.config.action_feature.shape[0]),
#                 dtype=dtype,
#                 device=device,
#                 generator=generator,
#             )
#         )

#         self.noise_scheduler.set_timesteps(self.num_inference_steps)

#         for t in self.noise_scheduler.timesteps:
#             # Predict model output
#             model_output = self.dit(
#                 sample,
#                 torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
#                 obs_enc,
#                 mask=mask,
#             )
#             # Compute previous image: x_t -> x_t-1
#             sample = self.noise_scheduler.step(model_output, t, sample, generator=generator).prev_sample

#         return sample

#     def _prepare_observation_encoding(self, batch: dict[str, Tensor]) -> Tensor:
#         """Prepare observation encoding for DiT conditioning.
        
#         Returns observation features as a sequence for transformer input.
#         """
#         batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
#         obs_feats = [batch[OBS_STATE]]  # (B, n_obs_steps, state_dim)

#         # Extract image features
#         if self.config.image_features:
#             if self.config.use_separate_rgb_encoder_per_camera:
#                 images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
#                 img_features_list = torch.cat(
#                     [
#                         encoder(images)
#                         for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
#                     ]
#                 )
#                 img_features = einops.rearrange(
#                     img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
#                 )
#             else:
#                 img_features = self.rgb_encoder(
#                     einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
#                 )
#                 img_features = einops.rearrange(
#                     img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
#                 )
#             obs_feats.append(img_features)

#         if self.config.env_state_feature:
#             obs_feats.append(batch[OBS_ENV_STATE])

#         # Concatenate all observation features: (B, n_obs_steps, obs_dim)
#         return torch.cat(obs_feats, dim=-1)

#     def generate_actions(
#         self, 
#         batch: dict[str, Tensor], 
#         noise: Tensor | None = None,
#         mask: Tensor | None = None,
#     ) -> Tensor:
#         """Generate actions from observations using the diffusion process.
        
#         Args:
#             batch: Batch of observations
#             noise: Optional initial noise
#             mask: (B, horizon, action_dim) optional external mask for conditioning
#         """
#         batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
#         assert n_obs_steps == self.config.n_obs_steps

#         # Prepare observation encoding
#         obs_enc = self._prepare_observation_encoding(batch)  # (B, n_obs_steps, obs_dim)

#         # Run sampling
#         actions = self.conditional_sample(batch_size, obs_enc=obs_enc, noise=noise, mask=mask)

#         # Extract n_action_steps worth of actions
#         start = n_obs_steps - 1
#         end = start + self.config.n_action_steps
#         actions = actions[:, start:end]

#         return actions

#     def compute_loss(self, batch: dict[str, Tensor], mask: Tensor | None = None) -> Tensor:
#         """Compute the diffusion training loss.
        
#         Args:
#             batch: Batch containing observations and actions
#             mask: (B, horizon, action_dim) optional external mask for conditioning
#         """
#         # Input validation
#         assert set(batch).issuperset({OBS_STATE, ACTION, "action_is_pad"})
#         assert OBS_IMAGES in batch or OBS_ENV_STATE in batch
#         n_obs_steps = batch[OBS_STATE].shape[1]
#         horizon = batch[ACTION].shape[1]
#         assert horizon == self.config.horizon
#         assert n_obs_steps == self.config.n_obs_steps

#         # Prepare observation encoding
#         obs_enc = self._prepare_observation_encoding(batch)  # (B, n_obs_steps, obs_dim)

#         # Forward diffusion
#         trajectory = batch[ACTION]
#         eps = torch.randn(trajectory.shape, device=trajectory.device)
#         timesteps = torch.randint(
#             low=0,
#             high=self.noise_scheduler.config.num_train_timesteps,
#             size=(trajectory.shape[0],),
#             device=trajectory.device,
#         ).long()
#         noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)

#         # Run the denoising network (with optional mask)
#         pred = self.dit(noisy_trajectory, timesteps, obs_enc, mask=mask)

#         # Compute the loss
#         if self.config.prediction_type == "epsilon":
#             target = eps
#         elif self.config.prediction_type == "sample":
#             target = batch[ACTION]
#         else:
#             raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")

#         loss = F.mse_loss(pred, target, reduction="none")

#         # Mask loss for padded actions 
#         # 这里的mask只去掉padding部分，（只计算非padding部分loss），没有用数据传入的mask
#         if self.config.do_mask_loss_for_padding:
#             if "action_is_pad" not in batch:
#                 raise ValueError(
#                     "You need to provide 'action_is_pad' in the batch when "
#                     f"{self.config.do_mask_loss_for_padding=}."
#                 )
#             in_episode_bound = ~batch["action_is_pad"]
#             loss = loss * in_episode_bound.unsqueeze(-1)

#         return loss.mean()


# # =============================================================================
# # DiTDP Policy
# # =============================================================================

# class DiTDPPolicy(PreTrainedPolicy):
#     """DiTDP (DiT Diffusion Policy) - Diffusion Policy with Transformer-based noise network.
    
#     This policy replaces the UNet in the original Diffusion Policy with a
#     Diffusion Transformer (DiT) architecture for improved performance.
#     """

#     config_class = DiTDPConfig
#     name = "ditdp"

#     def __init__(
#         self,
#         config: DiTDPConfig,
#         **kwargs,
#     ):
#         """
#         Args:
#             config: Policy configuration class instance.
#         """
#         super().__init__(config)
#         config.validate_features()
#         self.config = config

#         # Queues for observation and action caching
#         self._queues = None

#         self.diffusion = DiTDPModel(config)

#         self.reset()

#     def get_optim_params(self) -> dict:
#         return self.diffusion.parameters()

#     def reset(self):
#         """Clear observation and action queues. Should be called on `env.reset()`"""
#         self._queues = {
#             OBS_STATE: deque(maxlen=self.config.n_obs_steps),
#             ACTION: deque(maxlen=self.config.n_action_steps),
#         }
#         if self.config.image_features:
#             self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
#         if self.config.env_state_feature:
#             self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)

#     @torch.no_grad()
#     def predict_action_chunk(
#         self, 
#         batch: dict[str, Tensor], 
#         noise: Tensor | None = None,
#         mask: Tensor | None = None,
#     ) -> Tensor:
#         """Predict a chunk of actions given environment observations.
        
#         Args:
#             batch: Batch of observations
#             noise: Optional initial noise
#             mask: (B, horizon, action_dim) optional external mask for conditioning
#         """
#         batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
#         actions = self.diffusion.generate_actions(batch, noise=noise, mask=mask)
#         return actions

#     @torch.no_grad()
#     def select_action(
#         self, 
#         batch: dict[str, Tensor], 
#         noise: Tensor | None = None,
#         mask: Tensor | None = None,
#     ) -> Tensor:
#         """Select a single action given environment observations.
        
#         Args:
#             batch: Batch of observations
#             noise: Optional initial noise
#             mask: (B, horizon, action_dim) optional external mask for conditioning
#         """
#         # Remove action from batch if present (for offline evaluation)
#         if ACTION in batch:
#             batch.pop(ACTION)

#         if self.config.image_features:
#             batch = dict(batch)
#             batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        
#         self._queues = populate_queues(self._queues, batch)

#         if len(self._queues[ACTION]) == 0:
#             actions = self.predict_action_chunk(batch, noise=noise, mask=mask)
#             self._queues[ACTION].extend(actions.transpose(0, 1))

#         action = self._queues[ACTION].popleft()
#         return action

#     def forward(
#         self, 
#         batch: dict[str, Tensor],
#         mask: Tensor | None = None,
#     ) -> tuple[Tensor, None]:
#         """Run the batch through the model and compute the loss for training or validation.
        
#         Args:
#             batch: Batch containing observations and actions
#             mask: (B, horizon, action_dim) optional external mask for conditioning.
#                   Must be provided if config.use_mask=True.
#         """
#         if self.config.image_features:
#             batch = dict(batch)
#             batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
#         loss = self.diffusion.compute_loss(batch, mask=mask)
#         return loss, None

