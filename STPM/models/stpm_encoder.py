from pathlib import Path
from typing import Sequence, Union

import torch
import torch.nn as nn
from omegaconf import OmegaConf

try:
    from models.clip_encoder import FrozenCLIPEncoder
    from models.rewind_reward_model import RewardTransformer
    from utils.train_utils import get_normalizer_from_calculated
except ImportError:
    from STPM.models.clip_encoder import FrozenCLIPEncoder
    from STPM.models.rewind_reward_model import RewardTransformer
    from STPM.utils.train_utils import get_normalizer_from_calculated


class STPMEncoder(nn.Module):
    """
    Frozen STPM inference wrapper.

    Input:
    - rgbd: [T, 3/4, H, W], [B, T, 3/4, H, W],
      [T, N, 3/4, H, W], or [B, T, N, 3/4, H, W]
    - state: [T, state_dim] or [B, T, state_dim]

    Output:
    - progress: scalar for single sample, or [B] for batched input
      This is the predicted progress of the last timestep.
    """

    def __init__(
        self,
        ckpt_path: str,
        config_path: str = "STPM/config/rewind_maniskill.yaml",
        device: Union[str, torch.device, None] = None,
    ):
        super().__init__()
        self.cfg = OmegaConf.load(self._resolve_path(config_path))
        self.device = torch.device(device or self.cfg.general.device)
        self.task_name = str(self.cfg.general.task_name)
        self.task_description = str(
            getattr(self.cfg.general, "task_description", self.task_name)
        )
        self.state_dim = int(self.cfg.model.state_dim)
        self.state_paths = list(getattr(self.cfg.general, "state_paths", []))

        configured_camera_names = self.cfg.general.camera_names
        if isinstance(configured_camera_names, str) and configured_camera_names.lower() == "auto":
            raise ValueError(
                "STPMEncoder cannot use camera_names='auto'. "
                "Use the config.yaml saved by STPM training, which records concrete camera names."
            )
        self.camera_names = list(configured_camera_names)
        if len(self.camera_names) <= 0:
            raise ValueError("STPMEncoder requires at least one camera name in config.")
        self.camera_name = self.camera_names[0]

        self.clip_encoder = FrozenCLIPEncoder(self.cfg.encoders.vision_ckpt, self.device)
        self.reward_model = RewardTransformer(
            d_model=self.cfg.model.d_model,
            vis_emb_dim=512,
            text_emb_dim=512,
            state_dim=self.state_dim,
            n_layers=self.cfg.model.n_layers,
            n_heads=self.cfg.model.n_heads,
            dropout=self.cfg.model.dropout,
            num_cameras=len(self.camera_names),
        ).to(self.device)
        self.state_normalizer = get_normalizer_from_calculated(
            self.cfg.general.state_norm_path,
            self.device,
            state_dim=self.state_dim,
        )

        ckpt = torch.load(self._resolve_path(ckpt_path), map_location=self.device)
        self.reward_model.load_state_dict(ckpt["model"])

        self.eval()
        self.requires_grad_(False)

    @staticmethod
    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[2]

    @classmethod
    def _resolve_path(cls, path_str: str) -> str:
        path = Path(path_str).expanduser()
        if path.exists():
            return str(path)

        candidate = cls._repo_root() / path
        if candidate.exists():
            return str(candidate)

        return str(path)

    @staticmethod
    def _ensure_batch_dim(x: torch.Tensor, expected_ndim: int, name: str) -> tuple[torch.Tensor, bool]:
        if x.ndim == expected_ndim - 1:
            return x.unsqueeze(0), True
        if x.ndim != expected_ndim:
            raise ValueError(f"Expected {name} to have {expected_ndim - 1} or {expected_ndim} dims, got shape {tuple(x.shape)}")
        return x, False

    def _prepare_rgb(self, rgbd: torch.Tensor) -> tuple[torch.Tensor, bool]:
        num_cameras = len(self.camera_names)
        if rgbd.ndim == 4:
            rgbd = rgbd.unsqueeze(0)
            squeezed = True
        elif rgbd.ndim in (5, 6):
            squeezed = False
        else:
            raise ValueError(
                "Expected rgbd to have 4, 5, or 6 dims, "
                f"got shape {tuple(rgbd.shape)}"
            )

        if (
            rgbd.ndim == 5
            and num_cameras > 1
            and rgbd.shape[1] == num_cameras
            and rgbd.shape[2] in (3, 4)
        ):
            rgb = rgbd.unsqueeze(0)[:, :, :, :3, :, :]
            squeezed = True
        elif rgbd.ndim == 5:
            channels = rgbd.shape[2]
            if channels in (3, 4):
                if num_cameras != 1:
                    raise ValueError(
                        f"STPM checkpoint expects {num_cameras} cameras {self.camera_names}, "
                        f"but got a single-camera tensor with shape {tuple(rgbd.shape)}"
                    )
                rgb = rgbd[:, :, :3, :, :].unsqueeze(2)
            elif channels == 3 * num_cameras:
                rgb = rgbd.reshape(
                    rgbd.shape[0],
                    rgbd.shape[1],
                    num_cameras,
                    3,
                    rgbd.shape[3],
                    rgbd.shape[4],
                )
            elif channels == 4 * num_cameras:
                rgb = rgbd.reshape(
                    rgbd.shape[0],
                    rgbd.shape[1],
                    num_cameras,
                    4,
                    rgbd.shape[3],
                    rgbd.shape[4],
                )[:, :, :, :3, :, :]
            else:
                raise ValueError(
                    f"Expected channel dim 3/4 or 3/4*num_cameras={num_cameras}, "
                    f"got shape {tuple(rgbd.shape)}"
                )
        else:
            if rgbd.shape[2] != num_cameras:
                raise ValueError(
                    f"Expected camera dim {num_cameras}, got shape {tuple(rgbd.shape)}"
                )
            if rgbd.shape[3] not in (3, 4):
                raise ValueError(
                    f"Expected RGB/RGBD channel dim 3 or 4, got shape {tuple(rgbd.shape)}"
                )
            rgb = rgbd[:, :, :, :3, :, :]
        return rgb, squeezed

    def _normalize_tasks(self, tasks: Union[str, Sequence[str], None], batch_size: int) -> list[str]:
        if tasks is None:
            return [self.task_description] * batch_size
        if isinstance(tasks, str):
            return [tasks] * batch_size
        tasks = list(tasks)
        if len(tasks) != batch_size:
            raise ValueError(f"Expected {batch_size} task strings, got {len(tasks)}")
        return tasks

    @torch.no_grad()
    def predict_progress(
        self,
        rgbd: torch.Tensor,
        state: torch.Tensor,
        tasks: Union[str, Sequence[str], None] = None,
    ) -> torch.Tensor:
        rgb, squeezed_rgbd = self._prepare_rgb(rgbd)
        state, squeezed_state = self._ensure_batch_dim(state, expected_ndim=3, name="state")

        if squeezed_rgbd != squeezed_state:
            raise ValueError("rgbd and state must either both have batch dim or both be single trajectories.")
        if rgb.shape[:2] != state.shape[:2]:
            raise ValueError(
                f"Mismatched rgbd/state leading dims: rgbd {tuple(rgb.shape[:2])}, state {tuple(state.shape[:2])}"
            )
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state dim {self.state_dim}, got shape {tuple(state.shape)}")

        batch_size, horizon, num_cameras = rgb.shape[:3]
        lengths = torch.full((batch_size,), horizon, dtype=torch.long, device=self.device)
        task_list = self._normalize_tasks(tasks, batch_size)

        rgb = rgb.to(self.device)
        state = self.state_normalizer.normalize(state.to(self.device))
        if self.cfg.model.no_state:
            state = torch.zeros_like(state)

        rgb = rgb.permute(2, 0, 1, 3, 4, 5).reshape(
            num_cameras * batch_size * horizon,
            3,
            rgb.shape[-2],
            rgb.shape[-1],
        )
        img_emb = self.clip_encoder.encode_image(rgb)
        img_emb = img_emb.view(num_cameras, batch_size, horizon, -1).permute(1, 0, 2, 3)
        lang_emb = self.clip_encoder.encode_text(task_list)
        progress_seq = self.reward_model(img_emb, lang_emb, state, lengths)
        last_progress = progress_seq[:, -1]

        if squeezed_rgbd:
            return last_progress.squeeze(0)
        return last_progress

    @torch.no_grad()
    def forward(
        self,
        rgbd: torch.Tensor,
        state: torch.Tensor,
        tasks: Union[str, Sequence[str], None] = None,
    ) -> torch.Tensor:
        return self.predict_progress(rgbd=rgbd, state=state, tasks=tasks)
