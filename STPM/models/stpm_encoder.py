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
    - rgbd: [T, 4, H, W] or [B, T, 4, H, W]
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

        camera_names = list(self.cfg.general.camera_names)
        if len(camera_names) != 1:
            raise ValueError(f"STPMEncoder only supports single-camera inference, got cameras={camera_names}")
        self.camera_name = camera_names[0]

        self.clip_encoder = FrozenCLIPEncoder(self.cfg.encoders.vision_ckpt, self.device)
        self.reward_model = RewardTransformer(
            d_model=self.cfg.model.d_model,
            vis_emb_dim=512,
            text_emb_dim=512,
            state_dim=self.state_dim,
            n_layers=self.cfg.model.n_layers,
            n_heads=self.cfg.model.n_heads,
            dropout=self.cfg.model.dropout,
            num_cameras=1,
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

    @staticmethod
    def _select_rgb(rgbd: torch.Tensor) -> torch.Tensor:
        if rgbd.shape[2] < 3:
            raise ValueError(f"Expected RGBD tensor with at least 3 channels, got shape {tuple(rgbd.shape)}")
        return rgbd[:, :, :3, :, :]

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
        rgbd, squeezed_rgbd = self._ensure_batch_dim(rgbd, expected_ndim=5, name="rgbd")
        state, squeezed_state = self._ensure_batch_dim(state, expected_ndim=3, name="state")

        if squeezed_rgbd != squeezed_state:
            raise ValueError("rgbd and state must either both have batch dim or both be single trajectories.")
        if rgbd.shape[:2] != state.shape[:2]:
            raise ValueError(
                f"Mismatched rgbd/state leading dims: rgbd {tuple(rgbd.shape[:2])}, state {tuple(state.shape[:2])}"
            )
        if rgbd.shape[2] != 4:
            raise ValueError(f"Expected rgbd channel dim to be 4, got shape {tuple(rgbd.shape)}")
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state dim {self.state_dim}, got shape {tuple(state.shape)}")

        batch_size, horizon = rgbd.shape[:2]
        lengths = torch.full((batch_size,), horizon, dtype=torch.long, device=self.device)
        task_list = self._normalize_tasks(tasks, batch_size)

        rgbd = rgbd.to(self.device)
        state = self.state_normalizer.normalize(state.to(self.device))
        if self.cfg.model.no_state:
            state = torch.zeros_like(state)

        rgb = self._select_rgb(rgbd).flatten(0, 1)
        img_emb = self.clip_encoder.encode_image(rgb)
        img_emb = img_emb.view(batch_size, horizon, -1).unsqueeze(1)
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
