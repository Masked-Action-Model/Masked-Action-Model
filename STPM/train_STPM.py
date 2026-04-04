import os
from pathlib import Path
from omegaconf import OmegaConf
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from tqdm import tqdm
import wandb

from maniskill_dataset import FrameManiskillDataset
from utils.data_utils import get_valid_episodes, split_train_eval_episodes, adapt_maniskill_batch_rewind
from utils.train_utils import set_seed, save_ckpt, get_normalizer_from_calculated
from models.rewind_reward_model import RewardTransformer
from models.clip_encoder import FrozenCLIPEncoder

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_IGNORE_GLOBS"] = "**/rollout/**"


def _select_clip_rgb(images: torch.Tensor) -> torch.Tensor:
    if images.shape[2] < 3:
        raise ValueError(f"Expected at least 3 channels for CLIP input, got shape {tuple(images.shape)}")
    return images[:, :, :3, :, :]


class ReWiNDWorkspace:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.general.device if torch.cuda.is_available() else "cpu")
        print(f"[Init] Using device: {self.device}")
        set_seed(cfg.general.seed)
        self.camera_names = cfg.general.camera_names
        self.save_dir = Path(f'{cfg.general.project_name}/{cfg.general.task_name}')
        self.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Init] Logging & ckpts to: {self.save_dir}")

    def train(self):
        cfg = self.cfg
        OmegaConf.save(cfg, self.save_dir / "config.yaml")
        wandb_mode = os.environ.get("WANDB_MODE", "disabled")
        # --- wandb ---
        wandb.init(
            project=f'{cfg.general.project_name}-{cfg.general.task_name}',
            name=f'{datetime.now().strftime("%Y.%m.%d-%H.%M.%S")}',
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=wandb_mode,
        )

        # --- data ---
        valid_episodes = get_valid_episodes(cfg.general.repo_id)
        train_eps, val_eps = split_train_eval_episodes(valid_episodes, 1 - cfg.train.val_portion, seed=cfg.general.seed)

        dataset_train = FrameManiskillDataset(
            repo_id=cfg.general.repo_id,
            episodes=train_eps,
            n_obs_steps=cfg.model.n_obs_steps,
            frame_gap=cfg.model.frame_gap,
            image_names=cfg.general.camera_names,
            task_name=cfg.general.task_name,
        )

        dataset_val = FrameManiskillDataset(
            repo_id=cfg.general.repo_id,
            episodes=val_eps,
            n_obs_steps=cfg.model.n_obs_steps,
            frame_gap=cfg.model.frame_gap,
            image_names=cfg.general.camera_names,
            task_name=cfg.general.task_name,
        )

        dataloader_train = torch.utils.data.DataLoader(dataset_train, **cfg.dataloader)
        dataloader_val   = torch.utils.data.DataLoader(dataset_val, **cfg.val_dataloader)
        state_normalizer = get_normalizer_from_calculated(
            cfg.general.state_norm_path,
            self.device,
            state_dim=cfg.model.state_dim,
        )

        # CLIP encoder
        clip_encoder = FrozenCLIPEncoder(cfg.encoders.vision_ckpt, self.device)
        vis_dim = 512
        txt_dim = 512

        # --- reward_model ---
        reward_model = RewardTransformer(d_model=cfg.model.d_model, 
                                  vis_emb_dim=vis_dim, 
                                  text_emb_dim=txt_dim,
                                  state_dim=cfg.model.state_dim,
                                  n_layers=cfg.model.n_layers,
                                  n_heads=cfg.model.n_heads,
                                  dropout=cfg.model.dropout,
                                  num_cameras=len(self.camera_names),
                                  ).to(self.device)
        
        if cfg.model.resume_training:
            reward_model_path = Path(cfg.model.model_path)
            reward_ckpt = torch.load(reward_model_path, map_location=self.device)
            reward_model.load_state_dict(reward_ckpt["model"])
            reward_model.to(self.device)
            reward_model.train()

        # Optimizer
        reward_optimizer = torch.optim.AdamW(
            reward_model.parameters(),
            lr=cfg.optim.lr,
            betas=tuple(cfg.optim.betas),
            eps=cfg.optim.eps,
            weight_decay=cfg.optim.weight_decay,
        )
        
        # Schedulers
        reward_warmup_scheduler = LinearLR(
            reward_optimizer,
            start_factor=1e-6 / cfg.optim.lr,  
            end_factor=1.0,
            total_iters=cfg.optim.warmup_steps
        )
        reward_cosine_scheduler = CosineAnnealingLR(
            reward_optimizer,
            T_max=cfg.optim.total_steps - cfg.optim.warmup_steps,  
            eta_min=0.0
        )
        reward_scheduler = SequentialLR(
            reward_optimizer,
            schedulers=[reward_warmup_scheduler, reward_cosine_scheduler],
            milestones=[cfg.optim.warmup_steps]
        )

        # ==================== training loop ==================================
        best_val = float("inf")
        step = 0
        for epoch in range(1, cfg.train.num_epochs + 1):
            reward_model.train()
            with tqdm(dataloader_train, desc=f"Epoch {epoch}") as pbar:
                for batch in pbar:
                    batch = adapt_maniskill_batch_rewind(batch, camera_names=cfg.general.camera_names)

                    B, T = batch["image_frames"][self.camera_names[0]].shape[:2]
                    img_list = []
                    for key in self.camera_names:
                        imgs = _select_clip_rgb(batch["image_frames"][key]).flatten(0, 1).to(self.device)
                        img_list.append(imgs)
                    
                    lang_strs = batch["tasks"]
                    trg = batch["targets"].to(self.device)
                    lens = batch["lengths"].to(self.device)
                    state = batch["state"].to(self.device)
                    
                    with torch.no_grad():
                        state = state_normalizer.normalize(state)
                        # CLIP encoding
                        imgs_all = torch.cat(img_list, dim=0)  # (N * B * T, C, H, W)
                        img_emb = clip_encoder.encode_image(imgs_all)  # (N * B * T, D)
                        img_emb = img_emb.view(len(img_list), B, T, -1).permute(1, 0, 2, 3)  # (B, N, T, D)
                        lang_emb = clip_encoder.encode_text(lang_strs) # lang_emb: (B, txt_dim)

                    if cfg.model.no_state:
                        state = torch.zeros_like(state, device=self.device)
                    reward_pred = reward_model(img_emb, lang_emb, state, lens)
                    reward_loss = F.mse_loss(reward_pred, trg, reduction="mean")

                    reward_optimizer.zero_grad()
                    reward_loss.backward()
                    reward_unclipped = nn.utils.clip_grad_norm_(reward_model.parameters(), float("inf")).item()
                    _ = nn.utils.clip_grad_norm_(reward_model.parameters(), cfg.train.grad_clip)
                    reward_optimizer.step()
                    reward_scheduler.step()
                    
                    if step % cfg.train.log_every == 0:
                        wandb.log({
                            "train/total_loss": reward_loss.item(),
                            "train/lr": reward_scheduler.get_last_lr()[0],
                            "train/reward_grad_norm": reward_unclipped,
                            "epoch": epoch,
                        }, step=step)
                    
                    pbar.set_postfix(loss=f"{(reward_loss.item()):.4f}")

                    if step % cfg.train.save_every == 0:
                        save_ckpt(reward_model, reward_optimizer, epoch, self.save_dir, input_name=f"reward_step_{step:06d}_loss_{reward_loss.item():.3f}")
                    step += 1

            # --- validation ---
            if epoch % cfg.train.eval_every == 0:
                reward_model.eval()
                total_loss, num = 0.0, 0
                print("running validation...")
                with torch.no_grad():
                    for batch in dataloader_val:
                        batch = adapt_maniskill_batch_rewind(batch, camera_names=cfg.general.camera_names)
                        B, T = batch["image_frames"][self.camera_names[0]].shape[:2]
                        img_list = []
                        for key in self.camera_names:
                            imgs = _select_clip_rgb(batch["image_frames"][key]).flatten(0, 1).to(self.device)
                            img_list.append(imgs)
                        
                        lang_strs = batch["tasks"]
                        trg = batch["targets"].to(self.device)
                        lens = batch["lengths"].to(self.device)
                        state = batch["state"].to(self.device)
                        state = state_normalizer.normalize(state)

                        # CLIP encoding
                        imgs_all = torch.cat(img_list, dim=0)  # (N * B * T, C, H, W)
                        img_emb = clip_encoder.encode_image(imgs_all)  # (N * B * T, D)
                        img_emb = img_emb.view(len(img_list), B, T, -1).permute(1, 0, 2, 3)  # (B, N, T, D)
                        lang_emb = clip_encoder.encode_text(lang_strs) # lang_emb: (B, txt_dim)

                        if cfg.model.no_state:
                            state = torch.zeros_like(state, device=self.device)
                        reward_pred = reward_model(img_emb, lang_emb, state, lens)
                        reward_loss = F.mse_loss(reward_pred, trg, reduction="mean")
                        total_loss += reward_loss.item()
                        num += 1

                val_loss = total_loss / num 
                print(f"[Eval] Epoch {epoch} Val L1: {val_loss:.6f}")
                wandb.log({"val/loss": val_loss}, step=step)

            torch.cuda.empty_cache()

            # --- save checkpoints ---
            save_ckpt(reward_model, reward_optimizer, epoch, self.save_dir, input_name="reward_latest")
            
            if epoch == cfg.train.num_epochs:
                save_ckpt(reward_model, reward_optimizer, epoch, self.save_dir, input_name="reward_final")
            
            if val_loss < best_val:
                best_val = val_loss
                save_ckpt(reward_model, reward_optimizer, epoch, self.save_dir, input_name="reward_best")

        print(f"Training done. Best val_loss MSE = {best_val}")
        wandb.finish()


if __name__ == "__main__":
    config_path = Path(__file__).resolve().parent / "config" / "rewind_maniskill.yaml"
    workspace = ReWiNDWorkspace(cfg=OmegaConf.load(config_path))
    workspace.train()
