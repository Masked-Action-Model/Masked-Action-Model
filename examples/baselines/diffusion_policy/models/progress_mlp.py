import torch
import torch.nn as nn

from models.plain_conv import make_mlp


class ProgressMLP(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=32, out_dim=8, last_act=True):
        super().__init__()
        self.out_dim = out_dim
        self.mlp = make_mlp(in_dim, [hidden_dim, out_dim], last_act=last_act)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, progress: torch.Tensor) -> torch.Tensor:
        if progress.ndim != 2:
            raise ValueError(
                f"Expected progress to have shape (batch, 1), got {tuple(progress.shape)}"
            )
        if progress.shape[-1] != 1:
            raise ValueError(
                f"Expected progress last dim to be 1, got {tuple(progress.shape)}"
            )
        return self.mlp(progress)
