import torch
import torch.nn as nn

from models.plain_conv import make_mlp


class MasConv1D(nn.Module):
    def __init__(self, in_channels=2, mas_dim=8, out_dim=64, last_act=True):
        super().__init__()
        self.in_channels = in_channels
        self.mas_dim = mas_dim
        self.out_dim = out_dim
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = make_mlp(mas_dim * 32, [out_dim], last_act=last_act)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, mas_and_mask: torch.Tensor):
        if mas_and_mask.ndim != 4:
            raise ValueError(
                f"Expected mas_and_mask to have 4 dims, got shape {tuple(mas_and_mask.shape)}"
            )
        batch_size, channels, mas_dim, horizon = mas_and_mask.shape
        if channels != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, got {channels} in shape {tuple(mas_and_mask.shape)}"
            )
        if mas_dim != self.mas_dim:
            raise ValueError(
                f"Expected mas_dim={self.mas_dim}, got {mas_dim} in shape {tuple(mas_and_mask.shape)}"
            )

        x = mas_and_mask.permute(0, 2, 1, 3).reshape(
            batch_size * mas_dim, channels, horizon
        )
        x = self.temporal_cnn(x)
        x = self.pool(x).flatten(1)
        x = x.reshape(batch_size, mas_dim * x.shape[1])
        x = self.fc(x)
        return x
