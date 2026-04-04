import torch
import torch.nn as nn

from models.plain_conv import make_mlp


class MasConv(nn.Module):
    def __init__(self, in_channels=2, out_dim=64, last_act=True):
        super().__init__()
        self.out_dim = out_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = make_mlp(32, [out_dim], last_act=last_act)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, mas_and_mask: torch.Tensor):
        x = self.cnn(mas_and_mask)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
