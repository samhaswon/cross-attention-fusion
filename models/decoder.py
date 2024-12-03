import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, in_channels, num_features, out_channels):
        super().__init__()
        self.decode_0 = nn.Conv2d(in_channels, 1024, (1, 64))
        self.decode_1 = nn.Conv2d(1024, out_channels, (1, 16))
        # in_features = model features - 63 - 15
        self.decode_2 = nn.Linear(num_features - 63 - 15, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        x = self.decode_0(x)
        x = self.decode_1(x)
        x = self.decode_2(x)
        # x = self.sigmoid(x)
        return torch.squeeze(x).unsqueeze(-1)
