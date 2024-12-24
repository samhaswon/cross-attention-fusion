import torch
import torch.nn as nn


class BigDecoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_features: int,
            out_channels: int,
            decode_layer_spread_factor: int = 4,
            dropout_rate: float = 0.05
    ):
        """
        Initialize the extensible decoder.
        :param in_channels: The number of channels of the input tensor.
        :param num_features: The number of features of the input tensor.
        :param out_channels: The number of channels in the intermediate output of the decoder's
        inner layers (should be 1).
        :param decode_layer_spread_factor: The factor by which the inner layers of the decoder
        should be spread by. This controls the in and subsequent out parameters of the
        convolutional layers.
        :param dropout_rate: The dropout rate of the intermediate layers of the decoder.
        """
        super().__init__()
        torch._assert(in_channels % 16 == 0, "In channels must be divisible by 16")
        torch._assert(
            decode_layer_spread_factor > 2,
            "The spread factor of the decode layers must be greater than 2."
        )

        # The main layers of the decoder
        self.decode_layers = nn.ModuleList()
        self.dropouts: nn.ModuleList[nn.Dropout] = nn.ModuleList()
        intermediate_out = num_features - 1
        while in_channels > out_channels:
            self.decode_layers.append(nn.Conv2d(in_channels, intermediate_out, (1, 16)))
            self.dropouts.append(nn.Dropout(p=dropout_rate))  # Add dropout
            in_channels = intermediate_out
            intermediate_out //= decode_layer_spread_factor
        # in_features = num_features - ((kernel_size - 1) * kernel_applications)
        self.decode_final = nn.Linear(num_features - (15 * len(self.decode_layers)), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward call of the decoder.
        :param x: The input tensor.
        :return: Output tensor.
        """
        for decode_layer, dropout in zip(self.decode_layers, self.dropouts):
            x = decode_layer(x)
            x = dropout(x)  # Apply dropout
        x = self.decode_final(x)
        return torch.squeeze(x).unsqueeze(-1)
