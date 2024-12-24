import torch
import torch.nn as nn

from .vit import VisionTransformer
# from .decoder import Decoder
from .big_decoder import BigDecoder


class FullViT(nn.Module):
    def __init__(
            self,
            image_size: int,
            patch_size: int,
            num_layers: int,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float = 0.0,
            attention_dropout: float = 0.0
    ):
        super().__init__()

        # The transformer model
        self.vit = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            num_classes=1
        )

        self.decoder = BigDecoder(768, 1025, 1, dropout_rate=0.1)

    def forward(self, x: torch.Tensor):
        # Modality inferencing
        output = self.vit(x)

        # Rearrange the dimensions for self-attention
        output = output.permute(2, 0, 1)

        # Permute to (embed_dim, batch_size, seq_len)
        # output = feature_tensor.permute(2, 1, 0)

        # Feed the result to the CNN decoder
        output = self.decoder(output)
        return output
