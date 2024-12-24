import torch
import torch.nn as nn

from .vit import VisionTransformer
from .decoder import Decoder


class MHSAViT(nn.Module):
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

        # The "three modes" of the model
        self.__vit0 = VisionTransformer(
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
        self.__vit1 = VisionTransformer(
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
        self.__vit2 = VisionTransformer(
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

        self.__mhsa = nn.MultiheadAttention(
            embed_dim=hidden_dim,               # Basically, this just needs to be the hidden dim of the ViT
            num_heads=num_heads                 # Attention heads
        )

        self.__decoder = Decoder(768, 3075, 1)

    def forward(self, x: torch.Tensor):
        # Modality inferencing
        text = self.__vit0(x)
        image = self.__vit1(x)
        graph = self.__vit2(x)

        # Combine the features
        feature_tensor = torch.cat([text, image, graph], dim=1)

        # Rearrange the dimensions for self-attention
        feature_tensor = feature_tensor.permute(1, 0, 2)

        # Q, K, and V are the same for MHSA
        output, _ = self.__mhsa(feature_tensor, feature_tensor, feature_tensor)

        # Permute to (embed_dim, batch_size, seq_len)
        output = output.permute(2, 1, 0)

        # Feed the result to the CNN decoder
        output = self.__decoder(output)
        return output
