import torch
import torch.nn as nn

from .vit import VisionTransformer
from .decoder import Decoder


class EarlyConcat(nn.Module):
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

        self.__decoder = Decoder(768, 771, 1)

    def forward(self, x: torch.Tensor):
        # Modality inferencing
        text = self.__vit0(x)
        image = self.__vit1(x)
        graph = self.__vit2(x)

        # Combine the features
        feature_tensor = torch.cat([text, image, graph], dim=1)

        feature_tensor = feature_tensor.permute(2, 0, 1)

        # Feed the feature tensor to the CNN
        output = self.__decoder(feature_tensor)
        return output
