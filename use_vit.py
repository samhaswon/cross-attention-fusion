from models.vit import VisionTransformer
import torch

if __name__ == '__main__':
    net = VisionTransformer(
        image_size=512,
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        dropout=0.01,
        attention_dropout=0,
        num_classes=1
    )

    x = torch.rand((1, 3, 512, 512))
    result = net(x)
    print(result.data)
