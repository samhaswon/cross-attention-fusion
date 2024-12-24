from models.full_vit import FullViT
import torch

if __name__ == '__main__':
    # 855MB of memory
    net = FullViT(
        image_size=1024,
        patch_size=32,
        num_layers=6,
        num_heads=6,
        hidden_dim=768,
        mlp_dim=1536,
        dropout=0.01,
        attention_dropout=0
    )

    x = torch.rand((2, 3, 1024, 1024))
    result = net(x)
    print(result.data)
