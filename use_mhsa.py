from models.mhsa_vit import MHSAViT
import torch

if __name__ == '__main__':
    # 855MB of memory
    net = MHSAViT(
        image_size=512,
        patch_size=32,
        num_layers=6,
        num_heads=6,
        hidden_dim=768,
        mlp_dim=1536,
        dropout=0.01,
        attention_dropout=0
    )

    x = torch.rand((1, 3, 512, 512))
    result = net(x)
    print(result.data)
