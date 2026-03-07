# models/vit_backbone.py
import torch
import torch.nn as nn
import timm

class ViTBackbone(nn.Module):
    def __init__(self,
                 model_name: str = "vit_base_patch16_224",
                 pretrained: bool = True,
                 global_pool: str = "token",
                 freeze: bool = True):
        super().__init__()
        self.model_name = model_name
        self.global_pool = global_pool

        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
        )

        self.embed_dim = self.vit.num_features

        if freeze:
            self.freeze_backbone()

    def freeze_backbone(self):
        for p in self.vit.parameters():
            p.requires_grad = False
        self.vit.eval()

    def unfreeze_backbone(self):
        for p in self.vit.parameters():
            p.requires_grad = True
        self.vit.train()

    @torch.no_grad()
    def encode_image_nograd(self, x: torch.Tensor) -> torch.Tensor:
        self.vit.eval()
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.vit.forward_features(x)

        if isinstance(feats, dict):
            if "x" in feats:
                feats = feats["x"]
            elif "global" in feats:
                feats = feats["global"]
            else:
                raise ValueError(f"Unknown features dict keys: {feats.keys()}")

        if feats.dim() == 3:
            cls_token = feats[:, 0]
            if self.global_pool == "token":
                z0 = cls_token
            elif self.global_pool == "avg":
                patch_tokens = feats[:, 1:]  # [B, N, D]
                z0 = patch_tokens.mean(dim=1)
            else:
                raise ValueError(f"Unsupported global_pool: {self.global_pool}")
        elif feats.dim() == 2:
            z0 = feats
        else:
            raise ValueError(f"Unexpected feature shape: {feats.shape}")

        return z0
