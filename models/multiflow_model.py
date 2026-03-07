import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from models.vit_backbone import ViTBackbone


class TimeEmbedding(nn.Module):
    def __init__(self, time_embed_dim: int = 128):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, t_scalar: torch.Tensor) -> torch.Tensor:
        if t_scalar.dim() == 1:
            t_scalar = t_scalar.unsqueeze(-1)   # [B,1]
        return self.mlp(t_scalar)


class FlowBlock(nn.Module):
    def __init__(self, z_dim: int, time_embed_dim: int = 128, hidden_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(z_dim + time_embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, z_dim)
        self.act = nn.SiLU()

    def forward(self, z: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = torch.cat([z, t_emb], dim=-1)  # [B, D+T]
        h = self.act(self.fc1(h))
        dz = self.fc2(h)
        return z + dz  # 残差更新


class ModulatedLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, semantic_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.modulation = nn.Linear(semantic_dim, out_dim * 2)
        self.act = nn.GELU()
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_dim]
        s: [B, semantic_dim]
        """
        h = self.linear(x)
        mod = self.modulation(s)
        scale, shift = torch.chunk(mod, 2, dim=-1)

        h = h * (1.0 + scale) + shift
        return self.act(h)


class BranchFlow(nn.Module):

    def __init__(
        self,
        z_dim: int,
        hidden_dim: int = 1024,
        out_dim: Optional[int] = None,
        num_layers: int = 3,
        use_semantic: bool = True,
        semantic_dim: int = 1,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.out_dim = out_dim or z_dim
        self.use_semantic = use_semantic
        self.semantic_dim = semantic_dim

        if use_semantic:
            self.semantic_head = nn.Sequential(
                nn.Linear(z_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, semantic_dim),
            )

            layers: list[nn.Module] = []
            curr_dim = z_dim
            for _ in range(num_layers - 1):
                layers.append(ModulatedLayer(curr_dim, hidden_dim, semantic_dim))
                curr_dim = hidden_dim
            self.layers = nn.ModuleList(layers)
            self.out_layer = nn.Linear(curr_dim, self.out_dim)
        else:
            in_dim = z_dim
            layers: list[nn.Module] = []
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.GELU())
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, self.out_dim))
            self.net = nn.Sequential(*layers)

    def encode_semantic(self, z: torch.Tensor) -> torch.Tensor:
        if not self.use_semantic:
            return z.new_zeros(z.size(0), 0)
        return self.semantic_head(z)

    def forward(
        self,
        z: torch.Tensor,
        semantic_override: Optional[torch.Tensor] = None,
        return_semantic: bool = False,
    ):

        if not self.use_semantic:
            delta = self.net(z)
            if return_semantic:
                semantic = z.new_zeros(z.size(0), 0)
                return delta, semantic
            return delta

        if semantic_override is not None:
            s = semantic_override
            if s.dim() == 1:
                s = s.unsqueeze(-1)
        else:
            s = self.semantic_head(z)

        h = z
        for layer in self.layers:
            h = layer(h, s)

        delta = self.out_layer(h)

        if return_semantic:
            return delta, s
        return delta


class MainFlow(nn.Module):
    def __init__(
        self,
        z_dim: int,
        num_blocks: int = 8,
        time_embed_dim: int = 128,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.num_blocks = num_blocks
        self.time_embed = TimeEmbedding(time_embed_dim=time_embed_dim)
        self.blocks = nn.ModuleList(
            [FlowBlock(z_dim, time_embed_dim, hidden_dim) for _ in range(num_blocks)]
        )

    def _time_scalar(self, step_idx: int, num_blocks: int, batch_size: int, device) -> torch.Tensor:
        t_val = (step_idx + 0.5) / num_blocks
        t_scalar = torch.full((batch_size,), t_val, device=device, dtype=torch.float32)
        return t_scalar

    def forward_full(self, z0: torch.Tensor) -> torch.Tensor:
        z = z0
        B = z.size(0)
        device = z.device
        for i, block in enumerate(self.blocks):
            t_scalar = self._time_scalar(i, self.num_blocks, B, device)
            t_emb = self.time_embed(t_scalar)
            z = block(z, t_emb)
        return z

    def forward_split(
        self,
        z0: torch.Tensor,
        k_t: int,
        k_tx: int,
    ):

        assert 0 <= k_t <= k_tx <= self.num_blocks, "k_t, k_tx 必须满足 0 ≤ k_t ≤ k_tx ≤ num_blocks"

        z = z0
        B = z.size(0)
        device = z.device

        for i in range(0, k_t):
            t_scalar = self._time_scalar(i, self.num_blocks, B, device)
            t_emb = self.time_embed(t_scalar)
            z = self.blocks[i](z, t_emb)
        z_t = z

        for i in range(k_t, k_tx):
            t_scalar = self._time_scalar(i, self.num_blocks, B, device)
            t_emb = self.time_embed(t_scalar)
            z = self.blocks[i](z, t_emb)
        z_tx = z

        for i in range(k_tx, self.num_blocks):
            t_scalar = self._time_scalar(i, self.num_blocks, B, device)
            t_emb = self.time_embed(t_scalar)
            z = self.blocks[i](z, t_emb)
        z_end = z

        return z_t, z_tx, z_end
    
    def forward_with_continuous_time(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            t_emb = self.time_embed(t.unsqueeze(-1) if t.dim()==1 else t)
            z = x
            for block in self.blocks:
                z = block(z, t_emb)
            return z

class MultiFlowModel(nn.Module):
    def __init__(
        self,
        vit_name: str = "vit_base_patch16_224",
        vit_pretrained: bool = True,
        vit_freeze: bool = True,
        num_main_blocks: int = 8,
        k_t: int = 3,
        k_tx: int = 6,
        num_branches: int = 2,
        main_hidden_dim: int = 1024,
        branch_hidden_dim: int = 1024,
        time_embed_dim: int = 128,
        global_pool: str = "token",
        branch_semantic_dim: int = 1,
        branch_use_semantic: bool = True,
    ):
        super().__init__()

        self.num_branches = num_branches
        self.branch_semantic_dim = branch_semantic_dim
        self.branch_use_semantic = branch_use_semantic

        self.vit = ViTBackbone(
            model_name=vit_name,
            pretrained=vit_pretrained,
            global_pool=global_pool,
            freeze=vit_freeze,
        )
        z_dim = self.vit.embed_dim
        self.z_dim = z_dim

        self.main_flow = MainFlow(
            z_dim=z_dim,
            num_blocks=num_main_blocks,
            time_embed_dim=time_embed_dim,
            hidden_dim=main_hidden_dim,
        )

        self.branches = nn.ModuleList(
            [
                BranchFlow(
                    z_dim=self.z_dim,
                    hidden_dim=branch_hidden_dim,
                    out_dim=self.z_dim,
                    num_layers=3,
                    semantic_dim=self.branch_semantic_dim,
                    use_semantic=self.branch_use_semantic,
                )
                for _ in range(num_branches)
            ]
        )

        self.k_t = k_t
        self.k_tx = k_tx
        assert 0 <= k_t <= k_tx <= num_main_blocks


    def forward(self, images: torch.Tensor, return_intermediate: bool = False):
        z0 = self.vit(images)

        z_t, z_tx, z_base_end = self.main_flow.forward_split(
            z0, k_t=self.k_t, k_tx=self.k_tx
        )

        delta_list = []

        for branch in self.branches:
            delta_k = branch(z_t)
            delta_list.append(delta_k)
        delta_total = torch.stack(delta_list, dim=0).sum(0)

        z_fused_tx = z_tx + delta_total

        z = z_fused_tx
        B = z.size(0)
        device = z.device
        N = self.main_flow.num_blocks
        for i in range(self.k_tx, N):
            t_scalar = self.main_flow._time_scalar(i, N, B, device)
            t_emb = self.main_flow.time_embed(t_scalar)
            z = self.main_flow.blocks[i](z, t_emb)
        z_out = z

        if return_intermediate:
            return {
                "z0": z0,
                "z_t": z_t,
                "z_tx": z_tx,
                "z_base_end": z_base_end,
                "delta_list": delta_list,
                "delta_total": delta_total,
                "z_fused_tx": z_fused_tx,
                "z_out": z_out,
            }
        else:
            return z_out

class FlowBaselineModel(nn.Module):

    def __init__(
        self,
        z_dim: int = 768,
        num_main_blocks: int = 8,
        main_hidden_dim: int = 1024,
        time_embed_dim: int = 128,
    ):
        super().__init__()
        self.z_dim = z_dim
        
        self.main_flow = MainFlow(
            z_dim=z_dim,
            num_blocks=num_main_blocks,
            time_embed_dim=time_embed_dim,
            hidden_dim=main_hidden_dim,
        )

    def forward_from_latent(self, z0: torch.Tensor) -> torch.Tensor:
        return self.main_flow.forward_full(z0)

    def forward(self, z0: torch.Tensor) -> torch.Tensor:
        return self.forward_from_latent(z0)

    def forward_velocity(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self.main_flow.forward_with_continuous_time(x_t, t)