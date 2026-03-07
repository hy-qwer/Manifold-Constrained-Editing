import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleViTDecoder(nn.Module):
    def __init__(self, z_dim=768, base_channels=256, img_size=128):
        super().__init__()
        assert img_size == 128,"8->16->32->64->128"
        self.z_dim = z_dim
        self.base_channels = base_channels
        self.img_size = img_size

        self.fc = nn.Linear(z_dim, 8 * 8 * base_channels)

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels // 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels // 2),
            nn.LeakyReLU(0.2, inplace=True),
        )  # [B, C/2, 16, 16]

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels // 2, base_channels // 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels // 4),
            nn.LeakyReLU(0.2, inplace=True),
        )  # [B, C/4, 32, 32]

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels // 4, base_channels // 8, 4, 2, 1),
            nn.BatchNorm2d(base_channels // 8),
            nn.LeakyReLU(0.2, inplace=True),
        )  # [B, C/8, 64, 64]

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(base_channels // 8, base_channels // 16, 4, 2, 1),
            nn.BatchNorm2d(base_channels // 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels // 16, 3, 3, 1, 1),
            nn.Tanh(),  # 输出范围 [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, D]
        return: [B, 3, 128, 128], range ~ [-1, 1]
        """
        x = self.fc(z)  # [B, 8*8*C]
        x = x.view(z.size(0), self.base_channels, 8, 8)  # [B, C, 8, 8]
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = self.act(out + identity)
        return out


class MAEImageDecoder(nn.Module):
    def __init__(
        self,
        z_dim: int = 768,
        img_size: int = 128,
        base_channels: int = 256,
        num_res_blocks: int = 2,
    ):
        super().__init__()
        assert img_size in (128, 256), "128或256"

        self.z_dim = z_dim
        self.img_size = img_size
        self.base_channels = base_channels

        if img_size == 128:
            init_hw = 16
        else:
            init_hw = 32
        self.init_hw = init_hw

        self.fc = nn.Linear(z_dim, base_channels * init_hw * init_hw)

        ups = []
        in_ch = base_channels
        cur_hw = init_hw
        while cur_hw < img_size:
            out_ch = in_ch // 2
            block = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                *[ResidualBlock(out_ch) for _ in range(num_res_blocks)],
            )
            ups.append(block)
            in_ch = out_ch
            cur_hw *= 2

        self.up_blocks = nn.ModuleList(ups)

        self.to_rgb = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        x = self.fc(z)
        x = x.view(B, self.base_channels, self.init_hw, self.init_hw)

        for block in self.up_blocks:
            x = block(x)

        img = self.to_rgb(x)
        return img


def build_decoder_from_ckpt(ckpt_path: str, z_dim: int, img_size: int = 128) -> MAEImageDecoder:
    decoder = MAEImageDecoder(z_dim=z_dim, img_size=img_size)
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "decoder" in state:
        state = state["decoder"]
    decoder.load_state_dict(state, strict=True)
    return decoder
