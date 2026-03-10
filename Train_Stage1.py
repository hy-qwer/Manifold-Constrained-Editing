#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stage 1 training script.
"""

import argparse
from pathlib import Path
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm
from PIL import Image

from models.vit_backbone import ViTBackbone
from models.multiflow_model import FlowBaselineModel


class SimpleCelebADataset(Dataset):
    def __init__(self, root, attr_path=None, split="train", split_ratio=0.9, transform=None):
        self.root = root
        self.transform = transform
        self.split = split
        self.split_ratio = split_ratio
        valid_exts = {".jpg", ".png", ".jpeg", ".bmp", ".webp"}
        disk_rel_paths = []
        print(f"[S1] Scanning images in: {root} ...")
        for current_dir, _, filenames in os.walk(root):
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext in valid_exts:
                    full_path = os.path.join(current_dir, fname)
                    rel_path = os.path.relpath(full_path, root)
                    disk_rel_paths.append(rel_path)
        if attr_path and os.path.exists(attr_path):
            print(f"[S1] Filtering using attribute file: {attr_path}")
            valid_keys = self._load_valid_keys(attr_path)

            final_paths = []
            for p in disk_rel_paths:
                if (p in valid_keys) or (p.replace("\\", "/") in valid_keys):
                    final_paths.append(p)
                else:
                    root_name, ext = os.path.splitext(p)
                    if ext.lower() == ".png":
                        p_jpg = root_name + ".jpg"
                        if (p_jpg in valid_keys) or (p_jpg.replace("\\", "/") in valid_keys):
                            final_paths.append(p)

            print(f"[S1] Filtered: {len(disk_rel_paths)} -> {len(final_paths)} images.")
            self.rel_paths = final_paths
        else:
            print("[S1] No attr_path provided. Using all scanned images.")
            self.rel_paths = disk_rel_paths

        self.rel_paths.sort()

        total_imgs = len(self.rel_paths)
        if total_imgs == 0:
            raise RuntimeError("No images found after filtering.")

        split_idx = int(total_imgs * split_ratio)

        if split == "train":
            self.files = self.rel_paths[:split_idx]
            print(f"[S1 Dataset] Split='train': {len(self.files)} / {total_imgs}")
        elif split == "test":
            self.files = self.rel_paths[split_idx:]
            print(f"[S1 Dataset] Split='test': {len(self.files)} / {total_imgs}")
        else:
            self.files = self.rel_paths

    def _load_valid_keys(self, attr_path):
        valid_keys = set()
        print(f"Loading attributes from {attr_path} ...")
        with open(attr_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        print(f"Found {len(lines)} lines in attr file.")
        for i, line in enumerate(lines):
            if i < 2:
                continue
            parts = line.split()
            if len(parts) > 1:
                key = parts[0]
                valid_keys.add(key)
                valid_keys.add(key.replace("\\", "/"))
        print(f"Loaded {len(valid_keys)} valid keys.")
        return valid_keys

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        rel_path = self.files[idx]
        full_path = os.path.join(self.root, rel_path)
        try:
            img = Image.open(full_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, 0
        except Exception:
            return self.__getitem__((idx + 1) % len(self))


def build_dataloader(data_root: str, attr_path: str, batch_size: int = 32, num_workers: int = 0) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = SimpleCelebADataset(
        root=data_root,
        split="train",
        split_ratio=0.9,
        attr_path=attr_path,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


def train(
    data_root: str,
    ckpt_path: Path,
    attr_path: str = None,
    log_dir: str = "runs/stage1",
    num_epochs: int = 5,
    log_interval: int = 5,
    batch_size: int = 32,
    num_workers: int = 0,
    lr: float = 1e-4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    loader = build_dataloader(
        data_root,
        attr_path=attr_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    teacher = ViTBackbone(
        model_name="vit_base_patch16_224",
        pretrained=True,
        global_pool="token",
        freeze=True,
    ).to(device)

    model = FlowBaselineModel(
        z_dim=768,
        num_main_blocks=16,
        main_hidden_dim=4096,
        time_embed_dim=256,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(log_dir))

    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}", ncols=100, leave=True)

        for _, (images, _) in enumerate(pbar):
            images = images.to(device, non_blocking=True)

            with torch.no_grad():
                x_star = teacher.encode_image_nograd(images)

            z0 = 0.2 * torch.randn_like(x_star) + x_star
            z_out = model.forward_from_latent(z0)

            loss_mse = F.mse_loss(z_out, x_star)
            cos_sim = F.cosine_similarity(z_out, x_star, dim=-1).mean()
            loss_cos = 1.0 - cos_sim
            loss = loss_mse + loss_cos

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            if global_step % log_interval == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    mse=f"{loss_mse.item():.4f}",
                    cos=f"{cos_sim.item():.4f}",
                )
                tb_writer.add_scalar("loss/total", loss.item(), global_step)
                tb_writer.add_scalar("loss/mse", loss_mse.item(), global_step)
                tb_writer.add_scalar("loss/1-cos", loss_cos.item(), global_step)

        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }
        torch.save(state, ckpt_path)
        print(f"Epoch {epoch} done, saved checkpoint to {ckpt_path}")

    tb_writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train Stage-1 flow baseline.")
    parser.add_argument("--data-root", type=str, required=True, help="Path to the image root directory.")
    parser.add_argument("--attr-path", type=str, default="", help="Optional attribute annotation file used to filter valid images.")
    parser.add_argument("--ckpt", type=str, default="checkpoints/stage1.pth", help="Output path for the Stage-1 checkpoint.")
    parser.add_argument("--log-dir", type=str, default="runs/stage1", help="TensorBoard log directory.")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    train(
        data_root=args.data_root,
        attr_path=args.attr_path or None,
        ckpt_path=ckpt_path,
        log_dir=args.log_dir,
        num_epochs=args.epochs,
        log_interval=args.log_interval,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
