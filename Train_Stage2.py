# -*- coding: utf-8 -*-
"""
Stage 2 training script.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm

from datasets.dataset_celeba_pairs import CelebAPairsDataset
from models.vit_backbone import ViTBackbone
from models.multiflow_model import MultiFlowModel


def run_full_flow(main_flow, z0: torch.Tensor) -> torch.Tensor:
    z = z0
    batch_size = z0.size(0)
    device = z0.device
    num_blocks = main_flow.num_blocks

    for i in range(num_blocks):
        t_scalar = main_flow._time_scalar(i, num_blocks, batch_size, device)
        t_emb = main_flow.time_embed(t_scalar)
        z = main_flow.blocks[i](z, t_emb)
    return z


def compute_intra_alignment_loss(
    delta_a_src,
    delta_a_tgt,
    delta_b_src,
    delta_b_tgt,
    delta_c_src,
    delta_c_tgt,
):
    def pair_loss(d_src, d_tgt):
        d_src_n = F.normalize(d_src, dim=-1)
        d_tgt_n = F.normalize(d_tgt, dim=-1)
        cos = (d_src_n * d_tgt_n).sum(dim=-1)
        cos_abs = cos.abs()
        return (1.0 - cos_abs).mean()

    loss_a = pair_loss(delta_a_src, delta_a_tgt)
    loss_b = pair_loss(delta_b_src, delta_b_tgt)
    loss_c = pair_loss(delta_c_src, delta_c_tgt)

    return (loss_a + loss_b + loss_c) / 3.0


def compute_semantic_attr_loss(
    s_a_src,
    s_a_tgt,
    s_b_src,
    s_b_tgt,
    s_c_src,
    s_c_tgt,
    attr_src_vec=None,
    attr_tgt_vec=None,
):
    if (attr_src_vec is None) or (attr_tgt_vec is None):
        return s_a_src.new_tensor(0.0)

    bce = F.binary_cross_entropy_with_logits

    def to_sign_label(col: torch.Tensor) -> torch.Tensor:
        col = col.float()
        if col.min() >= 0.0 and col.max() <= 1.0:
            col = col * 2.0 - 1.0
        else:
            col = torch.sign(col)
        col[col == 0] = 1.0
        return col

    anchor_pos = 1.0
    anchor_neg = -1.0
    target_std = 0.75
    anchor_weight = 1.0
    std_weight = 0.5

    def branch_loss(s_src, s_tgt, col_src, col_tgt):
        if s_src.numel() == 0:
            return s_src.new_tensor(0.0)

        s_src = s_src.view(s_src.size(0), -1).mean(dim=-1)
        s_tgt = s_tgt.view(s_tgt.size(0), -1).mean(dim=-1)

        y_src = col_src.float()
        y_tgt = col_tgt.float()

        if y_src.min() < 0:
            y_src = (y_src > 0).float()
        if y_tgt.min() < 0:
            y_tgt = (y_tgt > 0).float()

        logit_src = s_src.view_as(y_src)
        logit_tgt = s_tgt.view_as(y_tgt)

        cls_loss = 0.5 * (bce(logit_src, y_src) + bce(logit_tgt, y_tgt))

        raw_src = col_src.float()
        raw_tgt = col_tgt.float()
        dy = raw_tgt - raw_src
        ds = (s_tgt - s_src).view_as(dy)

        mask = dy != 0
        if mask.any():
            dir_label = torch.sign(dy[mask])
            prod = ds[mask] * dir_label
            dir_loss = F.relu(-prod).mean()
        else:
            dir_loss = cls_loss.new_tensor(0.0)

        base_loss = cls_loss + dir_loss

        s_all = torch.cat([s_src, s_tgt], dim=0)
        y_all = torch.cat([col_src, col_tgt], dim=0)
        y_sign = to_sign_label(y_all)

        pos_mask = y_sign > 0
        neg_mask = y_sign < 0

        anchor_loss = s_src.new_tensor(0.0)
        if pos_mask.any():
            mean_pos = s_all[pos_mask].mean()
            anchor_loss = anchor_loss + (mean_pos - anchor_pos).pow(2)
        if neg_mask.any():
            mean_neg = s_all[neg_mask].mean()
            anchor_loss = anchor_loss + (mean_neg - anchor_neg).pow(2)
        if s_all.numel() > 1:
            std_s = s_all.std(unbiased=False)
            anchor_loss = anchor_loss + std_weight * (std_s - target_std).pow(2)

        return base_loss + anchor_weight * anchor_loss

    loss_a = branch_loss(s_a_src, s_a_tgt, attr_src_vec[:, 0], attr_tgt_vec[:, 0])
    loss_b = branch_loss(s_b_src, s_b_tgt, attr_src_vec[:, 1], attr_tgt_vec[:, 1])
    loss_c = branch_loss(s_c_src, s_c_tgt, attr_src_vec[:, 2], attr_tgt_vec[:, 2])

    return (loss_a + loss_b + loss_c) / 3.0


def end_distance_per_sample(z_out: torch.Tensor, z_tgt: torch.Tensor) -> torch.Tensor:
    mse = (z_out - z_tgt).pow(2).mean(dim=-1)
    cos = 1.0 - F.cosine_similarity(z_out, z_tgt, dim=-1)
    return mse + cos


def build_pair_loader(img_root, attr_path, attr_names, batch_size=64, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ds = CelebAPairsDataset(
        img_root=img_root,
        attr_path=attr_path,
        split="train",
        attr_names=tuple(attr_names),
        transform=transform,
    )

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dl


def train_stage2(
    img_root: str,
    attr_path: str,
    attr_names,
    stage1_ckpt: str,
    log_dir: str = "runs/stage2",
    stage2_ckpt: str = "checkpoints/stage2.pth",
    batch_size: int = 256,
    num_workers: int = 4,
    num_epochs: int = 100,
    lr: float = 5e-5,
    lambda_sem: float = 0.05,
    lambda_rank: float = 0.1,
    rank_margin: float = 0.05,
    lambda_pullback: float = 100.0,
    device=None,
):
    if len(attr_names) != 3:
        raise ValueError("Stage-2 training expects exactly three editable attributes.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    loader = build_pair_loader(
        img_root,
        attr_path,
        attr_names=attr_names,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    teacher = ViTBackbone(
        model_name="vit_base_patch16_224",
        pretrained=True,
        global_pool="token",
        freeze=True,
    ).to(device)
    teacher.eval()

    model = MultiFlowModel(
        vit_name="vit_base_patch16_224",
        vit_pretrained=True,
        vit_freeze=True,
        num_main_blocks=16,
        k_t=3,
        k_tx=6,
        num_branches=3,
        main_hidden_dim=4096,
        branch_hidden_dim=2048,
        time_embed_dim=256,
        global_pool="token",
    ).to(device)

    if os.path.exists(stage1_ckpt):
        ckpt = torch.load(stage1_ckpt, map_location=device)
        state_all = ckpt.get("model_state", ckpt)
        mainflow_state = {k: v for k, v in state_all.items() if k.startswith("main_flow.")}
        model.load_state_dict(mainflow_state, strict=False)
        print(f"Loaded main_flow from Stage1: {stage1_ckpt}")
    else:
        print(f"Stage1 checkpoint not found at {stage1_ckpt}.")

    for p in model.main_flow.parameters():
        p.requires_grad = False
    model.main_flow.eval()
    print("Locked Stage1 main_flow weights.")

    branch_params = []
    for i in range(3):
        branch_params += list(model.branches[i].parameters())

    optimizer_branches = torch.optim.AdamW(branch_params, lr=lr, weight_decay=1e-4)

    start_epoch = 0
    if os.path.exists(stage2_ckpt):
        print(f"Found Stage-2 checkpoint: {stage2_ckpt}. Resuming training...")
        checkpoint = torch.load(stage2_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if "optim_branches" in checkpoint:
            optimizer_branches.load_state_dict(checkpoint["optim_branches"])
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
            print(f"Resume from epoch {start_epoch} / {num_epochs}")
    else:
        print("No Stage-2 checkpoint found. Training starts from scratch.")

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    save_path = Path(stage2_ckpt)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    global_step = 0

    for epoch in range(start_epoch, num_epochs):
        loader_tqdm = tqdm(loader, desc=f"[Stage2] Epoch {epoch}", ncols=120)
        model.branches.train()

        for batch in loader_tqdm:
            if len(batch) >= 5:
                x, x_attr, attr_id, attr_src_vec, attr_tgt_vec = batch[:5]
            else:
                raise ValueError("Dataset format error.")

            x = x.to(device, non_blocking=True)
            x_attr = x_attr.to(device, non_blocking=True)
            attr_id = attr_id.to(device, non_blocking=True)
            attr_src_vec = attr_src_vec.to(device, non_blocking=True)
            attr_tgt_vec = attr_tgt_vec.to(device, non_blocking=True)

            with torch.no_grad():
                z_src = teacher.encode_image_nograd(x)
                z_tgt = teacher.encode_image_nograd(x_attr)
                z_anchor = run_full_flow(model.main_flow, z_src)

            z_null = 0.001 * torch.randn_like(z_src) + z_src

            branch_a, branch_b, branch_c = model.branches
            s_a_src = branch_a.encode_semantic(z_src)
            s_b_src = branch_b.encode_semantic(z_src)
            s_c_src = branch_c.encode_semantic(z_src)

            s_a_tgt = branch_a.encode_semantic(z_tgt)
            s_b_tgt = branch_b.encode_semantic(z_tgt)
            s_c_tgt = branch_c.encode_semantic(z_tgt)

            delta_a_src = branch_a(z_src, semantic_override=s_a_src)
            delta_b_src = branch_b(z_src, semantic_override=s_b_src)
            delta_c_src = branch_c(z_src, semantic_override=s_c_src)

            delta_a_tgt = branch_a(z_tgt, semantic_override=s_a_tgt)
            delta_b_tgt = branch_b(z_tgt, semantic_override=s_b_tgt)
            delta_c_tgt = branch_c(z_tgt, semantic_override=s_c_tgt)

            delta_a_tgt_wrong = branch_a(z_tgt, semantic_override=s_a_src)
            delta_b_tgt_wrong = branch_b(z_tgt, semantic_override=s_b_src)
            delta_c_tgt_wrong = branch_c(z_tgt, semantic_override=s_c_src)

            with torch.no_grad():
                if s_a_src.numel() > 0:
                    writer.add_scalar("semantic/a_mean_src", s_a_src.mean().item(), global_step)
                    writer.add_scalar("semantic/a_mean_tgt", s_a_tgt.mean().item(), global_step)
                    writer.add_scalar("semantic/a_std_src", s_a_src.std(unbiased=False).item(), global_step)
                    writer.add_scalar("semantic/a_std_tgt", s_a_tgt.std(unbiased=False).item(), global_step)
                    writer.add_scalar("semantic/a_delta_mean", (s_a_tgt - s_a_src).mean().item(), global_step)

                if s_b_src.numel() > 0:
                    writer.add_scalar("semantic/b_mean_src", s_b_src.mean().item(), global_step)
                    writer.add_scalar("semantic/b_mean_tgt", s_b_tgt.mean().item(), global_step)
                    writer.add_scalar("semantic/b_std_src", s_b_src.std(unbiased=False).item(), global_step)
                    writer.add_scalar("semantic/b_std_tgt", s_b_tgt.std(unbiased=False).item(), global_step)
                    writer.add_scalar("semantic/b_delta_mean", (s_b_tgt - s_b_src).mean().item(), global_step)

                if s_c_src.numel() > 0:
                    writer.add_scalar("semantic/c_mean_src", s_c_src.mean().item(), global_step)
                    writer.add_scalar("semantic/c_mean_tgt", s_c_tgt.mean().item(), global_step)
                    writer.add_scalar("semantic/c_std_src", s_c_src.std(unbiased=False).item(), global_step)
                    writer.add_scalar("semantic/c_std_tgt", s_c_tgt.std(unbiased=False).item(), global_step)
                    writer.add_scalar("semantic/c_delta_mean", (s_c_tgt - s_c_src).mean().item(), global_step)

            semantic_loss = compute_semantic_attr_loss(
                s_a_src,
                s_a_tgt,
                s_b_src,
                s_b_tgt,
                s_c_src,
                s_c_tgt,
                attr_src_vec=attr_src_vec,
                attr_tgt_vec=attr_tgt_vec,
            )

            d_a_all = torch.cat([delta_a_src, delta_a_tgt], dim=0)
            d_b_all = torch.cat([delta_b_src, delta_b_tgt], dim=0)
            d_c_all = torch.cat([delta_c_src, delta_c_tgt], dim=0)
            with torch.no_grad():
                norm_monitor = (d_a_all.norm(dim=-1).mean() + d_b_all.norm(dim=-1).mean() + d_c_all.norm(dim=-1).mean()) / 3.0

            intra_loss = compute_intra_alignment_loss(
                delta_a_src,
                delta_a_tgt,
                delta_b_src,
                delta_b_tgt,
                delta_c_src,
                delta_c_tgt,
            )

            loss_edit_all = x.new_tensor(0.0)
            loss_pullback_all = x.new_tensor(0.0)
            rank_all = x.new_tensor(0.0)
            count_groups = 0

            mask_a = attr_id == 0
            if mask_a.any():
                idx = mask_a.nonzero(as_tuple=False).squeeze(-1)
                d_a = delta_a_tgt[idx]
                d_b = delta_b_src[idx]
                d_c = delta_c_src[idx]
                z_in_full = z_null[idx] + d_a + d_b + d_c
                z_out_full = run_full_flow(model.main_flow, z_in_full)
                loss_pullback_all += end_distance_per_sample(z_out_full, z_anchor[idx]).mean()

                s_a_tgt_i = s_a_tgt[idx]
                d_a_edit = branch_a(z_src[idx], semantic_override=s_a_tgt_i)
                z_in_edit = z_null[idx] + d_a_edit + d_b + d_c
                z_out_edit = run_full_flow(model.main_flow, z_in_edit)
                loss_edit_all += end_distance_per_sample(z_out_edit, z_tgt[idx]).mean()

                d_a_wrong = delta_a_tgt_wrong[idx]
                z_in_neg = z_null[idx] + d_a_wrong + d_b + d_c
                z_out_neg = run_full_flow(model.main_flow, z_in_neg)
                dist_pos = end_distance_per_sample(z_out_full, z_tgt[idx])
                dist_neg = end_distance_per_sample(z_out_neg, z_tgt[idx])
                rank_all += F.relu(rank_margin + dist_pos - dist_neg).mean()
                count_groups += 1

            mask_b = attr_id == 1
            if mask_b.any():
                idx = mask_b.nonzero(as_tuple=False).squeeze(-1)
                d_a = delta_a_src[idx]
                d_b = delta_b_tgt[idx]
                d_c = delta_c_src[idx]
                z_in_full = z_null[idx] + d_a + d_b + d_c
                z_out_full = run_full_flow(model.main_flow, z_in_full)
                loss_pullback_all += end_distance_per_sample(z_out_full, z_anchor[idx]).mean()

                s_b_tgt_i = s_b_tgt[idx]
                d_b_edit = branch_b(z_src[idx], semantic_override=s_b_tgt_i)
                z_in_edit = z_null[idx] + d_a + d_b_edit + d_c
                z_out_edit = run_full_flow(model.main_flow, z_in_edit)
                loss_edit_all += end_distance_per_sample(z_out_edit, z_tgt[idx]).mean()

                d_b_wrong = delta_b_tgt_wrong[idx]
                z_in_neg = z_null[idx] + d_a + d_b_wrong + d_c
                z_out_neg = run_full_flow(model.main_flow, z_in_neg)
                dist_pos = end_distance_per_sample(z_out_full, z_tgt[idx])
                dist_neg = end_distance_per_sample(z_out_neg, z_tgt[idx])
                rank_all += F.relu(rank_margin + dist_pos - dist_neg).mean()
                count_groups += 1

            mask_c = attr_id == 2
            if mask_c.any():
                idx = mask_c.nonzero(as_tuple=False).squeeze(-1)
                d_a = delta_a_src[idx]
                d_b = delta_b_src[idx]
                d_c = delta_c_tgt[idx]
                z_in_full = z_null[idx] + d_a + d_b + d_c
                z_out_full = run_full_flow(model.main_flow, z_in_full)
                loss_pullback_all += end_distance_per_sample(z_out_full, z_anchor[idx]).mean()

                s_c_tgt_i = s_c_tgt[idx]
                d_c_edit = branch_c(z_src[idx], semantic_override=s_c_tgt_i)
                z_in_edit = z_null[idx] + d_a + d_b + d_c_edit
                z_out_edit = run_full_flow(model.main_flow, z_in_edit)
                loss_edit_all += end_distance_per_sample(z_out_edit, z_tgt[idx]).mean()

                d_c_wrong = delta_c_tgt_wrong[idx]
                z_in_neg = z_null[idx] + d_a + d_b + d_c_wrong
                z_out_neg = run_full_flow(model.main_flow, z_in_neg)
                dist_pos = end_distance_per_sample(z_out_full, z_tgt[idx])
                dist_neg = end_distance_per_sample(z_out_neg, z_tgt[idx])
                rank_all += F.relu(rank_margin + dist_pos - dist_neg).mean()
                count_groups += 1

            if count_groups == 0:
                continue

            loss_edit_all /= count_groups
            loss_pullback_all /= count_groups
            rank_all /= count_groups

            loss_branch = (
                loss_edit_all
                + lambda_pullback * loss_pullback_all
                + lambda_sem * semantic_loss
            )

            optimizer_branches.zero_grad()
            loss_branch.backward()
            torch.nn.utils.clip_grad_norm_(branch_params, max_norm=1.0)
            optimizer_branches.step()

            global_step += 1

            loader_tqdm.set_postfix({
                "L_total": f"{loss_branch.item():.3f}",
                "Edit": f"{loss_edit_all.item():.3f}",
                "Pull": f"{loss_pullback_all.item():.3f}",
                "Sem": f"{semantic_loss.item():.3f}",
                "Rank": f"{rank_all.item():.3f}",
            })

            writer.add_scalar("loss/total", loss_branch.item(), global_step)
            writer.add_scalar("loss/edit", loss_edit_all.item(), global_step)
            writer.add_scalar("loss/pullback", loss_pullback_all.item(), global_step)
            writer.add_scalar("loss/semantic", semantic_loss.item(), global_step)
            writer.add_scalar("loss/rank", rank_all.item(), global_step)
            writer.add_scalar("monitor/intra", intra_loss.item(), global_step)
            writer.add_scalar("monitor/norm", norm_monitor.item(), global_step)

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optim_branches": optimizer_branches.state_dict(),
            },
            save_path,
        )
        print(f"Epoch {epoch} saved to {save_path}")

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stage-2 manifold constrained editing model.")
    parser.add_argument("--img-root", type=str, required=True, help="Path to the training image root.")
    parser.add_argument("--attr-path", type=str, required=True, help="Path to the attribute annotation file.")
    parser.add_argument(
        "--attr-names",
        type=str,
        default="Smiling,Young,Male",
        help="Comma-separated names of the three editable attributes.",
    )
    parser.add_argument("--stage1-ckpt", type=str, default="checkpoints/stage1.pth", help="Path to the Stage-1 checkpoint.")
    parser.add_argument("--log-dir", type=str, default="runs/stage2", help="TensorBoard log directory.")
    parser.add_argument("--out-ckpt", type=str, default="checkpoints/stage2.pth", help="Output path for the Stage-2 checkpoint.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lambda-sem", type=float, default=0.5)
    parser.add_argument("--lambda-pullback", type=float, default=100)
    parser.add_argument("--lambda-rank", type=float, default=0.1)
    parser.add_argument("--rank-margin", type=float, default=0.05)

    args = parser.parse_args()
    attr_names = [x.strip() for x in args.attr_names.split(",") if x.strip()]

    train_stage2(
        img_root=args.img_root,
        attr_path=args.attr_path,
        attr_names=attr_names,
        stage1_ckpt=args.stage1_ckpt,
        log_dir=args.log_dir,
        stage2_ckpt=args.out_ckpt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_epochs=args.epochs,
        lr=args.lr,
        lambda_sem=args.lambda_sem,
        lambda_pullback=args.lambda_pullback,
        lambda_rank=args.lambda_rank,
        rank_margin=args.rank_margin,
    )
