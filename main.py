# -*- coding: utf-8 -*-
"""
main.py

Unified evaluation script for Manifold Constrained Editing.
Metrics:
    * Acc@0.5 / Stab@0.5
    * Main/Stable (Soft / Hard)
    * ID similarity (optional)
    * LPIPS (optional)
"""

import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm.auto import tqdm
import lpips

from evaluation.factor_metrics import evaluate_factor_soft_hard
from models.vit_backbone import ViTBackbone
from models.multiflow_model import MultiFlowModel
from models.image_decoder import SimpleViTDecoder
from datasets.dataset_celeba_pairs import CelebAPairsDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified evaluation for Manifold Constrained Editing"
    )

    parser.add_argument("--img_root", type=str, required=True, help="Path to dataset images")
    parser.add_argument("--attr_path", type=str, required=True, help="Path to attribute annotation file")

    parser.add_argument("--stage2_ckpt", type=str, required=True, help="Path to Stage-2 checkpoint")
    parser.add_argument("--probe_ckpt", type=str, default="", help="Path to attribute probe checkpoint")
    parser.add_argument("--decoder_ckpt", type=str, default="",  help="Path to decoder checkpoint")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=1000)

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--attr_names", type=str, default="Smiling,Young,Male", help="Comma-separated attribute names")
    parser.add_argument("--decoder_img_size", type=int, default=128)

    parser.add_argument("--use_arcface", action="store_true", help="Enable ArcFace identity similarity evaluation")
    parser.add_argument("--use_lpips", action="store_true", help="Enable LPIPS evaluation")

    return parser.parse_args()


def run_full_flow(main_flow, z0: torch.Tensor) -> torch.Tensor:
    z = z0
    B = z0.size(0)
    device = z0.device
    N = main_flow.num_blocks

    for i in range(N):
        t_scalar = main_flow._time_scalar(i, N, B, device)
        t_emb = main_flow.time_embed(t_scalar)
        z = main_flow.blocks[i](z, t_emb)
    return z


class ArcFaceEvaluator:
    def __init__(self, device="cuda"):
        import cv2
        from insightface.app import FaceAnalysis

        self.cv2 = cv2
        self.device = device
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(256, 256))
        self.rec_model = self.app.models.get("recognition", None)
        if self.rec_model is None:
            raise RuntimeError("buffalo_l recognition model not found.")

    @staticmethod
    def _center_crop_square(bgr_uint8: np.ndarray) -> np.ndarray:
        h, w = bgr_uint8.shape[:2]
        s = min(h, w)
        y0 = (h - s) // 2
        x0 = (w - s) // 2
        return bgr_uint8[y0:y0 + s, x0:x0 + s]

    @torch.no_grad()
    def get_embedding(self, img_tensor: torch.Tensor) -> torch.Tensor:
        img = img_tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
        img = img[..., ::-1]

        embs = []
        for i in range(img.shape[0]):
            bgr = self._center_crop_square(img[i])
            face112 = self.cv2.resize(bgr, (112, 112), interpolation=self.cv2.INTER_LINEAR)

            if hasattr(self.rec_model, "get_feat"):
                feat = self.rec_model.get_feat(face112)
            else:
                out = self.rec_model.get(face112)
                feat = getattr(out, "embedding", out)

            feat = np.asarray(feat).reshape(-1).astype(np.float32)
            embs.append(feat)

        embs = torch.from_numpy(np.stack(embs, axis=0)).to(self.device)
        return embs


class AttrProbe(nn.Module):
    def __init__(self, feat_dim, num_a=2, num_b=2, num_c=2):
        super().__init__()
        self.bn = nn.BatchNorm1d(feat_dim)
        self.head_a = nn.Linear(feat_dim, num_a)
        self.head_b = nn.Linear(feat_dim, num_b)
        self.head_c = nn.Linear(feat_dim, num_c)

    def forward(self, h):
        h = self.bn(h)
        la = self.head_a(h)
        lb = self.head_b(h)
        lc = self.head_c(h)
        logit = torch.stack(
            [la[:, 1] - la[:, 0], lb[:, 1] - lb[:, 0], lc[:, 1] - lc[:, 0]],
            dim=1
        )
        return logit


def compute_norm_stats(all_probs_list):
    if not all_probs_list:
        return None
    big = np.concatenate(all_probs_list, axis=0)
    return {
        "min": np.percentile(big, 2.5, axis=0),
        "max": np.percentile(big, 97.5, axis=0),
    }


@torch.no_grad()
def main():
    args = parse_args()
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    attr_names = tuple([x.strip() for x in args.attr_names.split(",")])

    teacher = ViTBackbone(
        model_name="vit_base_patch16_224",
        pretrained=True,
        global_pool="token",
        freeze=True,
    ).to(device).eval()

    probe = AttrProbe(feat_dim=teacher.embed_dim).to(device).eval()
    if args.probe_ckpt and os.path.exists(args.probe_ckpt):
        ckpt = torch.load(args.probe_ckpt, map_location=device)
        state = ckpt.get("probe", ckpt.get("model", ckpt))
        probe.load_state_dict(state, strict=False)

    stage2 = MultiFlowModel(
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
        branch_semantic_dim=1,
        branch_use_semantic=True,
        global_pool="token",
    ).to(device).eval()

    ckpt = torch.load(args.stage2_ckpt, map_location=device)
    stage2.load_state_dict(ckpt.get("model_state", ckpt.get("model", ckpt)), strict=False)

    decoder = SimpleViTDecoder(
        z_dim=teacher.embed_dim,
        base_channels=256,
        img_size=args.decoder_img_size,
    ).to(device).eval()

    if args.decoder_ckpt and os.path.exists(args.decoder_ckpt):
        ckpt = torch.load(args.decoder_ckpt, map_location=device)
        d_state = ckpt
        if isinstance(ckpt, dict):
            for key in ["decoder", "model", "state_dict"]:
                if key in ckpt:
                    d_state = ckpt[key]
                    break
        decoder.load_state_dict(d_state, strict=False)

    arcface = None
    if args.use_arcface:
        try:
            arcface = ArcFaceEvaluator(device=device)
        except Exception as e:
            print(f"[Warning] ArcFace initialization failed: {e}")

    lpips_fn = None
    if args.use_lpips:
        lpips_fn = lpips.LPIPS(net="alex").to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    ds = CelebAPairsDataset(
        img_root=args.img_root,
        attr_path=args.attr_path,
        attr_names=attr_names,
        split=args.split,
        transform=transform,
    )

    if len(ds) > args.max_samples:
        print(f"Truncating test set from {len(ds)} to {args.max_samples} for speed...")
        ds = Subset(ds, range(args.max_samples))
    else:
        print(f"Dataset size: {len(ds)} (no truncation)")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    attr_keys = ["A", "B", "C"]
    factor_idx = {"A": 0, "B": 1, "C": 2}

    store = {
        k: {
            "probs": {},
            "acc": [],
            "stab": [],
            "id": [],
            "lpips": [],
        } for k in attr_keys
    }
    all_trajs = []

    branches = list(stage2.branches)
    sample_idx = 0

    for batch in tqdm(loader, desc="Eval"):
        x_src, _, _, _labels, _ = batch
        x_src = x_src.to(device)
        h_src = teacher.encode_image_nograd(x_src)

        s_src_list = []
        d_src_list = []

        for k in range(3):
            s_k = branches[k].encode_semantic(h_src)
            s_src_list.append(s_k)
            d_k = branches[k](h_src, semantic_override=s_k)
            d_src_list.append(d_k)

        z_null = h_src

        for i in range(x_src.size(0)):
            z_in_src = z_null[i:i+1] + sum([d_src_list[j][i:i+1] for j in range(3)])
            z_out_src = run_full_flow(stage2.main_flow, z_in_src)
            pv_src = torch.sigmoid(probe(z_out_src))[0].detach().cpu().numpy()
            dec_src = torch.clamp((decoder(z_out_src) + 1.0) / 2.0, 0.0, 1.0)

            for k_str in attr_keys:
                k = factor_idx[k_str]
                if pv_src[k] >= 0.5:
                    continue

                s_target = -s_src_list[k][i:i+1]
                d_k_edit = branches[k](h_src[i:i+1], semantic_override=s_target)

                d_others = torch.zeros_like(h_src[i:i+1])
                for kk in range(3):
                    if kk != k:
                        d_others += d_src_list[kk][i:i+1]

                z_in_edit = z_null[i:i+1] + d_others + d_k_edit
                z_out_edit = run_full_flow(stage2.main_flow, z_in_edit)

                pv_edit = torch.sigmoid(probe(z_out_edit))[0].detach().cpu().numpy()
                traj = np.stack([pv_src, pv_edit], axis=0)

                key = f"{sample_idx}_{k_str}"
                store[k_str]["probs"][key] = traj
                all_trajs.append(traj)

                store[k_str]["acc"].append(float(pv_edit[k] >= 0.5))

                other_idxs = [j for j in range(3) if j != k]
                stab_vals = []
                for oid in other_idxs:
                    stab_vals.append(float((pv_edit[oid] >= 0.5) == (pv_src[oid] >= 0.5)))
                store[k_str]["stab"].append(float(np.mean(stab_vals)))

                dec_edit = torch.clamp((decoder(z_out_edit) + 1.0) / 2.0, 0.0, 1.0)

                if arcface is not None:
                    e1 = arcface.get_embedding(dec_src)
                    e2 = arcface.get_embedding(dec_edit)
                    e1 = F.normalize(e1, p=2, dim=1)
                    e2 = F.normalize(e2, p=2, dim=1)
                    sim = float((e1 * e2).sum().item())
                    store[k_str]["id"].append(sim)

                if lpips_fn is not None:
                    lp = float(lpips_fn(dec_src * 2 - 1, dec_edit * 2 - 1).item())
                    store[k_str]["lpips"].append(lp)

        sample_idx += x_src.size(0)

    norm_stats = compute_norm_stats(all_trajs)
    if norm_stats is None:
        print("No valid samples at all.")
        return

    print("\n" + "=" * 120)
    print("Method     | Attr  | N     | Acc      | Stab     | HardMain | SoftMain | HardStab | SoftStab | ID       | LPIPS")
    print("-" * 120)

    method_name = "MCE(Ours)"
    agg = {
        "Acc": [], "Stab": [],
        "HardMain": [], "SoftMain": [],
        "HardStab": [], "SoftStab": [],
        "ID": [], "LPIPS": [],
    }

    for k_str in attr_keys:
        d = store[k_str]
        N = len(d["acc"])
        if N == 0:
            continue

        main_idx = factor_idx[k_str]
        other_idx = [i for i in range(3) if i != main_idx]
        m = evaluate_factor_soft_hard(d["probs"], main_idx, other_idx, 0, 1, norm_stats=norm_stats)

        acc = float(np.mean(d["acc"]))
        stab = float(np.mean(d["stab"]))
        hm = float(m["HardMain"])
        sm = float(m["SoftMain"])
        hs = float(m["HardStable"])
        ss = float(m["SoftStable"])
        ids = float(np.mean(d["id"])) if len(d["id"]) > 0 else float("nan")
        lp = float(np.mean(d["lpips"])) if len(d["lpips"]) > 0 else float("nan")

        agg["Acc"].append(acc)
        agg["Stab"].append(stab)
        agg["HardMain"].append(hm)
        agg["SoftMain"].append(sm)
        agg["HardStab"].append(hs)
        agg["SoftStab"].append(ss)
        agg["ID"].append(ids)
        agg["LPIPS"].append(lp)

        print(
            f"{method_name:<10} | {k_str:<5} | {N:5d} | "
            f"{acc*100:7.2f}% | {stab*100:7.2f}% | "
            f"{hm*100:8.2f}% | {sm:8.4f} | "
            f"{hs*100:8.2f}% | {ss:8.4f} | "
            f"{ids:8.4f} | {lp:8.4f}"
        )

    if len(agg["Acc"]) > 0:
        acc_avg = float(np.mean(agg["Acc"]))
        stab_avg = float(np.mean(agg["Stab"]))
        hm_avg = float(np.mean(agg["HardMain"]))
        sm_avg = float(np.mean(agg["SoftMain"]))
        hs_avg = float(np.mean(agg["HardStab"]))
        ss_avg = float(np.mean(agg["SoftStab"]))
        id_avg = float(np.nanmean(np.array(agg["ID"], dtype=np.float32)))
        lp_avg = float(np.nanmean(np.array(agg["LPIPS"], dtype=np.float32)))

        print("-" * 120)
        print(
            f"{'AVERAGE':<10} | {'ALL':<5} | {'---':<5} | "
            f"{acc_avg*100:7.2f}% | {stab_avg*100:7.2f}% | "
            f"{hm_avg*100:8.2f}% | {sm_avg:8.4f} | "
            f"{hs_avg*100:8.2f}% | {ss_avg:8.4f} | "
            f"{id_avg:8.4f} | {lp_avg:8.4f}"
        )
        print("=" * 120)
    else:
        print("No valid samples at all.")


if __name__ == "__main__":
    main()