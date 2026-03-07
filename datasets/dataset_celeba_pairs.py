import os
import random
from typing import List, Dict, Sequence
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class CelebAPairsDataset(Dataset):

    def __init__(
        self,
        img_root: str,
        attr_path: str,
        attr_names: Sequence[str] = ("Smiling", "Young", "Male"),
        split: str = "train",
        split_ratio: float = 0.9,
        transform=None,
    ):
        super().__init__()
        self.img_root = img_root
        self.attr_path = attr_path
        self.attr_names = list(attr_names)
        self.transform = transform
        self.split = split
        self.split_ratio = split_ratio

        print(f"Loading attributes from {attr_path} ...")
        self.attr_dict, self.attr_indices = self._load_attr_dict(attr_path, self.attr_names)

        print(f"Scanning images recursively in {img_root} ...")
        disk_rel_paths = self._scan_images_recursively(img_root)

        valid_paths = []
        for p in disk_rel_paths:
            if p in self.attr_dict:
                valid_paths.append(p)
                continue

            p_linux = p.replace("\\", "/")
            if p_linux in self.attr_dict:
                self.attr_dict[p] = self.attr_dict[p_linux]
                valid_paths.append(p)
                continue

            root, ext = os.path.splitext(p)
            if ext.lower() == ".png":
                p_jpg = root + ".jpg"
                p_jpg_linux = p_jpg.replace("\\", "/")
                
                found_key = None
                if p_jpg in self.attr_dict:
                    found_key = p_jpg
                elif p_jpg_linux in self.attr_dict:
                    found_key = p_jpg_linux
                
                if found_key:
                    self.attr_dict[p] = self.attr_dict[found_key]
                    valid_paths.append(p)
                    continue

        if len(valid_paths) == 0:
            raise RuntimeError(
                f"  No matched images found\n"
            )

        valid_paths.sort()

        total = len(valid_paths)
        split_idx = int(total * split_ratio)
        
        if self.split == "train":
            self.final_paths = valid_paths[:split_idx]
            print(f"Split='train' ({split_ratio*100}%): {len(self.final_paths)} / {total}")
        elif self.split == "test":
            self.final_paths = valid_paths[split_idx:]
            print(f"Split='test' ({100 - split_ratio*100:.1f}%): {len(self.final_paths)} / {total}")
        else:
            self.final_paths = valid_paths
            print(f"Split='all': {len(self.final_paths)}")

        self.groups = self._build_groups(self.final_paths, self.attr_dict)

    
    def _scan_images_recursively(self, root):
        rel_paths = []
        valid_exts = {".jpg", ".png", ".jpeg", ".bmp"}
        for current_dir, _, filenames in os.walk(root):
            for fname in filenames:
                if os.path.splitext(fname)[1].lower() in valid_exts:
                    full_path = os.path.join(current_dir, fname)
                    rel_path = os.path.relpath(full_path, root)
                    rel_paths.append(rel_path)
        return rel_paths

    def _load_attr_dict(self, attr_path, target_names):
        with open(attr_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        
        header_idx = -1
        header = []
        for i, line in enumerate(lines[:10]):
            parts = line.split()
            if target_names[0] in parts:
                header_idx = i
                header = parts
                break
        
        if header_idx == -1:
            raise ValueError(f"Could not find header containing {target_names} in {attr_path}")
            
        col_indices = []
        for name in target_names:
            col_indices.append(header.index(name))

        attr_dict = {}
        for line in lines[header_idx+1:]:
            parts = line.split()
            if len(parts) < len(header): continue

            fname = parts[0]

            offset = 0
            if len(parts) == len(header) + 1:
                offset = 1
            
            try:
                curr_attrs = []
                for idx in col_indices:
                    val_str = parts[idx + offset]
                    val = int(val_str)
                    curr_attrs.append(1 if val == 1 else 0)
                
                attr_dict[fname] = curr_attrs
                attr_dict[fname.replace("\\", "/")] = curr_attrs 
                
            except Exception:
                continue
                
        return attr_dict, col_indices

    def _build_groups(self, paths, attr_dict):
        groups = {}
        for i, p in enumerate(paths):
            if p in attr_dict:
                attrs = tuple(attr_dict[p])
                if attrs not in groups:
                    groups[attrs] = []
                groups[attrs].append(i)
        return groups

    def __len__(self):
        return len(self.final_paths)

    def __getitem__(self, idx):
        idx = int(idx)
        fname_src = self.final_paths[idx]
        attrs_src = self.attr_dict[fname_src]
            
        K = len(self.attr_names)
        attr_id = random.randint(0, K - 1)
        
        target_attrs = list(attrs_src)
        target_attrs[attr_id] = 1 - target_attrs[attr_id]
        target_key = tuple(target_attrs)
        
        idx2 = self._sample_from_group(target_key, exclude_idx=idx)
        fname_tgt = self.final_paths[idx2]
        
        img1 = self._read_img(fname_src)
        img2 = self._read_img(fname_tgt)
        
        attr_src_vec = torch.tensor(attrs_src, dtype=torch.float32)
        attr_tgt_vec = torch.tensor(target_attrs, dtype=torch.float32)
        
        return img1, img2, attr_id, attr_src_vec, attr_tgt_vec

    def _sample_from_group(self, key, exclude_idx):
        idx_list = self.groups.get(key, [])
        if not idx_list: return exclude_idx
        for _ in range(10):
            j = random.choice(idx_list)
            if j != exclude_idx: return j
        return exclude_idx

    def _read_img(self, rel_path):
        full_path = os.path.join(self.img_root, rel_path)
        img = Image.open(full_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        return img