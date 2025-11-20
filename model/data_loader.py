# -*- coding:utf-8 -*-
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


# ============================================================
# 1) Excel → Label Dict 로딩 함수
# ============================================================
def load_label_dict(excel_path):
    df = pd.read_excel(excel_path)

    # hosp_id → 문자열 9자리로 맞추기
    label_dict = {
        str(row["hosp_id"]).zfill(9): int(row["HTf"])
        for _, row in df.iterrows()
    }

    print(f"[Label Loaded] {len(label_dict)} CT labels")
    return label_dict



# ============================================================
# 2) BagDataset (CT-level)
#    CT 하나에서 모든 패치를 모아서 하나의 Bag으로 반환
# ============================================================
class BagDataset(Dataset):
    def __init__(self, root_dir, ct_ids, label_dict):
        """
        root_dir: /home/brainlab/Workspace/jycha/HTF/patches
        ct_ids: CT 단위 리스트 (train, val, test split)
        """
        self.root_dir = root_dir
        self.ct_ids = ct_ids
        self.label_dict = label_dict

        # sanity check
        self.ct_ids = [
            ct for ct in self.ct_ids
            if os.path.isdir(os.path.join(root_dir, ct))
        ]

    def __len__(self):
        return len(self.ct_ids)

    def __getitem__(self, idx):

        ct_id = self.ct_ids[idx]
        ct_dir = os.path.join(self.root_dir, ct_id)
        label = self.label_dict[ct_id]

        # 패치 파일들 전부 로딩
        patch_files = sorted([f for f in os.listdir(ct_dir) if f.endswith(".npy")])

        patches = []
        for fname in patch_files:
            fpath = os.path.join(ct_dir, fname)
            patch = np.load(fpath).astype(np.float32)

            # Normalize
            patch = (patch - patch.mean()) / (patch.std() + 1e-6)

            patch = patch[None, :, :]   # (1, H, W)
            patches.append(patch)

        if len(patches) == 0:
            raise RuntimeError(f"[ERROR] No patches found in {ct_dir}")

        # stack → (N, 1, H, W)
        patches = np.stack(patches, axis=0)
        patches = torch.tensor(patches, dtype=torch.float32)

        return patches, torch.tensor(label, dtype=torch.long), ct_id


# ============================================================
# 3) CT 전체 패치를 로딩하는 Dataset (Inference용)
# ============================================================
class CTPatchDataset(Dataset):
    def __init__(self, ct_dir):
        """
        ct_dir: 하나의 CT 폴더
               ex) /home/brainlab/Workspace/jycha/HTF/patches/000006247/
        """
        self.ct_dir = ct_dir
        self.files = sorted([f for f in os.listdir(ct_dir) if f.endswith(".npy")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        fpath = os.path.join(self.ct_dir, fname)

        patch = np.load(fpath).astype(np.float32)
        patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-6)
        patch = np.expand_dims(patch, axis=0)

        patch = torch.tensor(patch, dtype=torch.float32)
        
        return patch, fname