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
# 2) 배경 패치 제거 기준 함수
# ============================================================
def is_background(patch, thr_nonzero=0.05, thr_mean=-800):
    """
    patch: (H, W)
    - thr_nonzero: non-zero 비율이 너무 낮으면 배경으로 판정
    - thr_mean: 평균 intensity가 극도로 낮으면 (공기값) 배경으로 판정
    """
    nonzero_ratio = np.count_nonzero(patch) / patch.size
    if nonzero_ratio < thr_nonzero:
        return True

    if patch.mean() < thr_mean:
        return True

    return False


# ============================================================
# 3) BagDataset (CT 별 모든 patch → 하나의 bag)
# ============================================================
class BagDataset(Dataset):
    def __init__(self, root_dir, ct_ids, label_dict):
        """
        root_dir: /home/brainlab/Workspace/jycha/HTF/patches
        ct_ids: CT 폴더명 리스트 (train, val, test split)
        """
        self.root_dir = root_dir
        self.ct_ids = [
            ct for ct in ct_ids
            if os.path.isdir(os.path.join(root_dir, ct))
        ]
        self.label_dict = label_dict

    def __len__(self):
        return len(self.ct_ids)

    def __getitem__(self, idx):
        ct_id = self.ct_ids[idx]
        ct_dir = os.path.join(self.root_dir, ct_id)
        label = self.label_dict[ct_id]

        patch_files = sorted([f for f in os.listdir(ct_dir) if f.endswith(".npy")])

        patches = []

        for fname in patch_files:
            patch = np.load(os.path.join(ct_dir, fname)).astype(np.float32)

            # -------------------------------
            # (A) 배경 패치 제거
            # -------------------------------
            if is_background(patch):
                continue

            # -------------------------------
            # (B) Normalize
            # -------------------------------
            patch = (patch - patch.mean()) / (patch.std() + 1e-6)
            patch = patch[None, :, :]  # (1, H, W)
            patches.append(patch)

        # -------------------------------
        # (C) 모든 패치 제거된 경우 → 오류
        # -------------------------------
        if len(patches) == 0:
            raise RuntimeError(f"[ERROR] No valid patches remain after filtering in {ct_dir}")

        patches = np.stack(patches, axis=0)  # (N, 1, H, W)
        patches = torch.tensor(patches, dtype=torch.float32)

        return patches, torch.tensor(label, dtype=torch.long), ct_id


# ============================================================
# 4) Inference용 Dataset — 배경 패치 제거 포함
# ============================================================
class CTPatchDataset(Dataset):
    def __init__(self, ct_dir):
        """
        ct_dir: /.../patches/000006247/
        """
        self.ct_dir = ct_dir
        self.files = sorted([f for f in os.listdir(ct_dir) if f.endswith(".npy")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        fpath = os.path.join(self.ct_dir, fname)

        patch = np.load(fpath).astype(np.float32)

        # -------------------------------
        # (A) inference에서도 배경 patch 제외
        # -------------------------------
        if is_background(patch):
            # dummy patch라도 반환 (시각화에서 스킵 가능)
            patch = np.zeros_like(patch)

        # -------------------------------
        # (B) Normalize
        # -------------------------------
        patch = (patch - patch.mean()) / (patch.std() + 1e-6)
        patch = np.expand_dims(patch, axis=0)

        patch = torch.tensor(patch, dtype=torch.float32)

        return patch, fname