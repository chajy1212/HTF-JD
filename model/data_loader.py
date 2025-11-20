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
# 2) Patch Dataset (훈련/검증용)
# ============================================================
class PatchDataset(Dataset):
    def __init__(self, root_dir, label_dict, split_ids):
        """
        root_dir: /home/jycha/HTF/patches
        split_ids: 학습에 사용할 CT
        """
        self.root_dir = root_dir
        self.label_dict = label_dict
        self.samples = []
        self.labels = []

        # CT ID 순회
        for ct_id in split_ids:
            ct_path = os.path.join(root_dir, ct_id)

            if not os.path.isdir(ct_path):
                continue

            # 패치 파일 로딩
            for fname in os.listdir(ct_path):
                if fname.endswith(".npy"):
                    fpath = os.path.join(ct_path, fname)

                    label = label_dict.get(ct_id, None)
                    if label is None:
                        continue

                    self.samples.append((fpath, label))
                    self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]

        patch = np.load(fpath).astype(np.float32)

        # Normalize
        patch = (patch - patch.mean()) / (patch.std() + 1e-6)

        patch = patch[None, :, :]  # (1, H, W)

        return torch.tensor(patch, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# ============================================================
# 3) CT 전체 패치를 로딩하는 Dataset (Inference용)
# ============================================================
class CTPatchDataset(Dataset):
    def __init__(self, ct_dir):
        """
        ct_dir: 하나의 CT 폴더
               ex) /home/jycha/HTF/patches/000006247/
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