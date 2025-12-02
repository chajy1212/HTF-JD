# -*- coding:utf-8 -*-
import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1) Slice → Patchify
# ============================================================
def create_patches_from_slice(slice_img, patch_size=512, stride=32, fg_ratio=0.05):
    H, W = slice_img.shape
    patches = []

    min_fg_pixels = int(patch_size * patch_size * fg_ratio)

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = slice_img[y:y + patch_size, x:x + patch_size]

            # foreground filtering
            if np.sum(patch > 0) < min_fg_pixels:
                continue

            patches.append(patch)

    return patches


# ============================================================
# 2) Full CT volume → slice → patchify
# ============================================================
def volume_to_patches(volume, patch_size=512, stride=32, fg_ratio=0.05):
    H, W, Z = volume.shape
    all_patches = []

    for z in range(Z):
        slice_img = volume[:, :, z]

        # skip empty slices
        if np.sum(slice_img > 0) < patch_size * patch_size * fg_ratio:
            continue

        patches = create_patches_from_slice(
            slice_img,
            patch_size=patch_size,
            stride=stride,
            fg_ratio=fg_ratio
        )

        all_patches.extend(patches)

    return all_patches  # list of (64×64) patches


# ============================================================
# 3) Dataset for MAE SSL
# ============================================================
class CTMAEDataset(Dataset):
    def __init__(self, nii_root, ct_id_list, patch_size=512, stride=32, fg_ratio=0.05):
        self.nii_root = nii_root
        self.ct_id_list = ct_id_list
        self.patch_size = patch_size
        self.stride = stride
        self.fg_ratio = fg_ratio

        print(f"[Dataset Init] Loading CT volumes & patchifying...")
        self.patch_list = []   # will store all patches (as numpy)

        for ct_id in ct_id_list:
            nii_path = os.path.join(nii_root, f"{ct_id}.nii.gz")
            vol = nib.load(nii_path).get_fdata()

            patches = volume_to_patches(
                vol,
                patch_size=self.patch_size,
                stride=self.stride,
                fg_ratio=self.fg_ratio
            )

            self.patch_list.extend(patches)

        print(f"[Total patches loaded] {len(self.patch_list)}")


    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, idx):
        patch = self.patch_list[idx].astype(np.float32)

        # normalize
        patch = (patch - patch.mean()) / (patch.std() + 1e-6)

        # ======== Resize from (512,512) → (300,300) ========
        import torch.nn.functional as F
        patch = torch.tensor(patch).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        patch = F.interpolate(patch, size=(300, 300), mode="bilinear", align_corners=False)
        patch = patch.squeeze(0)  # (1,300,300)

        return patch


# ============================================================
# Helper: make loaders
# ============================================================
def make_mae_dataloaders(nii_root, all_ct_ids, batch_size=64):
    # train 80% / val 20%
    n = len(all_ct_ids)
    train_ids = all_ct_ids[:int(n * 0.7)]
    val_ids   = all_ct_ids[int(n * 0.7):]

    train_ds = CTMAEDataset(nii_root, train_ids)
    val_ds   = CTMAEDataset(nii_root, val_ids)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader