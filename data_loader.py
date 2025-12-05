# -*- coding:utf-8 -*-
import os, cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from nibabel.dft import pydicom
import torch.nn.functional as F
import scipy.ndimage as ndi


# ============================================================
# Brain region crop
# ============================================================
def clean_brain_region(img, final_size=384, threshold=0.05):
    """
    img: (H, W) 0~1 scaled CT slice
    Performs full brain extraction:
        1) Remove bottom 15% (table)
        2) Keep largest connected component
        3) Crop bounding box
        4) Pad to square
        5) Resize to final_size
    """
    H = img.shape[0]

    # Remove bottom 15%
    img = img[: int(H * 0.85), :]

    # Largest connected component
    mask = img > threshold
    labeled, num = ndi.label(mask)
    sizes = ndi.sum(mask, labeled, range(1, num + 1))

    if num > 0:
        largest = np.argmax(sizes) + 1
        mask = (labeled == largest)
        img = img * mask

    # Crop bounding box
    coords = np.argwhere(img > threshold)
    if len(coords) > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        img = img[y0:y1 + 1, x0:x1 + 1]

    # Pad to square
    h, w = img.shape
    size = max(h, w)
    pad = np.zeros((size, size), dtype=np.float32)

    y_off = (size - h) // 2
    x_off = (size - w) // 2
    pad[y_off:y_off + h, x_off:x_off + w] = img
    img = pad

    # Resize to final_size
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    img = F.interpolate(img, size=(final_size, final_size), mode="bilinear", align_corners=False)
    img = img.squeeze().numpy().astype(np.float32)

    return img


# ============================================================
# Load single CT volume (stack DICOM slices)
# ============================================================
def load_dicom_volume(folder_path):
    """폴더 안의 모든 DICOM을 읽어 (H, W, Z) volume으로 반환"""
    files = sorted(
        [os.path.join(folder_path, f)
         for f in os.listdir(folder_path)
         if f.lower().endswith(".dcm")]
    )

    images = []

    target_size = None  # 첫 슬라이스 기준

    for f in files:
        d = pydicom.dcmread(f)

        # ---- Fix missing Transfer Syntax UID ----
        if "TransferSyntaxUID" not in d.file_meta:
            from pydicom.uid import ExplicitVRLittleEndian
            d.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        img = d.pixel_array.astype(np.float32)

        # --- HU Transform ---
        if "RescaleSlope" in d:
            img = img * float(d.RescaleSlope)
        if "RescaleIntercept" in d:
            img = img + float(d.RescaleIntercept)

        # --- Brain windowing ---
        WL, WW = 40, 80
        lower = WL - WW / 2
        upper = WL + WW / 2

        img = np.clip(img, lower, upper)
        img = (img - lower) / (upper - lower)   # scale 0~1

        # ---- Determine target size using first slice ----
        if target_size is None:
            target_size = img.shape  # (H, W)

        # ---- Resize if needed ----
        if img.shape != target_size:
            img = cv2.resize(img, (target_size[1], target_size[0]),
                             interpolation=cv2.INTER_AREA)

        images.append(img)

    volume = np.stack(images, axis=-1)          # (H, W, Z)
    return volume


# ============================================================
# Dataset for MAE SSL
# ============================================================
class CTMAEDataset(Dataset):
    def __init__(self, dicom_root, ct_id_list, final_size=384):
        self.samples = []
        self.final_size = final_size

        print("[CTMAEDataset] Loading CT dicom volumes...")

        for ct_id in ct_id_list:
            folder = os.path.join(dicom_root, ct_id)
            if not os.path.isdir(folder):
                continue

            # ---- Load DICOM volume ----
            vol = load_dicom_volume(folder)   # (H,W,Z)

            # ---- Middle slice 선택 ----
            mid = vol.shape[2] // 2
            sl = vol[:, :, mid]

            # ---- Clean brain region ----
            sl = clean_brain_region(sl, final_size=self.final_size)

            # ---- Normalize ----
            # sl = (sl - sl.mean()) / (sl.std() + 1e-6)

            self.samples.append(sl)

        print(f"[CTMAEDataset] Loaded {len(self.samples)} slices.")


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        img = self.samples[idx]                 # (final_size, final_size)
        img = torch.tensor(img).unsqueeze(0)    # (1,H,W)
        return img


# ============================================================
# Data loader maker
# ============================================================
def make_mae_dataloaders(dicom_root, all_ct_ids, batch_size=8, final_size=384):

    n = len(all_ct_ids)
    train_ids = all_ct_ids[:int(n * 0.7)]
    val_ids   = all_ct_ids[int(n * 0.7):]

    train_ds = CTMAEDataset(dicom_root, train_ids, final_size)
    val_ds   = CTMAEDataset(dicom_root, val_ids, final_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader