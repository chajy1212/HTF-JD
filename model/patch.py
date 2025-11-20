# -*- coding:utf-8 -*-
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1) 단일 slice → 여러 patch 생성
# ============================================================
def create_patches_from_slice(slice_img, patch_size=64, stride=128, fg_ratio=0.2):
    """
    slice_img: (H, W)
    patch_size: 패치 크기
    stride: 패치 간격
    fg_ratio: foreground 비율(0보다 큰 픽셀 비율)
    """
    H, W = slice_img.shape
    patches = []
    min_fg_pixels = int(patch_size * patch_size * fg_ratio)

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = slice_img[y:y+patch_size, x:x+patch_size]

            # foreground(>0) 비율 체크
            if np.sum(patch > 0) < min_fg_pixels:
                continue

            patches.append(((y, x), patch))

    return patches


# ============================================================
# 2) 3D volume → 2D slice → patch 생성
# ============================================================
def volume_to_patches(volume, patch_size=64, stride=128, fg_ratio=0.2):
    """
    volume: (H, W, Z)
    """
    H, W, Z = volume.shape
    all_patches = []

    for z in range(Z):
        slice_img = volume[:, :, z]

        # 완전 빈 slice 스킵
        if np.sum(slice_img > 0) < patch_size * patch_size * fg_ratio:
            continue

        patches = create_patches_from_slice(
            slice_img,
            patch_size=patch_size,
            stride=stride,
            fg_ratio=fg_ratio
        )

        if len(patches) > 0:
            all_patches.append((z, patches))

    return all_patches


# ============================================================
# 3) 패치 저장
# ============================================================
def save_patches(patches_3d, save_dir, base_name):
    os.makedirs(save_dir, exist_ok=True)

    for (z_idx, patch_list) in patches_3d:
        for idx, (coord, patch) in enumerate(patch_list):
            y, x = coord
            save_path = os.path.join(
                save_dir,
                f"{base_name}_z{z_idx}_y{y}_x{x}_p{idx}.npy"
            )
            np.save(save_path, patch)


# ============================================================
# 4) 전체 파이프라인 (CT 한 개 처리)
# ============================================================
def process_ct_file(path, patch_save_root):
    base_name = os.path.basename(path).replace(".nii.gz", "")
    print(f"\n[Processing] {base_name}")

    img = nib.load(path)
    volume = img.get_fdata()  # (H, W, Z)

    # ① Patch 생성
    patches = volume_to_patches(volume, patch_size=64, stride=128, fg_ratio=0.2)

    # ② Patch 저장
    save_dir = os.path.join(patch_save_root, base_name)
    save_patches(patches, save_dir, base_name)

    print(f"[Done] patches saved at {save_dir}")


# ============================================================
# main
# ============================================================
if __name__ == "__main__":

    data_dir = "/data/Cloud-basic/shared/Dataset/HTF/nifti_masked"
    patch_save_root = "/home/jycha/HTF/patches"

    nii_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".nii.gz")])

    print(f"Total NIfTI files: {len(nii_files)}")
    print("Processing first 10 files...\n")

    for fname in nii_files:   # 테스트 → 나중에 전체로 변경 가능
        fpath = os.path.join(data_dir, fname)
        process_ct_file(fpath, patch_save_root)