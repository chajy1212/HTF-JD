import os
import nibabel as nib
import numpy as np

# 각 slice 안에서 여러 patch 중 ➜ 값이 전부 0인 patch들은 개별적으로 필터링되고,
# 만약 그 슬라이스의 모든 patch가 전부 0이라서➜ 살아남은 patch가 1개도 없으면, 그 slice 자체가 “실질적으로 필터링된 슬라이스(skipped_slices)”가 됨

# 1) 슬라이스 하나에서 패치 생성 (특정 패치가가 0이면 patch 버림)
def create_patches_from_slice(slice_img, patch_size=64, stride=32):
    H, W = slice_img.shape
    patches = []
    dropped = 0   # 추가됨

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = slice_img[y:y + patch_size, x:x + patch_size]

            if np.all(patch == 0):
                dropped += 1          # ✨ all-zero patch count
                continue

            patches.append(((y, x), patch))

    return patches, dropped


# 2) 3D volume → (z, patch_list) + 통계
def volume_to_patches(volume, patch_size=64, stride=32):
    H, W, Z = volume.shape
    patches_3d = []

    total_slices = Z
    used_slices = 0
    skipped_slices = 0 #이거는 아마 0이어야 할 것임
    kept_patches = 0
    dropped_patches = 0

    for z in range(Z):
        slice_img = volume[:, :, z]

        patch_list = []
        # 여기서는 for문 직접 쓰지 말고 함수 호출
        patch_candidates, dropped = create_patches_from_slice(
            slice_img,
            patch_size=patch_size,
            stride=stride
        )
        dropped_patches += dropped
        # 통계 위해 다시 체크 (all-zero 버려진 개수 셀 거면 여기서 처리해도 되고,
        # create_patches_from_slice에서 개수 리턴하게 바꿔도 됨)
        # 지금은 "살아남은 것만" 기반으로만 통계 쓰면 간단함
        kept_patches += len(patch_candidates)

        if len(patch_candidates) > 0: #유효한 patch가 하나도 없었다면
            used_slices += 1
            patches_3d.append((z, patch_candidates))
        else:
            skipped_slices += 1 #slice도 filtering됨

    stats = {
        "total_slices": total_slices,
        "used_slices": used_slices,
        "skipped_slices": skipped_slices,
        "kept_patches": kept_patches,
        "dropped_patches": dropped_patches,
    }

    return patches_3d, stats


# 3) 저장
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


# 4) CT 한 개 처리
def process_ct_file(path, patch_save_root):
    base_name = os.path.basename(path).replace(".nii.gz", "")
    print(f"\n[Processing] {base_name}")

    img = nib.load(path)
    volume = img.get_fdata()

    patches_3d, stats = volume_to_patches(volume, patch_size=64, stride=32)

    save_dir = os.path.join(patch_save_root, base_name)
    save_patches(patches_3d, save_dir, base_name)

    print(f"[{base_name}] Slice Summary ------------------")
    print(f" Total slices   : {stats['total_slices']}")
    print(f" Used slices    : {stats['used_slices']}")
    print(f" Skipped slices : {stats['skipped_slices']}")
    print(f"-------------------------------------------")
    print(f" Patch Summary")
    print(f"  Kept patches    : {stats['kept_patches']}")
    print(f"  Dropped patches : {stats['dropped_patches']}")
    print("-------------------------------------------")
    print(f"[Done] patches saved at {save_dir}")

# ============================================================
# main
# ============================================================
# if __name__ == "__main__":
#
#     # data_dir = "/data/Cloud-basic/shared/Dataset/HTF/nifti_masked"
    # patch_save_root = "/home/daeun/Workspace/HTF-JD/patches"
    #
    # nii_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".nii.gz")])
    #
    # print(f"Total NIfTI files: {len(nii_files)}")
    #
    # for fname in nii_files:
    #     fpath = os.path.join(data_dir, fname)
    #     process_ct_file(fpath, patch_save_root)


    ##################지멘스+120/315 프로토콜


# ============================================================
# main
# ============================================================
if __name__ == "__main__":

    data_dir = "/data/Cloud-basic/shared/Dataset/HTF/nifti_masked"
    patch_save_root = "/home/daeun/Workspace/HTF-JD/patches"

    # 1) 최종 사용할 ID 리스트 읽기
    id_list_path = "/home/daeun/Workspace/HTF-JD/model/ct_ids_setting1.txt"  # 경로 맞게!
    with open(id_list_path) as f:
        final_ids = [line.strip() for line in f if line.strip()]

    final_ids_set = set(final_ids)
    print(f"최종 사용할 환자 ID 개수: {len(final_ids_set)}")

    # 2) NIfTI 파일 목록
    nii_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".nii.gz")])
    print(f"폴더 안 NIfTI 파일 개수(전체): {len(nii_files)}")

    used, skipped = 0, 0

    for fname in nii_files:
        # 파일 이름에서 ID 추출: 000000507.nii.gz -> 000000507
        base_id = fname.replace(".nii.gz", "")

        # 최종 ID 목록에 없는 애들은 스킵
        if base_id not in final_ids_set:
            skipped += 1
            continue

        used += 1
        fpath = os.path.join(data_dir, fname)
        process_ct_file(fpath, patch_save_root)

    print(f"\n최종 패치 생성에 사용된 NIfTI 수: {used}")
    print(f"스킵된 NIfTI 수: {skipped}")
