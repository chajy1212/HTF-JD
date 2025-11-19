import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# === 1) 파일 불러오기 ===
path = "/data/Cloud-basic/shared/Dataset/HTF/nifti_masked/000000507.nii.gz"   # 너의 CT 파일
img = nib.load(path)
ct = img.get_fdata()

print("CT shape:", ct.shape)  # (512,512,30)


# === 2) z축 절반 위치 슬라이스 선택 ===
z = ct.shape[2] // 2
slice2d = ct[:, :, z]

print("Selected slice shape:", slice2d.shape)


# === 3) 시각화 ===
plt.figure(figsize=(6,6))
plt.imshow(slice2d, cmap='gray')
plt.title(f"CT Slice z = {z}")
plt.axis("off")
plt.show()