import nibabel as nib
import numpy as np
import os
import pydicom
import matplotlib.pyplot as plt
nii_path = "/data/Cloud-basic/shared/Dataset/HTF/nifti/000020367.nii.gz"

img = nib.load(nii_path)
data = img.get_fdata() #512,512,32 (Height, Width, Slices)

print("NIfTI shape:", data.shape)
print("dtype:", data.dtype)

# 각 축 크기 출력
print(f"Height (Y): {data.shape[0]}")
print(f"Width  (X): {data.shape[1]}")
print(f"Slices(Z): {data.shape[2]}")

print("------------------------------")

H, W, Z = data.shape
num_slices = Z

cols = 4
rows = int(np.ceil(num_slices / cols))

plt.figure(figsize=(25, 40))   # 충분히 크게

for i in range(num_slices):
    ax = plt.subplot(rows, cols, i + 1)
    ax.imshow(data[:, :, i], cmap='gray')
    ax.set_title(f"Slice {i}", fontsize=12)
    ax.axis('off')

plt.tight_layout()
plt.show()



# dicom_dir = "/data/Cloud-basic/shared/Dataset/HTF/dicom/000020367"

# # 폴더에서 첫 DICOM 파일만 읽기
# dcm_path = None
# for root, dirs, files in os.walk(dicom_dir):
#     for f in files:
#         if f.lower().endswith(".dcm"):
#             dcm_path = os.path.join(root, f)
#             break
#     if dcm_path:
#         break
#
# ds = pydicom.dcmread(dcm_path)
#
# print("DICOM Rows     :", ds.Rows)
# print("DICOM Columns  :", ds.Columns)
# print("DICOM Shape    :", (ds.Rows, ds.Columns))
