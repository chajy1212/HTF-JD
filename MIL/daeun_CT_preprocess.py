import os
import numpy as np
import pandas as pd
import pydicom
from data_loader import load_label_dict   # 🔥 label 쓰려면 추가

#가장 많은 제조사 -> 1st, 2nd Setting ID.txt 저장

dicom_dir_path = '/data/Cloud-basic/shared/Dataset/HTF/dicom'
excel_path = "/home/daeun/Workspace/HTF-JD/patient_whole_add.xlsx"

df = pd.read_excel(excel_path)

dicom_dir_ids = sorted([
    os.path.join(dicom_dir_path, d) for d in os.listdir(dicom_dir_path)
])

clean_dicom = []
seen_base = set()
for patient in dicom_dir_ids:
    base = patient.split("_")[0]
    if base in seen_base:
        continue
    seen_base.add(base)
    clean_dicom.append(patient)

clean_dicom_ids = [os.path.basename(p) for p in clean_dicom]

print("엑셀 환자수 :", len(df))
print("DICOM 원래환자수 :", len(dicom_dir_ids))
print("DICOM 정리된환자수 :", len(clean_dicom_ids))

overlap_ids = sorted(set(df['hosp_id']) & set(clean_dicom_ids))
print("엑셀_DICOM 겹치는 환자수 : ", len(overlap_ids))

# ===== 각 환자에서 제조사/voltage/current setting =====
id_to_dir = {os.path.basename(p): p for p in clean_dicom}

def find_one_dicom(folder_path):
    """폴더(및 하위 폴더)에서 .dcm 파일 하나만 찾기"""
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            if f.lower().endswith(".dcm"):
                return os.path.join(dirpath, f)
    return None

records = []

for pid in overlap_ids:   # 1495명 대상
    folder = id_to_dir.get(pid)
    if folder is None:
        print(f"[경고] {pid} 폴더를 못 찾음")
        continue

    dcm_path = find_one_dicom(folder)
    if dcm_path is None:
        print(f"[경고] {pid} 폴더에 DICOM 파일이 없음")
        continue

    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)

    manu = getattr(ds, "Manufacturer", "UNKNOWN")
    kvp  = getattr(ds, "KVP", None)               # tube voltage (kVp)
    tube = getattr(ds, "XRayTubeCurrent", None)   # tube current (mA)

    records.append({
        "hosp_id": pid,
        "Manufacturer": manu,
        "KVP": kvp,
        "TubeCurrent": tube,
    })

meta_df = pd.DataFrame(records)
print(meta_df.head())
print("총 메타데이터 개수:", len(meta_df))

# ===== 가장 많은 조합 + 두 번째 조합 뽑기 =====
print(meta_df["Manufacturer"].value_counts())

top_manu = meta_df["Manufacturer"].value_counts().idxmax()
print("가장 많은 제조사:", top_manu)

top_df = meta_df[meta_df["Manufacturer"] == top_manu].copy()
top_df = top_df.dropna(subset=["KVP", "TubeCurrent"])

top_df["setting"] = top_df["KVP"].astype(str) + " | " + top_df["TubeCurrent"].astype(str)

print("\n이 제조사에서 세팅별 환자 수 top 10:")
setting_counts = top_df["setting"].value_counts()
print(setting_counts.head(10))

# 1등 / 2등 세팅
best_setting   = setting_counts.index[0]
second_setting = setting_counts.index[1] if len(setting_counts) > 1 else None

print("\n선택된 1등 세팅:", best_setting)
if second_setting is not None:
    print("선택된 2등 세팅:", second_setting)

best_ids   = top_df.loc[top_df["setting"] == best_setting, "hosp_id"].tolist()
second_ids = top_df.loc[top_df["setting"] == second_setting, "hosp_id"].tolist() \
             if second_setting is not None else []

print("1등 세팅 환자 수:", len(best_ids))
print("2등 세팅 환자 수:", len(second_ids))

# ===== label 0/1 카운트 (엑셀 라벨 딕셔너리 이용) =====
label_dict = load_label_dict(excel_path)   # {hosp_id: 0 or 1}
# ============================================
# 🔥 전체 1495명(label 분포) 출력
# ============================================
all_labels = []
for pid in overlap_ids:
    if pid in label_dict:
        all_labels.append(int(label_dict[pid]))
    else:
        print(f"[경고] 라벨 없음: {pid}")

total = len(all_labels)
n0 = sum(1 for v in all_labels if v == 0)
n1 = sum(1 for v in all_labels if v == 1)

print("\n=== 전체 1495명 라벨 분포 ===")
print(f"총 인원: {total}")
print(f"정상(label 0): {n0}")
print(f"환자(label 1): {n1}")
# ============================================

def count_labels(id_list, name):
    labels = []
    for pid in id_list:
        if pid in label_dict:
            labels.append(int(label_dict[pid]))
        else:
            print(f"[경고] label_dict에 {pid}가 없음")
    n_total = len(labels)
    n0 = sum(1 for v in labels if v == 0)
    n1 = sum(1 for v in labels if v == 1)
    print(f"\n{name} 세팅 라벨 분포:")
    print(f" - 총 {n_total}명")
    print(f" - label 0: {n0}")
    print(f" - label 1: {n1}")

count_labels(best_ids,   "1등")
if second_ids:
    count_labels(second_ids, "2등")

# ===== ID를 각각 텍스트 파일로 저장 =====
with open("ct_ids_setting1.txt", "w") as f:
    for pid in best_ids:
        f.write(str(pid) + "\n")

if second_ids:
    with open("ct_ids_setting2.txt", "w") as f:
        for pid in second_ids:
            f.write(str(pid) + "\n")

print("\nct_ids_setting1.txt / ct_ids_setting2.txt 저장 완료")
