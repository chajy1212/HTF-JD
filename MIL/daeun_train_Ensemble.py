import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import nibabel as nib
from torch.utils.data import DataLoader
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import confusion_matrix, roc_curve, auc
import torch.optim as optim

from data_loader import BagDataset, CTPatchDataset, load_label_dict
from daeun_train_Basic import MILClassifier   # 혹은 같은 파일에 이미 정의돼 있으면 필요 X


def split_ct_ids(patch_root, label_dict, final_ids):
    ct_ids = sorted([
        d for d in os.listdir(patch_root)
        if os.path.isdir(os.path.join(patch_root, d))
        and d in label_dict
        and d in final_ids
    ])

    labels = [int(label_dict[cid]) for cid in ct_ids]

    train_ids, temp_ids, y_train, y_temp = train_test_split(
        ct_ids, labels, test_size=0.40, stratify=labels, random_state=42
    )

    val_ids, test_ids, y_val, y_test = train_test_split(
        temp_ids, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    print("[Split]")
    print("Train:", len(train_ids), Counter(y_train))
    print("Val:", len(val_ids), Counter(y_val))
    print("Test:", len(test_ids), Counter(y_test))

    return train_ids, val_ids, test_ids

class VolumeCTDataset(Dataset):
    def __init__(self, nii_root, ct_ids, label_dict, target_shape=(32, 128, 128)):
        """
        nii_root: NIfTI 폴더 (이미 visualize에서 쓰는 경로)
        ct_ids:   사용할 CT ID 리스트 (train_ids / val_ids / test_ids 등)
        label_dict: ID -> 라벨 (0/1)
        target_shape: (D, H, W) 로 리샘플해서 3D CNN에 넣을 크기
        """
        self.nii_root = nii_root
        self.ct_ids = ct_ids
        self.label_dict = label_dict
        self.target_shape = target_shape

    def __len__(self):
        return len(self.ct_ids)

    def __getitem__(self, idx):
        ct_id = self.ct_ids[idx]
        nii_path = os.path.join(self.nii_root, f"{ct_id}.nii.gz")
        img = nib.load(nii_path)
        vol = img.get_fdata().astype(np.float32)  # (H, W, D)

        # normalize
        vol = (vol - vol.mean()) / (vol.std() + 1e-6)

        # (H, W, D) -> (1, 1, D, H, W)
        vol_t = torch.from_numpy(vol).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)

        # 3D resize to target_shape (D,H,W)
        vol_t = F.interpolate(
            vol_t,
            size=self.target_shape,
            mode="trilinear",
            align_corners=False
        )  # (1,1,D',H',W')

        vol_t = vol_t.squeeze(0)  # (1, D', H', W')

        label = int(self.label_dict[ct_id])
        label_t = torch.tensor(label, dtype=torch.long)

        return vol_t, label_t, ct_id

class VolumeCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),        # (D,H,W) -> (D/2,H/2,W/2)

            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),        # again half

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1) # -> (B,32,1,1,1)
        )
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        # x: (B,1,D,H,W)
        h = self.conv(x)              # (B,32,1,1,1)
        h = h.view(h.size(0), -1)     # (B,32)
        logits = self.fc(h)           # (B,num_classes)
        return logits

def train_mil_model_with_given_split(train_ids, val_ids, label_dict, device):
    patch_root = "/home/daeun/Workspace/HTF-JD/patches"

    train_ds = BagDataset(patch_root, train_ids, label_dict)
    val_ds   = BagDataset(patch_root, val_ids,   label_dict)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = MILClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_state = None
    best_val_acc = 0

    for ep in range(50):
        model.train()
        total_loss = 0

        for bag, label, _ in train_loader:
            bag = bag.to(device).squeeze(0)
            label = label.to(device).squeeze().long()
            logits, _, _, _ = model(bag)
            loss = loss_fn(logits, label.unsqueeze(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for bag, label, _ in val_loader:
                bag = bag.to(device).squeeze(0)
                label = label.to(device).squeeze().long()
                logits, _, _, _ = model(bag)
                pred = torch.argmax(logits, dim=1).item()
                correct += (pred == label.item())
                total += 1

        val_acc = correct / total
        print(f"[MIL][Epoch {ep+1}] TrainLoss={total_loss/len(train_loader):.4f} | ValAcc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, patch_root
def train_volume_model(train_ids, val_ids, label_dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nii_root = "/data/Cloud-basic/shared/Dataset/HTF/nifti_masked"

    train_ds = VolumeCTDataset(nii_root, train_ids, label_dict)
    val_ds   = VolumeCTDataset(nii_root, val_ids,   label_dict)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=2, shuffle=False)

    model = VolumeCNN(num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    epochs = 50
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for vol, label, _ in train_loader:
            vol = vol.to(device)          # (B,1,D,H,W)
            label = label.to(device)      # (B,)

            logits = model(vol)
            loss = loss_fn(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # ---- validation ----
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for vol, label, _ in val_loader:
                vol = vol.to(device)
                label = label.to(device)
                logits = model(vol)
                pred = torch.argmax(logits, dim=1)
                correct += (pred == label).sum().item()
                total += label.size(0)

        val_acc = correct / total if total > 0 else 0.0
        print(f"[VolCNN][Epoch {ep+1}] TrainLoss={total_loss/len(train_loader):.4f} | ValAcc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            print(f"  -> New BEST VolumeCNN (Epoch {ep+1}, ValAcc={best_val_acc:.4f})")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model

def evaluate_ct_level_ensemble(mil_model, vol_model,
                               patch_root, eval_ids, device, label_dict):
    print("\n==========================")
    print(" CT-level Evaluation (MIL + Vol Ensemble)")
    print("==========================")

    nii_root = "/data/Cloud-basic/shared/Dataset/HTF/nifti_masked"
    save_dir = "/home/daeun/Workspace/HTF-JD/patches/plot"

    y_true, y_pred, y_score = [], [], []
    last_ct_for_vis = None

    mil_model.eval()
    vol_model.eval()

    alpha = 0.5   # MIL/Volume 가중치 비율

    for ct_id in eval_ids:
        # ---------- 1) MIL 기반 예측 ----------
        ct_dir = os.path.join(patch_root, ct_id)
        dataset = CTPatchDataset(ct_dir)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)

        patches = []
        for p, _ in loader:
            patches.append(p)
        patches = torch.cat(patches, dim=0).to(device)

        with torch.no_grad():
            logits_mil, _, _, _ = mil_model(patches)
            prob_mil = torch.softmax(logits_mil, dim=1)[0, 1].item()

        # ---------- 2) Volume CNN 기반 예측 ----------
        nii_path = os.path.join(nii_root, f"{ct_id}.nii.gz")
        img = nib.load(nii_path)
        vol = img.get_fdata().astype(np.float32)
        vol = (vol - vol.mean()) / (vol.std() + 1e-6)

        vol_t = torch.from_numpy(vol).permute(2,0,1).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
        vol_t = F.interpolate(vol_t, size=(32,128,128), mode="trilinear", align_corners=False)
        vol_t = vol_t.to(device)

        with torch.no_grad():
            logits_vol = vol_model(vol_t)
            prob_vol = torch.softmax(logits_vol, dim=1)[0, 1].item()

        # ---------- 3) Ensemble ----------
        prob_ens = alpha * prob_mil + (1.0 - alpha) * prob_vol
        pred = int(prob_ens >= 0.5)

        true_label = int(label_dict[ct_id])
        y_true.append(true_label)
        y_pred.append(pred)
        y_score.append(prob_ens)

        print(f"CT {ct_id} → GT: {true_label}, "
              f"Prob_MIL={prob_mil:.4f}, Prob_VOL={prob_vol:.4f}, "
              f"Prob_ENS={prob_ens:.4f}, Pred={pred}")

        last_ct_for_vis = ct_id

    # -------- Metrics (← 루프 바깥!!) --------
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    accuracy   = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN + 1e-6)
    specificity = TN / (TN + FP + 1e-6)
    precision   = TP / (TP + FP + 1e-6)
    f1          = 2 * precision * sensitivity / (precision + sensitivity + 1e-6)

    print("\n==== Ensemble Test Result ====")
    print("Accuracy   :", accuracy)
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)
    print("Precision  :", precision)
    print("F1         :", f1)

    return accuracy, sensitivity, specificity, precision, f1
    # 이후 confusion matrix, ROC, PR curve 부분은
    # 지금 너 코드의 evaluate_ct_level()에서 가져와서 그대로 쓰면 돼.
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    excel_path = "/home/daeun/Workspace/HTF-JD/patient_whole_add.xlsx"
    label_dict = load_label_dict(excel_path)

    patch_root = "/home/daeun/Workspace/HTF-JD/patches"
    final_id_txt = "/home/daeun/Workspace/HTF-JD/model/ct_ids_setting1.txt"

    with open(final_id_txt) as f:
        final_ids = {line.strip() for line in f if line.strip()}

    # 1) 공통 split
    train_ids, val_ids, test_ids = split_ct_ids(patch_root, label_dict, final_ids)

    # 2) MIL 학습
    mil_model, patch_root = train_mil_model_with_given_split(train_ids, val_ids, label_dict, device)

    # 3) 3D Volume CNN 학습
    vol_model = train_volume_model(train_ids, val_ids, label_dict)

    # 4) 앙상블 평가
    evaluate_ct_level_ensemble(mil_model, vol_model, patch_root, test_ids, device, label_dict)
