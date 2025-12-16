# -*- coding:utf-8 -*-
import os, re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split

from data_loader import load_label_dict, BagDataset, CTPatchDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt
import copy

#기본 MIL

"""
# ============================================================
# 1) CNN patch encoder (patch → feature)
# ============================================================
class PatchEncoder(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64→32
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32→16
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, feat_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x  # (B, feat_dim)
"""


# ============================================================
# 1) 3-Layer MLP patch encoder
# ============================================================
class PatchEncoder(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),                      # (1,64,64) → (4096)
            nn.Linear(64*64, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, feat_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.mlp(x)  # (B, feat_dim)

# ============================================================
# 2) CT-level MIL classifier
# ============================================================
class MILClassifier(nn.Module):
    def __init__(self, feat_dim=128, num_classes=2):
        super().__init__()
        self.encoder = PatchEncoder(feat_dim)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, bag, batch_size=64):
        """
        bag : (N, 1, 64, 64)
        return:
          logits:    (1, num_classes)  - bag-level prediction
          feats:     (N, feat_dim)     - patch-level features
          bag_feat:  (feat_dim,)       - pooled bag feature
          attn_dummy: None             - attention 안 쓰므로 None
        """
        all_feats = []
        N = bag.size(0)

        # 1) patch encoder (chunk 단위로)
        for i in range(0, N, batch_size):
            batch = bag[i:i + batch_size]   # (B, 1, 64, 64)
            f = self.encoder(batch)         # (B, feat_dim)
            all_feats.append(f)

        feats = torch.cat(all_feats, dim=0)     # (N, feat_dim)

        # 2) Simple MIL aggregation: Mean pooling
        bag_feat = feats.mean(dim=0)           # (feat_dim,)

        # 3) CT-level logits
        logits = self.classifier(bag_feat.unsqueeze(0))  # (1, num_classes)

        # attention은 안 쓰니까 None 리턴 (train/eval 코드 깨지지 않게 4개 그대로 반환)
        attn_scores = None

        return logits, feats, bag_feat, attn_scores



def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    # Load labels
    excel_path = "/home/daeun/Workspace/HTF-JD/patient_whole_add.xlsx"
    label_dict = load_label_dict(excel_path)

    # ★ DICOM/NIfTI 조건 맞춰서 고른 최종 ID txt 불러오기
    final_id_txt =  "/home/daeun/Workspace/HTF-JD/model/ct_ids_setting1.txt"
    with open(final_id_txt) as f:
        final_ids = {line.strip() for line in f if line.strip()}

    print(f"[Filter] txt에 저장된 최종 ID 개수: {len(final_ids)}")

    patch_root = "/home/daeun/Workspace/HTF-JD/patches"

    # patch 폴더 + label_dict + final_ids 세 조건 모두 만족하는 애들만 사용
    ct_ids = sorted([
        d for d in os.listdir(patch_root)
        if (
            os.path.isdir(os.path.join(patch_root, d))
            and d in label_dict
            and d in final_ids
        )
    ])

    n = len(ct_ids)
    print(f"[Total CT] {n}")

    # 0/1 라벨 리스트 만들기
    labels = [int(label_dict[cid]) for cid in ct_ids]

    # 1단계: Train vs (Val+Test)  → 60% / 40%
    train_ids, temp_ids, y_train, y_temp = train_test_split(
        ct_ids,
        labels,
        test_size=0.40,
        stratify=labels,
        random_state=42,
    )

    # 2단계: (Val+Test) → Val / Test  (각각 전체의 15%씩 → temp의 50%/50%)
    val_ids, test_ids, y_val, y_test = train_test_split(
        temp_ids,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=42,
    )

    print("\n[Split]")
    print(" Train:", len(train_ids))
    print(" Val  :", len(val_ids))
    print(" Test :", len(test_ids))

    print("\n[Label distribution after stratified split]")
    print(" Train:", Counter(y_train))
    print(" Val  :", Counter(y_val))
    print(" Test :", Counter(y_test))

    # BagDataset 사용
    train_ds = BagDataset(patch_root, train_ids, label_dict)
    val_ds   = BagDataset(patch_root, val_ids, label_dict)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False)

    print(f"\n[BagDataset] Train bags: {len(train_ds)}")
    print(f"[BagDataset] Val bags:   {len(val_ds)}\n")

    # Model
    model = MILClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 50
    accumulation_steps = 4

    # 🔹 Best model tracking용 변수들 (Acc + Loss)
    best_val_acc = 0.0
    best_val_loss = float("inf")    # ★ 추가
    best_epoch = 0
    best_state_dict = None
    best_model_path = "/home/daeun/Workspace/HTF-JD/model_best.pth"

    # Train loop
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, (bag, label, _) in enumerate(train_loader):
            bag = bag.to(device).squeeze(0)          # (N,1,64,64)
            label = label.to(device).squeeze().long()  # scalar

            logits, feats, bag_feat, attn_scores = model(bag)

            raw_loss = loss_fn(logits, label.unsqueeze(0))
            total_loss += raw_loss.item()

            loss = raw_loss / accumulation_steps
            loss.backward()

            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # leftover gradient 처리
        if (step + 1) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        # ================= Validation =================
        model.eval()
        correct = 0
        total = 0
        val_loss_sum = 0.0           # ★ 추가

        with torch.no_grad():
            for bag, label, _ in val_loader:
                bag = bag.to(device).squeeze(0)
                label = label.to(device).squeeze().long()

                logits, _, _,_ = model(bag)

                # ★ validation loss 계산
                loss_val = loss_fn(logits, label.unsqueeze(0))
                val_loss_sum += loss_val.item()

                pred = torch.argmax(logits, dim=1).item()
                correct += (pred == label.item())
                total += 1

        val_acc = correct / total if total > 0 else 0.0
        val_loss = val_loss_sum / len(val_loader)   # ★ epoch 평균 loss

        print(f"[Epoch {ep+1}] Loss={total_loss/len(train_loader):.4f} "
              f"| Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f}")

        # 🔹 Best model 갱신 & 저장 (Acc 우선, 같으면 Loss 작은 걸 선택)
        is_better = False
        if val_acc > best_val_acc:
            is_better = True
        elif np.isclose(val_acc, best_val_acc) and val_loss < best_val_loss:
            is_better = True

        if is_better:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = ep + 1
            #torch.save(model.state_dict(), best_model_path)
            best_state_dict = copy.deepcopy(model.state_dict())
            print(f"  -> New BEST model saved! Epoch {best_epoch}, "
                  f"Val Acc={best_val_acc:.4f}, Val Loss={best_val_loss:.4f}")

    # # 마지막 epoch 모델도 따로 저장 (선택)
    # last_path = "/home/daeun/Workspace/HTF-JD/model_last.pth"
    # torch.save(model.state_dict(), last_path)
    # print(f"[Saved LAST] {last_path}")
    #
    # # 🔹 Best 모델을 로드해서 이후 평가에 사용
    # model.load_state_dict(torch.load(best_model_path))
    # print(f"[Best] Epoch {best_epoch}, Val Acc={best_val_acc:.4f}, Val Loss={best_val_loss:.4f}")
    # print(f"[Saved BEST] {best_model_path}")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        torch.save(best_state_dict, best_model_path)  # 🔥 여기서 한 번만 save
        print(f"[Best] Epoch {best_epoch}, "
              f"Val Acc={best_val_acc:.4f}, Val Loss={best_val_loss:.4f}")
        print(f"[Saved BEST] {best_model_path}")
    else:
        print("[Warn] best_state_dict가 없습니다. (val을 안 돌렸거나, 데이터가 없을 수 있음)")

    return model, test_ids, patch_root, device

def visualize_ct_prediction(model, ct_id, patch_root, nii_root, device, save_dir):
    """
    (1) patch 위치 표시
    (2) patch 예측값 표시
    (3) heatmap overlay
    (4) CT-level 최종 결과까지 표시
    patch embedding의 norm을 heatmap score로 사용
    """
    nii_path = os.path.join(nii_root, f"{ct_id}.nii.gz")
    img = nib.load(nii_path)
    vol = img.get_fdata()

    z_mid = vol.shape[2] // 2
    slice_img = vol[:, :, z_mid]

    ct_dir = os.path.join(patch_root, ct_id)
    patch_files = sorted([f for f in os.listdir(ct_dir) if f.endswith(".npy")])

    patches_list = []
    coords = []

    # 패치 불러오기 (여기서는 단순히 numpy로만 모음)
    for fname in patch_files:
        m = re.search(r"_z(\d+)_y(\d+)_x(\d+)_p", fname)
        if m is None:
            continue

        z, y, x = map(int, m.groups())
        if z != z_mid:
            continue

        patch = np.load(os.path.join(ct_dir, fname)).astype(np.float32)
        patch = (patch - patch.mean()) / (patch.std() + 1e-6)
        patches_list.append(patch)
        coords.append((y, x))

    if len(patches_list) == 0:
        print(f"[Warn] {ct_id}: no patches found on mid-slice (z={z_mid})")
        return

    # ⚠️ warning 없애기: list → numpy array → tensor
    patches_arr = np.stack(patches_list, axis=0)          # (N,64,64)
    patches_tensor = torch.from_numpy(patches_arr).unsqueeze(1).to(device)  # (N,1,64,64)

    model.eval()
    with torch.no_grad():
        logits, feats, _, _ = model(patches_tensor)
        ct_pred = torch.argmax(logits, dim=1).item()

    # feat-norm heatmap (no_grad 안에서 나왔으니 requires_grad=False)
    scores = torch.norm(feats, dim=1).cpu().numpy()       # (N,)

    heatmap = np.zeros_like(slice_img)
    countmap = np.zeros_like(slice_img)

    for (y, x), s in zip(coords, scores):
        heatmap[y:y + 64, x:x + 64] += s
        countmap[y:y + 64, x:x + 64] += 1

    mask = countmap > 0
    final_map = np.zeros_like(heatmap)
    final_map[mask] = heatmap[mask] / countmap[mask]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(slice_img, cmap='gray')
    ax.imshow(final_map, cmap='jet', alpha=0.35)
    ax.set_title(f"CT {ct_id} — MIL Pred: {ct_pred}")
    ax.axis('off')

    os.makedirs(save_dir, exist_ok=True)
    plt.show()
    plt.close()


# ============================================================
# 5) Evaluate + Visualize
# ============================================================
def evaluate_ct_level(model, patch_root, eval_ids, device, label_dict):
    print("\n==========================")
    print(" CT-level Evaluation (MIL)")
    print("==========================")

    nii_root = "/data/Cloud-basic/shared/Dataset/HTF/nifti_masked"
    save_dir = "/home/daeun/Workspace/HTF-JD/patches/plot"

    y_true = []
    y_pred = []
    y_score = []        # ★ ROC/PR curve 위한 확률값 저장

    for ct_id in eval_ids:
        ct_dir = os.path.join(patch_root, ct_id)
        dataset = CTPatchDataset(ct_dir)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)

        patches = []
        for p, _ in loader:
            patches.append(p)
        patches = torch.cat(patches, dim=0).to(device)

        with torch.no_grad():
            logits, _, _, _ = model(patches)
            prob = torch.softmax(logits, dim=1)[0, 1].item()  # ★ Positive 확률
            pred = int(prob >= 0.5)

        true_label = int(label_dict[ct_id])

        y_true.append(true_label)
        y_pred.append(pred)
        y_score.append(prob)

        print(f"CT {ct_id} → GT: {true_label}, Pred: {pred}, Prob={prob:.4f}")

        last_ct_for_vis = ct_id

    # 🔥 루프 끝나고 마지막 CT 한 장만 플롯
    if last_ct_for_vis is not None:
        print(f"\n[Visualization] 마지막 CT {last_ct_for_vis}만 heatmap 플롯")
        visualize_ct_prediction(model, last_ct_for_vis, patch_root, nii_root, device, save_dir)

    # ==========================================================
    # Confusion Matrix & Metrics
    # ==========================================================
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    accuracy    = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN + 1e-6)
    specificity = TN / (TN + FP + 1e-6)
    precision   = TP / (TP + FP + 1e-6)
    f1          = 2 * precision * sensitivity / (precision + sensitivity + 1e-6)

    print("\n===== Test Performance =====")
    print("Confusion Matrix:")
    print(cm)
    print(f"\nAccuracy   : {accuracy:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"F1-score   : {f1:.4f}")

    # ==========================================================
    # ROC Curve + AUC
    # ==========================================================
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close()

    # ==========================================================
    # Precision–Recall Curve
    # ==========================================================
    precision_list, recall_list, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall_list, precision_list)

    plt.figure()
    plt.plot(recall_list, precision_list, label=f"PR Curve (AUC={pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close()

    return accuracy, sensitivity, specificity, precision, f1
# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    model, test_ids, patch_root, device = train_model()

    # label_dict 다시 로드
    excel_path = "/home/daeun/Workspace/HTF-JD/patient_whole_add.xlsx"
    label_dict = load_label_dict(excel_path)

    evaluate_ct_level(model, patch_root, test_ids, device, label_dict)