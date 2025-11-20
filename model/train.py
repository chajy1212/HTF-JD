# -*- coding:utf-8 -*-
import os, re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from data_loader import load_label_dict, PatchDataset, CTPatchDataset


# ============================================================
# 1) Simple CNN 모델
# ============================================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 → 32

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 → 16

            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 2) CT-level inference (패치 → majority vote)
# ============================================================
def predict_ct(model, ct_dir, device):
    """
    ct_dir: ex) /home/jycha/HTF/patches/000000507
    """
    dataset = CTPatchDataset(ct_dir)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    preds = []
    model.eval()

    with torch.no_grad():
        for patches, _ in loader:
            patches = patches.to(device)
            logits = model(patches)
            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.cpu().tolist())

    if len(preds) == 0:
        raise RuntimeError(f"[ERROR] No patches in CT folder: {ct_dir}")

    # Majority Vote
    final_pred = max(set(preds), key=preds.count)
    return final_pred


# ============================================================
# 3) Train + Validation + Test 분리 학습
# ============================================================
def train_model():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    # Load labels (from Excel)
    excel_path = "/home/jycha/HTF/patient_whole_add.xlsx"
    label_dict = load_label_dict(excel_path)

    # Load CT patch folders
    patch_root = "/home/jycha/HTF/patches"

    ct_ids = sorted([
        d for d in os.listdir(patch_root)
        if os.path.isdir(os.path.join(patch_root, d)) and d in label_dict
    ])

    n = len(ct_ids)
    if n < 10:
        raise ValueError("CT 개수가 너무 적습니다. 데이터 확인 필요.")

    # Train / Valid / Test Split
    train_ids = ct_ids[:int(n * 0.70)]
    val_ids   = ct_ids[int(n * 0.70):int(n * 0.85)]
    test_ids  = ct_ids[int(n * 0.85):]

    print("\n[Split Info]")
    print(" Train CT :", len(train_ids))
    print(" Val CT   :", len(val_ids))
    print(" Test CT  :", len(test_ids))

    # Dataset
    train_ds = PatchDataset(patch_root, label_dict, split_ids=train_ids)
    val_ds   = PatchDataset(patch_root, label_dict, split_ids=val_ids)

    print(f"[PatchDataset] Train patches: {len(train_ds)}")
    print(f"[PatchDataset] Val patches:   {len(val_ds)}\n")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)

    # Model/Optimizer/Loss
    model = SimpleCNN(num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Training Loop
    epochs = 10

    for ep in range(epochs):
        model.train()
        total_loss = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"\n[Epoch {ep+1}] Train Loss: {total_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = torch.argmax(model(X), dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        print(f"Patch-level Val Acc: {correct / total:.4f}")

    # Save model
    save_path = "/home/jycha/HTF/model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"[Saved] {save_path}")

    return model, test_ids, patch_root, device


# ============================================================
# 4) 시각화 — Patch heatmap + Pred + Rectangles
# ============================================================
def visualize_ct_prediction(model, ct_id, patch_root, nii_root, device, save_dir):
    """
    (1) patch 위치 표시
    (2) patch 예측값 표시
    (3) heatmap overlay
    (4) CT-level 최종 결과까지 표시
    """
    nii_path = os.path.join(nii_root, f"{ct_id}.nii.gz")
    img = nib.load(nii_path)
    vol = img.get_fdata()

    z_mid = vol.shape[2] // 2
    slice_img = vol[:, :, z_mid]

    ct_dir = os.path.join(patch_root, ct_id)
    patch_files = sorted([f for f in os.listdir(ct_dir) if f.endswith(".npy")])

    heatmap = np.zeros_like(slice_img, dtype=float)
    countmap = np.zeros_like(slice_img, dtype=float)

    patch_pred_list = []

    model.eval()
    with torch.no_grad():

        for fname in patch_files:
            m = re.search(r"_z(\d+)_y(\d+)_x(\d+)_p", fname)
            if m is None:
                continue

            z, y, x = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if z != z_mid:
                continue

            patch = np.load(os.path.join(ct_dir, fname)).astype(np.float32)
            patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-6)
            patch_tensor = torch.tensor(patch[None, None], dtype=torch.float32).to(device)

            logits = model(patch_tensor)
            prob = torch.softmax(logits, dim=1)[0]
            pred = torch.argmax(prob).item()
            score = float(prob[pred].item())

            patch_pred_list.append({
                "coord": (y, x),
                "pred": pred,
                "score": score
            })

            heatmap[y:y + 64, x:x + 64] += score
            countmap[y:y + 64, x:x + 64] += 1

    final_heatmap = np.zeros_like(heatmap)
    mask = countmap > 0
    final_heatmap[mask] = heatmap[mask] / countmap[mask]

    if len(patch_pred_list) > 0:
        ct_pred = max(
            [p["pred"] for p in patch_pred_list],
            key=[p["pred"] for p in patch_pred_list].count
        )
    else:
        ct_pred = -1

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(slice_img, cmap="gray")

    for p in patch_pred_list:
        y, x = p["coord"]
        pred = p["pred"]
        score = p["score"]

        color = "red" if pred == 1 else "blue"

        rect = patches.Rectangle((x, y), 64, 64, linewidth=1.5,
                                 edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        ax.text(x, y - 5, f"{pred}:{score:.2f}",
                color=color, fontsize=8, weight="bold")

    ax.imshow(final_heatmap, cmap="jet", alpha=0.35)
    ax.set_title(f"CT {ct_id} — Predicted: {ct_pred}")
    ax.axis("off")

    save_path = os.path.join(save_dir, f"{ct_id}_viz.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    # plt.show()
    plt.close()

    print(f"[Saved] {save_path}")


# ============================================================
# 5) Test + Visualization
# ============================================================
def evaluate_ct_level(model, patch_root, test_ids, device):
    print("\n==========================")
    print(" CT-level Evaluation")
    print("==========================")

    nii_root = "/data/Cloud-basic/shared/Dataset/HTF/nifti_masked"
    save_dir = "/home/jycha/HTF/plot"

    for ct_id in test_ids:
        ct_dir = os.path.join(patch_root, ct_id)
        pred = predict_ct(model, ct_dir, device)
        print(f"CT {ct_id} → Predicted label: {pred}")

        visualize_ct_prediction(
            model=model,
            ct_id=ct_id,
            patch_root=patch_root,
            nii_root=nii_root,
            device=device,
            save_dir=save_dir
        )


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    model, test_ids, patch_root, device = train_model()
    evaluate_ct_level(model, patch_root, test_ids, device)