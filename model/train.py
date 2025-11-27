# -*- coding:utf-8 -*-
import os, re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from data_loader import load_label_dict, BagDataset, CTPatchDataset


# ============================================================
# 1) CNN patch encoder
# ============================================================
class PatchEncoder(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 64 → 32

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 32 → 16

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 16 → 8
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, feat_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


# ============================================================
# 2) Attention-MIL Model
# ============================================================
class AttentionMIL(nn.Module):
    def __init__(self, feat_dim=128, hidden_dim=64, num_classes=2):
        super().__init__()

        self.encoder = PatchEncoder(feat_dim)

        self.att_V = nn.Linear(feat_dim, hidden_dim)
        self.att_U = nn.Linear(hidden_dim, 1)

        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, bag, batch_size=64):
        all_feats = []
        N = bag.size(0)

        # Patch-level mini-batch inference
        for i in range(0, N, batch_size):
            batch = bag[i:i+batch_size]
            f = self.encoder(batch)
            all_feats.append(f)

        feats = torch.cat(all_feats, dim=0)     # (N, feat_dim)

        # Attention (score for each patch)
        A = torch.tanh(self.att_V(feats))
        A = self.att_U(A)
        A = torch.softmax(A.squeeze(1), dim=0)

        # Bag feature
        bag_feat = torch.sum(feats * A.unsqueeze(1), dim=0)

        # CT-level prediction
        logits = self.classifier(bag_feat.unsqueeze(0))

        return logits, feats, bag_feat, A


# ============================================================
# 3) Train Loop
# ============================================================
def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    excel_path = "/home/brainlab/Workspace/jycha/HTF/patient_whole_add.xlsx"
    label_dict = load_label_dict(excel_path)

    patch_root = "/home/brainlab/Workspace/jycha/HTF/patches"

    # Load CT IDs
    ct_ids = sorted([
        d for d in os.listdir(patch_root)
        if os.path.isdir(os.path.join(patch_root, d)) and d in label_dict
    ])

    n = len(ct_ids)
    print(f"[Total CT] {n}")

    # Split
    train_ids = ct_ids[:int(n * 0.60)]
    val_ids   = ct_ids[int(n * 0.60):int(n * 0.80)]
    test_ids  = ct_ids[int(n * 0.80):]

    print("\n[Split]")
    print(" Train:", len(train_ids))
    print(" Val  :", len(val_ids))
    print(" Test :", len(test_ids))

    # Dataset
    train_ds = BagDataset(patch_root, train_ids, label_dict)
    val_ds   = BagDataset(patch_root, val_ids, label_dict)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False)

    # Model
    model = AttentionMIL().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Training Config
    epochs = 200
    accumulation_steps = 4

    # Best Acc Tracking
    best_acc = 0
    best_epoch = 0
    best_state = None

    # ========================================================
    # Training Loop
    # ========================================================
    for ep in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        # Train
        for step, (bag, label, _) in enumerate(train_loader):
            bag = bag.to(device).squeeze(0)
            label = label.to(device).squeeze().long()

            logits, _, _, _ = model(bag)

            raw_loss = loss_fn(logits, label.unsqueeze(0))
            total_loss += raw_loss.item()

            loss = raw_loss / accumulation_steps
            loss.backward()

            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # 마지막 step 누락된 경우
        if (step + 1) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        # ====================================================
        # Validation
        # ====================================================
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for bag, label, _ in val_loader:
                bag = bag.to(device).squeeze(0)
                label = label.to(device).squeeze().long()

                logits, _, _, _ = model(bag)
                pred = torch.argmax(logits, dim=1).item()

                correct += (pred == label.item())
                total += 1

        val_acc = correct / total

        # Best model 저장
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = ep + 1
            best_state = model.state_dict()

        print(
            f"[Epoch {ep+1}] "
            f"Loss={total_loss/len(train_loader):.4f} | "
            f"Val Acc={val_acc:.4f} | "
            f"Best={best_acc:.4f} (ep {best_epoch})"
        )

    # ========================================================
    # Save Best Model
    # ========================================================
    best_path = "/home/brainlab/Workspace/jycha/HTF/best_model.pth"
    torch.save(best_state, best_path)
    print(f"[BEST MODEL SAVED] {best_path}")

    return model, test_ids, patch_root, device


# ============================================================
# 4) Visualization
# ============================================================
def visualize_ct_prediction(model, ct_id, patch_root, nii_root, device, save_dir):
    nii_path = os.path.join(nii_root, f"{ct_id}.nii.gz")
    img = nib.load(nii_path)
    vol = img.get_fdata()

    z_mid = vol.shape[2] // 2
    slice_img = vol[:, :, z_mid]

    ct_dir = os.path.join(patch_root, ct_id)
    patch_files = sorted([f for f in os.listdir(ct_dir) if f.endswith(".npy")])

    patches_list = []
    coords = []

    with torch.no_grad():
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

    patches_tensor = torch.tensor(np.stack(patches_list), dtype=torch.float32).unsqueeze(1).to(device)

    logits, feats, bag_feat, att = model(patches_tensor)

    att = att.detach().cpu().numpy()
    ct_pred = torch.argmax(logits, dim=1).item()

    heatmap = np.zeros_like(slice_img)
    countmap = np.zeros_like(slice_img)

    for (y, x), score in zip(coords, att):
        heatmap[y:y+64, x:x+64] += score
        countmap[y:y+64, x:x+64] += 1

    mask = countmap > 0
    final_map = np.zeros_like(heatmap)
    final_map[mask] = heatmap[mask] / countmap[mask]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(slice_img, cmap="gray")
    ax.imshow(final_map, cmap="jet", alpha=0.35)
    ax.set_title(f"CT {ct_id} — Pred: {ct_pred}")
    ax.axis("off")

    save_path = os.path.join(save_dir, f"{ct_id}_viz.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[Saved] {save_path}")


# ============================================================
# 5) Evaluation
# ============================================================
def evaluate_ct_level(model, patch_root, test_ids, device):
    print("\n==========================")
    print(" CT-level Evaluation (MIL)")
    print("==========================")

    nii_root = "/home/brainlab/Workspace/jycha/HTF/nifti_masked"
    save_dir = "/home/brainlab/Workspace/jycha/HTF/plot"

    for ct_id in test_ids:
        ct_dir = os.path.join(patch_root, ct_id)
        dataset = CTPatchDataset(ct_dir)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)

        patches = []
        for p, _ in loader:
            patches.append(p)
        patches = torch.cat(patches, dim=0).to(device)

        with torch.no_grad():
            logits, feats, bag_feat, att = model(patches)
            pred = torch.argmax(logits, dim=1).item()

        print(f"CT {ct_id} → Pred label: {pred}")

        visualize_ct_prediction(model, ct_id, patch_root, nii_root, device, save_dir)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    model, test_ids, patch_root, device = train_model()
    evaluate_ct_level(model, patch_root, test_ids, device)