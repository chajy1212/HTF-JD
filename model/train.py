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


# ============================================================
# 2) CT-level MIL classifier
# ============================================================
class MILClassifier(nn.Module):
    def __init__(self, feat_dim=128, num_classes=2):
        super().__init__()
        self.encoder = PatchEncoder(feat_dim)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, bag):
        """
        bag: (N, 1, 64, 64)  -> N개의 patch
        """
        feats = self.encoder(bag)        # (N, 128)
        bag_feat = feats.mean(dim=0)     # (128,) 평균 MIL
        logits = self.classifier(bag_feat.unsqueeze(0))  # (1, num_classes)
        return logits, feats, bag_feat


# ============================================================
# 3) Train (MIL)
# ============================================================
def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    # Load labels
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

    train_ids = ct_ids[:int(n * 0.70)]
    val_ids   = ct_ids[int(n * 0.70):int(n * 0.85)]
    test_ids  = ct_ids[int(n * 0.85):]

    print("\n[Split]")
    print(" Train:", len(train_ids))
    print(" Val  :", len(val_ids))
    print(" Test :", len(test_ids))

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

    # Train loop
    for ep in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for step, (bag, label, _) in enumerate(train_loader):
            bag = bag.to(device).squeeze(0)
            label = label.to(device).squeeze().long()

            logits, feats, bag_feat = model(bag)

            # accumulation 전 loss 값 저장
            raw_loss = loss_fn(logits, label.unsqueeze(0))
            total_loss += raw_loss.item()

            # accumulation용 normalizing
            loss = raw_loss / accumulation_steps
            loss.backward()

            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # leftover gradient 처리
        if (step + 1) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for bag, label, _ in val_loader:
                bag = bag.to(device).squeeze(0)
                label = label.to(device).squeeze().long()

                logits, _, _ = model(bag)
                pred = torch.argmax(logits, dim=1).item()

                correct += (pred == label.item())
                total += 1

        val_acc = correct / total

        print(f"[Epoch {ep+1}] Loss={total_loss/len(train_loader):.4f} | Val Acc={val_acc:.4f}")

    save_path = "/home/brainlab/Workspace/jycha/HTF/model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"[Saved] {save_path}")

    return model, test_ids, patch_root, device


# ============================================================
# 4) Patch-level Visualization
# ============================================================
def visualize_ct_prediction(model, ct_id, patch_root, nii_root, device, save_dir):
    """
    (1) patch 위치 표시
    (2) patch 예측값 표시
    (3) heatmap overlay
    (4) CT-level 최종 결과까지 표시
    MIL에서 patch attention이 없기 때문에, patch embedding의 norm을 heatmap score로 사용
    """
    nii_path = os.path.join(nii_root, f"{ct_id}.nii.gz")
    img = nib.load(nii_path)
    vol = img.get_fdata()

    z_mid = vol.shape[2] // 2
    slice_img = vol[:, :, z_mid]

    ct_dir = os.path.join(patch_root, ct_id)
    patch_files = sorted([f for f in os.listdir(ct_dir) if f.endswith(".npy")])

    # Load all patches for MIL embedding
    patches_list = []
    coords = []
    model.eval()

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

    patches_tensor = torch.tensor(patches_list, dtype=torch.float32).unsqueeze(1).to(device)
    logits, feats, _ = model(patches_tensor)
    ct_pred = torch.argmax(logits, dim=1).item()

    # feat-norm heatmap
    scores = torch.norm(feats, dim=1).cpu().numpy()
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
    save_path = os.path.join(save_dir, f"{ct_id}_mil_viz.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    # plt.show()
    plt.close()

    print(f"[Saved] {save_path}")


# ============================================================
# 5) Evaluate + Visualize
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

        # MIL forward 전체
        patches = []
        for p, _ in loader:
            patches.append(p)

        patches = torch.cat(patches, dim=0).to(device)

        with torch.no_grad():
            logits, feats, bag_feat = model(patches)
            pred = torch.argmax(logits, dim=1).item()

        print(f"CT {ct_id} → Pred label: {pred}")

        visualize_ct_prediction(model, ct_id, patch_root, nii_root, device, save_dir)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    model, test_ids, patch_root, device = train_model()
    evaluate_ct_level(model, patch_root, test_ids, device)