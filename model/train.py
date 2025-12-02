# -*- coding:utf-8 -*-
import os
import random
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from data_loader import make_mae_dataloaders
from model import MaskedAutoencoderViT


# ============================================================
# Seed (Reproducibility)
# ============================================================
def set_seed(seed=777):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Config
# ============================================================
class Config:
    image_size = 300          # 최종 입력 크기
    patch_size = 15           # MAE의 internal patch size
    mask_ratio = 0.75
    embed_dim = 256
    depth = 6
    num_heads = 8
    mlp_ratio = 4.0

    batch_size = 64
    accum_steps = 4
    num_epochs = 3000
    lr = 1e-4

    data_root = "/home/brainlab/Workspace/jycha/HTF/nifti_masked"
    save_path = "/home/brainlab/Workspace/jycha/HTF/model.pth"
    vis_dir = "/home/brainlab/Workspace/jycha/HTF/plot"


# ============================================================
# Visualization (input / mask / reconstruction)
# ============================================================
def visualize(img, pred, mask, cfg, save_name):
    """
    img  : (1, 128, 128)
    pred : (L, patch_dim)
    mask : (L)
    """
    img = img[0].detach().cpu().numpy()
    p = cfg.patch_size

    # ----- masked target patch map -----
    mask = mask.detach().cpu().numpy()  # (L,)
    h = w = int((mask.shape[0]) ** 0.5)
    mask_2d = mask.reshape(h, w)

    # ----- reconstruction -----
    pred = pred.detach().cpu().numpy()  # (L, patch_dim)
    patches = pred.reshape(h, w, p, p)
    rec_img = np.block([[patches[i, j] for j in range(w)] for i in range(h)])

    # ----- plot -----
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].imshow(img, cmap="gray")
    ax[0].set_title("Original")
    ax[0].axis("off")

    ax[1].imshow(mask_2d, cmap="gray")
    ax[1].set_title("Mask Map")
    ax[1].axis("off")

    ax[2].imshow(rec_img, cmap="gray")
    ax[2].set_title("Reconstruction")
    ax[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_name)
    plt.show()
    plt.close()


# ============================================================
# MAE Training Code
# ============================================================
def train_mae():
    set_seed(777)
    cfg = Config()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    # --------------------------------------------------------
    # Load Dataset
    # --------------------------------------------------------
    all_ct_ids = sorted([
        f.replace(".nii.gz", "")
        for f in os.listdir(cfg.data_root)
        if f.endswith(".nii.gz")
    ])

    # --- Quick Test ---
    all_ct_ids = all_ct_ids[:20]
    print(f"[Quick Test Mode] Using {len(all_ct_ids)} CT volumes")

    train_loader, val_loader = make_mae_dataloaders(
        cfg.data_root,
        all_ct_ids,
        batch_size=cfg.batch_size
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = MaskedAutoencoderViT(
        img_size=cfg.image_size,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    print("\n[Model Created]")
    print(model)


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------
    best_val_loss = 999

    for epoch in range(cfg.num_epochs):
        model.train()
        total_loss = 0

        for step, img in enumerate(train_loader):
            img = img.to(device)

            loss, pred, mask = model(img, mask_ratio=cfg.mask_ratio)
            loss = loss / cfg.accum_steps  # loss scaling
            loss.backward()

            if (step + 1) % cfg.accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------
        model.eval()
        val_total = 0

        with torch.no_grad():
            for img in val_loader:
                img = img.to(device)
                loss, _, _ = model(img, mask_ratio=cfg.mask_ratio)
                val_total += loss.item()

        val_loss = val_total / len(val_loader)

        print(f"[Epoch {epoch+1:03d}] Train={train_loss:.4f} | Val={val_loss:.4f}")

        # ----------------------------------------------------
        # Save Best
        # ----------------------------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), cfg.save_path)
            print(f"[BEST SAVED] epoch={epoch+1} | val_loss={val_loss:.4f}")

        # ----------------------------------------------------
        # Visualization (only first batch of val set)
        # ----------------------------------------------------
        with torch.no_grad():
            img = next(iter(val_loader)).to(device)
            _, pred, mask = model(img, mask_ratio=cfg.mask_ratio)

            vis_path = os.path.join(cfg.vis_dir, f"epoch_{epoch+1}.png")
            visualize(img[0], pred[0], mask[0], cfg, vis_path)
            print(f"[VIS SAVED] {vis_path}")

    print(f"\nTraining Done. Best model: {cfg.save_path}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    train_mae()