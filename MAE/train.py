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
    image_size = 384
    patch_size = 32
    mask_ratio = 0.75
    embed_dim = 256
    depth = 6
    num_heads = 8
    mlp_ratio = 4.0

    batch_size = 8
    accum_steps = 4
    num_epochs = 3000
    lr = 1e-4

    data_root = "/home/brainlab/Workspace/jycha/HTF/dicom"
    vis_dir = "/home/brainlab/Workspace/jycha/HTF/plot"
    save_dir = "/home/brainlab/Workspace/jycha/HTF/checkpoints"


# ============================================================
# Reconstruction Accuracy
# ============================================================
def reconstruction_accuracy(orig, recon):
    mse = torch.mean((orig - recon) ** 2, dim=[1, 2, 3])  # per sample
    acc = 1.0 - mse
    acc = torch.clamp(acc, 0.0, 1.0)
    return acc.mean().item()


# ============================================================
# Visualization (input / mask / reconstruction)
# ============================================================
def visualize(img, pred, mask, cfg, save_name):
    """
    img  : (1, 384, 384)
    pred : (L, patch_dim)
    mask : (L)
    """
    img = img[0].detach().cpu().numpy()
    p = cfg.patch_size

    # ----- masked target patch map -----
    mask = mask.detach().cpu().numpy()
    L = mask.shape[0]
    h = w = int(np.sqrt(L))
    mask_2d = mask.reshape(h, w)

    # ----- reconstruction -----
    pred = pred.detach().cpu().numpy()  # (L, patch_dim)
    patches = pred.reshape(h, w, p, p)
    recon = np.block([[patches[i, j] for j in range(w)] for i in range(h)])

    # ----- masked image -----
    mask_expanded = mask_2d.repeat(p, axis=0).repeat(p, axis=1)
    masked_img = img.copy()
    masked_img[mask_expanded == 1] = 0.5

    # ----- plot -----
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].imshow(img, cmap="gray")
    ax[0].set_title("Original")
    ax[0].axis("off")

    ax[1].imshow(masked_img, cmap="gray")
    ax[1].set_title("Masked Input")
    ax[1].axis("off")

    ax[2].imshow(recon, cmap="gray")
    ax[2].set_title("Reconstruction")
    ax[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_name)
    plt.show()
    plt.close()


# ============================================================
# MAE Training Loop
# ============================================================
def train_mae():
    set_seed(777)
    cfg = Config()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    # Load CT IDs (folder names)
    all_ct_ids = sorted([
        d for d in os.listdir(cfg.data_root)
        if os.path.isdir(os.path.join(cfg.data_root, d))
    ])

    print(f"[Test] Using {len(all_ct_ids)} CT subjects")

    train_loader, val_loader = make_mae_dataloaders(
        cfg.data_root, all_ct_ids, batch_size=cfg.batch_size
    )

    # Model
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


    # Train
    best_val_loss = 999

    for epoch in range(cfg.num_epochs):

        model.train()
        total_loss = 0
        total_train_acc = 0
        optimizer.zero_grad()

        for step, img in enumerate(train_loader):

            img = img.to(device)
            loss, pred, mask = model(img, mask_ratio=cfg.mask_ratio)

            # reconstruction
            recon = model.unpatchify(pred)
            batch_acc = reconstruction_accuracy(img, recon)
            total_train_acc += batch_acc

            # loss backward
            loss = loss / cfg.accum_steps
            loss.backward()

            if (step + 1) % cfg.accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * cfg.accum_steps

        train_loss = total_loss / len(train_loader)
        train_acc = total_train_acc / len(train_loader)

        # Validation
        model.eval()
        val_total = 0
        val_acc_total = 0

        with torch.no_grad():
            for img in val_loader:
                img = img.to(device)
                loss, pred, mask = model(img, mask_ratio=cfg.mask_ratio)
                val_total += loss.item()

                recon = model.unpatchify(pred)
                val_acc_total += reconstruction_accuracy(img, recon)

        val_loss = val_total / len(val_loader)
        val_acc = val_acc_total / len(val_loader)

        print(f"[Epoch {epoch + 1:03d}] Train Loss={train_loss:.4f} | Train Acc={train_acc:.4f} | Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(cfg.save_dir, f"best_epoch{epoch + 1}.pth")
            torch.save(model.state_dict(), best_path)

            # overwrite best.pth
            torch.save(model.state_dict(), os.path.join(cfg.save_dir, "best.pth"))

            print(f"[BEST SAVED] {best_path}")

        # Always save last
        torch.save(model.state_dict(), os.path.join(cfg.save_dir, "last.pth"))

        # Visualization every 10 epochs
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                img = next(iter(val_loader)).to(device)
                _, pred, mask = model(img, mask_ratio=cfg.mask_ratio)

                save_path = os.path.join(cfg.vis_dir, f"epoch_{epoch + 1}.png")
                visualize(img[0], pred[0], mask[0], cfg, save_path)
                print(f"[VIS SAVED] {save_path}")

    print("\nTraining Completed.")
    print(f"Best model saved inside: {cfg.save_dir}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    train_mae()