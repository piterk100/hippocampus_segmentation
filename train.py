import torch
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from monai.losses import DiceLoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import math
import random
import numpy as np
import time
import matplotlib.pyplot as plt
import os

from dataloader import DatasetFromNii
from unet import MyUNet2D
from metrics import soft_dice

torch.set_num_threads(4)

model = MyUNet2D(in_channels=1, num_classes=1)

device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print("Using device:", device)
model = model.to(device)

torch.manual_seed(42); np.random.seed(42); random.seed(42)
if device.type == "cuda":
    torch.cuda.manual_seed_all(42)

writer = SummaryWriter("runs/unet2d_run12")

os.makedirs("debug_slices", exist_ok=True)

train_data = DatasetFromNii("train_set/", patches_per_patient=16, mode="train")
val_data = DatasetFromNii("val_set/",   patches_per_patient=2, mode="val")

train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)
val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

def estimate_pos_weight(ds, n=64):
    pos = neg = 0
    for i, s in enumerate(ds):
        if i >= n: break
        m = (s['mask'] > 0).sum().item()
        vox = s['mask'].numel()
        pos += m
        neg += vox - m
    pw = neg / max(1, pos)
    return float(min(max(pw, 5.0), 50.0))

pw = estimate_pos_weight(train_data)
print("Estimated pos_weight:", pw)

bce = BCEWithLogitsLoss(reduction="mean", pos_weight=torch.tensor([5.0], device=device))
dice = DiceLoss(sigmoid=True, smooth_nr=1e-6, smooth_dr=1e-6)

def combined_loss(pred, target):
    return 0.3 * bce(pred, target) + 0.7 * dice(pred, target)

optimizer = Adam(params=model.parameters(), lr=1e-4, weight_decay=1e-4)

min_valid_loss = math.inf

best_dice_all = -1.0
best_thr_last = None

for epoch in range(30):
    show_example = True  # tylko raz na epokę
    train_loss = 0.0
    val_dice_fg_sum = 0.0
    val_dice_all_sum = 0.0
    val_all = 0
    val_fg = 0
    infer_time_sum_ms = 0.0
    infer_time_cnt = 0
    train_examples = 0
    val_examples = 0

    thr_list = [0.3, 0.4, 0.5, 0.6, 0.7]
    dice_thr_sum = {thr: 0.0 for thr in thr_list}
    dice_thr_cnt = {thr: 0   for thr in thr_list}

    model.train()

    for data in train_dataloader:
        image, ground_truth = data['image'], data['mask']
        image = image.unsqueeze(1).float().to(device)        # (B,1,H,W,D)
        ground_truth = ground_truth.unsqueeze(1).float().to(device)
        B, C, H, W, D = image.shape

        D = image.size(-1)
        pos_idx = [i for i in range(D) if (ground_truth[..., i] > 0).any()]
        neg_idx = [i for i in range(D) if (ground_truth[..., i] == 0).all()]
        k = min(len(neg_idx), len(pos_idx))
        sampled = pos_idx + (random.sample(neg_idx, k=k) if k > 0 else [])
        if not sampled:  # w razie pustych masek w całym patchu
            sampled = list(range(D))

        idx     = torch.tensor(sampled, device=device)
        x5d     = image.permute(0,4,1,2,3)[:, idx]
        imgs2d  = x5d.reshape(-1, 1, H, W)
        y5d     = ground_truth.permute(0,4,1,2,3)[:, idx]
        gts2d   = y5d.reshape(-1, 1, H, W)

        optimizer.zero_grad()
        target = model(imgs2d)
        loss   = combined_loss(target, gts2d)
        loss.backward()
        optimizer.step()

        train_loss     += loss.item()
        train_examples += imgs2d.size(0)

    valid_loss = 0.0
    model.eval()
    with torch.inference_mode():
        mean_prob = 0.0
        mean_prob_cnt = 0
        precision_sum = 0.0; recall_sum = 0.0; val_batches = 0
        for val_batch_idx, data in enumerate(val_dataloader):
            image, ground_truth = data['image'], data['mask']
            image = image.unsqueeze(1).float().to(device)
            ground_truth = ground_truth.unsqueeze(1).float().to(device)

            B, C, H, W, D = image.shape
            idx    = torch.arange(D, device=device)
            x5d    = image.permute(0,4,1,2,3)[:, idx]
            imgs2d = x5d.reshape(-1, C, H, W)
            y5d    = ground_truth.permute(0,4,1,2,3)[:, idx]
            gts2d  = y5d.reshape(-1, C, H, W)

            t0 = time.perf_counter()
            target = model(imgs2d)
            if device.type == "cuda": torch.cuda.synchronize()
            elif device.type == "mps" and hasattr(torch, "mps"): torch.mps.synchronize()
            infer_time_ms = (time.perf_counter() - t0) * 1000.0
            infer_time_sum_ms += infer_time_ms; infer_time_cnt += 1

            prob = torch.sigmoid(target)

            if epoch % 5 == 0 and val_batch_idx == 0:
                slice_idx = D // 2   # środkowy slice w osi Z

                # slice_idx odnosi się do osi Z, więc bierzemy imgs2d[slice_idx]
                img_np  = imgs2d[slice_idx, 0].detach().cpu().numpy()
                prob_np = prob[slice_idx, 0].detach().cpu().numpy()
                gt_np   = gts2d[slice_idx, 0].detach().cpu().numpy()

                fig, axes = plt.subplots(1, 3, figsize=(9, 3))
                axes[0].imshow(img_np, cmap="gray")
                axes[0].set_title("input")
                axes[0].axis("off")

                axes[1].imshow(gt_np, cmap="gray")
                axes[1].set_title("gt")
                axes[1].axis("off")

                axes[2].imshow(prob_np, cmap="gray")
                axes[2].set_title(f"pred (ep {epoch})")
                axes[2].axis("off")

                fig.tight_layout()
                fig.savefig(f"debug_slices/epoch{epoch:03d}_slice{slice_idx:03d}.png", dpi=150)
                plt.close(fig)

            mean_prob += prob.mean().item()
            mean_prob_cnt += 1

            # Dice_all (zakładając że soft_dice liczy per-sample i uśrednia)
            val_dice_all_sum += soft_dice(prob, gts2d).item()
            val_all += 1

            # Dice_fg_only
            has_fg = (gts2d.sum(dim=(1,2,3)) > 0)
            if has_fg.any():
                val_dice_fg_sum += soft_dice(prob[has_fg], gts2d[has_fg]).item()
                val_fg += 1

            # sweep progów (wektorowo)
            for thr in thr_list:
                bin_pred = (prob > thr).float()
                inter = (bin_pred * gts2d).sum(dim=(1,2,3))
                den   = bin_pred.sum(dim=(1,2,3)) + gts2d.sum(dim=(1,2,3))
                dice_thr_sum[thr] += (2*inter / (den + 1e-8)).mean().item()
                dice_thr_cnt[thr] += 1

            # Prec/Rec @0.5 uśrednione po próbkach
            pred05 = (prob > 0.5).float()
            tp = ((pred05 == 1) & (gts2d == 1)).flatten(1).sum(1).float()
            fp = ((pred05 == 1) & (gts2d == 0)).flatten(1).sum(1).float()
            fn = ((pred05 == 0) & (gts2d == 1)).flatten(1).sum(1).float()
            precision_sum += (tp / (tp + fp + 1e-8)).mean().item()
            recall_sum    += (tp / (tp + fn + 1e-8)).mean().item()

            valid_loss  += combined_loss(target, gts2d).item()
            val_examples += imgs2d.size(0)
            val_batches += 1

    if epoch in [5, 8]:
        for g in optimizer.param_groups:
            g["lr"] *= 0.5

    precision_mean = precision_sum / max(val_batches, 1)
    recall_mean    = recall_sum    / max(val_batches, 1)
    infer_ms_mean  = infer_time_sum_ms / max(infer_time_cnt, 1)
    train_l = train_loss / train_examples
    val_l = valid_loss / val_examples
    dice_fg = val_dice_fg_sum / max(val_fg, 1)
    dice_all = val_dice_all_sum / max(val_all, 1)

    # === wybór najlepszego progu ===
    best_thr = None
    best_dice = -1
    if best_thr is None:
        best_thr  = 0.5
        best_dice = dice_all

    for thr in thr_list:
        if dice_thr_cnt[thr] > 0:
            d = dice_thr_sum[thr] / dice_thr_cnt[thr]
            if d > best_dice:
                best_dice = d
                best_thr = thr

    writer.add_scalar("Metrics/BestThr", best_thr, epoch)
    writer.add_scalar("Metrics/Dice_fg_bestThr", best_dice, epoch)

    writer.add_scalar("Loss/Train", train_loss / train_examples, epoch)
    writer.add_scalar("Loss/Validation", valid_loss / val_examples, epoch)
    writer.add_scalar("Metrics/Dice_fg",  val_dice_fg_sum  / max(val_fg, 1),  epoch)
    writer.add_scalar("Metrics/Dice_all", val_dice_all_sum / max(val_all, 1), epoch)
    writer.add_scalar("Metrics/Precision", precision_mean, epoch)
    writer.add_scalar("Metrics/Recall",    recall_mean,    epoch)
    writer.add_scalar("Time/Infer_ms_mean", infer_ms_mean, epoch)

    mp = mean_prob / max(mean_prob_cnt, 1)
    print(f"mean_prob_val = {mp:.3f}")
    print(
        f"Epoch {epoch+1:02d} | "
        f"Train: {train_l:.3f} | Val: {val_l:.3f} | "
        f"Dice_fg: {dice_fg:.3f} | Dice_all: {dice_all:.3f} | "
        f"Prec: {precision_mean:.3f} | Rec: {recall_mean:.3f} | "
        f"Infer(ms): {infer_ms_mean:.1f}"
        f" | BestThr: {best_thr} | Dice@Best: {best_dice:.3f}"
    )

    if epoch == 0 or dice_all > best_dice_all:
        best_dice_all = dice_all
        torch.save(model.state_dict(), "checkpoints/best_dice.pth")


writer.flush()
writer.close()

if best_thr_last is not None:
    with open("best_thr.txt", "w") as f:
        f.write(f"{best_thr_last:.3f}\n")
