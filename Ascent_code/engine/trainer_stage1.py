"""
engine/trainer_stage1.py
------------------------
Stage 1: fully-supervised calcium segmentation on COCA.
Produces (a) the student's basic skill, (b) the paper's "fully-supervised
upper bound", (c) the EMA teacher's initialization for Stage 2.

Engineering issues handled here:
  * Per-case slices have DIFFERENT H,W  -> train: random-crop to a fixed size
    so a batch can stack; val: batch_size=1 so no cropping needed.
  * Class imbalance -> SegLoss (DiceFocal) + calcium-oversampled train index.
  * Dice computed on the validation set; best checkpoint saved by val Dice.
"""

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from monai.metrics import DiceMetric

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import COCASliceDataset
from models.segresnet import build_student
from engine.losses import SegLoss


# ---------------------------------------------------------------------
# random crop / pad a batch item to a fixed size (train only)
# ---------------------------------------------------------------------
def _rand_crop_pad(img, hu, lab, size):
    """img:(C,H,W) hu:(1,H,W) lab:(1,H,W) -> all cropped/padded to size x size."""
    C, H, W = img.shape
    th = tw = size
    # pad if smaller than target
    ph, pw = max(0, th - H), max(0, tw - W)
    if ph or pw:
        img = torch.nn.functional.pad(img, (0, pw, 0, ph))
        hu = torch.nn.functional.pad(hu, (0, pw, 0, ph), value=-1000.0)  # pad HU as air
        lab = torch.nn.functional.pad(lab, (0, pw, 0, ph))
        C, H, W = img.shape
    # random top-left
    top = random.randint(0, H - th)
    left = random.randint(0, W - tw)
    sl = (slice(top, top + th), slice(left, left + tw))
    return img[:, sl[0], sl[1]], hu[:, sl[0], sl[1]], lab[:, sl[0], sl[1]]


def collate_train(batch, crop=256):
    imgs, labs = [], []
    for b in batch:
        i, _, l = _rand_crop_pad(b["image"], b["hu"], b["label"], crop)
        imgs.append(i); labs.append(l)
    return {"image": torch.stack(imgs), "label": torch.stack(labs)}


def collate_val(batch):
    # batch_size=1 -> just add batch dim, keep native size
    b = batch[0]
    return {"image": b["image"][None], "label": b["label"][None]}

def _pad_to_multiple(x, mult=16):
    """pad last two dims (H,W) up to a multiple of `mult`. returns padded x + pads."""
    H, W = x.shape[-2], x.shape[-1]
    ph = (mult - H % mult) % mult
    pw = (mult - W % mult) % mult
    x = torch.nn.functional.pad(x, (0, pw, 0, ph))
    return x, ph, pw
# ---------------------------------------------------------------------
def train_stage1(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])

    train_ds = COCASliceDataset(cfg["split_json"], fold="train",
                                n_adjacent=cfg["n_adjacent"],
                                calcium_oversample=True)
    val_ds = COCASliceDataset(cfg["split_json"], fold="val",
                              n_adjacent=cfg["n_adjacent"],
                              calcium_oversample=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], drop_last=True,
        collate_fn=lambda b: collate_train(b, cfg["crop"]),
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=cfg["num_workers"], collate_fn=collate_val,
    )

    model = build_student(in_channels=2 * cfg["n_adjacent"] + 1,
                          init_filters=cfg["init_filters"],
                          dropout_prob=cfg["dropout"]).to(device)
    crit = SegLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    os.makedirs(cfg["ckpt_dir"], exist_ok=True)
    best_dice = -1.0

    for epoch in range(cfg["epochs"]):
        # ---- train ----
        model.train()
        running = 0.0
        for it, batch in enumerate(train_loader):
            img = batch["image"].to(device)
            lab = batch["label"].to(device)
            opt.zero_grad()
            logits = model(img)
            loss = crit(logits, lab)
            loss.backward()
            opt.step()
            running += loss.item()
            if it % cfg["log_every"] == 0:
                print(f"  e{epoch} it{it}/{len(train_loader)} loss {loss.item():.4f}")
        sched.step()

        # ---- validate ----
        model.eval()
        dice_metric.reset()
        with torch.no_grad():
            for batch in val_loader:

                img = batch["image"].to(device)
                lab = batch["label"].to(device)
                img_p, ph, pw = _pad_to_multiple(img, 16)         # pad to /16
                logits = model(img_p)
                # crop the padding back off so it matches the label size
                H, W = lab.shape[-2], lab.shape[-1]
                logits = logits[..., :H, :W]
                prob = torch.sigmoid(logits)
                pred = (prob > 0.5).float()







                dice_metric(y_pred=pred, y=lab)
        val_dice = dice_metric.aggregate().item()
        print(f"[epoch {epoch}] train_loss {running/len(train_loader):.4f} "
              f"| val_dice {val_dice:.4f}")

        # ---- save best ----
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_dice": val_dice, "cfg": cfg},
                       os.path.join(cfg["ckpt_dir"], "stage1_best.pth"))
            print(f"  -> saved best (dice {best_dice:.4f})")

    print(f"\nDONE. best val dice = {best_dice:.4f}")
    print("checkpoint:", os.path.join(cfg["ckpt_dir"], "stage1_best.pth"))