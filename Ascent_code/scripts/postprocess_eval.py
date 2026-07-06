"""
scripts/postprocess_eval.py  (v2: reads .nrrd heart masks)
----------------------------------------------------------
Apply clinically-motivated post-processing to Stage-1 predictions and measure
how much predicted-mask -> Agatston-score correlation recovers from 0.72.

Cumulative ablation:
  raw   no post-proc, no min-area  (= the 0.72 baseline)
  +C    HU gate (>=130 enforced by the scorer)
  +C+A  also keep only voxels inside the heart mask
  +C+A+B also drop per-slice lesions < MIN_AREA_MM2 (clinical Agatston rule)

Heart masks are .nrrd in heartPredTr/, named <case>.nrrd, same shape/affine
as the CT (verified). Read with pynrrd.

Run:
    python scripts/postprocess_eval.py
    python scripts/postprocess_eval.py --no_heart    # C+B only
"""

import os, sys, json, argparse
import numpy as np
import torch
import nibabel as nib
import nrrd
from scipy import ndimage
from scipy.stats import spearmanr, pearsonr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.segresnet import build_student
from data.datasets import _window_normalize

CKPT = "checkpoints/stage1/stage1_best.pth"
SPLIT = "data/splits/coca_split.json"

# heart masks (.nrrd), named <case>.nrrd
HEART_DIR = ("/media/yuankai/3.5T/NN-Unet/nnUNet-master/DATASET/"
             "nnUNet_raw_data/Dataset018_STF_HeartCrop/heartPredTr")
HEART_EXT = ".nrrd"

HU_THRESHOLD = 130.0
MIN_AREA_MM2 = 1.0


def dw(p):
    return 4 if p >= 400 else 3 if p >= 300 else 2 if p >= 200 else 1 if p >= 130 else 0


def load_heart(case):
    """read a .nrrd heart mask -> float32 array, or None if missing."""
    path = os.path.join(HEART_DIR, case + HEART_EXT)
    if not os.path.exists(path):
        return None
    data, _ = nrrd.read(path)
    return data.astype(np.float32)


def standard_agatston_from_binary(ct, mask, spacing, apply_minarea=True):
    sx, sy, sz = spacing; pa, sw = sx * sy, sz / 3.0
    total = 0.0
    for s in range(ct.shape[2]):
        m = (mask[:, :, s] > 0.5) & (ct[:, :, s] >= HU_THRESHOLD)
        if not m.any(): continue
        lab, n = ndimage.label(m)
        for c in range(1, n + 1):
            cm = lab == c; area = cm.sum() * pa
            if apply_minarea and area < MIN_AREA_MM2:
                continue
            total += area * dw(ct[:, :, s][cm].max()) * sw
    return float(total)


@torch.no_grad()
def predict_prob(model, ct, n_adj, device):
    H, W, S = ct.shape; prob = np.zeros_like(ct, np.float32)
    for s in range(S):
        chans = [_window_normalize(ct[:, :, min(max(s+o,0),S-1)]) for o in range(-n_adj,n_adj+1)]
        x = torch.from_numpy(np.stack(chans)[None]).to(device)
        ph,pw = (16-H%16)%16,(16-W%16)%16
        x = torch.nn.functional.pad(x,(0,pw,0,ph))
        prob[:, :, s] = torch.sigmoid(model(x)[...,:H,:W])[0,0].cpu().numpy()
    return prob


def min_area_filter(binmask, ct, spacing):
    sx, sy, _ = spacing; pa = sx * sy
    out = np.zeros_like(binmask)
    for s in range(binmask.shape[2]):
        m = (binmask[:, :, s] > 0.5) & (ct[:, :, s] >= HU_THRESHOLD)
        if not m.any(): continue
        lab, n = ndimage.label(m)
        keep = np.zeros_like(m)
        for c in range(1, n + 1):
            cm = lab == c
            if cm.sum() * pa >= MIN_AREA_MM2:
                keep |= cm
        out[:, :, s] = keep
    return out


def corr(std, pred):
    std, pred = np.array(std), np.array(pred)
    if len(std) < 3: return float("nan")
    return spearmanr(std, pred)[0]


def fit_corr_report(name, std, raw):
    raw = np.array(raw); s = np.array(std)
    a = (s*raw).sum()/(raw*raw).sum() if (raw*raw).sum()>0 else 1.0
    pred = a*raw
    sp_all = corr(s, pred)
    pe = pearsonr(np.log1p(s), np.log1p(pred))[0] if len(s)>=3 else float("nan")
    mae = float(np.mean(np.abs(pred - s)))
    print(f"\n=== {name}  (alpha={a:.3f}) ===")
    print(f"  overall Spearman={sp_all:.3f}  Pearson(log)={pe:.3f}  MAE={mae:.1f}")
    for lo,hi in [(0,100),(100,400),(400,1000),(1000,1e9)]:
        band=[(s[i],pred[i]) for i in range(len(s)) if lo<=s[i]<hi]
        if len(band)>=3:
            print(f"    band {lo}-{hi if hi<1e9 else 'inf'} (n={len(band)}): "
                  f"Spearman={corr([x[0] for x in band],[x[1] for x in band]):.3f}")
    return sp_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_heart", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(CKPT, map_location=device, weights_only=False); cfg = ck["cfg"]
    model = build_student(in_channels=2*cfg["n_adjacent"]+1,
                          init_filters=cfg["init_filters"], dropout_prob=cfg["dropout"]).to(device)
    model.load_state_dict(ck["model"]); model.eval()

    sp = json.load(open(SPLIT)); img_dir, lbl_dir = sp["img_dir"], sp["lbl_dir"]

    std, raw, c_only, ca, cab = [], [], [], [], []
    n_missing_heart = 0
    for case in sp["test"]:
        nii = nib.load(os.path.join(img_dir, case+"_0000.nii.gz"))
        ct = nii.get_fdata().astype(np.float32)
        gt = nib.load(os.path.join(lbl_dir, case+".nii.gz")).get_fdata().astype(np.float32)
        spc = nii.header.get_zooms()[:3]

        std.append(standard_agatston_from_binary(ct, gt, spc, apply_minarea=True))

        prob = predict_prob(model, ct, cfg["n_adjacent"], device)
        pred = (prob > 0.5).astype(np.float32)

        raw.append(standard_agatston_from_binary(ct, pred, spc, apply_minarea=False))
        c_only.append(standard_agatston_from_binary(ct, pred, spc, apply_minarea=False))

        pred_ca = pred.copy()
        if not args.no_heart:
            heart = load_heart(case)
            if heart is not None and heart.shape == ct.shape:
                pred_ca = pred_ca * (heart > 0.5)
            else:
                n_missing_heart += 1
        ca.append(standard_agatston_from_binary(ct, pred_ca, spc, apply_minarea=False))

        pred_cab = min_area_filter(pred_ca, ct, spc)
        cab.append(standard_agatston_from_binary(ct, pred_cab, spc, apply_minarea=True))

    print("\n################ POST-PROCESS ABLATION (TEST) ################")
    fit_corr_report("raw (= 0.72 baseline)", std, raw)
    fit_corr_report("+C HU gate", std, c_only)
    if not args.no_heart:
        if n_missing_heart:
            print(f"\n(WARN: {n_missing_heart} cases missing/!=shape heart mask)")
        fit_corr_report("+C +A heart", std, ca)
    fit_corr_report("+C +A +B min-area", std, cab)
    print("\n(Large lift from +A+B => post-proc fixes most FP.")
    print(" Residual gap = stents / intracardiac non-coronary => Stage-2 target.)")


if __name__ == "__main__":
    main()