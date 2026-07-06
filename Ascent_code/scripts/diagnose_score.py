"""
scripts/diagnose_score.py
-------------------------
Why did predicted-mask -> score drop to Spearman 0.72 while Dice is 0.88?
This script diagnoses it on the TEST set by:
  1. computing the score three ways from the predicted mask:
       (a) soft probability (current approach)
       (b) binarized at 0.5
       (c) binarized at 0.5 AND HU>=130 gate hard-applied
  2. reporting Spearman / Pearson(log) for each, overall and split by
     score band (0-100 / 100-400 / 400-1000 / >1000),
  3. printing per-case GT-mask-voxels vs pred-mask-voxels for the highest
     GT-score cases, to see if big lesions are under-segmented.

Run:
    python scripts/diagnose_score.py
"""

import os, sys, json
import numpy as np
import torch
import nibabel as nib
from scipy import ndimage
from scipy.stats import spearmanr, pearsonr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.segresnet import build_student
from diffagatston.layer import DiffAgatston
from data.datasets import _window_normalize

CKPT = "checkpoints/stage1/stage1_best.pth"
SPLIT = "data/splits/coca_split.json"


def density_weight_hard(p):
    return 4 if p >= 400 else 3 if p >= 300 else 2 if p >= 200 else 1 if p >= 130 else 0


def standard_agatston(ct, mask, spacing):
    sx, sy, sz = spacing; pa, sw = sx * sy, sz / 3.0
    total = 0.0
    for s in range(ct.shape[2]):
        m = (mask[:, :, s] > 0.5) & (ct[:, :, s] >= 130)
        if not m.any(): continue
        lab, n = ndimage.label(m)
        for c in range(1, n + 1):
            cm = lab == c; area = cm.sum() * pa
            if area < 1.0: continue
            total += area * density_weight_hard(ct[:, :, s][cm].max()) * sw
    return float(total)


@torch.no_grad()
def predict_prob(model, ct, n_adj, device):
    H, W, S = ct.shape; prob = np.zeros_like(ct, np.float32)
    for s in range(S):
        chans = [ _window_normalize(ct[:, :, min(max(s+o,0),S-1)]) for o in range(-n_adj, n_adj+1) ]
        x = torch.from_numpy(np.stack(chans)[None]).to(device)
        ph, pw = (16-H%16)%16, (16-W%16)%16
        x = torch.nn.functional.pad(x,(0,pw,0,ph))
        prob[:, :, s] = torch.sigmoid(model(x)[...,:H,:W])[0,0].cpu().numpy()
    return prob


def corr(std, pred):
    std, pred = np.array(std), np.array(pred)
    if len(std) < 3: return (float("nan"), float("nan"))
    return spearmanr(std, pred)[0], pearsonr(np.log1p(std), np.log1p(pred))[0]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(CKPT, map_location=device); cfg = ck["cfg"]
    model = build_student(in_channels=2*cfg["n_adjacent"]+1,
                          init_filters=cfg["init_filters"], dropout_prob=cfg["dropout"]).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    diff = DiffAgatston(alpha=1.0).to(device)

    sp = json.load(open(SPLIT)); img_dir, lbl_dir = sp["img_dir"], sp["lbl_dir"]
    rows = []  # (case, std, soft, bin, bin_hu, gt_vox, pred_vox)
    for case in sp["test"]:
        nii = nib.load(os.path.join(img_dir, case+"_0000.nii.gz"))
        ct = nii.get_fdata().astype(np.float32)
        gt = nib.load(os.path.join(lbl_dir, case+".nii.gz")).get_fdata().astype(np.float32)
        spc = nii.header.get_zooms()[:3]; pa = spc[0]*spc[1]
        s_std = standard_agatston(ct, gt, spc)
        prob = predict_prob(model, ct, cfg["n_adjacent"], device)
        hu = torch.from_numpy(ct).to(device)
        rd = tuple(range(hu.dim()))
        soft = float(diff(torch.from_numpy(prob).to(device), hu, pixel_area=pa, reduce_dims=rd))
        binm = (prob > 0.5).astype(np.float32)
        b = float(diff(torch.from_numpy(binm).to(device), hu, pixel_area=pa, reduce_dims=rd))
        binhu = binm * (ct >= 130)
        bh = float(diff(torch.from_numpy(binhu).to(device), hu, pixel_area=pa, reduce_dims=rd))
        rows.append((case, s_std, soft, b, bh, int((gt>0.5).sum()), int(binm.sum())))

    std = [r[1] for r in rows]
    for name, idx in [("soft", 2), ("bin@0.5", 3), ("bin@0.5+HU", 4)]:
        # fit a per-variant alpha so scale is fair, then correlate
        raw = np.array([r[idx] for r in rows]); s = np.array(std)
        a = (s*raw).sum()/(raw*raw).sum() if (raw*raw).sum()>0 else 1.0
        sp_all, pe_all = corr(s, a*raw)
        print(f"\n=== variant: {name}  (alpha={a:.3f}) ===")
        print(f"  overall: Spearman={sp_all:.3f}  Pearson(log)={pe_all:.3f}")
        for lo, hi in [(0,100),(100,400),(400,1000),(1000,1e9)]:
            band = [(s[i], a*raw[i]) for i in range(len(s)) if lo<=s[i]<hi] 