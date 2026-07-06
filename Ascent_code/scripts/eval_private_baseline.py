"""
scripts/eval_private_segresnet.py
---------------------------------
Zero-shot baseline of the Stage-1 SegResNet on the private (LHCH) cohort,
with resampling to the COCA spacing for a FAIR comparison.

Per case:
  private CT (.nrrd, ~0.75mm) --resample--> COCA spacing (0.38/0.38/3.0)
  -> SegResNet 2.5D prediction -> resample prob BACK to original spacing
  -> heart-mask constraint + HU gate + min-area
  -> Agatston (computed at ORIGINAL spacing, physically correct)
  -> compare to Excel 'Agatston Total'

This is the clean control the score-adapted SegResNet must beat (same net,
the only added ingredient later is score supervision).

Run:
    python scripts/eval_private_segresnet.py
"""
import os, sys, glob
import numpy as np
import pandas as pd
import nrrd
import torch
from scipy.ndimage import zoom
from scipy import ndimage
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import cohen_kappa_score, accuracy_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.segresnet import build_student
from data.datasets import _window_normalize

D = "/media/yuankai/3.5T/NN-Unet/nnUNet-master/DATASET1/LHCH_for_Dataset018_inference"
IMG_DIR = f"{D}/imagesTs"
HEART_DIR = f"{D}/heartMasksTs"
XLSX = f"{D}/GT.xlsx"
CKPT = "checkpoints/stage1/stage1_best.pth"

# COCA training spacing (target for resampling)
COCA_SPACING = (0.3828, 0.3828, 3.0)
HU_THRESHOLD = 130.0
MIN_AREA_MM2 = 1.0


def dw(p):
    return 4 if p >= 400 else 3 if p >= 300 else 2 if p >= 200 else 1 if p >= 130 else 0


def risk_cat(s):
    if s == 0: return 0
    if s < 100: return 1
    if s < 400: return 2
    return 3


def read_nrrd(path):
    data, hdr = nrrd.read(path)
    sp = (1.0, 1.0, 1.0)
    if "space directions" in hdr:
        sd = np.array(hdr["space directions"], dtype=float)
        sp = tuple(float(np.linalg.norm(sd[i])) for i in range(3))
    return data.astype(np.float32), sp


def agatston(ct, mask, spacing, min_area=True):
    sx, sy, sz = spacing; pa, sw = sx * sy, sz / 3.0
    total = 0.0
    for s in range(ct.shape[2]):
        m = (mask[:, :, s] > 0.5) & (ct[:, :, s] >= HU_THRESHOLD)
        if not m.any(): continue
        lab, n = ndimage.label(m)
        for c in range(1, n + 1):
            cm = lab == c; area = cm.sum() * pa
            if min_area and area < MIN_AREA_MM2: continue
            total += area * dw(ct[:, :, s][cm].max()) * sw
    return float(total)


def pad16(x):
    H, W = x.shape[-2], x.shape[-1]
    ph, pw = (16 - H % 16) % 16, (16 - W % 16) % 16
    return torch.nn.functional.pad(x, (0, pw, 0, ph)), ph, pw


@torch.no_grad()
def predict_resampled(model, ct_orig, spacing, n_adj, device):
    """resample CT to COCA spacing, predict, resample prob back to original."""
    # zoom factors to reach COCA spacing
    fx = spacing[0] / COCA_SPACING[0]
    fy = spacing[1] / COCA_SPACING[1]
    fz = spacing[2] / COCA_SPACING[2]
    ct_rs = zoom(ct_orig, (fx, fy, fz), order=1)          # linear for image
    H, W, S = ct_rs.shape

    prob_rs = np.zeros_like(ct_rs, dtype=np.float32)
    for s in range(S):
        chans = [_window_normalize(ct_rs[:, :, min(max(s + o, 0), S - 1)])
                 for o in range(-n_adj, n_adj + 1)]
        x = torch.from_numpy(np.stack(chans)[None]).to(device)
        xp, ph, pw = pad16(x)
        logit = model(xp)[..., :H, :W]
        prob_rs[:, :, s] = torch.sigmoid(logit)[0, 0].cpu().numpy()

    # resample prob back to original shape
    inv = (ct_orig.shape[0] / prob_rs.shape[0],
           ct_orig.shape[1] / prob_rs.shape[1],
           ct_orig.shape[2] / prob_rs.shape[2])
    prob_back = zoom(prob_rs, inv, order=1)
    # guard exact shape
    prob_back = prob_back[:ct_orig.shape[0], :ct_orig.shape[1], :ct_orig.shape[2]]
    if prob_back.shape != ct_orig.shape:
        pb = np.zeros_like(ct_orig, dtype=np.float32)
        sl = tuple(slice(0, min(prob_back.shape[i], ct_orig.shape[i])) for i in range(3))
        pb[sl] = prob_back[sl]
        prob_back = pb
    return prob_back


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(CKPT, map_location=device, weights_only=False); cfg = ck["cfg"]
    model = build_student(in_channels=2 * cfg["n_adjacent"] + 1,
                          init_filters=cfg["init_filters"],
                          dropout_prob=cfg["dropout"]).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    print(f"loaded stage1 SegResNet (val_dice {ck['val_dice']:.4f})")

    df = pd.read_excel(XLSX)
    gt_map = {str(r["RBQ_No"]).strip(): float(r["Agatston Total"])
              for _, r in df.iterrows() if pd.notna(r["Agatston Total"])}

    imgs = sorted(glob.glob(f"{IMG_DIR}/RBQ*_0000.nrrd"))
    std_gt, pred_score, cat_gt = [], [], []
    used = 0
    for ip in imgs:
        case = os.path.basename(ip).replace("_0000.nrrd", "")
        rbq = case.split("_")[0]
        if rbq not in gt_map:
            continue
        hp = os.path.join(HEART_DIR, case + ".nrrd")
        if not os.path.exists(hp):
            continue
        ct, spc = read_nrrd(ip)
        heart, _ = read_nrrd(hp)
        if heart.shape != ct.shape:
            print(f"skip {case}: heart shape {heart.shape} != ct {ct.shape}")
            continue

        prob = predict_resampled(model, ct, spc, cfg["n_adjacent"], device)
        pred_pp = (prob > 0.5).astype(np.float32) * (heart > 0.5)
        s_pred = agatston(ct, pred_pp, spc, min_area=True)

        std_gt.append(gt_map[rbq]); pred_score.append(s_pred)
        cat_gt.append(risk_cat(gt_map[rbq]))
        used += 1
        if used % 20 == 0:
            print(f"  {used} cases done...")

    std_gt = np.array(std_gt); pred_score = np.array(pred_score)
    a = (std_gt * pred_score).sum() / (pred_score ** 2).sum() if (pred_score ** 2).sum() > 0 else 1.0
    pred_cal = a * pred_score
    cat_pred = [risk_cat(x) for x in pred_cal]

    sp_rho = spearmanr(std_gt, pred_score)[0]
    pe_log = pearsonr(np.log1p(std_gt), np.log1p(pred_cal))[0]
    mae = float(np.mean(np.abs(pred_cal - std_gt)))
    acc = accuracy_score(cat_gt, cat_pred)
    kappa = cohen_kappa_score(cat_gt, cat_pred, weights="linear")

    print(f"\n######## PRIVATE BASELINE (SegResNet zero-shot, resampled) ########")
    print(f" cases used            : {used}")
    print(f" calibration alpha     : {a:.4f}")
    print(f" Spearman rho          : {sp_rho:.4f}")
    print(f" Pearson r (log)       : {pe_log:.4f}")
    print(f" MAE (score)           : {mae:.1f}")
    print(f" 4-class risk accuracy : {acc:.4f}")
    print(f" weighted kappa        : {kappa:.4f}")
    print("###################################################################")
    from collections import Counter
    print("GT class dist  :", dict(sorted(Counter(cat_gt).items())))
    print("Pred class dist:", dict(sorted(Counter(cat_pred).items())))


if __name__ == "__main__":
    main()