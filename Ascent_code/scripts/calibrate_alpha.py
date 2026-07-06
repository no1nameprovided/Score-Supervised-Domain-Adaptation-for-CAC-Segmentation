"""
scripts/calibrate_alpha.py
--------------------------
1. Load the Stage-1 student.
2. On VAL cases: predict soft calcium masks, run DiffAgatston (raw), and
   least-squares fit alpha against the standard Agatston computed from GT masks.
   -> this alpha is correct for SOFT probabilities (Stage-2 score loss uses it).
3. On TEST cases (locked, untouched): report how well
   "predicted soft mask -> DiffAgatston score" agrees with the standard score.
   -> a clean paper number proving the full prediction chain works.

Run from project root:
    conda activate ascent
    python scripts/calibrate_alpha.py
"""

import os
import sys
import json
import numpy as np
import torch
import nibabel as nib
from scipy import ndimage
from scipy.stats import pearsonr, spearmanr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.segresnet import build_student
from diffagatston.layer import DiffAgatston
from data.datasets import _window_normalize, HU_CLIP_MIN  # reuse same windowing

CKPT = "checkpoints/stage1/stage1_best.pth"
SPLIT = "data/splits/coca_split.json"
HU_THRESHOLD = 130.0
MIN_AREA_MM2 = 1.0


# ---- standard (non-diff) Agatston from a binary mask, as the target ----
def density_weight_hard(peak):
    if peak >= 400: return 4
    if peak >= 300: return 3
    if peak >= 200: return 2
    if peak >= 130: return 1
    return 0


def standard_agatston(ct, mask, spacing):
    sx, sy, sz = spacing
    pa, sw = sx * sy, sz / 3.0
    total = 0.0
    for s in range(ct.shape[2]):
        m = (mask[:, :, s] > 0.5) & (ct[:, :, s] >= HU_THRESHOLD)
        if not m.any():
            continue
        lab, n = ndimage.label(m)
        for c in range(1, n + 1):
            cm = lab == c
            area = cm.sum() * pa
            if area < MIN_AREA_MM2:
                continue
            total += area * density_weight_hard(ct[:, :, s][cm].max()) * sw
    return float(total)


@torch.no_grad()
def predict_volume_prob(model, ct, n_adj=1, device="cuda"):
    """run the 2.5D student over every slice -> soft prob volume (H,W,S)."""
    H, W, S = ct.shape
    prob = np.zeros_like(ct, dtype=np.float32)
    for s in range(S):
        chans = []
        for off in range(-n_adj, n_adj + 1):
            idx = min(max(s + off, 0), S - 1)
            chans.append(_window_normalize(ct[:, :, idx]))
        x = torch.from_numpy(np.stack(chans)[None]).to(device)   # (1,C,H,W)
        # pad to /16
        ph = (16 - H % 16) % 16
        pw = (16 - W % 16) % 16
        x = torch.nn.functional.pad(x, (0, pw, 0, ph))
        logit = model(x)[..., :H, :W]
        prob[:, :, s] = torch.sigmoid(logit)[0, 0].cpu().numpy()
    return prob


def run_fold(model, cfg, fold, diff_layer, device):
    """return arrays: std_scores, raw_diff_scores (alpha=1)."""
    with open(SPLIT) as f:
        sp = json.load(f)
    img_dir, lbl_dir = sp["img_dir"], sp["lbl_dir"]
    std_list, raw_list = [], []
    diff_layer.set_alpha(1.0)
    for case in sp[fold]:
        nii = nib.load(os.path.join(img_dir, case + "_0000.nii.gz"))
        ct = nii.get_fdata().astype(np.float32)
        mask = nib.load(os.path.join(lbl_dir, case + ".nii.gz")).get_fdata().astype(np.float32)
        spacing = nii.header.get_zooms()[:3]
        pa = spacing[0] * spacing[1]

        s_std = standard_agatston(ct, mask, spacing)
        prob = predict_volume_prob(model, ct, cfg["n_adjacent"], device)
        pr_t = torch.from_numpy(prob).to(device)
        hu_t = torch.from_numpy(ct).to(device)
        s_raw = float(diff_layer(pr_t, hu_t, pixel_area=pa,
                                 reduce_dims=tuple(range(pr_t.dim()))))
        std_list.append(s_std); raw_list.append(s_raw)
        print(f"  [{fold}] {case}: std={s_std:8.1f}  raw_diff={s_raw:8.1f}")
    return np.array(std_list), np.array(raw_list)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(CKPT, map_location=device)
    cfg = ckpt["cfg"]
    model = build_student(in_channels=2 * cfg["n_adjacent"] + 1,
                          init_filters=cfg["init_filters"],
                          dropout_prob=cfg["dropout"]).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    print(f"loaded stage1 (val_dice={ckpt['val_dice']:.4f})")

    diff = DiffAgatston().to(device)

    # ---- fit alpha on VAL ----
    print("\n== calibrating alpha on VAL (predicted soft masks) ==")
    std_v, raw_v = run_fold(model, cfg, "val", diff, device)
    alpha = float((std_v * raw_v).sum() / (raw_v ** 2).sum())
    print(f"\n>>> calibrated alpha (soft prob) = {alpha:.4f}")

    # ---- evaluate on TEST with the fitted alpha ----
    print("\n== evaluating predicted mask -> score on TEST ==")
    std_t, raw_t = run_fold(model, cfg, "test", diff, device)
    diff_cal = alpha * raw_t
    sp_rho = spearmanr(std_t, diff_cal)[0]
    pe_log = pearsonr(np.log1p(std_t), np.log1p(diff_cal))[0]
    mae = float(np.mean(np.abs(diff_cal - std_t)))

    print("\n================ TEST RESULT ================")
    print(f" alpha used            : {alpha:.4f}")
    print(f" Spearman rho          : {sp_rho:.4f}")
    print(f" Pearson r (log space) : {pe_log:.4f}")
    print(f" MAE (score)           : {mae:.1f}")
    print("=============================================")

    # save alpha for Stage 2
    out = {"alpha_soft": alpha, "tau_g": diff.tau_g, "tau_w": diff.tau_w,
           "test_spearman": sp_rho, "test_pearson_log": pe_log, "test_mae": mae}
    with open("checkpoints/stage1/diffagatston_calib.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved -> checkpoints/stage1/diffagatston_calib.json")

    # scatter
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 5))
        plt.scatter(std_t, diff_cal, s=20, alpha=0.7)
        lim = max(std_t.max(), diff_cal.max()) * 1.05
        plt.plot([0, lim], [0, lim], "r--", lw=1)
        plt.xlabel("Standard Agatston (GT mask)")
        plt.ylabel("DiffAgatston (predicted soft mask)")
        plt.title(f"TEST  Spearman={sp_rho:.3f}  Pearson(log)={pe_log:.3f}")
        plt.tight_layout(); plt.savefig("pred_mask_to_score_test.png", dpi=130)
        print("saved -> pred_mask_to_score_test.png")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()