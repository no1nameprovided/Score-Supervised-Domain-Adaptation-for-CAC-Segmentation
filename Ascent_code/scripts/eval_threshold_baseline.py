"""
scripts/eval_threshold_baseline.py
----------------------------------
Classical (non-learning) baseline for the main comparison table.

No network at all: within the heart region, take all voxels >= 130 HU as
calcium candidates, group into connected components per slice, drop lesions
< 1 mm^2, and compute the Agatston score directly. This is the traditional
CAC pipeline and shows how much a pure intensity rule over-counts
high-attenuation non-coronary structures (aorta, sternum, valves).

Evaluated on the SAME locked test-20 split, with the SAME post-processing
and the SAME native-spacing Agatston as every other method, for fairness.

Run:
    python scripts/eval_threshold_baseline.py
"""
import os, sys, json
import numpy as np
import nrrd
from scipy.ndimage import label as cc_label
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import accuracy_score, cohen_kappa_score

D = "/media/yuankai/3.5T/NN-Unet/nnUNet-master/DATASET1/LHCH_for_Dataset018_inference"
IMG_DIR, HEART_DIR = f"{D}/imagesTs", f"{D}/heartMasksTs"
SPLIT = "data/splits/private_split.json"
SCORES = "data/splits/private_scores.json"
HU_THRESHOLD, MIN_AREA_MM2 = 130.0, 1.0


def dw(p): return 4 if p>=400 else 3 if p>=300 else 2 if p>=200 else 1 if p>=130 else 0
def risk_cat(s): return 0 if s==0 else 1 if s<100 else 2 if s<400 else 3


def read_nrrd(path):
    data, hdr = nrrd.read(path)
    sp = (1.,1.,1.)
    if "space directions" in hdr:
        sd = np.array(hdr["space directions"], float)
        sp = tuple(float(np.linalg.norm(sd[i])) for i in range(3))
    return data.astype(np.float32), sp


def agatston(ct, mask, spacing):
    """standard Agatston at native spacing, given a binary calcium mask."""
    sx, sy, sz = spacing; pa, sw = sx*sy, sz/3.0; tot = 0.0
    for s in range(ct.shape[2]):
        m = (mask[:,:,s] > 0.5) & (ct[:,:,s] >= HU_THRESHOLD)
        if not m.any(): continue
        lab, n = cc_label(m)
        for c in range(1, n+1):
            cm = lab == c; ar = cm.sum()*pa
            if ar < MIN_AREA_MM2: continue
            tot += ar * dw(ct[:,:,s][cm].max()) * sw
    return float(tot)


def metrics(std_gt, pred_sc):
    std_gt = np.array(std_gt); pred_sc = np.array(pred_sc)
    a = (std_gt*pred_sc).sum()/(pred_sc**2).sum() if (pred_sc**2).sum()>0 else 1.0
    pc = a*pred_sc
    cg = [risk_cat(x) for x in std_gt]; cp = [risk_cat(x) for x in pc]
    nz_ok = sum(1 for g,p in zip(cg,cp) if g==0 and p==0)
    nz = sum(1 for g in cg if g==0)
    return {
        "spearman": spearmanr(std_gt, pred_sc)[0],
        "pearson_log": pearsonr(np.log1p(std_gt), np.log1p(pc))[0],
        "mae": float(np.mean(np.abs(pc-std_gt))),
        "acc": accuracy_score(cg, cp),
        "kappa": cohen_kappa_score(cg, cp, weights="linear"),
        "zero_recall": f"{nz_ok}/{nz}",
        "alpha": a,
    }


def main():
    sp = json.load(open(SPLIT)); scores = json.load(open(SCORES))
    test = sp["test"]
    std_gt, pred_sc = [], []
    for case in test:
        ct, spc = read_nrrd(os.path.join(IMG_DIR, case + "_0000.nrrd"))
        heart, _ = read_nrrd(os.path.join(HEART_DIR, case + ".nrrd"))
        if heart.shape != ct.shape:
            print(f"skip {case}: shape mismatch"); continue
        # pure threshold inside heart region (no network)
        mask = (heart > 0.5).astype(np.float32)   # agatston() re-applies 130HU
        std_gt.append(scores[case]); pred_sc.append(agatston(ct, mask, spc))

    m = metrics(std_gt, pred_sc)
    print("\n######## THRESHOLD BASELINE (130HU + heart, no learning) ########")
    print(f" test cases            : {len(std_gt)}")
    print(f" calibration alpha     : {m['alpha']:.4f}")
    print(f" Spearman rho          : {m['spearman']:.4f}")
    print(f" Pearson r (log)       : {m['pearson_log']:.4f}")
    print(f" MAE (score)           : {m['mae']:.1f}")
    print(f" 4-class risk accuracy : {m['acc']:.4f}")
    print(f" weighted kappa        : {m['kappa']:.4f}")
    print(f" zero-recall           : {m['zero_recall']}")
    print("#################################################################")
    from collections import Counter
    cg = [risk_cat(x) for x in std_gt]
    cp = [risk_cat(x) for x in (m['alpha']*np.array(pred_sc))]
    print("GT class dist  :", dict(sorted(Counter(cg).items())))
    print("Pred class dist:", dict(sorted(Counter(cp).items())))
    print("\n--> Table row:  Threshold (130HU)  &  "
          f"{m['spearman']:.3f} & {m['acc']:.2f} & {m['kappa']:.2f} & {m['zero_recall']}")


if __name__ == "__main__":
    main()