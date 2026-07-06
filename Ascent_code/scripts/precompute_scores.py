"""
scripts/precompute_scores.py
----------------------------
Pre-compute the standard Agatston score for every COCA case from its GT mask,
and cache to data/splits/coca_scores.json. Stage-2a uses these as the weak
supervision signal (score-only, mask hidden during training).

Also bins each case into the 4 clinical risk categories for later reporting.

Run:
    python scripts/precompute_scores.py
"""

import os, sys, json
import numpy as np
import nibabel as nib
from scipy import ndimage

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SPLIT = "data/splits/coca_split.json"
OUT = "data/splits/coca_scores.json"
HU_THRESHOLD = 130.0
MIN_AREA_MM2 = 1.0


def dw(p):
    return 4 if p >= 400 else 3 if p >= 300 else 2 if p >= 200 else 1 if p >= 130 else 0


def standard_agatston(ct, mask, spacing):
    sx, sy, sz = spacing; pa, sw = sx * sy, sz / 3.0
    total = 0.0
    for s in range(ct.shape[2]):
        m = (mask[:, :, s] > 0.5) & (ct[:, :, s] >= HU_THRESHOLD)
        if not m.any():
            continue
        lab, n = ndimage.label(m)
        for c in range(1, n + 1):
            cm = lab == c; area = cm.sum() * pa
            if area < MIN_AREA_MM2:
                continue
            total += area * dw(ct[:, :, s][cm].max()) * sw
    return float(total)


def risk_category(score):
    if score == 0: return 0
    if score < 100: return 1
    if score < 400: return 2
    return 3


def main():
    sp = json.load(open(SPLIT))
    img_dir, lbl_dir = sp["img_dir"], sp["lbl_dir"]
    all_cases = sp["train"] + sp["val"] + sp["test"]

    scores, cats = {}, {0: 0, 1: 0, 2: 0, 3: 0}
    for i, case in enumerate(all_cases):
        nii = nib.load(os.path.join(img_dir, case + "_0000.nii.gz"))
        ct = nii.get_fdata().astype(np.float32)
        mask = nib.load(os.path.join(lbl_dir, case + ".nii.gz")).get_fdata().astype(np.float32)
        spacing = nii.header.get_zooms()[:3]
        s = standard_agatston(ct, mask, spacing)
        scores[case] = s
        cats[risk_category(s)] += 1
        if i % 50 == 0:
            print(f"  {i}/{len(all_cases)}  {case}: {s:.1f}")

    json.dump(scores, open(OUT, "w"), indent=2)
    print(f"\nsaved {len(scores)} scores -> {OUT}")
    print(f"risk categories: 0(=0): {cats[0]}  1(1-99): {cats[1]}  "
          f"2(100-399): {cats[2]}  3(>=400): {cats[3]}")


if __name__ == "__main__":
    main()