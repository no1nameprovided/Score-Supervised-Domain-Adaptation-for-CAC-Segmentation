"""
plot_diffagatston_paper.py
--------------------------
Render a publication-quality version of the DiffAgatston-vs-standard-Agatston
scatter for the paper (Fig. 1). Reuses the same computation as
verify_diffagatston.py but with clean, paper-ready styling:
  * larger fonts, thin spines, light grid
  * log-log axes (Agatston spans 0..>3000, so linear crowds small scores)
  * identity line + fitted line, with rho / r annotated in a box
  * saves a vector PDF (best for LaTeX) and a high-dpi PNG

Run (point dirs at your COCA HeartCrop data):
    python plot_diffagatston_paper.py \
      --img_dir .../Dataset018_STF_HeartCrop/imagesTr \
      --lbl_dir .../Dataset018_STF_HeartCrop/labelsTr --n 90
"""
import os, glob, argparse
import numpy as np
import nibabel as nib
import torch
from scipy import ndimage
from scipy.stats import pearsonr, spearmanr

HU_THRESHOLD, MIN_AREA_MM2 = 130.0, 1.0


def dwt(p): return 4 if p>=400 else 3 if p>=300 else 2 if p>=200 else 1 if p>=130 else 0

def standard_agatston(ct, mask, spacing):
    sx, sy, sz = spacing; pa, sw = sx*sy, sz/3.0; tot = 0.0
    for s in range(ct.shape[2]):
        m = (mask[:,:,s] > 0.5) & (ct[:,:,s] >= HU_THRESHOLD)
        if not m.any(): continue
        lab, n = ndimage.label(m)
        for c in range(1, n+1):
            cm = lab==c; ar = cm.sum()*pa
            if ar < MIN_AREA_MM2: continue
            tot += ar*dwt(ct[:,:,s][cm].max())*sw
    return float(tot)

def diff_agatston(ct, prob, spacing, tau_g=0.1, tau_w=0.08):
    sx, sy, sz = spacing; pa, sw = sx*sy, sz/3.0
    g = torch.sigmoid(tau_g*(ct-130.0))
    w = (1.0 + torch.sigmoid(tau_w*(ct-200.0)) + torch.sigmoid(tau_w*(ct-300.0))
         + torch.sigmoid(tau_w*(ct-400.0)))
    return float((prob*g*w*pa*sw).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--lbl_dir", required=True)
    ap.add_argument("--n", type=int, default=90)
    ap.add_argument("--out", default="fig_diffagatston")
    args = ap.parse_args()

    lbls = sorted(glob.glob(os.path.join(args.lbl_dir, "*.nii.gz")))[:args.n]
    std, diff = [], []
    for lp in lbls:
        case = os.path.basename(lp).replace(".nii.gz", "")
        ip = os.path.join(args.img_dir, case + "_0000.nii.gz")
        if not os.path.exists(ip): continue
        nii = nib.load(ip); ct = nii.get_fdata().astype(np.float32)
        mask = nib.load(lp).get_fdata().astype(np.float32)
        spc = nii.header.get_zooms()[:3]
        std.append(standard_agatston(ct, mask, spc))
        diff.append(diff_agatston(torch.from_numpy(ct), torch.from_numpy(mask), spc))
    std, diff = np.array(std), np.array(diff)
    a = (std*diff).sum()/(diff**2).sum()
    diff_cal = a*diff
    rho = spearmanr(std, diff_cal)[0]
    r = pearsonr(np.log1p(std), np.log1p(diff_cal))[0]
    print(f"n={len(std)}  alpha={a:.3f}  Spearman={rho:.4f}  Pearson(log)={r:.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 13, "font.family": "serif",
        "axes.linewidth": 0.8, "axes.edgecolor": "#444444",
        "xtick.direction": "out", "ytick.direction": "out",
    })

    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    # clip zeros to a small floor so they appear on log axes
    floor = 1.0
    x = np.clip(std, floor, None); y = np.clip(diff_cal, floor, None)
    lim_lo, lim_hi = 0.8, max(x.max(), y.max())*1.4

    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], ls="--", lw=1.0,
            color="#999999", zorder=1, label="identity")
    ax.scatter(x, y, s=26, c="#2c6fbb", alpha=0.7, edgecolors="white",
               linewidths=0.4, zorder=3)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim_lo, lim_hi); ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("Standard Agatston score", fontsize=13)
    ax.set_ylabel("DiffAgatston (calibrated)", fontsize=13)
    ax.set_aspect("equal")
    ax.grid(True, which="major", ls=":", lw=0.5, color="#dddddd", zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    txt = f"Spearman $\\rho$ = {rho:.3f}\nPearson (log) = {r:.3f}\n$n$ = {len(std)}"
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=11.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", lw=0.8))

    fig.tight_layout()
    fig.savefig(args.out + ".pdf", bbox_inches="tight")
    fig.savefig(args.out + ".png", dpi=300, bbox_inches="tight")
    print("saved", args.out + ".pdf", "and", args.out + ".png")


if __name__ == "__main__":
    main()