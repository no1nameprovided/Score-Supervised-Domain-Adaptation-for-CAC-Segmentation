"""
scripts/save_predictions.py
---------------------------
Save Stage-1 predictions as nii.gz for visual inspection in ITK-SNAP / Slicer.

For each requested case it writes (into pred_vis/):
  <case>_img.nii.gz    the CT (copied, so you can load everything together)
  <case>_gt.nii.gz     ground-truth calcium mask
  <case>_prob.nii.gz   predicted probability (0..1, load as image/heatmap)
  <case>_pred.nii.gz   predicted binary mask at 0.5 (load as segmentation)

All share the SAME affine/header as the CT, so they overlay correctly.

Open in ITK-SNAP: load <case>_img as main image, then load <case>_gt and
<case>_pred as 'Segmentation' (one at a time, or as additional overlays) to
compare. The false positives (e.g. aorta) will be obvious.

Run:
    python scripts/save_predictions.py
"""

import os, sys, json
import numpy as np
import torch
import nibabel as nib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.segresnet import build_student
from data.datasets import _window_normalize

CKPT = "checkpoints/stage1/stage1_best.pth"
SPLIT = "data/splits/coca_split.json"
OUT_DIR = "pred_vis"

# cases to inspect: the worst over-predictions + a couple of normal/under ones
CASES = ["img_0090", "img_0312", "img_0190", "img_0022",   # over-predicted
         "img_0173", "img_0342"]                            # under / normal


@torch.no_grad()
def predict_prob(model, ct, n_adj, device):
    H, W, S = ct.shape
    prob = np.zeros_like(ct, np.float32)
    for s in range(S):
        chans = [_window_normalize(ct[:, :, min(max(s + o, 0), S - 1)])
                 for o in range(-n_adj, n_adj + 1)]
        x = torch.from_numpy(np.stack(chans)[None]).to(device)
        ph, pw = (16 - H % 16) % 16, (16 - W % 16) % 16
        x = torch.nn.functional.pad(x, (0, pw, 0, ph))
        prob[:, :, s] = torch.sigmoid(model(x)[..., :H, :W])[0, 0].cpu().numpy()
    return prob


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(CKPT, map_location=device); cfg = ck["cfg"]
    model = build_student(in_channels=2 * cfg["n_adjacent"] + 1,
                          init_filters=cfg["init_filters"],
                          dropout_prob=cfg["dropout"]).to(device)
    model.load_state_dict(ck["model"]); model.eval()

    sp = json.load(open(SPLIT)); img_dir, lbl_dir = sp["img_dir"], sp["lbl_dir"]
    os.makedirs(OUT_DIR, exist_ok=True)

    for case in CASES:
        if case not in sp["test"]:
            print(f"WARN {case} not in test split, skipping"); continue
        nii = nib.load(os.path.join(img_dir, case + "_0000.nii.gz"))
        ct = nii.get_fdata().astype(np.float32)
        gt = nib.load(os.path.join(lbl_dir, case + ".nii.gz")).get_fdata().astype(np.float32)
        affine, header = nii.affine, nii.header

        prob = predict_prob(model, ct, cfg["n_adjacent"], device)
        pred = (prob > 0.5).astype(np.uint8)

        nib.save(nib.Nifti1Image(ct, affine, header),
                 os.path.join(OUT_DIR, f"{case}_img.nii.gz"))
        nib.save(nib.Nifti1Image(gt.astype(np.uint8), affine, header),
                 os.path.join(OUT_DIR, f"{case}_gt.nii.gz"))
        nib.save(nib.Nifti1Image(prob, affine, header),
                 os.path.join(OUT_DIR, f"{case}_prob.nii.gz"))
        nib.save(nib.Nifti1Image(pred, affine, header),
                 os.path.join(OUT_DIR, f"{case}_pred.nii.gz"))

        # quick textual summary: where are the false positives in HU terms?
        fp = (pred > 0) & (gt < 0.5)        # predicted calcium, not in GT
        fn = (pred == 0) & (gt > 0.5)        # missed real calcium
        tp = (pred > 0) & (gt > 0.5)
        fp_hu = ct[fp]
        print(f"\n{case}: GTvox={int((gt>0.5).sum())}  PREDvox={int(pred.sum())}")
        print(f"  TP={int(tp.sum())}  FP={int(fp.sum())}  FN={int(fn.sum())}")
        if fp_hu.size:
            print(f"  FP HU: mean={fp_hu.mean():.0f}  "
                  f">=130:{(fp_hu>=130).mean():.2f}  >=200:{(fp_hu>=200).mean():.2f}")

    print(f"\nsaved to ./{OUT_DIR}/  -- open *_img + *_gt + *_pred in ITK-SNAP")


if __name__ == "__main__":
    main()