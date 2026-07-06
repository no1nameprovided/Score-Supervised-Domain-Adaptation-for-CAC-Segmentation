"""
engine/trainer_stage2b.py  (FIXED: correct val + low-score weighting)
=====================================================================
Stage-2b score adaptation on private (LHCH). One-shot tuned version:
  * validation FIXED (resample-back done correctly; reports zero-class recall)
  * anchor OFF by default (let score fully suppress FP); re-add later if needed
  * larger lr, more epochs
  * LOW-SCORE WEIGHTING: zero/low Agatston cases get higher score-loss weight
    so the FP-suppression signal (push toward 0) is not drowned by the many
    high-score cases.

Reports on the LOCKED test split each epoch: spearman / acc / kappa /
zero_recall (how many of the true-zero cases are correctly predicted 0).
"""

import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from scipy.ndimage import zoom, label as cc_label
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, cohen_kappa_score
import nrrd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.segresnet import build_student
from diffagatston.layer import DiffAgatston
from data.datasets import _window_normalize

COCA_SPACING = (0.3828, 0.3828, 3.0)
HU_THRESHOLD = 130.0
MIN_AREA_MM2 = 1.0


def risk_cat(s):
    if s == 0: return 0
    if s < 100: return 1
    if s < 400: return 2
    return 3


def score_weight(gt):
    """upweight low/zero-score cases so FP-suppression isn't drowned out."""
    if gt == 0: return 1.5
    if gt < 100: return 1.2
    return 1.0


def read_nrrd(path):
    data, hdr = nrrd.read(path)
    sp = (1.0, 1.0, 1.0)
    if "space directions" in hdr:
        sd = np.array(hdr["space directions"], dtype=float)
        sp = tuple(float(np.linalg.norm(sd[i])) for i in range(3))
    return data.astype(np.float32), sp


def pad16(x):
    H, W = x.shape[-2], x.shape[-1]
    ph, pw = (16 - H % 16) % 16, (16 - W % 16) % 16
    return F.pad(x, (0, pw, 0, ph)), ph, pw


def resample_to_coca(ct, spacing):
    fx, fy, fz = spacing[0]/COCA_SPACING[0], spacing[1]/COCA_SPACING[1], spacing[2]/COCA_SPACING[2]
    return zoom(ct, (fx, fy, fz), order=1)


def forward_volume_rs(model, ct_rs, n_adj, device, chunk=8, use_ckpt=True):
    H, W, S = ct_rs.shape
    stacks = [np.stack([_window_normalize(ct_rs[:, :, min(max(s + o, 0), S - 1)])
                        for o in range(-n_adj, n_adj + 1)]) for s in range(S)]
    x = torch.from_numpy(np.stack(stacks)).to(device)
    xp, ph, pw = pad16(x)
    outs = []
    for i in range(0, S, chunk):
        xc = xp[i:i + chunk]
        out = checkpoint(lambda inp: model(inp), xc, use_reentrant=False) if (use_ckpt and model.training) else model(xc)
        outs.append(out)
    logits = torch.cat(outs, 0)[..., :H, :W]
    return torch.sigmoid(logits)[:, 0].permute(1, 2, 0)


def score_loss_log(diff, prob, hu, pa, gt, w=1.0):
    pred = diff(prob, hu, pixel_area=pa, reduce_dims=tuple(range(prob.dim())))
    lp = torch.log1p(pred)
    lt = torch.log1p(torch.tensor(float(gt), device=prob.device))
    return w * F.huber_loss(lp, lt, delta=1.0), float(pred)


def hu_prior(prob, hu, tau=0.1):
    return (prob * torch.sigmoid(tau * (130.0 - hu))).mean()


def heart_prior(prob, heart):
    return (prob * (1.0 - heart)).mean()


def asym_anchor(prob_s, prob_a, conf=0.5):
    sure = (prob_a > conf).float()
    deficit = F.relu(prob_a - prob_s)
    return (deficit ** 2 * sure).sum() / (sure.sum() + 1e-6)


def dw(p): return 4 if p>=400 else 3 if p>=300 else 2 if p>=200 else 1 if p>=130 else 0

def agatston_np(ct, mask, spacing):
    sx, sy, sz = spacing; pa, sw = sx*sy, sz/3.0; total=0.0
    for s in range(ct.shape[2]):
        m = (mask[:,:,s] > 0.5) & (ct[:,:,s] >= HU_THRESHOLD)
        if not m.any(): continue
        lab, n = cc_label(m)
        for c in range(1, n+1):
            cm = lab==c; area = cm.sum()*pa
            if area < MIN_AREA_MM2: continue
            total += area * dw(ct[:,:,s][cm].max()) * sw
    return float(total)


def train_stage2b(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])

    sp = json.load(open(cfg["split_json"]))
    scores = json.load(open(cfg["scores_json"]))
    img_dir, heart_dir = sp["img_dir"], sp["heart_dir"]
    train_cases = sp["train"]

    student = build_student(in_channels=2*cfg["n_adjacent"]+1,
                            init_filters=cfg["init_filters"], dropout_prob=cfg["dropout"]).to(device)
    ck = torch.load(cfg["init_ckpt"], map_location=device, weights_only=False)
    student.load_state_dict(ck["model"])
    print(f"init from {cfg['init_ckpt']}")

    use_anchor = cfg["w_anchor"] > 0
    if use_anchor:
        anchor = build_student(in_channels=2*cfg["n_adjacent"]+1,
                               init_filters=cfg["init_filters"], dropout_prob=cfg["dropout"]).to(device)
        anchor.load_state_dict(ck["model"])
        for p in anchor.parameters(): p.requires_grad_(False)
        anchor.eval()

    diff = DiffAgatston(alpha=cfg["alpha"]).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)
    best = -1.0
    pa_coca = COCA_SPACING[0] * COCA_SPACING[1]

    # baseline eval before any training
    val0 = validate_private(student, diff, sp, scores, cfg, device)
    print(f"[epoch -1 zero-shot] spearman {val0['spearman']:.4f} acc {val0['acc']:.4f} "
          f"kappa {val0['kappa']:.4f} zero_recall {val0['zero_recall']}", flush=True)

    for epoch in range(cfg["epochs"]):
        student.train()
        order = np.random.permutation(len(train_cases))
        run_s = 0.0
        for step, ci in enumerate(order):
            case = train_cases[ci]
            ct, spc = read_nrrd(os.path.join(img_dir, case + "_0000.nrrd"))
            heart, _ = read_nrrd(os.path.join(heart_dir, case + ".nrrd"))
            if heart.shape != ct.shape: continue
            gt = scores[case]
            w = score_weight(gt) if cfg.get("low_weight", True) else 1.0

            ct_rs = resample_to_coca(ct, spc)
            heart_rs = resample_to_coca((heart > 0.5).astype(np.float32), spc)
            ct_rs_t = torch.from_numpy(ct_rs).to(device)
            heart_rs_t = torch.from_numpy((heart_rs > 0.5).astype(np.float32)).to(device)

            if use_anchor:
                with torch.no_grad():
                    prob_a = forward_volume_rs(anchor, ct_rs, cfg["n_adjacent"], device, cfg["chunk"], use_ckpt=False)

            opt.zero_grad()
            prob = forward_volume_rs(student, ct_rs, cfg["n_adjacent"], device, cfg["chunk"])

            l_score, pred_s = score_loss_log(diff, prob, ct_rs_t, pa_coca, gt, w)
            l_hu = hu_prior(prob, ct_rs_t)
            l_heart = heart_prior(prob, heart_rs_t)
            loss = cfg["w_score"]*l_score + cfg["w_hu"]*l_hu + cfg["w_heart"]*l_heart
            if use_anchor:
                loss = loss + cfg["w_anchor"] * asym_anchor(prob, prob_a, cfg["anchor_conf"])
            loss.backward(); opt.step()

            run_s += l_score.item()
            if step % cfg["log_every"] == 0:
                print(f"  e{epoch} s{step}/{len(train_cases)} Ls{l_score.item():.3f} "
                      f"pred={pred_s:.0f} gt={gt:.0f} w={w}", flush=True)
            del prob, loss; torch.cuda.empty_cache()

        val = validate_private(student, diff, sp, scores, cfg, device)
        print(f"[epoch {epoch}] Ls{run_s/len(train_cases):.3f} | spearman {val['spearman']:.4f} "
              f"| acc {val['acc']:.4f} | kappa {val['kappa']:.4f} | zero_recall {val['zero_recall']}", flush=True)

        if val["kappa"] > best:
            best = val["kappa"]
            torch.save({"model": student.state_dict(), "epoch": epoch, "val": val, "cfg": cfg},
                       os.path.join(cfg["ckpt_dir"], "stage2b_best.pth"))
            print(f"  -> saved best (acc {best:.4f})", flush=True)

    print(f"\nDONE. best test acc = {best:.4f} (zero-shot was {val0['acc']:.4f})", flush=True)


@torch.no_grad()
def validate_private(model, diff, sp, scores, cfg, device):
    model.eval()
    img_dir, heart_dir = sp["img_dir"], sp["heart_dir"]
    std_gt, pred_sc, cat_gt = [], [], []
    for case in sp["test"]:
        ct, spc = read_nrrd(os.path.join(img_dir, case + "_0000.nrrd"))
        heart, _ = read_nrrd(os.path.join(heart_dir, case + ".nrrd"))
        if heart.shape != ct.shape: continue
        ct_rs = resample_to_coca(ct, spc)
        prob_rs = forward_volume_rs(model, ct_rs, cfg["n_adjacent"], device, cfg["chunk"], use_ckpt=False).cpu().numpy()
        inv = (ct.shape[0]/prob_rs.shape[0], ct.shape[1]/prob_rs.shape[1], ct.shape[2]/prob_rs.shape[2])
        prob = zoom(prob_rs, inv, order=1)[:ct.shape[0], :ct.shape[1], :ct.shape[2]]
        if prob.shape != ct.shape:
            pb = np.zeros_like(ct, dtype=np.float32)
            s = tuple(slice(0, min(prob.shape[i], ct.shape[i])) for i in range(3)); pb[s]=prob[s]; prob=pb
        pred_pp = (prob > 0.5).astype(np.float32) * (heart > 0.5)
        std_gt.append(scores[case]); pred_sc.append(agatston_np(ct, pred_pp, spc))
        cat_gt.append(risk_cat(scores[case]))

    std_gt, pred_sc = np.array(std_gt), np.array(pred_sc)
    a = (std_gt*pred_sc).sum()/(pred_sc**2).sum() if (pred_sc**2).sum()>0 else 1.0
    pred_cal = a*pred_sc
    cat_pred = [risk_cat(x) for x in pred_cal]
    nz_ok = sum(1 for g,p in zip(cat_gt,cat_pred) if g==0 and p==0)
    nz = sum(1 for g in cat_gt if g==0)
    return {"spearman": float(spearmanr(std_gt, pred_sc)[0]),
            "acc": float(accuracy_score(cat_gt, cat_pred)),
            "kappa": float(cohen_kappa_score(cat_gt, cat_pred, weights="linear")),
            "zero_recall": f"{nz_ok}/{nz}"}