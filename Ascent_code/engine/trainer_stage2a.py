"""
engine/trainer_stage2a.py  (frozen Stage-1 anchor, asymmetric, confidence-gated)
"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import nibabel as nib
import nrrd
from scipy.stats import spearmanr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.segresnet import build_student
from diffagatston.layer import DiffAgatston
from data.datasets import _window_normalize

HU_THRESHOLD = 130.0


def load_heart(heart_dir, case, shape):
    path = os.path.join(heart_dir, case + ".nrrd")
    if not os.path.exists(path):
        return None
    data, _ = nrrd.read(path)
    if data.shape != shape:
        return None
    return data.astype(np.float32)


def pad16(x):
    H, W = x.shape[-2], x.shape[-1]
    ph, pw = (16 - H % 16) % 16, (16 - W % 16) % 16
    return F.pad(x, (0, pw, 0, ph)), ph, pw


def forward_volume(model, ct_np, n_adj, device, chunk=8, use_ckpt=True):
    H, W, S = ct_np.shape
    stacks = []
    for s in range(S):
        chans = [_window_normalize(ct_np[:, :, min(max(s + o, 0), S - 1)])
                 for o in range(-n_adj, n_adj + 1)]
        stacks.append(np.stack(chans))
    x = torch.from_numpy(np.stack(stacks)).to(device)
    xp, ph, pw = pad16(x)
    outs = []
    for i in range(0, S, chunk):
        xc = xp[i:i + chunk]
        if use_ckpt and model.training:
            out = checkpoint(lambda inp: model(inp), xc, use_reentrant=False)
        else:
            out = model(xc)
        outs.append(out)
    logits = torch.cat(outs, 0)[..., :H, :W]
    prob = torch.sigmoid(logits)[:, 0]
    return prob.permute(1, 2, 0)


def score_loss_log(diff_layer, prob, hu, pixel_area, gt_score):
    pred = diff_layer(prob, hu, pixel_area=pixel_area,
                      reduce_dims=tuple(range(prob.dim())))
    lp = torch.log1p(pred)
    lt = torch.log1p(torch.tensor(float(gt_score), device=prob.device))
    return F.huber_loss(lp, lt, delta=1.0), float(pred)


def hu_prior(prob, hu, tau=0.1):
    return (prob * torch.sigmoid(tau * (130.0 - hu))).mean()


def heart_prior(prob, heart):
    return (prob * (1.0 - heart)).mean()


def tv_loss(prob):
    dh = (prob[1:, :, :] - prob[:-1, :, :]).abs().mean()
    dw_ = (prob[:, 1:, :] - prob[:, :-1, :]).abs().mean()
    return dh + dw_


def asymmetric_anchor(prob_student, prob_anchor, conf=0.5):
    """penalize ONLY where frozen anchor is confident calcium (prob>conf)
    AND student is erasing it (student<anchor). Adding calcium is free."""
    sure = (prob_anchor > conf).float()
    deficit = F.relu(prob_anchor - prob_student)
    return (deficit ** 2 * sure).sum() / (sure.sum() + 1e-6)


def train_stage2a(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])

    sp = json.load(open(cfg["split_json"]))
    img_dir, lbl_dir = sp["img_dir"], sp["lbl_dir"]
    scores = json.load(open(cfg["scores_json"]))
    train_cases = sp["train"]

    student = build_student(in_channels=2 * cfg["n_adjacent"] + 1,
                            init_filters=cfg["init_filters"],
                            dropout_prob=cfg["dropout"]).to(device)
    ck = torch.load(cfg["init_ckpt"], map_location=device, weights_only=False)
    student.load_state_dict(ck["model"])
    print(f"initialized from {cfg['init_ckpt']} (dice {ck['val_dice']:.4f})")

    anchor = build_student(in_channels=2 * cfg["n_adjacent"] + 1,
                           init_filters=cfg["init_filters"],
                           dropout_prob=cfg["dropout"]).to(device)
    anchor.load_state_dict(ck["model"])
    for p in anchor.parameters():
        p.requires_grad_(False)
    anchor.eval()

    diff = DiffAgatston(alpha=cfg["alpha"]).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=cfg["lr"], weight_decay=1e-5)

    os.makedirs(cfg["ckpt_dir"], exist_ok=True)
    best = -1.0

    for epoch in range(cfg["epochs"]):
        student.train()
        order = np.random.permutation(len(train_cases))
        run_score = run_anchor = 0.0
        for step, ci in enumerate(order):
            case = train_cases[ci]
            nii = nib.load(os.path.join(img_dir, case + "_0000.nii.gz"))
            ct = nii.get_fdata().astype(np.float32)
            spc = nii.header.get_zooms()[:3]; pa = spc[0] * spc[1]
            heart = load_heart(cfg["heart_dir"], case, ct.shape)
            if heart is None:
                continue
            gt_score = scores[case]

            ct_t = torch.from_numpy(ct).to(device)
            heart_t = torch.from_numpy((heart > 0.5).astype(np.float32)).to(device)

            with torch.no_grad():
                prob_a = forward_volume(anchor, ct, cfg["n_adjacent"], device,
                                        cfg["chunk"], use_ckpt=False)

            opt.zero_grad()
            prob = forward_volume(student, ct, cfg["n_adjacent"], device, cfg["chunk"])

            l_score, pred_s = score_loss_log(diff, prob, ct_t, pa, gt_score)
            l_anchor = asymmetric_anchor(prob, prob_a, cfg["anchor_conf"])
            l_hu = hu_prior(prob, ct_t)
            l_heart = heart_prior(prob, heart_t)
            l_tv = tv_loss(prob)
            loss = (cfg["w_score"] * l_score + cfg["w_anchor"] * l_anchor
                    + cfg["w_hu"] * l_hu + cfg["w_heart"] * l_heart
                    + cfg["w_tv"] * l_tv)
            loss.backward()
            opt.step()

            run_score += l_score.item(); run_anchor += l_anchor.item()
            if step % cfg["log_every"] == 0:
                print(f"  e{epoch} s{step}/{len(train_cases)} "
                      f"Ls{l_score.item():.3f} Lanc{l_anchor.item():.4f} "
                      f"pred={pred_s:.0f} gt={gt_score:.0f}", flush=True)
            del prob, prob_a, loss
            torch.cuda.empty_cache()

        val = validate(student, diff, sp, scores, cfg, device)
        print(f"[epoch {epoch}] Ls{run_score/len(train_cases):.3f} "
              f"Lanc{run_anchor/len(train_cases):.4f} | val_dice {val['dice']:.4f} "
              f"| score_spearman {val['spearman']:.4f} | MAE {val['mae']:.1f}", flush=True)

        if val["spearman"] > best:
            best = val["spearman"]
            torch.save({"model": student.state_dict(), "epoch": epoch,
                        "val": val, "cfg": cfg},
                       os.path.join(cfg["ckpt_dir"], "stage2a_best.pth"))
            print(f"  -> saved best (spearman {best:.4f}, dice {val['dice']:.4f})", flush=True)

    print(f"\nDONE. best val spearman = {best:.4f}", flush=True)


@torch.no_grad()
def validate(model, diff, sp, scores, cfg, device):
    model.eval()
    img_dir, lbl_dir = sp["img_dir"], sp["lbl_dir"]
    dices, std_s, pred_s = [], [], []
    for case in sp["val"]:
        nii = nib.load(os.path.join(img_dir, case + "_0000.nii.gz"))
        ct = nii.get_fdata().astype(np.float32)
        gt = nib.load(os.path.join(lbl_dir, case + ".nii.gz")).get_fdata().astype(np.float32)
        spc = nii.header.get_zooms()[:3]; pa = spc[0] * spc[1]

        prob = forward_volume(model, ct, cfg["n_adjacent"], device, cfg["chunk"], use_ckpt=False)
        pred = (prob > 0.5).float()
        gt_t = torch.from_numpy((gt > 0.5).astype(np.float32)).to(device)
        inter = (pred * gt_t).sum()
        dices.append((2 * inter / (pred.sum() + gt_t.sum() + 1e-6)).item())

        ct_t = torch.from_numpy(ct).to(device)
        ps = float(diff(prob, ct_t, pixel_area=pa, reduce_dims=tuple(range(prob.dim()))))
        std_s.append(scores[case]); pred_s.append(ps)
        del prob; torch.cuda.empty_cache()

    std_s, pred_s = np.array(std_s), np.array(pred_s)
    a = (std_s * pred_s).sum() / (pred_s ** 2).sum() if (pred_s ** 2).sum() > 0 else 1.0
    sp_rho = spearmanr(std_s, a * pred_s)[0]
    mae = float(np.mean(np.abs(a * pred_s - std_s)))
    return {"dice": float(np.mean(dices)), "spearman": float(sp_rho), "mae": mae}
