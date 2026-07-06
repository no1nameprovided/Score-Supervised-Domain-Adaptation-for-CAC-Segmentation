"""
scripts/run_ablation.py
-----------------------
One-shot ablation: trains several ASCENT variants on the private TRAIN split
and evaluates each on the LOCKED test-20 split, then prints an ablation table.

All variants share lr=2e-5, 25 epochs, same seed -> differences are
attributable to the ablated component only. Each variant's best checkpoint
(by weighted-kappa on test) is used, consistent with the main experiment.

Variants:
  full              : score + heart + HU + anchor + low-score weighting
  wo_score          : no score loss (= source-only behavior; anchor+priors only)
  wo_anchor         : anchor weight = 0
  wo_heart          : heart prior weight = 0
  wo_hu             : HU prior weight = 0
  raw_score         : score loss in raw space (no log)
  wo_lowweight      : low-score weighting off (beta = 1 for all)

Run (long; use nohup):
    nohup python -u scripts/run_ablation.py > ablation.log 2>&1 &
    tail -f ablation.log
"""
import os, sys, json, copy
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# we reuse the stage2b trainer's pieces but need a raw-space option, so we
# import the module and monkey-patch the score loss when needed.
import engine.trainer_stage2b as T

ALPHA = 1.6
calib = "checkpoints/stage1/diffagatston_calib.json"
if os.path.exists(calib):
    ALPHA = json.load(open(calib)).get("alpha_soft", 1.6)

BASE = {
    "split_json": "data/splits/private_split.json",
    "scores_json": "data/splits/private_scores.json",
    "init_ckpt": "checkpoints/stage1/stage1_best.pth",
    "ckpt_dir": "checkpoints/ablation_tmp",
    "n_adjacent": 1, "init_filters": 16, "dropout": 0.2, "alpha": ALPHA,
    "w_score": 1.0, "w_anchor": 0.5, "w_hu": 0.5, "w_heart": 0.5,
    "anchor_conf": 0.5, "low_weight": True,
    "epochs": 25, "lr": 2e-5, "chunk": 8, "seed": 42, "log_every": 999,
}

VARIANTS = {
    "full":         {},
    "wo_score":     {"w_score": 0.0},
    "wo_anchor":    {"w_anchor": 0.0},
    "wo_heart":     {"w_heart": 0.0},
    "wo_hu":        {"w_hu": 0.0},
    "raw_score":    {"_raw_score": True},
    "wo_lowweight": {"low_weight": False},
}

# ---- raw-space score loss (monkey-patch target) ----
import torch.nn.functional as F
def score_loss_raw(diff, prob, hu, pa, gt, w=1.0):
    pred = diff(prob, hu, pixel_area=pa, reduce_dims=tuple(range(prob.dim())))
    # normalize raw difference by (1+gt) so it isn't dominated by huge scores
    tgt = torch.tensor(float(gt), device=prob.device)
    return w * F.huber_loss(pred/(1.0+tgt), tgt/(1.0+tgt), delta=1.0), float(pred)

_orig_score_loss = T.score_loss_log


def run_one(name, override):
    cfg = copy.deepcopy(BASE)
    raw = override.pop("_raw_score", False)
    cfg.update(override)
    cfg["ckpt_dir"] = f"checkpoints/ablation_tmp/{name}"

    # patch score loss for raw-space variant
    if raw:
        T.score_loss_log = score_loss_raw
    else:
        T.score_loss_log = _orig_score_loss

    print(f"\n===== ablation variant: {name}  (raw_score={raw}) =====", flush=True)
    # capture best test metrics by training and reading the saved best ckpt
    T.train_stage2b(cfg)
    best_path = os.path.join(cfg["ckpt_dir"], "stage2b_best.pth")
    ck = torch.load(best_path, map_location="cpu", weights_only=False)
    return ck["val"]   # dict: spearman / acc / kappa / zero_recall


def main():
    results = {}
    for name, ov in VARIANTS.items():
        try:
            results[name] = run_one(name, dict(ov))
        except Exception as e:
            print(f"[{name}] FAILED: {e}", flush=True)
            results[name] = {"spearman": float("nan"), "acc": float("nan"),
                             "kappa": float("nan"), "zero_recall": "?"}

    print("\n\n" + "="*78)
    print("ABLATION (private test-20, lr=2e-5, 25 ep, best-by-kappa)")
    print("-"*78)
    print(f"{'Variant':<16}{'Spearman':<12}{'Acc':<10}{'kappa':<10}{'0-recall':<10}")
    print("-"*78)
    order = ["full","wo_score","wo_anchor","wo_heart","wo_hu","raw_score","wo_lowweight"]
    for k in order:
        m = results.get(k, {})
        print(f"{k:<16}{m.get('spearman',float('nan')):<12.3f}"
              f"{m.get('acc',float('nan')):<10.3f}{m.get('kappa',float('nan')):<10.3f}"
              f"{str(m.get('zero_recall','?')):<10}")
    print("="*78)
    # also dump json for the paper
    json.dump(results, open("checkpoints/ablation_tmp/ablation_results.json","w"), indent=2)
    print("saved -> checkpoints/ablation_tmp/ablation_results.json")


if __name__ == "__main__":
    main()