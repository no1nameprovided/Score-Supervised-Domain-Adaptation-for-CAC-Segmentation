"""
scripts/train_stage2b.py - private score-adaptation (one-shot tuned)
"""
import os, sys, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.trainer_stage2b import train_stage2b

ALPHA = 1.6
calib = "checkpoints/stage1/diffagatston_calib.json"
if os.path.exists(calib):
    ALPHA = json.load(open(calib)).get("alpha_soft", 1.6)

CONFIG = {
    "split_json": "data/splits/private_split.json",
    "scores_json": "data/splits/private_scores.json",
    "init_ckpt": "checkpoints/stage1/stage1_best.pth",
    "ckpt_dir": "checkpoints/stage2b",
    "n_adjacent": 1,
    "init_filters": 16,
    "dropout": 0.2,
    "alpha": ALPHA,
    "w_score": 1.0,
    "w_anchor": 0.5,       # OFF: let score fully suppress FP first
    "w_hu": 0.5,
    "w_heart": 0.5,
    "anchor_conf": 0.5,
    "low_weight": True,    # upweight zero/low-score cases
    "epochs": 40,
    "lr": 2e-5,            # 5x larger than before
    "chunk": 8,
    "seed": 42,
    "log_every": 20,
}

if __name__ == "__main__":
    train_stage2b(CONFIG)