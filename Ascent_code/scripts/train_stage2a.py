import os, sys, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.trainer_stage2a import train_stage2a

ALPHA = 1.6
calib_path = "checkpoints/stage1/diffagatston_calib.json"
if os.path.exists(calib_path):
    ALPHA = json.load(open(calib_path)).get("alpha_soft", 1.6)

CONFIG = {
    "split_json": "data/splits/coca_split.json",
    "scores_json": "data/splits/coca_scores.json",
    "heart_dir": ("/media/yuankai/3.5T/NN-Unet/nnUNet-master/DATASET/"
                  "nnUNet_raw_data/Dataset018_STF_HeartCrop/heartPredTr"),
    "init_ckpt": "checkpoints/stage1/stage1_best.pth",
    "ckpt_dir": "checkpoints/stage2a",
    "n_adjacent": 1,
    "init_filters": 16,
    "dropout": 0.2,
    "alpha": ALPHA,
    "w_score": 1.0,
    "w_anchor": 5.0,
    "w_hu": 0.5,
    "w_heart": 0.5,
    "w_tv": 0.01,
    "anchor_conf": 0.5,
    "epochs": 2,
    "lr": 1e-5,
    "chunk": 8,
    "num_workers": 0,
    "seed": 42,
    "log_every": 40,
}

if __name__ == "__main__":
    train_stage2a(CONFIG)
