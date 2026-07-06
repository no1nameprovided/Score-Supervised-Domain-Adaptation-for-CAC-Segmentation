"""
scripts/train_stage1.py
-----------------------
Entry point for Stage 1 (fully-supervised COCA segmentation).

Run from project root:
    conda activate ascent
    python scripts/train_stage1.py

Edit the CONFIG dict below, or later wire it to configs/stage1_coca.yaml.
Start with a SHORT run (epochs=2) to confirm everything trains, then scale up.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.trainer_stage1 import train_stage1


CONFIG = {
    "split_json": "data/splits/coca_split.json",
    "ckpt_dir": "checkpoints/stage1",

    # data / 2.5D
    "n_adjacent": 1,          # 1 -> 3 channels
    "crop": 256,              # train crop size (fits 4090 easily)

    # model
    "init_filters": 16,
    "dropout": 0.2,

    # optimization
    "epochs": 50,
    "batch_size": 16,         # lower to 8 if OOM on 4090
    "lr": 2e-4,
    "weight_decay": 1e-5,
    "num_workers": 8,
    "seed": 42,
    "log_every": 50,
}


if __name__ == "__main__":
    # tip for the first run: set epochs=2 to smoke-test the whole loop fast.
    train_stage1(CONFIG)