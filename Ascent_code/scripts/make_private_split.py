"""
scripts/make_private_split.py
-----------------------------
Split the private (LHCH) cohort into train/test ONCE, STRATIFIED by the 4
Agatston risk categories so the small test set still contains every class
(critical: there are only 9 zero-score cases, and beating the FP problem is
judged largely on recovering class 0).

Also caches RBQ_No -> Agatston Total to data/splits/private_scores.json.

Outputs:
  data/splits/private_split.json   (train/test case lists by RBQ_No)
  data/splits/private_scores.json  (RBQ_No -> true Agatston Total)

Run:
    python scripts/make_private_split.py
"""
import os, glob, json, random
import numpy as np
import pandas as pd

D = "/media/yuankai/3.5T/NN-Unet/nnUNet-master/DATASET1/LHCH_for_Dataset018_inference"
IMG_DIR = f"{D}/imagesTs"
HEART_DIR = f"{D}/heartMasksTs"
XLSX = f"{D}/GT.xlsx"
OUT_SPLIT = "data/splits/private_split.json"
OUT_SCORES = "data/splits/private_scores.json"

N_TEST = 20
SEED = 42


def risk_cat(s):
    if s == 0: return 0
    if s < 100: return 1
    if s < 400: return 2
    return 3


def main():
    df = pd.read_excel(XLSX)
    gt_map = {str(r["RBQ_No"]).strip(): float(r["Agatston Total"])
              for _, r in df.iterrows() if pd.notna(r["Agatston Total"])}

    # keep only cases that have BOTH an image and a heart mask
    cases = []
    for ip in sorted(glob.glob(f"{IMG_DIR}/RBQ*_0000.nrrd")):
        case = os.path.basename(ip).replace("_0000.nrrd", "")
        rbq = case.split("_")[0]
        hp = os.path.join(HEART_DIR, case + ".nrrd")
        if rbq in gt_map and os.path.exists(hp):
            cases.append((case, rbq, gt_map[rbq]))

    print(f"usable paired cases: {len(cases)}")

    # group by risk category
    by_cat = {0: [], 1: [], 2: [], 3: []}
    for case, rbq, s in cases:
        by_cat[risk_cat(s)].append((case, rbq, s))
    print("class sizes:", {k: len(v) for k, v in by_cat.items()})

    # stratified: take ~ N_TEST * (class_frac) from each class for test
    rng = random.Random(SEED)
    n_total = len(cases)
    test, train = [], []
    for k, lst in by_cat.items():
        rng.shuffle(lst)
        n_test_k = max(1, round(N_TEST * len(lst) / n_total)) if lst else 0
        n_test_k = min(n_test_k, len(lst))
        test += lst[:n_test_k]
        train += lst[n_test_k:]

    # adjust to hit exactly N_TEST if rounding drifted
    while len(test) > N_TEST:
        train.append(test.pop())
    while len(test) < N_TEST and train:
        test.append(train.pop())

    split = {
        "img_dir": IMG_DIR, "heart_dir": HEART_DIR,
        "seed": SEED,
        "train": sorted([c[0] for c in train]),
        "test": sorted([c[0] for c in test]),
    }
    scores = {c[0]: c[2] for c in cases}   # case -> score (case includes date)

    os.makedirs(os.path.dirname(OUT_SPLIT), exist_ok=True)
    json.dump(split, open(OUT_SPLIT, "w"), indent=2)
    json.dump(scores, open(OUT_SCORES, "w"), indent=2)

    from collections import Counter
    tr_cat = Counter(risk_cat(c[2]) for c in train)
    te_cat = Counter(risk_cat(c[2]) for c in test)
    print(f"\ntrain={len(train)}  test={len(test)}")
    print("train class dist:", dict(sorted(tr_cat.items())))
    print("test  class dist:", dict(sorted(te_cat.items())))
    print(f"saved -> {OUT_SPLIT}")
    print(f"saved -> {OUT_SCORES}")
    print("\nNOTE: test cases LOCKED. Score-adaptation trains on TRAIN only.")


if __name__ == "__main__":
    main()