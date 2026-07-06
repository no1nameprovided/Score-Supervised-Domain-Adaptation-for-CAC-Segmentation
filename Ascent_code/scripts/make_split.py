"""
make_split.py
-------------
Split the COCA cases into train / val / test ONCE, with a fixed seed, and
save to data/splits/coca_split.json. Every later experiment loads this file,
so the test set is locked and never leaks.

Run from project root:
    python scripts/make_split.py
"""

import os
import glob
import json
import random
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--img_dir",
        default="/media/yuankai/3.5T/NN-Unet/nnUNet-master/DATASET/"
                "nnUNet_raw_data/Dataset018_STF_HeartCrop/imagesTr",
    )
    ap.add_argument(
        "--lbl_dir",
        default="/media/yuankai/3.5T/NN-Unet/nnUNet-master/DATASET/"
                "nnUNet_raw_data/Dataset018_STF_HeartCrop/labelsTr",
    )
    ap.add_argument("--out", default="data/splits/coca_split.json")
    ap.add_argument("--n_test", type=int, default=90)
    ap.add_argument("--n_val", type=int, default=44)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # ---- collect case ids from labels, keep only those with an image ----
    lbl_files = sorted(glob.glob(os.path.join(args.lbl_dir, "*.nii.gz")))
    cases = []
    for lp in lbl_files:
        cid = os.path.basename(lp).replace(".nii.gz", "")          # img_0001
        ip = os.path.join(args.img_dir, cid + "_0000.nii.gz")
        if os.path.exists(ip):
            cases.append(cid)
        else:
            print("WARN: no image for", cid, "-> skipped")

    print(f"found {len(cases)} paired cases")

    # ---- shuffle with fixed seed, then split ----
    rng = random.Random(args.seed)
    rng.shuffle(cases)

    test = sorted(cases[: args.n_test])
    val = sorted(cases[args.n_test: args.n_test + args.n_val])
    train = sorted(cases[args.n_test + args.n_val:])

    split = {
        "img_dir": args.img_dir,
        "lbl_dir": args.lbl_dir,
        "seed": args.seed,
        "train": train,
        "val": val,
        "test": test,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(split, f, indent=2)

    print(f"train={len(train)}  val={len(val)}  test={len(test)}")
    print("saved ->", args.out)
    print("\nNOTE: test cases are now LOCKED. Never train on them.")


if __name__ == "__main__":
    main()