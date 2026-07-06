# Score-Supervised-Domain-Adaptation-for-CAC-Segmentation# ASCENT

**Score-Supervised Domain Adaptation for Coronary Artery Calcium Segmentation on Non-Contrast CT without Target Masks**

ASCENT adapts a coronary-artery-calcium (CAC) segmentation network to an unlabeled
target cohort using **only case-level Agatston scores** — no target voxel masks.

The Agatston score is not a black box: it is a **known, closed-form function** of the
calcium mask and the CT intensities. ASCENT makes that function differentiable
(**DiffAgatston**) and uses the scalar score to supervise the voxel mask "in reverse,"
regularized by anatomical priors and an anchor to the source model.

> **The score fixes _how much_ calcium; the priors fix _where_ it can be; the anchor preserves confident source calcium.**

---

## Key idea

Supervised segmentation needs per-voxel masks, which are expensive and usually
unavailable in a new clinical cohort. But every cohort already stores **case-level
Agatston scores**. ASCENT turns that scalar into voxel-level supervision:

- **DiffAgatston** — a differentiable surrogate of the clinical Agatston operator.
  Given a soft calcium probability map and raw HU values it returns a differentiable
  score, so a case-level score can back-propagate to the mask.
- **Anatomical priors** — an HU prior (calcium is > 130 HU) and a heart-region prior
  (calcium lies inside the heart) constrain *where* the mask may place calcium.
- **Source anchor** — a frozen copy of the source model prevents the mask from drifting
  to "cheat" the score while degrading localization.

DiffAgatston reproduces the clinical operator almost exactly — **Spearman ρ = 0.997 /
log-Pearson = 0.993** against standard Agatston on COCA ground-truth masks — which is
what makes score-only supervision trustworthy.

---

## Method

```
Stage 1  ─ source model ────────────────────────────────────────────────
           full supervision on COCA (masks available)  ->  Dice ~0.877

Stage 2  ─ score-supervised adaptation to the target ───────────────────
           init from Stage 1; supervise with case-level scores only.
           Four forces on the student:

             L_score   DiffAgatston(M) vs GT score   (log-space Huber,
                       low-score weighting β(s))            <- "how much"
             L_anchor  asymmetric anchor to the frozen
                       source model                         <- anti-drift
             L_HU      penalize prob where HU < 130          <- "where"
             L_heart   penalize prob outside heart mask      <- "where"
```

`trainer_stage2b.py` is the deployed adaptation (frozen, asymmetric anchor —
penalizes *erasing* confident source calcium but leaves the score free to add/remove).
`trainer_stage2a.py` is a source-side sanity check (mask-blind fine-tuning on COCA with
an EMA-teacher anchor); it validates the mechanism but does not improve an
already-strong source model, which motivates applying the method on the target domain.

**Cross-spacing.** Source (COCA) and target differ in voxel spacing. Stage-2b resamples
the target volume to the source spacing for the network forward pass and maps the mask
back; the Agatston computation always runs at the volume's native spacing.

---

## Results

Locked 20-case private test set. ASCENT uses only case-level scores from the target
training split — no target masks. nnU-Net is a fully-supervised reference (upper bound).

| Method                              | Spearman ρ | Accuracy | Weighted κ | Zero-recall |
| ----------------------------------- | :--------: | :------: | :--------: | :---------: |
| Source-only (SegResNet)             |   0.726    |   0.40   |    0.35    |     0/2     |
| nnU-Net (fully-supervised ref.)     |   0.875    |   0.69   |    0.70    |     1/2     |
| **ASCENT (ours)**                   | **0.746**  | **0.60** |  **0.53**  |   **2/2**   |

Accuracy is averaged over three runs. The headline effect is **risk-category accuracy
0.40 → 0.60**, **weighted κ 0.35 → 0.53**, and recovery of **all calcium-free cases
(0/2 → 2/2)** — the source-only model produces cross-domain false positives on the aorta,
sternum, and stents, which score supervision suppresses.

---

## Installation

```bash
conda create -n ascent python=3.10 -y
conda activate ascent

# PyTorch (CUDA 12.1 build)
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# core deps — numpy MUST stay < 2 (OpenCV/MONAI ABI conflict otherwise)
pip install monai==1.5.2 "numpy<2" \
            nibabel pynrrd scipy scikit-image \
            opencv-python-headless pandas openpyxl
```

Hardware: developed on a single **RTX 4090 (24 GB)**; Stage-2 uses gradient
checkpointing to fit. A larger GPU (e.g. H100) removes the memory constraint.

---

## Data

Two cohorts, both in nnU-Net-style raw layout with a separate heart-region mask.

**Source — Stanford COCA** (heart-cropped, masks available):

```
Dataset018_STF_HeartCrop/
├── imagesTr/img_XXXX_0000.nii.gz    # non-contrast CT, raw HU
├── labelsTr/img_XXXX.nii.gz         # calcium mask (0/1)
└── heartPredTr/img_XXXX.nrrd        # heart-region mask
```

**Target — private cohort** (scores only, no masks):

```
LHCH_for_Dataset018_inference/
├── imagesTs/RBQ..._0000.nrrd
├── heartMasksTs/RBQ....nrrd
├── predictions/RBQ....nii.gz        # source-model predictions (baseline)
└── GT.xlsx                          # RBQ_No <-> "Agatston Total"  (case-level score)
```

Splits and precomputed scores live in `data/splits/`:

| File                    | Contents                                     |
| ----------------------- | -------------------------------------------- |
| `coca_split.json`       | source 312 / 44 / 90 train/val/test (seed 42)|
| `coca_scores.json`      | reference Agatston scores for source cases   |
| `private_split.json`    | target 68 train / 20 test (stratified)       |
| `private_scores.json`   | case-level scores for target cases           |

Update the dataset paths at the top of each `scripts/*.py` CONFIG to match your machine.

---

## Project structure

```
Ascent/
├── data/
│   ├── splits/               # json splits + precomputed scores
│   └── datasets.py           # windowing/normalization, 2.5D stacking, calcium oversampling
├── models/
│   └── segresnet.py          # build_student(): MONAI SegResNet, 2.5D (3 adjacent slices)
├── diffagatston/
│   └── layer.py              # DiffAgatston — differentiable Agatston operator
├── engine/
│   ├── losses.py
│   ├── trainer_stage1.py     # full supervision on source
│   ├── trainer_stage2a.py    # mask-blind score sup. on source (EMA anchor; validation)
│   └── trainer_stage2b.py    # score-supervised adaptation to target (main method)
├── scripts/                  # thin entry points: a CONFIG dict + one call
│   ├── make_split.py / make_private_split.py
│   ├── precompute_scores.py
│   ├── calibrate_alpha.py
│   ├── train_stage1.py
│   ├── train_stage2a.py
│   ├── train_stage2b.py
│   ├── eval_private_baseline.py
│   ├── eval_stage2b.py
│   └── eval_test20_table.py  # 3-method comparison w/ bootstrap 95% CIs
└── checkpoints/
```

**Convention:** everything under `engine/` *defines* logic (`def train_*`, long files);
everything under `scripts/` *runs* it (a `CONFIG = {...}` dict then the call, short files).

---

## Usage

Run all commands from the project root with the `ascent` env active.

**1 — Splits and reference scores**

```bash
python -u scripts/make_split.py           # source split
python -u scripts/precompute_scores.py    # reference Agatston scores for source
python -u scripts/make_private_split.py   # target split (stratified)
```

**2 — Calibrate DiffAgatston** (fits the global scale α; writes
`checkpoints/stage1/diffagatston_calib.json`)

```bash
python -u scripts/calibrate_alpha.py
```

**3 — Stage 1: source model**

```bash
python -u scripts/train_stage1.py         # -> checkpoints/stage1/stage1_best.pth
```

**4 — Stage 2: score-supervised adaptation**

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u scripts/train_stage2b.py        # -> checkpoints/stage2b/stage2b_best.pth
```

Optional source-side sanity check: `python -u scripts/train_stage2a.py`.

**5 — Evaluate**

```bash
python -u scripts/eval_private_baseline.py   # source-only baseline on test-20
python -u scripts/eval_stage2b.py            # ASCENT on test-20
python -u scripts/eval_test20_table.py       # full comparison table
```

Key hyperparameters live in each script's `CONFIG` dict, which is the source of truth.

---

## Notes & gotchas

- **DiffAgatston takes RAW HU, not normalized intensities.** The network input is
  windowed to `[-1, 1]`, but the scoring layer must receive the original Hounsfield
  values. The dataset returns both an `image` (normalized) and an `hu` (raw) field.
- **Stage-2 trains per volume** (the score is case-level), which is memory-heavy.
  Gradient checkpointing is on in the Stage-2 trainers; if you still hit OOM, lower
  `chunk` in the CONFIG (8 → 4 → 2), or move to a larger GPU.
- **The anchor is asymmetric.** It only penalizes *erasing* confident source calcium,
  so the score is free to correct false positives without being pulled back toward them.
  A symmetric (plain MSE) anchor over-constrains and negates the score signal.
- **SegResNet needs input dims divisible by 16** — the trainers pad/crop accordingly.
- **Checkpoints are selected by weighted-κ**, which is more stable than raw accuracy on
  the small target test set.
- If a run exits silently with no output, an entry script and its trainer were probably
  swapped — check that `scripts/train_*.py` is the short CONFIG file and
  `engine/trainer_*.py` is the long `def train_*` file.

---

## Citation

Manuscript under review. Please update once published.

```bibtex
@inproceedings{ascent,
  title     = {Score-Supervised Domain Adaptation for Coronary Artery Calcium
               Segmentation on Non-Contrast CT without Target Masks},
  author    = {Anonymous},
  booktitle = {Under review},
  year      = {2026}
}
```
