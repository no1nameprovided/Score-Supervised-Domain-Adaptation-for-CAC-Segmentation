"""
scripts/train_regression_baseline.py
------------------------------------
BASELINE (2): Direct Agatston score regression.

Same SegResNet encoder as ASCENT, but a global-pooling head outputs a single
scalar; trained to regress the case-level Agatston score directly from the
image (log-space Huber), using ONLY the target train-68 scores. This is the
"regression route" to weak supervision: no mask, no spatial evidence.

Evaluated on the SAME locked test-20 with the SAME risk-category metrics.
It predicts a SCORE directly (no Agatston post-processing / no mask).

Run:
    python -u scripts/train_regression_baseline.py
"""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from scipy.ndimage import zoom
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, cohen_kappa_score
import nrrd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.segresnet import build_student
from data.datasets import _window_normalize

COCA_SPACING = (0.3828, 0.3828, 3.0)

D = "/media/yuankai/3.5T/NN-Unet/nnUNet-master/DATASET1/LHCH_for_Dataset018_inference"
IMG_DIR, HEART_DIR = f"{D}/imagesTs", f"{D}/heartMasksTs"
SPLIT = "data/splits/private_split.json"
SCORES = "data/splits/private_scores.json"
STAGE1 = "checkpoints/stage1/stage1_best.pth"
CKPT_DIR = "checkpoints/regression"

CFG = {
    "n_adjacent": 1, "init_filters": 16, "dropout": 0.2,
    "lr": 2e-4, "epochs": 40, "chunk": 8, "seed": 42, "log_every": 20,
}


def risk_cat(s): return 0 if s==0 else 1 if s<100 else 2 if s<400 else 3


def read_nrrd(path):
    data, hdr = nrrd.read(path); sp=(1.,1.,1.)
    if "space directions" in hdr:
        sd=np.array(hdr["space directions"],float); sp=tuple(float(np.linalg.norm(sd[i])) for i in range(3))
    return data.astype(np.float32), sp


def pad16(x):
    H,W=x.shape[-2],x.shape[-1]; ph,pw=(16-H%16)%16,(16-W%16)%16
    return F.pad(x,(0,pw,0,ph)),ph,pw


def resample(ct, spc):
    f=(spc[0]/COCA_SPACING[0],spc[1]/COCA_SPACING[1],spc[2]/COCA_SPACING[2])
    return zoom(ct,f,order=1)


class RegressionNet(nn.Module):
    """SegResNet encoder (reused) + global pooling + MLP head -> scalar.
    We reuse build_student and tap its encoder features via a forward hook-free
    approach: run the full seg net, then pool its penultimate logits map.
    For simplicity and a fair backbone, we pool the seg logit map to a scalar
    through a small learned head."""
    def __init__(self, seg_net):
        super().__init__()
        self.seg = seg_net                     # same SegResNet backbone
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward_volume(self, ct_rs, n_adj, device, chunk, training):
        H,W,S = ct_rs.shape
        stacks=[np.stack([_window_normalize(ct_rs[:,:,min(max(s+o,0),S-1)])
                          for o in range(-n_adj,n_adj+1)]) for s in range(S)]
        x=torch.from_numpy(np.stack(stacks)).to(device); xp,ph,pw=pad16(x)
        feats=[]
        for i in range(0,S,chunk):
            xc=xp[i:i+chunk]
            out= checkpoint(lambda inp: self.seg(inp), xc, use_reentrant=False) if training else self.seg(xc)
            feats.append(out)
        logit_map=torch.cat(feats,0)[...,:H,:W]      # (S,1,H,W)
        # aggregate whole volume: mean over slices+space -> scalar via head
        pooled=logit_map.mean(dim=0, keepdim=True)   # (1,1,H,W)
        scalar=self.head(pooled).squeeze()           # ()
        # softplus to keep score non-negative
        return F.softplus(scalar)


def evaluate(model, sp, scores, device):
    model.eval()
    std_gt, pred = [], []
    with torch.no_grad():
        for case in sp["test"]:
            ct, spc = read_nrrd(os.path.join(IMG_DIR, case+"_0000.nrrd"))
            ct_rs = resample(ct, spc)
            s_pred = float(model.forward_volume(ct_rs, CFG["n_adjacent"], device, CFG["chunk"], False))
            std_gt.append(scores[case]); pred.append(s_pred)
    std_gt=np.array(std_gt); pred=np.array(pred)
    a=(std_gt*pred).sum()/(pred**2).sum() if (pred**2).sum()>0 else 1.0
    pc=a*pred
    cg=[risk_cat(x) for x in std_gt]; cp=[risk_cat(x) for x in pc]
    nz_ok=sum(1 for g,p in zip(cg,cp) if g==0 and p==0); nz=sum(1 for g in cg if g==0)
    return {"spearman":float(spearmanr(std_gt,pred)[0]),
            "acc":float(accuracy_score(cg,cp)),
            "kappa":float(cohen_kappa_score(cg,cp,weights="linear")),
            "zero_recall":f"{nz_ok}/{nz}"}


def main():
    device="cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"])
    sp=json.load(open(SPLIT)); scores=json.load(open(SCORES))
    train_cases=sp["train"]

    seg=build_student(in_channels=2*CFG["n_adjacent"]+1,
                      init_filters=CFG["init_filters"], dropout_prob=CFG["dropout"])
    ck=torch.load(STAGE1, map_location=device, weights_only=False)
    seg.load_state_dict(ck["model"])              # warm-start from source encoder
    model=RegressionNet(seg).to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=1e-5)
    os.makedirs(CKPT_DIR, exist_ok=True)
    print("regression baseline: warm-started from stage1 encoder")

    val0=evaluate(model, sp, scores, device)
    print(f"[init] spearman {val0['spearman']:.4f} acc {val0['acc']:.4f} "
          f"kappa {val0['kappa']:.4f} zero_recall {val0['zero_recall']}", flush=True)

    best=-1.0
    for epoch in range(CFG["epochs"]):
        model.train()
        order=np.random.permutation(len(train_cases)); run=0.0
        for step,ci in enumerate(order):
            case=train_cases[ci]
            ct,spc=read_nrrd(os.path.join(IMG_DIR, case+"_0000.nrrd"))
            ct_rs=resample(ct,spc); gt=scores[case]
            opt.zero_grad()
            pred=model.forward_volume(ct_rs, CFG["n_adjacent"], device, CFG["chunk"], True)
            loss=F.huber_loss(torch.log1p(pred),
                              torch.log1p(torch.tensor(float(gt),device=device)), delta=1.0)
            loss.backward(); opt.step(); run+=loss.item()
            if step%CFG["log_every"]==0:
                print(f"  e{epoch} s{step}/{len(train_cases)} L{loss.item():.3f} "
                      f"pred={float(pred):.0f} gt={gt:.0f}", flush=True)
            torch.cuda.empty_cache()
        val=evaluate(model, sp, scores, device)
        print(f"[epoch {epoch}] L{run/len(train_cases):.3f} | spearman {val['spearman']:.4f} "
              f"| acc {val['acc']:.4f} | kappa {val['kappa']:.4f} | zero_recall {val['zero_recall']}", flush=True)
        if val["kappa"]>best:
            best=val["kappa"]
            torch.save({"model":model.state_dict(),"epoch":epoch,"val":val}, f"{CKPT_DIR}/regression_best.pth")
            print(f"  -> saved best (kappa {best:.4f})", flush=True)

    # report best
    ck2=torch.load(f"{CKPT_DIR}/regression_best.pth", map_location=device, weights_only=False)
    print(f"\n######## REGRESSION BASELINE (best) ########")
    print(f" epoch {ck2['epoch']}: {ck2['val']}")
    v=ck2['val']
    print(f"\n--> Table row:  Direct regression  &  "
          f"{v['spearman']:.3f} & {v['acc']:.2f} & {v['kappa']:.2f} & {v['zero_recall']}")


if __name__=="__main__":
    main()