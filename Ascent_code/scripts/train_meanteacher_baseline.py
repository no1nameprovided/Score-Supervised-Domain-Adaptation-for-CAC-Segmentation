"""
scripts/train_meanteacher_baseline.py
-------------------------------------
BASELINE (3): Mean-Teacher unsupervised domain adaptation.

Standard self-ensembling UDA: adapt the source SegResNet to the target images
using teacher-student consistency (EMA teacher, augmentation consistency),
plus the same heart/HU priors -- but WITHOUT the case-level scores. This tests
whether a standard target-image-only adaptation can remove the cross-domain
false positives. Expectation: it cannot, because it has no target-side
quantitative signal telling it the FP are wrong.

Evaluated on the SAME locked test-20 with SAME post-processing / Agatston.

Run:
    python -u scripts/train_meanteacher_baseline.py
"""
import os, sys, json, copy
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
from data.datasets import _window_normalize

COCA_SPACING = (0.3828, 0.3828, 3.0)
HU_THRESHOLD, MIN_AREA_MM2 = 130.0, 1.0

D = "/media/yuankai/3.5T/NN-Unet/nnUNet-master/DATASET1/LHCH_for_Dataset018_inference"
IMG_DIR, HEART_DIR = f"{D}/imagesTs", f"{D}/heartMasksTs"
SPLIT = "data/splits/private_split.json"
SCORES = "data/splits/private_scores.json"
STAGE1 = "checkpoints/stage1/stage1_best.pth"
CKPT_DIR = "checkpoints/meanteacher"

CFG = {
    "n_adjacent":1, "init_filters":16, "dropout":0.2,
    "lr":2e-5, "epochs":40, "chunk":8, "seed":42, "log_every":20,
    "ema_decay":0.95, "w_cons":1.0, "w_hu":0.5, "w_heart":0.5,
    "noise_std":0.10,   # input perturbation for consistency
}


def dw(p): return 4 if p>=400 else 3 if p>=300 else 2 if p>=200 else 1 if p>=130 else 0
def risk_cat(s): return 0 if s==0 else 1 if s<100 else 2 if s<400 else 3


def read_nrrd(path):
    data,hdr=nrrd.read(path); sp=(1.,1.,1.)
    if "space directions" in hdr:
        sd=np.array(hdr["space directions"],float); sp=tuple(float(np.linalg.norm(sd[i])) for i in range(3))
    return data.astype(np.float32), sp

def pad16(x):
    H,W=x.shape[-2],x.shape[-1]; ph,pw=(16-H%16)%16,(16-W%16)%16
    return F.pad(x,(0,pw,0,ph)),ph,pw

def resample(ct,spc):
    f=(spc[0]/COCA_SPACING[0],spc[1]/COCA_SPACING[1],spc[2]/COCA_SPACING[2])
    return zoom(ct,f,order=1)

def agatston_np(ct,mask,spacing):
    sx,sy,sz=spacing; pa,sw=sx*sy,sz/3.0; tot=0.
    for s in range(ct.shape[2]):
        m=(mask[:,:,s]>0.5)&(ct[:,:,s]>=HU_THRESHOLD)
        if not m.any(): continue
        lab,n=cc_label(m)
        for c in range(1,n+1):
            cm=lab==c; ar=cm.sum()*pa
            if ar<MIN_AREA_MM2: continue
            tot+=ar*dw(ct[:,:,s][cm].max())*sw
    return float(tot)


def forward_prob(model, ct_rs, n_adj, device, chunk, training, noise=0.0):
    H,W,S=ct_rs.shape
    stacks=[np.stack([_window_normalize(ct_rs[:,:,min(max(s+o,0),S-1)])
                      for o in range(-n_adj,n_adj+1)]) for s in range(S)]
    x=torch.from_numpy(np.stack(stacks)).to(device)
    if noise>0: x=x+torch.randn_like(x)*noise
    xp,ph,pw=pad16(x); outs=[]
    for i in range(0,S,chunk):
        xc=xp[i:i+chunk]
        out= checkpoint(lambda inp: model(inp), xc, use_reentrant=False) if training else model(xc)
        outs.append(out)
    logits=torch.cat(outs,0)[...,:H,:W]
    return torch.sigmoid(logits)[:,0].permute(1,2,0)   # (H,W,S)


@torch.no_grad()
def ema_update(teacher, student, decay):
    for tp,spm in zip(teacher.parameters(), student.parameters()):
        tp.mul_(decay).add_(spm, alpha=1-decay)
    for tb,sb in zip(teacher.buffers(), student.buffers()):
        tb.copy_(sb)


def hu_prior(prob, hu, tau=0.1): return (prob*torch.sigmoid(tau*(130.0-hu))).mean()
def heart_prior(prob, heart): return (prob*(1.0-heart)).mean()


@torch.no_grad()
def evaluate(model, sp, scores, device):
    model.eval(); std_gt,pred_sc=[],[]
    for case in sp["test"]:
        ct,spc=read_nrrd(os.path.join(IMG_DIR,case+"_0000.nrrd"))
        heart,_=read_nrrd(os.path.join(HEART_DIR,case+".nrrd"))
        if heart.shape!=ct.shape: continue
        ct_rs=resample(ct,spc)
        prob_rs=forward_prob(model,ct_rs,CFG["n_adjacent"],device,CFG["chunk"],False).cpu().numpy()
        inv=(ct.shape[0]/prob_rs.shape[0],ct.shape[1]/prob_rs.shape[1],ct.shape[2]/prob_rs.shape[2])
        prob=zoom(prob_rs,inv,order=1)[:ct.shape[0],:ct.shape[1],:ct.shape[2]]
        if prob.shape!=ct.shape:
            pb=np.zeros_like(ct); s=tuple(slice(0,min(prob.shape[i],ct.shape[i])) for i in range(3)); pb[s]=prob[s]; prob=pb
        pp=(prob>0.5).astype(np.float32)*(heart>0.5)
        std_gt.append(scores[case]); pred_sc.append(agatston_np(ct,pp,spc))
    std_gt=np.array(std_gt); pred_sc=np.array(pred_sc)
    a=(std_gt*pred_sc).sum()/(pred_sc**2).sum() if (pred_sc**2).sum()>0 else 1.0
    cg=[risk_cat(x) for x in std_gt]; cp=[risk_cat(x) for x in a*pred_sc]
    nz_ok=sum(1 for g,p in zip(cg,cp) if g==0 and p==0); nz=sum(1 for g in cg if g==0)
    return {"spearman":float(spearmanr(std_gt,pred_sc)[0]),
            "acc":float(accuracy_score(cg,cp)),
            "kappa":float(cohen_kappa_score(cg,cp,weights="linear")),
            "zero_recall":f"{nz_ok}/{nz}"}


def main():
    device="cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"])
    sp=json.load(open(SPLIT)); scores=json.load(open(SCORES)); train_cases=sp["train"]

    student=build_student(in_channels=2*CFG["n_adjacent"]+1,
                          init_filters=CFG["init_filters"], dropout_prob=CFG["dropout"]).to(device)
    ck=torch.load(STAGE1,map_location=device,weights_only=False)
    student.load_state_dict(ck["model"])
    teacher=copy.deepcopy(student).to(device)
    for p in teacher.parameters(): p.requires_grad_(False)
    teacher.eval()
    opt=torch.optim.AdamW(student.parameters(), lr=CFG["lr"], weight_decay=1e-5)
    os.makedirs(CKPT_DIR, exist_ok=True)
    print("Mean-Teacher UDA: init from stage1 (no scores used)")

    val0=evaluate(student, sp, scores, device)
    print(f"[init] spearman {val0['spearman']:.4f} acc {val0['acc']:.4f} "
          f"kappa {val0['kappa']:.4f} zero_recall {val0['zero_recall']}", flush=True)

    best=-1.0
    for epoch in range(CFG["epochs"]):
        student.train()
        order=np.random.permutation(len(train_cases)); run=0.0
        for step,ci in enumerate(order):
            case=train_cases[ci]
            ct,spc=read_nrrd(os.path.join(IMG_DIR,case+"_0000.nrrd"))
            heart,_=read_nrrd(os.path.join(HEART_DIR,case+".nrrd"))
            if heart.shape!=ct.shape: continue
            ct_rs=resample(ct,spc)
            heart_rs=resample((heart>0.5).astype(np.float32),spc)
            ct_t=torch.from_numpy(ct_rs).to(device)
            heart_t=torch.from_numpy((heart_rs>0.5).astype(np.float32)).to(device)

            # teacher on clean input (no grad), student on noised input
            with torch.no_grad():
                prob_t=forward_prob(teacher,ct_rs,CFG["n_adjacent"],device,CFG["chunk"],False,noise=0.0)
            opt.zero_grad()
            prob_s=forward_prob(student,ct_rs,CFG["n_adjacent"],device,CFG["chunk"],True,noise=CFG["noise_std"])

            l_cons=F.mse_loss(prob_s, prob_t)
            l_hu=hu_prior(prob_s, ct_t)
            l_heart=heart_prior(prob_s, heart_t)
            loss=CFG["w_cons"]*l_cons + CFG["w_hu"]*l_hu + CFG["w_heart"]*l_heart
            loss.backward(); opt.step()
            ema_update(teacher, student, CFG["ema_decay"])
            run+=l_cons.item()
            if step%CFG["log_every"]==0:
                print(f"  e{epoch} s{step}/{len(train_cases)} Lcons{l_cons.item():.4f}", flush=True)
            del prob_s, prob_t, loss; torch.cuda.empty_cache()

        val=evaluate(student, sp, scores, device)
        print(f"[epoch {epoch}] Lcons{run/len(train_cases):.4f} | spearman {val['spearman']:.4f} "
              f"| acc {val['acc']:.4f} | kappa {val['kappa']:.4f} | zero_recall {val['zero_recall']}", flush=True)
        if val["kappa"]>best:
            best=val["kappa"]
            torch.save({"model":student.state_dict(),"epoch":epoch,"val":val}, f"{CKPT_DIR}/meanteacher_best.pth")
            print(f"  -> saved best (kappa {best:.4f})", flush=True)

    ck2=torch.load(f"{CKPT_DIR}/meanteacher_best.pth", map_location=device, weights_only=False)
    print(f"\n######## MEAN-TEACHER BASELINE (best) ########")
    print(f" epoch {ck2['epoch']}: {ck2['val']}")
    v=ck2['val']
    print(f"\n--> Table row:  Mean-Teacher (img only)  &  "
          f"{v['spearman']:.3f} & {v['acc']:.2f} & {v['kappa']:.2f} & {v['zero_recall']}")


if __name__=="__main__":
    main()