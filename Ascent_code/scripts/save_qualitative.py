"""
scripts/save_qualitative.py  (v2: dump ALL test cases + rank by FP contrast)
============================================================================
Saves CT + source-only + ASCENT masks for EVERY test case, and prints a table
ranked so you can pick the most illustrative examples:
  - for zero-score cases: largest source-only FP that ASCENT removes
  - for high-score cases: true calcium that both keep (ASCENT cleaner)

Look at the printed table, pick the best case names, then screenshot those in
ITK-SNAP.
"""
import os, sys, json
import numpy as np
import nrrd, nibabel as nib
import torch
from scipy.ndimage import zoom

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.segresnet import build_student
from data.datasets import _window_normalize

COCA_SPACING = (0.3828, 0.3828, 3.0)
D = "/media/yuankai/3.5T/NN-Unet/nnUNet-master/DATASET1/LHCH_for_Dataset018_inference"
IMG_DIR, HEART_DIR = f"{D}/imagesTs", f"{D}/heartMasksTs"
SPLIT = "data/splits/private_split.json"
SCORES = "data/splits/private_scores.json"
STAGE1 = "checkpoints/stage1/stage1_best.pth"
STAGE2B = "checkpoints/stage2b/stage2b_best.pth"
OUT = "qual_vis"
HU_THRESHOLD = 130.0


def read_nrrd(path):
    data, hdr = nrrd.read(path); sp=(1.,1.,1.)
    if "space directions" in hdr:
        sd=np.array(hdr["space directions"],float); sp=tuple(float(np.linalg.norm(sd[i])) for i in range(3))
    return data.astype(np.float32), sp

def pad16(x):
    H,W=x.shape[-2],x.shape[-1]; ph,pw=(16-H%16)%16,(16-W%16)%16
    return torch.nn.functional.pad(x,(0,pw,0,ph)),ph,pw

@torch.no_grad()
def predict(model, ct, spc, n_adj, device):
    fx,fy,fz=spc[0]/COCA_SPACING[0],spc[1]/COCA_SPACING[1],spc[2]/COCA_SPACING[2]
    ct_rs=zoom(ct,(fx,fy,fz),order=1); H,W,S=ct_rs.shape
    prob_rs=np.zeros_like(ct_rs,np.float32)
    for s in range(S):
        ch=[_window_normalize(ct_rs[:,:,min(max(s+o,0),S-1)]) for o in range(-n_adj,n_adj+1)]
        x=torch.from_numpy(np.stack(ch)[None]).to(device); xp,ph,pw=pad16(x)
        prob_rs[:,:,s]=torch.sigmoid(model(xp)[...,:H,:W])[0,0].cpu().numpy()
    inv=tuple(ct.shape[i]/prob_rs.shape[i] for i in range(3))
    prob=zoom(prob_rs,inv,order=1)[:ct.shape[0],:ct.shape[1],:ct.shape[2]]
    if prob.shape!=ct.shape:
        pb=np.zeros_like(ct); sl=tuple(slice(0,min(prob.shape[i],ct.shape[i])) for i in range(3)); pb[sl]=prob[sl]; prob=pb
    return prob

def save_nii(arr, spacing, path):
    nib.save(nib.Nifti1Image(arr.astype(np.float32), np.diag([spacing[0],spacing[1],spacing[2],1.0])), path)

def risk_cat(s): return 0 if s==0 else 1 if s<100 else 2 if s<400 else 3


def main():
    device="cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT, exist_ok=True)
    sp=json.load(open(SPLIT)); scores=json.load(open(SCORES))

    ck=torch.load(STAGE1,map_location=device,weights_only=False); cfg=ck["cfg"]
    m_src=build_student(in_channels=2*cfg["n_adjacent"]+1,init_filters=cfg["init_filters"],dropout_prob=cfg["dropout"]).to(device)
    m_src.load_state_dict(ck["model"]); m_src.eval()
    ck2=torch.load(STAGE2B,map_location=device,weights_only=False)
    m_asc=build_student(in_channels=2*cfg["n_adjacent"]+1,init_filters=cfg["init_filters"],dropout_prob=cfg["dropout"]).to(device)
    m_asc.load_state_dict(ck2["model"]); m_asc.eval()
    print("loaded models; dumping ALL test cases...\n")

    rows=[]
    for case in sp["test"]:
        ct, spc = read_nrrd(os.path.join(IMG_DIR, case+"_0000.nrrd"))
        heart, _ = read_nrrd(os.path.join(HEART_DIR, case+".nrrd"))
        gt = scores[case]
        prob_src = predict(m_src, ct, spc, cfg["n_adjacent"], device)
        prob_asc = predict(m_asc, ct, spc, cfg["n_adjacent"], device)
        mask_src = ((prob_src>0.5)&(heart>0.5)&(ct>=HU_THRESHOLD)).astype(np.float32)
        mask_asc = ((prob_asc>0.5)&(heart>0.5)&(ct>=HU_THRESHOLD)).astype(np.float32)

        base = os.path.join(OUT, f"cat{risk_cat(gt)}_{case}")
        save_nii(ct, spc, base+"_ct.nii.gz")
        save_nii(mask_src, spc, base+"_srconly.nii.gz")
        save_nii(mask_asc, spc, base+"_ascent.nii.gz")
        save_nii((heart>0.5).astype(np.float32), spc, base+"_heart.nii.gz")

        sv, av = int(mask_src.sum()), int(mask_asc.sum())
        rows.append((risk_cat(gt), gt, case, sv, av, sv-av))

    # print ranked table
    print(f"{'cat':<4}{'gt':>7}{'case':<26}{'src_vox':>9}{'ascent_vox':>12}{'removed':>9}")
    print("-"*70)
    # zero-score first, ranked by how much FP removed
    for cat in [0,1,2,3]:
        sub=[r for r in rows if r[0]==cat]
        sub.sort(key=lambda r:-r[5])
        for r in sub:
            print(f"{r[0]:<4}{r[1]:>7.0f}{r[2]:<26}{r[3]:>9}{r[4]:>12}{r[5]:>9}")
    print("\nPICK:")
    print(" * zero-score case with LARGE src_vox and small ascent_vox = best FP-suppression demo")
    print(" * high-score case where both keep calcium but ascent_vox slightly < src_vox = clean detection")
    print(f"\nAll files in {OUT}/ named cat<risk>_<case>_*.nii.gz")


if __name__=="__main__":
    main()