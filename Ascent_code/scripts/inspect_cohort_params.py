"""
scripts/inspect_cohort_params.py
--------------------------------
Read all private nrrd volumes and summarize acquisition parameters for the
paper's dataset section: in-plane spacing, slice thickness, matrix size,
HU range. Also tries to read DICOM-style fields from the nrrd header if
present (vendor/kernel are usually NOT in nrrd, so likely need the original
DICOM -- the script will tell you what it could and could not find).
"""
import glob, os
import numpy as np
import nrrd

D = "/media/yuankai/3.5T/NN-Unet/nnUNet-master/DATASET1/LHCH_for_Dataset018_inference"
IMG_DIR = f"{D}/imagesTs"

def main():
    files = sorted(glob.glob(f"{IMG_DIR}/RBQ*_0000.nrrd"))
    print(f"total volumes: {len(files)}\n")

    xy_sp, z_sp, nx, ny, nz = [], [], [], [], []
    hu_min, hu_max = [], []
    header_keys_seen = set()

    for f in files:
        data, hdr = nrrd.read(f)
        if "space directions" in hdr:
            sd = np.array(hdr["space directions"], dtype=float)
            sp = [float(np.linalg.norm(sd[i])) for i in range(3)]
            xy_sp.append(round(sp[0], 3)); z_sp.append(round(sp[2], 3))
        nx.append(data.shape[0]); ny.append(data.shape[1]); nz.append(data.shape[2])
        hu_min.append(float(data.min())); hu_max.append(float(data.max()))
        header_keys_seen.update(hdr.keys())

    def rng(a):
        a = np.array(a)
        return f"{a.min():.3f}-{a.max():.3f} (median {np.median(a):.3f})"

    print("=== Acquisition parameters (from nrrd headers) ===")
    print(f"in-plane spacing (mm): {rng(xy_sp)}")
    print(f"slice thickness (mm) : {rng(z_sp)}")
    print(f"matrix X             : {min(nx)}-{max(nx)} (median {int(np.median(nx))})")
    print(f"matrix Y             : {min(ny)}-{max(ny)} (median {int(np.median(ny))})")
    print(f"num slices           : {min(nz)}-{max(nz)} (median {int(np.median(nz))})")
    print(f"HU range across all  : [{min(hu_min):.0f}, {max(hu_max):.0f}]")

    print("\n=== Unique spacing values (to report range) ===")
    from collections import Counter
    print("in-plane:", dict(sorted(Counter(xy_sp).items())))
    print("z-thick :", dict(sorted(Counter(z_sp).items())))

    print("\n=== nrrd header fields available ===")
    print(sorted(header_keys_seen))
    print("\nNOTE: vendor / reconstruction kernel are typically NOT stored in")
    print("nrrd. If you need them, read the ORIGINAL DICOM tags:")
    print("  (0008,0070) Manufacturer, (0008,1090) ManufacturerModelName,")
    print("  (0018,1210) ConvolutionKernel, (0018,0050) SliceThickness,")
    print("  (0018,0060) KVP, (0018,1150) ExposureTime.")

if __name__ == "__main__":
    main()
    