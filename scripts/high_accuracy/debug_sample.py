import os
import sys
import glob
import traceback
import torch

sys.path.insert(0, '/mnt/c/Users/CharanOp/xmv-ad')
from scripts.high_accuracy.extract_canonical_features import CanonicalExtractor

def main():
    extractor = CanonicalExtractor(device="cuda")
    print("Extractor initialized.")
    
    cat = "peach"
    cat_dir = os.path.join("/opt/mvtec3d", cat)
    out_cat_dir = os.path.join("/opt/xmv_h_fast/canonical", cat)
    
    for split in ["train", "validation", "test"]:
        split_dir = os.path.join(cat_dir, split)
        if not os.path.exists(split_dir):
            continue
        for defect in sorted(os.listdir(split_dir)):
            d_dir = os.path.join(split_dir, defect)
            if not os.path.isdir(d_dir):
                continue
            rgb_files = sorted(glob.glob(os.path.join(d_dir, "rgb", "*.png")))
            tiff_files = sorted(glob.glob(os.path.join(d_dir, "xyz", "*.tiff")))
            gt_files = sorted(glob.glob(os.path.join(d_dir, "gt", "*.png"))) if split == "test" and defect != "good" else []
            
            label = 0 if defect == "good" else 1
            for i in range(len(rgb_files)):
                rgb_p = rgb_files[i]
                tiff_p = tiff_files[i]
                gt_p = gt_files[i] if i < len(gt_files) else None
                sname = os.path.splitext(os.path.basename(rgb_p))[0]
                sid = f"{defect}_{sname}" if defect != "good" else sname
                out_p = os.path.join(out_cat_dir, split, f"{sid}.pt")
                if os.path.exists(out_p) and os.path.getsize(out_p) > 10000:
                    continue
                print(f"Testing sample: {split}/{defect}/{sname} ({rgb_p})", flush=True)
                try:
                    meta = {"sample_id": sid, "category": cat, "split": split, "defect_type": defect}
                    sd, timing = extractor.extract_sample(rgb_p, tiff_p, gt_p, label, meta)
                    print(f"Success! Time: {timing['total']:.3f}s", flush=True)
                except Exception as e:
                    print(f"FAILED on sample {sid}!")
                    traceback.print_exc()
                    return

if __name__ == "__main__":
    main()
