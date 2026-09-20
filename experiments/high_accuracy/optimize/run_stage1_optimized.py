import argparse
import os
import sys
import time
import logging
import json
import hashlib
import torch
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from tqdm import tqdm

REPO = Path("/opt/m3dm_ws/m3dm")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.chdir(str(REPO))

import lean_coreset
lean_coreset.apply_patch()

import pandas as pd
from m3dm_runner import M3DM
from dataset import get_data_loader, TrainDataset, TestDataset, DataLoader, mvtec3d_classes
import dataset as ds_module

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("stage1_opt")

TRACKER = Path("/opt/preprocess_tracker.txt")
MV = Path("/opt/mvtec3d")
CACHE_DIR = Path("/opt/tmp/feature_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
TIMING_CSV = Path("/opt/tmp/H0R_timing_breakdown.csv")


def class_tiffs_ready(class_name):
    done = set()
    with open(TRACKER) as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(line)
    tiffs = set(str(p) for p in (MV / class_name).rglob("*.tiff"))
    ready = tiffs <= done
    missing = tiffs - done
    return ready, len(tiffs), len(done & tiffs), sorted(missing)[:3]


def compute_cache_key(class_name, sample_idx, args):
    m = hashlib.sha256()
    m.update(class_name.encode())
    m.update(str(sample_idx).encode())
    m.update(str(args.img_size).encode())
    m.update(str(args.group_size).encode())
    m.update(str(args.num_group).encode())
    m.update(str(args.method_name).encode())
    m.update(str(args.random_state).encode() if args.random_state else b"none")
    return m.hexdigest()


def check_feature_cache(cache_key):
    cache_path = CACHE_DIR / f"{cache_key}.pt"
    if cache_path.exists():
        return True, cache_path
    return False, cache_path


def save_feature_cache(cache_key, feature_tensor):
    cache_path = CACHE_DIR / f"{cache_key}.pt"
    torch.save(feature_tensor, cache_path)


def load_feature_cache(cache_key):
    cache_path = CACHE_DIR / f"{cache_key}.pt"
    return torch.load(cache_path)


def load_models_once(method):
    t0 = time.perf_counter()
    method.eval()
    t1 = time.perf_counter()
    log.info(f"Model eval() set. Load time: {t1-t0:.3f}s")
    return t1 - t0


class TimedStageRunner:
    def __init__(self, args):
        self.args = args
        self.stage_timings = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.model_load_time = 0

    def stage_feature_extraction_train(self, args):
        t0 = time.perf_counter()
        from feature_extractors.multiple_features import DoubleRGBPointFeatures

        feat_args = argparse.Namespace(**vars(args))
        method = DoubleRGBPointFeatures(feat_args)
        load_models_once(method)

        train_loader = ds_module.get_data_loader("train", class_name=args.class_name, img_size=args.img_size, args=args)
        flag = 0
        for sample_idx, (sample, _) in enumerate(train_loader):
            if flag > args.max_sample:
                flag = 0
                break
            cache_key = compute_cache_key(args.class_name, sample_idx, args)
            cached, cache_path = check_feature_cache(cache_key)
            if cached and args.use_cache:
                self.cache_hits += 1
                patch_lib = load_feature_cache(cache_key)
                method.patch_lib.append(patch_lib)
            else:
                self.cache_misses += 1
                method.add_sample_to_mem_bank(sample)
                if args.save_feature and len(method.patch_lib) > 0:
                    save_feature_cache(cache_key, method.patch_lib[-1])
            flag += 1

        t1 = time.perf_counter()
        self.stage_timings["feature_extraction_train"] = t1 - t0
        self.feature_method = method
        self.train_loader = train_loader
        return method

    def stage_coreset(self, method):
        t0 = time.perf_counter()
        for method_name, m in method.methods.items() if hasattr(method, 'methods') else [("DINO+Point_MAE", method)]:
            log.info(f"Running coreset for {method_name} on class {args.class_name}...")
            m.run_coreset()
        t1 = time.perf_counter()
        self.stage_timings["coreset"] = t1 - t0
        log.info(f"Coreset time: {self.stage_timings['coreset']:.3f}s")

    def stage_late_fusion(self, method):
        t0 = time.perf_counter()
        flag = 0
        for sample, _ in tqdm(self.train_loader, desc=f'Running late fusion for {args.class_name}'):
            for method_name, m in self.method.methods.items() if hasattr(self.method, 'methods') else [("DINO+Point_MAE", method)]:
                m.add_sample_to_late_fusion_mem_bank(sample)
                flag += 1
            if flag > args.max_sample:
                flag = 0
                break
        for method_name, m in self.method.methods.items() if hasattr(self.method, 'methods') else [("DINO+Point_MAE", method)]:
            log.info(f'Training Decision Layer Fusion for {method_name} on class {args.class_name}...')
            m.run_late_fusion()
        t1 = time.perf_counter()
        self.stage_timings["late_fusion"] = t1 - t0
        log.info(f"Late fusion time: {self.stage_timings['late_fusion']:.3f}s")

    def stage_evaluate(self, args):
        t0 = time.perf_counter()
        path_list = []
        test_loader = ds_module.get_data_loader("test", class_name=args.class_name, img_size=args.img_size, args=args)
        method = self.feature_method
        method.eval()
        with torch.no_grad():
            for sample, mask, label, rgb_path in test_loader:
                for m in [method]:
                    m.predict(sample, mask, label)
                    path_list.append(rgb_path)
        image_rocaucs = {}
        pixel_rocaucs = {}
        au_pros = {}
        for method_name, m in [("DINO+Point_MAE", method)]:
            m.calculate_metrics()
            image_rocaucs[method_name] = round(m.image_rocauc, 3)
            pixel_rocaucs[method_name] = round(m.pixel_rocauc, 3)
            au_pros[method_name] = round(m.au_pro, 3)
        t1 = time.perf_counter()
        self.stage_timings["evaluation"] = t1 - t0
        self.test_total = t1 - t0
        self.image_rocaucs = image_rocaucs
        self.pixel_rocaucs = pixel_rocaucs
        self.au_pros = au_pros
        self.path_list = path_list
        return image_rocaucs, pixel_rocaucs, au_pros

    def write_timing_csv(self):
        rows = []
        total = self.test_total
        for stage_name, stage_time in self.stage_timings.items():
            rows.append({"stage": stage_name, "seconds": round(stage_time, 3), "percentage_of_total": round(100 * stage_time / total if total > 0 else 0, 1)})
        rows.append({"stage": "model_load", "seconds": round(self.model_load_time, 3), "percentage_of_total": round(100 * self.model_load_time / total if total > 0 else 0, 1)})
        rows.append({"stage": "io_total", "seconds": round(sum(self.stage_timings.values()), 3), "percentage_of_total": round(100 * sum(self.stage_timings.values()) / total if total > 0 else 0, 1)})
        df = pd.DataFrame(rows)
        df.to_csv(TIMING_CSV, mode='a', header=not TIMING_CSV.exists(), index=False)
        return rows

    def run(self, args):
        overall_t0 = time.perf_counter()
        self.args = args

        env_info = {
            "class_name": args.class_name,
            "method": args.method_name,
            "num_group": args.num_group,
            "num_workers": args.num_workers,
            "dino_batch_size": args.dino_batch_size,
            "use_cache": args.use_cache,
            "torch_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
        }
        log.info(f"Environment: {json.dumps(env_info)}")

        if args.stage in ["all", "features"]:
            method = self.stage_feature_extraction_train(args)
            self.model_load_time = 0

        if args.stage in ["all", "coreset"]:
            self.stage_coreset(method)

        if args.memory_bank == "multiple" and args.stage in ["all", "score"]:
            self.stage_late_fusion(method)

        if args.stage in ["all", "eval"]:
            self.stage_evaluate(args)

        self.write_timing_csv()
        total_time = time.perf_counter() - overall_t0
        log.info(f"Total wall-clock time: {total_time:.3f}s")

        summary = {
            "class": args.class_name,
            "total_seconds": round(total_time, 3),
            "stages": {k: round(v, 3) for k, v in self.stage_timings.items()},
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "num_workers": args.num_workers,
            "dino_batch_size": args.dino_batch_size,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method_name", default="DINO+Point_MAE", type=str)
    parser.add_argument("--max_sample", default=400, type=int)
    parser.add_argument("--memory_bank", default="multiple", type=str)
    parser.add_argument("--rgb_backbone_name", default="vit_base_patch8_224_dino", type=str)
    parser.add_argument("--xyz_backbone_name", default="Point_MAE", type=str)
    parser.add_argument("--fusion_module_path", default="checkpoints/checkpoint-0.pth", type=str)
    parser.add_argument("--save_feature", default=False, action="store_true")
    parser.add_argument("--use_uff", default=False, action="store_true")
    parser.add_argument("--save_feature_path", default="datasets/patch_lib", type=str)
    parser.add_argument("--save_preds", default=False, action="store_true")
    parser.add_argument("--group_size", default=128, type=int)
    parser.add_argument("--num_group", default=1024, type=int)
    parser.add_argument("--random_state", default=None, type=int)
    parser.add_argument("--dataset_type", default="mvtec3d", type=str)
    parser.add_argument("--dataset_path", default="/opt/mvtec3d", type=str)
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--xyz_s_lambda", default=1.0, type=float)
    parser.add_argument("--xyz_smap_lambda", default=1.0, type=float)
    parser.add_argument("--rgb_s_lambda", default=0.1, type=float)
    parser.add_argument("--rgb_smap_lambda", default=0.1, type=float)
    parser.add_argument("--fusion_s_lambda", default=1.0, type=float)
    parser.add_argument("--fusion_smap_lambda", default=1.0, type=float)
    parser.add_argument("--coreset_eps", default=0.9, type=float)
    parser.add_argument("--f_coreset", default=0.1, type=float)
    parser.add_argument("--asy_memory_bank", default=None, type=int)
    parser.add_argument("--ocsvm_nu", default=0.5, type=float)
    parser.add_argument("--ocsvm_maxiter", default=1000, type=int)
    parser.add_argument("--rm_zero_for_project", default=False, action="store_true")
    parser.add_argument("--class_name", required=True, type=str)
    parser.add_argument("--wait_until_ready", default=False, action="store_true")
    parser.add_argument("--out_csv", default=None, type=str)
    parser.add_argument("--num_workers", default=0, type=int, help="DataLoader workers")
    parser.add_argument("--dino_batch_size", default=1, type=int, help="Batch size for DINO extraction")
    parser.add_argument("--use_cache", default=True, action="store_true", help="Use feature cache")
    parser.add_argument("--stage", default="all", type=str, choices=["all", "features", "coreset", "score", "eval"])
    parser.add_argument("--enable_inference_mode", default=True, action="store_true")
    args = parser.parse_args()

    # Patch DataLoader num_workers
    original_init = TrainDataset.__init__
    def patched_train_init(self, *a, **kw):
        original_init(self, *a, **kw)
        self.loader_workers = args.num_workers

    ds_module.get_data_loader = lambda split, class_name, img_size, args: (
        DataLoader(
            dataset=(TrainDataset(class_name=class_name, img_size=img_size, dataset_path=args.dataset_path) if split == "train" else TestDataset(class_name=class_name, img_size=img_size, dataset_path=args.dataset_path)),
            batch_size=1, shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True
        )
    )

    if args.wait_until_ready:
        while True:
            ready, total, ndone, missing = class_tiffs_ready(args.class_name)
            if ready:
                log.info("class %s fully preprocessed (%d tiffs)", args.class_name, total)
                break
            log.info("waiting... %s: %d/%d done (missing e.g. %s)", args.class_name, ndone, total, missing[:2])
            time.sleep(300)

    runner = TimedStageRunner(args)
    summary = runner.run(args)

    if args.out_csv:
        out = Path(args.out_csv)
        df = pd.DataFrame([{
            "class": summary["class"],
            "method": args.method_name,
            "image_rocauc": summary.get("image_rocauc", 0),
            "pixel_rocauc": summary.get("pixel_rocauc", 0),
            "au_pro": summary.get("au_pro", 0),
            "total_seconds": summary["total_seconds"],
            "cache_hits": summary["cache_hits"],
            "cache_misses": summary["cache_misses"],
        }])
        if out.exists():
            pd.concat([pd.read_csv(out), df], ignore_index=True).to_csv(out, index=False)
        else:
            df.to_csv(out, index=False)
        log.info(f"Appended row to {args.out_csv}")

    log.info(f"Optimization summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
