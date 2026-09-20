"""Reproducible CPU latency + memory benchmark (Phase 6, milestones 8-12).

Measures, per category, per backend:
  - model-only latency  : exported neural graph only (runtime run/infer)
  - end-to-end latency  : model-only + post-processing (calibration, 50/50
    fusion with validity fallback, image score = max) on identical inputs
  - peak RSS during each phase (psutil)

Protocol: 1 thread unless --threads; warmup 50 runs then 500 timed runs;
report p50/p90/p95/p99 and FPS. Fixed random inputs (seeded). Conversion
time is never included. Same machine for all backends so the Pareto plot
is comparable.

Usage:
  python scripts/benchmark_cpu.py --backend onnx --categories bagel rope ...
  python scripts/benchmark_cpu.py --backend openvino_int8 --threads 1
Writes experiments/tables/deployment_benchmark.csv (append) + per-backend JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd
import psutil
import torch

from xmvad.deployment.model_wrapper import build_deployment_e4
from xmvad.fusion.score_fusion import calibrate
from xmvad.metrics.scores import variant_scores
from xmvad.utils.config import load_config


def make_inputs(seed: int, batch: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = np.random.default_rng(seed)
    rgb = g.standard_normal((batch, 3, 256, 256)).astype(np.float32)
    depth = g.standard_normal((batch, 1, 256, 256)).astype(np.float32) * 0.5
    valid = (g.uniform(0, 1, (batch, 1, 256, 256)) > 0.1).astype(np.float32)
    return rgb, depth, valid


class Runner:
    def __init__(self, backend: str, cfg: dict, threads: int) -> None:
        self.backend = backend
        self.cfg = cfg
        self.threads = threads
        self._ort = None
        self._ov = None
        self._ov_prop = None

    def load(self, cat: str) -> None:
        import openvino as ov
        if self.backend.startswith("openvino"):
            if self._ov is None:
                self._ov = ov.Core()
                try:
                    self._ov.set_property("CPU", {"INFERENCE_NUM_THREADS": str(self.threads),
                                                  "NUM_STREAMS": "1"})
                except Exception:
                    pass
            ir = REPO_ROOT / "deployment" / ("openvino_int8" if self.backend == "openvino_int8"
                                             else "openvino_fp32")
            self._compiled = self._ov.compile_model(str(ir / f"{cat}.xml"), "CPU")
        elif self.backend == "onnx":
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = self.threads
            so.inter_op_num_threads = 1
            self._ort = ort.InferenceSession(str(REPO_ROOT / "deployment" / "onnx" / f"{cat}.onnx"),
                                             sess_options=so, providers=["CPUExecutionProvider"])
        elif self.backend == "pytorch":
            r = sorted((REPO_ROOT / "checkpoints" / "E1_baseline_rgb").glob(f"{cat}_best.pt"))[0]
            d = sorted((REPO_ROOT / "checkpoints" / "E2_baseline_depth").glob(f"{cat}_best.pt"))[0]
            f = sorted((REPO_ROOT / "checkpoints" / "E4_concat_fusion").glob(f"{cat}_best.pt"))[0]
            self._model = build_deployment_e4(self.cfg, r, d, f, "cpu").eval()
        self._post = json.loads((REPO_ROOT / "experiments" / "tables" / "E4_concat_calibration.json")
                                .read_text(encoding="utf-8"))["categories"][cat]

    def _run_neural(self, rgb: np.ndarray, depth: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.backend.startswith("openvino"):
            o = self._compiled({"rgb": rgb, "depth": depth, "valid": valid})
            return np.asarray(o[0]), np.asarray(o[1])
        if self.backend == "onnx":
            o = self._ort.run(None, {"rgb": rgb, "depth": depth, "valid": valid})
            return np.asarray(o[0]), np.asarray(o[1])
        with torch.no_grad():
            o = self._model(torch.from_numpy(rgb), torch.from_numpy(depth), torch.from_numpy(valid))
        return o["A_rgb"].numpy(), o["A_depth"].numpy()

    def _postprocess(self, a_r: np.ndarray, a_d: np.ndarray, valid: np.ndarray, post: dict) -> float:
        zr = calibrate(a_r, post["rgb"]["median"], post["rgb"]["scale"])
        zd = calibrate(a_d, post["depth"]["median"], post["depth"]["scale"])
        v = valid[0, 0] > 0.5
        fused = np.where(v, 0.5 * zr + 0.5 * zd, zr)
        return variant_scores(fused, None)["max"]

    def benchmark(self, cat: str, warmup: int, runs: int) -> dict:
        rgb, depth, valid = make_inputs(0)
        self.load(cat)
        # warmup
        for _ in range(warmup):
            self._run_neural(rgb, depth, valid)
        # model-only
        lat = []
        rss = []
        proc = psutil.Process()
        for _ in range(runs):
            t0 = time.perf_counter()
            self._run_neural(rgb, depth, valid)
            lat.append((time.perf_counter() - t0) * 1000.0)
            rss.append(proc.memory_info().rss / 1e6)
        mo = np.array(lat)
        # end-to-end
        lat_e, rss_e = [], []
        for _ in range(runs):
            t0 = time.perf_counter()
            a_r, a_d = self._run_neural(rgb, depth, valid)
            self._postprocess(a_r, a_d, valid, self._post)
            lat_e.append((time.perf_counter() - t0) * 1000.0)
            rss_e.append(proc.memory_info().rss / 1e6)
        ee = np.array(lat_e)
        pct = lambda a: dict(zip(("p50", "p90", "p95", "p99"),
                                 [float(np.percentile(a, p)) for p in (50, 90, 95, 99)]))
        return {"backend": self.backend, "category": cat, "threads": self.threads,
                "model_only_ms": json.dumps(pct(mo)), "model_only_fps": float(1000.0 / mo.mean()),
                "model_only_ms_mean": float(mo.mean()),
                "e2e_ms": json.dumps(pct(ee)), "e2e_fps": float(1000.0 / ee.mean()),
                "e2e_ms_mean": float(ee.mean()),
                "peak_rss_model_only_mb": float(max(rss)), "peak_rss_e2e_mb": float(max(rss_e))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["pytorch", "onnx", "openvino_fp32", "openvino_int8"])
    ap.add_argument("--config", default="configs/concat_fusion.yaml")
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--runs", type=int, default=500)
    args = ap.parse_args()

    cfg = load_config(args.config)
    from xmvad.data.mvtec3d import EXPECTED_CATEGORIES
    cats = args.categories or list(EXPECTED_CATEGORIES)
    torch.set_num_threads(args.threads)

    runner = Runner(args.backend, cfg, args.threads)
    rows = []
    for cat in cats:
        row = runner.benchmark(cat, args.warmup, args.runs)
        rows.append(row)
        print(f"{args.backend}/{cat}: model p50={row['model_only_ms']} e2e p95={json.loads(row['e2e_ms'])['p95']:.2f} ms",
              flush=True)

    df = pd.DataFrame(rows)
    out_csv = REPO_ROOT / "experiments" / "tables" / "deployment_benchmark.csv"
    if out_csv.exists():
        old = pd.read_csv(out_csv)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(out_csv, index=False)
    (REPO_ROOT / "experiments" / "tables" / f"benchmark_{args.backend}.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    print(f"appended {len(rows)} rows -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())