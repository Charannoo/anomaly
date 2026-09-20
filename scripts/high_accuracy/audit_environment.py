"""H0.1 — Hardware & software environment audit for the XMV-AD-H track.

Detects CPU, RAM, GPU, CUDA, VRAM, PyTorch, Python and records a stable
JSON snapshot at experiments/high_accuracy/environment.json.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
assert REPO.name == "xmv-ad", REPO

ENV = {}


def _sysinfo() -> None:
    ENV["os"] = platform.system()
    ENV["os_release"] = platform.release()
    ENV["os_version"] = platform.version()
    ENV["machine"] = platform.machine()
    ENV["processor"] = platform.processor()
    ENV["python"] = platform.python_version()
    ENV["python_implementation"] = platform.python_implementation()
    ENV["hostname"] = platform.node()


def _cpu() -> None:
    try:
        import psutil
        ENV["cpu_physical_cores"] = psutil.cpu_count(logical=False)
        ENV["cpu_logical_cores"] = psutil.cpu_count(logical=True)
        vm = psutil.virtual_memory()
        ENV["ram_total_gb"] = round(vm.total / 1e9, 2)
        ENV["ram_available_gb"] = round(vm.available / 1e9, 2)
    except Exception as e:  # pragma: no cover
        ENV["cpu_error"] = repr(e)


def _gpu() -> None:
    import torch
    ENV["torch_version"] = torch.__version__
    ENV["cuda_available"] = bool(torch.cuda.is_available())
    ENV["cuda_version"] = torch.version.cuda
    if torch.cuda.is_available():
        ENV["gpu_count"] = torch.cuda.device_count()
        devs = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            devs.append({
                "index": i,
                "name": p.name,
                "total_memory_mb": round(p.total_memory / 1e6, 1),
                "capability": f"{p.major}.{p.minor}",
            })
        ENV["gpus"] = devs
        ENV["cuda_device_name"] = torch.cuda.get_device_name(0)
    else:
        ENV["gpus"] = []


def _toolchain() -> None:
    mods = ["numpy", "onnx", "onnxruntime", "openvino", "nncf", "torchvision",
            "transformers", "timm", "einops", "scipy", "sklearn", "matplotlib", "opencv-python"]
    out = {}
    for m in mods:
        try:
            mod = __import__(m)
            out[m] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[m] = None
    ENV["modules"] = out
    try:
        import torch
        ENV["torch_build_config"] = {
            "cuda_build": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "debug": torch.version.debug,
        }
    except Exception as e:
        ENV["torch_build_config"] = {"error": repr(e)}
    try:
        r = subprocess.run(["git", "--version"], capture_output=True, text=True)
        ENV["git"] = r.stdout.strip() or r.stderr.strip()
    except Exception:
        ENV["git"] = None


def main() -> int:
    _sysinfo()
    _cpu()
    _gpu()
    _toolchain()
    ENV["recorded_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = REPO / "experiments" / "high_accuracy"
    out.mkdir(parents=True, exist_ok=True)
    (out / "environment.json").write_text(json.dumps(ENV, indent=2, default=str), encoding="utf-8")
    print(json.dumps(ENV, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())