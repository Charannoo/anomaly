"""Environment capture for experiment tracking (Phase 1: minimal)."""

from __future__ import annotations

import platform
import sys
from typing import Any


def collect_env(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Collect Python/OS/CPU (+ torch if installed) metadata."""
    env: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "os": platform.system(),
        "cpu": platform.processor() or platform.machine(),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        env["torch"] = None
        env["cuda_available"] = False
    try:
        import psutil

        env["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 2)
        env["cpu_count"] = psutil.cpu_count(logical=True)
    except ImportError:
        pass
    if extra:
        env.update(extra)
    return env
