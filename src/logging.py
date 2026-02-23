from __future__ import annotations

import datetime as dt
import json
import os
import platform
import socket
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml


def _short_git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        return out
    except Exception:
        return "unknown"


def _cuda_driver_version() -> str:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if not out:
            return "unknown"
        return out.splitlines()[0]
    except Exception:
        return "unknown"


def build_run_id() -> str:
    utc = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    commit = _short_git_commit()[:7]
    return f"{utc}-{commit}"


def prepare_run_dir(base_dir: str | Path, run_id: str) -> Path:
    run_dir = Path(base_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "plots").mkdir(exist_ok=True)
    return run_dir


def write_resolved_config(run_dir: Path, config: dict[str, Any]) -> Path:
    out_path = run_dir / "config_resolved.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return out_path


def runtime_metadata(
    model_cfg: dict[str, Any],
    hw_cfg: dict[str, Any],
    generation_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gpu_name = "cpu"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    meta: dict[str, Any] = {
        "git_commit": _short_git_commit(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "gpu_name": gpu_name,
        "driver": _cuda_driver_version(),
        "cuda": torch.version.cuda or "none",
        "torch_version": torch.__version__,
        "model_name": model_cfg.get("name_or_path"),
        "model_revision": model_cfg.get("revision", "main"),
        "dtype": model_cfg.get("dtype"),
        "device": hw_cfg.get("device"),
        "max_length_hint": model_cfg.get("max_length_hint"),
        "notes": None,
        # reserved for Week 2+ quant logs
        "w_bits": None,
        "a_bits": None,
        "group_size": None,
        "per_channel": None,
        "quant_backend": None,
        "calib_method": None,
    }
    if generation_cfg:
        meta["generation_settings"] = {
            k: v for k, v in generation_cfg.items() if v is not None
        }
    return meta


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + os.linesep)
