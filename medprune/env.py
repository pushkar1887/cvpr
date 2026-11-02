"""Compute profile and paths, so the same code runs on Kaggle T4x2, an SSH A100, or a laptop CPU (plan.md Part E)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Profile:
    name: str  # "t4x2" | "a100" | "cpu"
    dtype: str  # torch dtype name
    device_map: str | dict
    max_memory: dict | None


PROFILES = {
    "t4x2": Profile("t4x2", "float16", "auto", {0: "14GiB", 1: "14GiB"}),
    "a100": Profile("a100", "bfloat16", {"": 0}, None),
    "cpu": Profile("cpu", "float32", {"": "cpu"}, None),
}


def on_kaggle() -> bool:
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def detect_profile() -> Profile:
    """MEDPRUNE_PROFILE overrides; otherwise A100-class (sm >= 8.0, >= 40 GB) → a100, two T4s → t4x2, else cpu.

    Must agree with scripts/setup_env.sh.
    """
    forced = os.environ.get("MEDPRUNE_PROFILE")
    if forced:
        return PROFILES[forced]
    import torch

    if not torch.cuda.is_available():
        return PROFILES["cpu"]
    gpus = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
    if gpus[0].major >= 8 and gpus[0].total_memory >= 40 * 2**30:
        return PROFILES["a100"]
    if len(gpus) == 2 and all("T4" in g.name for g in gpus):
        return PROFILES["t4x2"]
    raise RuntimeError(f"unsupported GPUs {[g.name for g in gpus]}; set MEDPRUNE_PROFILE explicitly")


def torch_dtype(profile: Profile):
    import torch

    return getattr(torch, profile.dtype)


def data_root() -> Path:
    if "MEDPRUNE_DATA" in os.environ:
        return Path(os.environ["MEDPRUNE_DATA"])
    return Path("/kaggle/input/medprune-data") if on_kaggle() else Path.home() / "medprune-data"


def results_root() -> Path:
    if "MEDPRUNE_RESULTS" in os.environ:
        return Path(os.environ["MEDPRUNE_RESULTS"])
    return Path("/kaggle/working/results") if on_kaggle() else Path("results")
