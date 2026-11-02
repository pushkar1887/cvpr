#!/usr/bin/env bash
# Same environment on Kaggle and on SSH GPU servers (plan.md, Part E).
#   Kaggle notebook cell:  !bash scripts/setup_env.sh
#   SSH server (in a venv): bash scripts/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."

python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

python - <<'PY'
import os

import torch
import transformers

print(f"torch {torch.__version__} | CUDA {torch.version.cuda} | transformers {transformers.__version__}")
gpus = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
for i, p in enumerate(gpus):
    print(f"GPU {i}: {p.name}, {p.total_memory / 2**30:.1f} GiB, compute capability {p.major}.{p.minor}")

# Must match medprune/env.py:detect_profile().
forced = os.environ.get("MEDPRUNE_PROFILE")
if forced:
    profile = forced
elif gpus and gpus[0].major >= 8 and gpus[0].total_memory >= 40 * 2**30:
    profile = "a100"   # bf16, single GPU
elif len(gpus) == 2 and all("T4" in p.name for p in gpus):
    profile = "t4x2"   # fp16, model split over both GPUs
else:
    profile = "unsupported"
print(f"profile: {profile}  (override with MEDPRUNE_PROFILE)")
PY
