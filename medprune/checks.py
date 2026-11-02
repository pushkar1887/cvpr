"""Stage 3 (E0) checks: is the GPU pipeline numerically sound, is logging inert, do offline keep-sets match?

    python -m medprune.checks --model lingshu7b --bundle $DATA --out $RES --n 50

Writes $RES/e0_report__<model>__<profile>.json with one entry per check and an overall "pass" (plan.md Stage 3).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from .data import Sample, load_manifest
from .run_eval import RunSpec, build_state, bundle_for
from .scoring import is_correct
from .selection import METHODS, keep_count, select_topk


def _generate(adapter, samples: list[Sample], root: Path, method: str | None = None, budget: float = 0.10):
    """Answers (and the states used) for each sample; method None = no pruning state at all."""
    from .attention_hook import pruning

    results = []
    for s in samples:
        inputs = adapter.prepare(s, Image.open(root / s.image_path).convert("RGB"))
        state = None
        if method is not None:
            spec = RunSpec(adapter.name, s.dataset, method, 1.0 if METHODS[method].rule == "none" else budget)
            state = build_state(spec, METHODS[method], s, inputs, None, None, adapter, None)
        with pruning(state):
            text, _ = adapter.generate(inputs, 64)
        results.append((text, state))
    return results


def fp16_check(adapter, samples: list[Sample], root: Path) -> dict:
    """E0b: answers in fp16 vs with the vision tower in fp32; next-token logits finite in fp16."""
    fp16 = [t for t, _ in _generate(adapter, samples, root)]
    finite = all(np.isfinite(adapter.last_logits(adapter.prepare(s, Image.open(root / s.image_path).convert("RGB")))).all()
                 for s in samples[:10])
    adapter.set_vision_fp32(True)
    try:
        fp32 = [t for t, _ in _generate(adapter, samples, root)]
    finally:
        adapter.set_vision_fp32(False)
    same = sum(a == b for a, b in zip(fp16, fp32))
    differing = [{"qid": s.qid, "fp16": a, "vision_fp32": b} for s, a, b in zip(samples, fp16, fp32) if a != b][:5]
    return {"n": len(samples), "identical": same, "logits_finite": bool(finite), "differing_examples": differing,
            "pass": bool(finite and same >= 0.96 * len(samples))}


def logging_is_inert(adapter, samples: list[Sample], root: Path) -> dict:
    """E0c: answers with the `full` logging state equal answers with no state."""
    plain = [t for t, _ in _generate(adapter, samples, root)]
    logged = [t for t, _ in _generate(adapter, samples, root, method="full")]
    same = sum(a == b for a, b in zip(plain, logged))
    return {"n": len(samples), "identical": same, "pass": bool(same >= 0.98 * len(samples))}


def offline_keepsets_match(adapter, samples: list[Sample], root: Path, budget: float = 0.10) -> dict:
    """E0d: FastV keep-sets recomputed from the `full` run's logged layer-1 scores equal the keep-sets FastV chose.

    This licenses every Stage 4 analysis done from logs. Also reports FastV accuracy at 50% and at `budget` next to
    Full on these samples (information only; 50 questions are too few to gate on).
    """
    full = _generate(adapter, samples, root, method="full")
    fastv = _generate(adapter, samples, root, method="fastv", budget=budget)
    fastv50 = _generate(adapter, samples, root, method="fastv", budget=0.50)
    matches = 0
    for (_, full_state), (_, fastv_state) in zip(full, fastv):
        scores = full_state.logs["scores_l1"]
        offline = select_topk(scores, keep_count(len(scores), budget))
        matches += bool(np.array_equal(offline, fastv_state.keep))

    def accuracy(results) -> float:
        return 100.0 * float(np.mean([is_correct(s, t)[0] for s, (t, _) in zip(samples, results)]))

    return {"n": len(samples), "keepsets_identical": matches, "accuracy_full": accuracy(full),
            f"accuracy_fastv_{round(budget * 100)}": accuracy(fastv), "accuracy_fastv_50": accuracy(fastv50),
            "pass": bool(matches >= 0.98 * len(samples))}


def timing_summary(run_jsonl: Path) -> dict:
    """E0f: seconds per question of a finished run."""
    if not run_jsonl.exists():
        return {"run": run_jsonl.name, "available": False}
    times = [json.loads(line)["elapsed_s"] for line in run_jsonl.read_text(encoding="utf-8").splitlines() if line]
    return {"run": run_jsonl.name, "available": True, "n": len(times), "mean_s": float(np.mean(times)),
            "p50_s": float(np.percentile(times, 50)), "p95_s": float(np.percentile(times, 95))}


def run_checks(adapter, samples: list[Sample], root: Path, out: Path, profile: str) -> dict:
    report = {
        "model": adapter.name, "profile": profile, "dataset": samples[0].dataset if samples else None,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "E0b_fp16": fp16_check(adapter, samples, root),
        "E0c_logging_inert": logging_is_inert(adapter, samples, root),
        "E0d_offline_keepsets": offline_keepsets_match(adapter, samples, root),
    }
    if samples:
        run_id = RunSpec(adapter.name, samples[0].dataset, "full").run_id(profile)
        report["E0f_timing"] = timing_summary(out / "runs" / f"{run_id}.jsonl")
    report["pass"] = all(report[k]["pass"] for k in ("E0b_fp16", "E0c_logging_inert", "E0d_offline_keepsets"))
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"e0_report__{adapter.name}__{profile}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    from .env import data_root, detect_profile, results_root
    from .models import load_adapter

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--bundle", type=Path, action="append", default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--dataset", default="slake_en")
    p.add_argument("--n", type=int, default=50)
    args = p.parse_args()

    profile = detect_profile()
    root = bundle_for(args.bundle or [data_root()], args.dataset)
    samples = load_manifest(root, args.dataset)[: args.n]
    adapter = load_adapter(args.model, profile)
    report = run_checks(adapter, samples, root, args.out or results_root(), profile.name)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
