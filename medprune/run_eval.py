"""Run evaluations. One run = model x dataset x method x budget x seed (plan.md §B0, §C6, §D3).

    python -m medprune.run_eval --list --exp E1 --model lingshu7b          # runs + estimated hours, runs nothing
    python -m medprune.run_eval --exp E1 --model lingshu7b --limit 5       # smoke test, 5 questions per run
    python -m medprune.run_eval --exp E1 --model lingshu7b --max-hours 6   # every pending run of the block
    python -m medprune.run_eval --dry-run --exp E1 --model lingshu7b --limit 20   # CPU, fake model: tests the loop

Run IDs do not contain the experiment: identical computations share one ID and are computed once (E1's runs are
reused by E3). Each run appends one JSON line per question and resumes where it stopped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from .data import Sample, load_manifest
from .masks import MaskStore
from .scoring import is_correct
from .selection import (METHODS, MethodSpec, keep_count, pdrop_schedule, rng_for, sample_map, select_divprune,
                        select_inverse, select_random, select_topk, select_uniform, select_uniform_quota)

LOG_LAYERS = (1, 7, 15)  # full runs log FastV scores here (layers K-1 for K = 2, 8, 16)
OUTLIER_Z = 5.0  # high-norm "artifact" tokens: robust z-score above this (plan.md §C7)


# ----------------------------------------------------------------------------------------------- run specs

@dataclass(frozen=True)
class RunSpec:
    model: str
    dataset: str
    method: str
    budget: float = 1.0
    seed: int = 0
    exps: tuple[str, ...] = field(default=(), compare=False)

    def run_id(self, profile: str) -> str:
        return f"{self.model}__{self.dataset}__{self.method}__r{round(self.budget * 100):03d}__s{self.seed}__{profile}"


def _rate_key(dataset: str) -> str:
    for prefix, key in (("slake", "slake"), ("omni", "omni"), ("rin", "rin"), ("natural", "natural")):
        if dataset.startswith(prefix):
            return key
    raise KeyError(dataset)


def load_runs(path: Path, exp: str | None = None, model: str | None = None, account: str | None = None,
              dataset: str | None = None, method: str | None = None) -> tuple[list[RunSpec], dict]:
    """Expand configs/runs.yaml (matrix form) into unique runs, optionally filtered. Returns (runs, rates)."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    groups, merged = cfg["groups"], {}
    exps = set(exp.split(",")) if exp else None  # "E6,E3": several experiments, in configs/runs.yaml block order
    models = set(model.split(",")) if model else None  # several models load one after another
    methods = set(method.split(",")) if method else None
    for block in cfg["blocks"]:
        if (exps and block["exp"] not in exps) or (models and block["model"] not in models) or \
                (account and block.get("account") != account):
            continue
        datasets = [d for g in block["groups"] for d in groups.get(g, [g])]
        for ds in datasets:
            for m in block["methods"]:
                budgets = [1.0] if METHODS[m].rule == "none" else block.get("budgets", [0.10])
                for b in budgets:
                    for s in block.get("seeds", [0]):
                        spec = RunSpec(block["model"], ds, m, float(b), int(s))
                        if (dataset and ds not in dataset.split(",")) or (methods and m not in methods):
                            continue
                        old = merged.get(spec)
                        merged[spec] = RunSpec(spec.model, ds, m, float(b), int(s),
                                               tuple(sorted(set((old.exps if old else ()) + (block["exp"],)))))
    return list(merged.values()), cfg["rates_s_per_q"]


TWO_PASS_FACTOR = 1.22  # fastv_prellm / fastv, measured on Lingshu (SLAKE 1.30, RIN 1.16; Stage 4)


def _rate(model_rates: dict, dataset: str, text_only: bool) -> float:
    """Seconds per question: a dataset-specific entry if present, else its family's; text-only may be per family."""
    family = _rate_key(dataset)
    if text_only:
        entry = model_rates["text_only"]
        return float(entry.get(dataset, entry.get(family))) if isinstance(entry, dict) else float(entry)
    return float(model_rates.get(dataset, model_rates.get(family)))


def estimate_hours(spec: RunSpec, n_questions: int, rates: dict) -> float:
    rate = _rate(rates[spec.model], spec.dataset, spec.method == "text_only")
    return n_questions * rate * (TWO_PASS_FACTOR if METHODS[spec.method].two_pass else 1.0) / 3600


# ----------------------------------------------------------------------------------------------- outputs

def _done_qids(path: Path, expected_ids: set[str] | None = None) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        done = set()
        for line in f:
            if not line.strip():
                continue
            qid = json.loads(line)["qid"]
            if qid in done:
                raise ValueError(f"Duplicate qid {qid!r} in completed run {path}; refusing unsafe resume")
            done.add(qid)
        if expected_ids is not None and not done <= expected_ids:
            raise ValueError(f"Run {path} contains question IDs outside the current manifest; "
                             "preserve or recover the old run before resuming")
        return done


class DiagWriter:
    """Buffers per-question diagnostic arrays and writes them in shards of 200 questions."""

    def __init__(self, directory: Path, shard_size: int = 200):
        self.directory, self.shard_size, self.rows = directory, shard_size, []
        directory.mkdir(parents=True, exist_ok=True)
        self.next_part = len(list(directory.glob("part-*.npz")))

    def add(self, qid: str, logs: dict[str, np.ndarray]) -> None:
        if logs:
            self.rows.append((qid, logs))
        if len(self.rows) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        keys = sorted({k for _, logs in self.rows for k in logs})
        arrays: dict[str, np.ndarray] = {"qid": np.array([q for q, _ in self.rows], dtype=str)}
        for k in keys:
            parts = [np.asarray(logs.get(k, np.zeros(0)), dtype=np.float16) for _, logs in self.rows]
            arrays[k] = np.concatenate(parts) if parts else np.zeros(0, np.float16)
            arrays[f"{k}__offsets"] = np.cumsum([0] + [p.size for p in parts]).astype(np.int64)
        np.savez_compressed(self.directory / f"part-{self.next_part:04d}.npz", **arrays)
        self.next_part, self.rows = self.next_part + 1, []


def _git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


# ----------------------------------------------------------------------------------------------- planning

def _robust_z(values: np.ndarray) -> np.ndarray:
    med = np.median(values)
    mad = np.median(np.abs(values - med)) * 1.4826
    return (values - med) / max(mad, 1e-8)


def build_state(spec: RunSpec, method: MethodSpec, sample: Sample, inputs, gate_frac: np.ndarray | None,
                oracle_frac: np.ndarray | None, adapter, static_map: np.ndarray | None):
    """The PruneState for one question (None = no pruning and no logging)."""
    from .attention_hook import PruneState, Stage

    if not method.image:
        return None
    if method.rule == "none":
        if method.profile:  # every layer, both readouts, sink scores (plan.md §A10 D1)
            return PruneState(vis_pos=inputs.vis_pos, text_pos=inputs.text_pos, profile=True)
        return PruneState(vis_pos=inputs.vis_pos, log_layers=tuple(l for l in LOG_LAYERS if l < adapter.num_layers),
                          log_ropefree=True)
    n = len(inputs.vis_pos)
    k = keep_count(n, spec.budget)
    cells = inputs.cell_index
    allowed = None
    if method.gate == "foreground":
        allowed = gate_frac.reshape(-1)[cells] >= method.tau
    elif method.gate == "oracle":
        allowed = oracle_frac.reshape(-1)[cells] >= 0.5
    if method.exclude_outliers:
        not_outlier = _robust_z(np.linalg.norm(adapter.visual_embeddings(inputs), axis=1)) <= OUTLIER_Z
        allowed = not_outlier if allowed is None else allowed & not_outlier

    state = PruneState(vis_pos=inputs.vis_pos, text_pos=inputs.text_pos)
    rule = method.rule
    if rule == "random":
        state.apply_keep(select_random(n, k, rng_for(spec.seed, sample.qid), allowed), method.mask_from)
    elif rule == "uniform":
        if method.bg_quota > 0 and allowed is not None:  # E20: background quota for the coverage selector
            keep = select_uniform_quota(n, k, allowed, keep_count(n, method.bg_quota))
        else:
            keep = select_uniform(n, k, allowed)
        state.apply_keep(keep, method.mask_from)
    elif rule == "inverse":
        state.apply_keep(select_inverse(allowed, k), method.mask_from)
    elif rule == "static":
        if static_map is None:
            raise FileNotFoundError("static_map needs results/static_maps/<model>__for_<dataset>.npy (Stage 6)")
        scores = sample_map(static_map, inputs.grid_hw)[cells]
        state.apply_keep(select_topk(scores, k, allowed), method.mask_from)
    elif rule == "divprune":
        state.apply_keep(select_divprune(adapter.visual_embeddings(inputs), k, allowed), method.mask_from)
    elif rule == "topk":
        k_bg = keep_count(n, method.bg_quota) if method.bg_quota > 0 else 0
        sign = -1.0 if method.invert else 1.0

        def select(scores: np.ndarray, survivors: np.ndarray) -> np.ndarray:
            scores = sign * np.asarray(scores, dtype=np.float64)
            candidates = survivors if allowed is None else survivors & allowed
            if not k_bg:
                return select_topk(scores, k, candidates)
            taken = np.zeros(n, dtype=bool)  # background quota: evenly spaced background tokens, rest by score
            taken[select_uniform(n, k_bg, survivors & ~allowed)] = True
            rest = select_topk(np.where(taken, -np.inf, scores), k - k_bg, candidates & ~taken)
            return np.union1d(np.flatnonzero(taken), rest)

        state.need_raters = method.score == "raters"  # SparseVLM's raters with one FastV-style stage (E16)
        state.stages.append(Stage(method.score_layer, method.mask_from, select, score_kind=method.score))
    elif rule == "pdrop":
        for boundary, fraction in pdrop_schedule(spec.budget, adapter.num_layers):
            k_stage = keep_count(n, fraction)

            def select(scores, survivors, k_stage=k_stage):
                return select_topk(scores, k_stage, survivors if allowed is None else survivors & allowed)

            state.stages.append(Stage(boundary - 1, boundary, select, score_kind="attn"))
    elif rule in BASELINES:
        if allowed is not None and rule not in GATED_BASELINES:
            raise ValueError(f"{method.name}: this published pruner runs without the gate")
        extra = {"allowed": allowed} if rule in GATED_BASELINES else {}
        BASELINES[rule](state, spec, sample, inputs, adapter, n, k, **extra)
    else:
        raise ValueError(f"unknown rule {rule!r}")
    return state


# --------------------------------------------------------------------------------------- published pruners

def _sparsevlm(state, spec, sample, inputs, adapter, n: int, k: int) -> None:
    from .attention_hook import Stage
    from .baselines.sparsevlm import sparsevlm_schedule, stage_keep

    state.need_raters = True
    for score_layer, mask_from, fraction in sparsevlm_schedule(spec.budget, adapter.num_layers):
        count = keep_count(n, fraction)
        state.stages.append(Stage(score_layer, mask_from, lambda s, surv, count=count: stage_keep(s, surv, count),
                                  score_kind="raters"))


def _vispruner(state, spec, sample, inputs, adapter, n: int, k: int) -> None:
    from .baselines.vispruner import select_vispruner

    importance, feats = adapter.vision_cues(inputs)
    state.apply_keep(select_vispruner(importance, feats, k), 0)


def _prunesid(state, spec, sample, inputs, adapter, n: int, k: int) -> None:
    from .baselines.prunesid import select_prunesid

    seed = int(rng_for(spec.seed, sample.qid).integers(2**31))
    state.apply_keep(select_prunesid(adapter.visual_embeddings(inputs), k, seed), 0)


def _btp(state, spec, sample, inputs, adapter, n: int, k: int) -> None:
    from .attention_hook import Stage
    from .baselines.btp import btp_first_fraction, btp_layers, stage_attention, stage_balanced

    l1, l2, l3, l4 = btp_layers(adapter.num_layers)
    k1 = keep_count(n, btp_first_fraction(spec.budget, adapter.num_layers))
    counts = (k1, k1 // 2, k1 // 4)

    def on_survivors(fn):  # BTP indexes the surviving image tokens in raster order; map back to 0..N-1
        def select(scores, survivors, hidden):
            alive = np.flatnonzero(survivors)
            return alive[fn(scores[alive], hidden[alive])]
        return select

    stages = [
        (l1, on_survivors(lambda a, h: stage_balanced(a, h, counts[0], attn_share=0.1, extend=1.2))),
        (l2, on_survivors(lambda a, h: stage_balanced(a, h, counts[1], attn_share=0.4, extend=1.0))),
        (l3, on_survivors(lambda a, h: stage_attention(a, counts[2]))),
        (l4, lambda scores, survivors, hidden: np.empty(0, dtype=np.int64)),  # every visual token leaves
    ]
    for mask_from, select in stages:
        state.stages.append(Stage(mask_from - 1, mask_from, select, score_kind="attn", needs_hidden=True))


def _visionzip(state, spec, sample, inputs, adapter, n: int, k: int, allowed=None) -> None:
    from .baselines.visionzip import select_visionzip

    cues = adapter.vision_attention(inputs)
    kept, merge = select_visionzip(cues["attn"][max(cues["attn"])], cues["keys"], k, allowed)
    state.merge = None if METHODS[spec.method].merge == "off" else merge
    state.apply_keep(kept, 0)


def _hiprune(state, spec, sample, inputs, adapter, n: int, k: int) -> None:
    from .baselines.hiprune import object_block, select_hiprune

    obj = object_block(adapter.name)
    cues = adapter.vision_attention(inputs, (obj,))
    state.apply_keep(select_hiprune(cues["attn"][obj], cues["attn"][max(cues["attn"])], inputs.grid_hw, k), 0)


def _cdpruner(state, spec, sample, inputs, adapter, n: int, k: int, allowed=None) -> None:
    from .baselines.cdpruner import select_cdpruner

    kept = select_cdpruner(adapter.visual_embeddings(inputs), adapter.instruction_embeddings(inputs), k, allowed,
                           negate=METHODS[spec.method].invert)
    state.apply_keep(kept, 0)


def _feather(state, spec, sample, inputs, adapter, n: int, k: int) -> None:
    from .attention_hook import Stage
    from .baselines.feather import STAGE_LAYERS, feather_fractions, stage1_keep, stage2_keep, uniform_grid

    f1, f2 = feather_fractions(spec.budget)
    k1, k2 = keep_count(n, f1), keep_count(n, f2)
    uniform = uniform_grid(inputs.rc)
    first, second = STAGE_LAYERS
    state.stages.append(Stage(first - 1, first, lambda s, surv: stage1_keep(s, surv, uniform, k1),
                              score_kind="ropefree"))
    state.stages.append(Stage(second - 1, second, lambda s, surv: stage2_keep(s, surv, k2), score_kind="ropefree"))


def _vscan(state, spec, sample, inputs, adapter, n: int, k: int) -> None:
    from .attention_hook import Stage
    from .baselines.vscan import LOCAL_BLOCK, stage1, stage2_keep, vscan_schedule

    f1, layer, f2 = vscan_schedule(spec.budget, adapter.num_layers)
    k1, k2 = keep_count(n, f1), keep_count(n, f2)
    local = LOCAL_BLOCK[adapter.name]
    cues = adapter.vision_attention(inputs, (local,))
    kept, merge = stage1(cues["attn"][local], cues["attn"][max(cues["attn"])], cues["embeds"], inputs.grid_hw, k1)
    state.merge = None if METHODS[spec.method].merge == "off" else merge
    state.apply_keep(kept, 0)
    state.stages.append(Stage(layer, layer, lambda s, surv: stage2_keep(s, surv, k2), score_kind="ropefree"))


def _transprune(state, spec, sample, inputs, adapter, n: int, k: int) -> None:
    from .attention_hook import Stage
    from .baselines.transprune import ACCUMULATING_LAYERS, stage_keep, transprune_schedule

    state.ttv_layers = ACCUMULATING_LAYERS
    for layer, fraction in transprune_schedule(spec.budget):
        count = keep_count(n, fraction)

        def select(scores, survivors, layer=layer, count=count):
            ttv = sum((state.ttv[l] for l in ACCUMULATING_LAYERS if l < layer), np.zeros(n))
            return stage_keep(scores, ttv, survivors, count)

        state.stages.append(Stage(layer, layer, select, score_kind="text"))


BASELINES = {"sparsevlm": _sparsevlm, "vispruner": _vispruner, "prunesid": _prunesid, "btp": _btp,
             "visionzip": _visionzip, "hiprune": _hiprune, "cdpruner": _cdpruner, "feather": _feather,
             "vscan": _vscan, "transprune": _transprune}
GATED_BASELINES = {"visionzip", "cdpruner"}  # pruners with a gated variant (gate_visionzip, gate_cdpruner)


# ----------------------------------------------------------------------------------------------- running

def run_one(adapter, spec: RunSpec, bundle: Path, out: Path, profile: str, limit: int | None = None,
            max_new_tokens: int = 64, strict_masks: bool = True) -> dict:
    """Run (or resume) one run. `strict_masks=False` only for --dry-run: missing gates become all-foreground,
    and the check that the pruning hook actually fires is skipped (the fake adapter has no attention)."""
    from . import attention_hook as hook
    from .attention_hook import PruneState, pruning

    method = METHODS[spec.method]
    manifest = load_manifest(bundle, spec.dataset)
    samples = manifest[:limit] if limit else manifest
    run_id = spec.run_id(profile)
    runs_dir = out / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    jsonl = runs_dir / f"{run_id}.jsonl"
    done = _done_qids(jsonl, {s.qid for s in manifest})
    todo = [s for s in samples if s.qid not in done]

    gates = oracle = None
    if method.image:
        gate_path = bundle / "masks" / f"{spec.dataset}__{spec.model}.npz"
        if strict_masks and not gate_path.exists():
            raise FileNotFoundError(f"{gate_path} missing: run `python -m medprune.data masks --bundle {bundle}`")
        gates = MaskStore.load(gate_path) if gate_path.exists() else None
        gates_copy = out / "gates" / gate_path.name  # analysis reads gates from results (RadImageNet's bundle is session-only)
        if gate_path.exists() and not gates_copy.exists():
            gates_copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(gate_path, gates_copy)
        oracle_path = bundle / "masks" / f"{spec.dataset}__{spec.model}__oracle.npz"
        oracle = MaskStore.load(oracle_path) if oracle_path.exists() else None
    static_path = out / "static_maps" / f"{spec.model}__for_{spec.dataset}.npy"
    static_map = np.load(static_path) if static_path.exists() else None
    log_diag = spec.method == "full" or method.score is not None or method.profile
    diag = DiagWriter(out / "diag" / run_id) if log_diag else None

    meta_path = runs_dir / f"{run_id}.meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {
        "run_id": run_id, "model": spec.model, "repo": getattr(adapter, "repo", "fake"), "dataset": spec.dataset,
        "method": spec.method, "budget": spec.budget, "seed": spec.seed, "profile": profile, "exps": list(spec.exps),
        "max_new_tokens": max_new_tokens, "git": _git_hash(), "python": platform.python_version(),
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        import torch
        import transformers
        meta.update(torch=torch.__version__, transformers=transformers.__version__)
    except ImportError:
        pass

    elapsed = []
    with jsonl.open("a", encoding="utf-8") as f:
        for s in todo:
            t0 = time.perf_counter()
            image = Image.open(bundle / s.image_path).convert("RGB") if method.image else None
            inputs = adapter.prepare(s, image)
            gate_frac = oracle_frac = None
            bg_frac = None
            if method.image:
                grid, gate_frac = gates.get(s.image_id) if gates is not None and s.image_id in gates else \
                    (inputs.grid_hw, np.ones(inputs.grid_hw, dtype=np.float32))
                if tuple(grid) != tuple(inputs.grid_hw):
                    raise RuntimeError(f"{s.qid}: mask grid {grid} != model grid {inputs.grid_hw}")
                bg_frac = float(np.mean(gate_frac < 0.5))
                if oracle is not None and s.image_id in oracle:
                    oracle_frac = oracle.get(s.image_id)[1]
            if method.gate == "oracle" and oracle_frac is None:
                raise FileNotFoundError(f"oracle gate needs {spec.dataset}__{spec.model}__oracle.npz")

            state = build_state(spec, method, s, inputs, gate_frac, oracle_frac, adapter, static_map)
            if method.two_pass:  # pass 1: FastV scores from an unpruned prefill; pass 2: tokens removed before the LLM
                state.stop_after_scoring = True
                with pruning(state):
                    adapter.score_pass(inputs)
                scored = state
                state = PruneState(vis_pos=inputs.vis_pos)
                state.apply_keep(scored.keep, method.mask_from)
                state.logs = scored.logs
                if method.merge == "nearest":  # E18: the dropped tokens merge into the kept ones (VScan's rule)
                    from .baselines.vscan import merge_weights

                    import torch

                    emb = torch.as_tensor(adapter.visual_embeddings(inputs))
                    state.merge = (state.keep, merge_weights(emb, torch.as_tensor(state.keep)))
            calls_before = hook.CALLS["decoder"]
            with pruning(state):
                response, n_new = adapter.generate(inputs, max_new_tokens)
            if strict_masks and state is not None and not elapsed and hook.CALLS["decoder"] == calls_before:
                raise RuntimeError("the pruning attention was never called: it is not installed on the text decoder")
            correct, recall = is_correct(s, response)
            dt = time.perf_counter() - t0
            elapsed.append(dt)
            row = {
                "qid": s.qid, "image_id": s.image_id, "dataset": s.dataset, "modality": s.modality, "lang": s.lang,
                "answer_type": s.answer_type, "question": s.question, "answer": s.answer, "response": response,
                "correct": bool(correct), "open_recall": recall, "n_vis": int(len(inputs.vis_pos)),
                "k_keep": None if state is None or state.keep is None else int(len(state.keep)),
                "grid_hw": list(inputs.grid_hw),
                "keep_idx": None if state is None or state.keep is None else state.keep.tolist(),
                "keep_stages": [[layer, kept.tolist()] for layer, kept in state.history]
                if state is not None and len(state.history) > 1 else None,  # multi-stage methods (PDrop, SparseVLM, BTP)
                "bg_frac": bg_frac, "elapsed_s": round(dt, 3), "n_new_tokens": n_new,
                "truncated": n_new >= max_new_tokens,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if diag is not None and state is not None:
                diag.add(s.qid, state.logs)
    if diag is not None:
        diag.flush()

    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    meta.update(finished=datetime.now(timezone.utc).isoformat(timespec="seconds"), n_done=len(rows),
                n_manifest=len(samples), accuracy=float(np.mean([r["correct"] for r in rows])) if rows else None,
                mean_s_per_q=float(np.mean(elapsed)) if elapsed else meta.get("mean_s_per_q"))
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"run_id": run_id, "n": len(rows), "accuracy": meta["accuracy"], "s_per_q": meta["mean_s_per_q"]}


class FakeAdapter:
    """CPU stand-in for --dry-run and tests: 4x4 visual tokens; emulates logging and scoring stages with scores that
    depend only on (question, layer), as the real model's would."""

    repo = "fake"
    num_layers = 28

    def __init__(self, model: str):
        self.name = model

    def prepare(self, sample: Sample, image):
        import torch
        from .models import PreparedInputs

        if image is None:
            return PreparedInputs({"qid": sample.qid}, torch.empty(0, dtype=torch.long), (0, 0), np.zeros((0, 2), int), 8)
        rc = np.array([(r, c) for r in range(4) for c in range(4)])
        return PreparedInputs({"qid": sample.qid}, torch.arange(10, 26), (4, 4), rc, 40, torch.arange(26, 40))

    @staticmethod
    def _scores(qid: str, layer, n: int) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(f"{qid}:{layer}".encode()).digest()[:8], "little")
        return np.random.default_rng(seed).random(n)

    def _emulate(self, inputs) -> None:
        from .attention_hook import ACTIVE, StopForward

        state, qid = ACTIVE, inputs.model_kwargs.get("qid", "")
        if state is None:
            return
        for layer in state.log_layers:
            state.logs[f"scores_l{layer}"] = self._scores(qid, layer, state.n)
        if state.log_layers:
            state.logs["vit_norm"] = 1 + self._scores(qid, "vit", state.n)
            state.logs["llm_norm_l1"] = 1 + self._scores(qid, "llm", state.n)
        if state.profile:
            for layer in range(self.num_layers):
                state.logs[f"lp_last_l{layer}"] = np.log(self._scores(qid, layer, state.n))
                state.logs[f"lp_text_l{layer}"] = np.log(self._scores(qid, f"text{layer}", state.n))
                state.logs[f"sink_l{layer}"] = 40 * self._scores(qid, f"sink{layer}", state.n)
        if state.need_raters:  # SparseVLM: every other token after the image rates
            state.rater_pos = state.text_pos[::2].clone()
        for layer in state.ttv_layers:  # TransPrune's token transition variation
            state.ttv[layer] = self._scores(qid, f"ttv{layer}", state.n)
        for stage in sorted(state.stages, key=lambda s: s.mask_from):
            if not stage.done:
                key = f"{stage.score_kind}{stage.score_layer}" if stage.score_kind != "attn" else stage.score_layer
                scores = self._scores(qid, key, state.n)
                state.logs.setdefault(f"scores_{stage.score_kind}_l{stage.score_layer}", scores)
                if stage.needs_hidden:
                    hidden = np.random.default_rng(stage.mask_from).normal(size=(state.n, 8))
                    state.apply_keep(stage.select(scores, state.survivors(), hidden), stage.mask_from)
                else:
                    state.apply_keep(stage.select(scores, state.survivors()), stage.mask_from)
                stage.done = True
        if state.stop_after_scoring and state.stages and all(s.done for s in state.stages):
            raise StopForward

    def score_pass(self, inputs) -> None:
        from .attention_hook import StopForward

        try:
            self._emulate(inputs)
        except StopForward:
            pass

    def generate(self, inputs, max_new_tokens: int) -> tuple[str, int]:
        self._emulate(inputs)
        qid = inputs.model_kwargs.get("qid", "")
        return ("yes" if len(qid) % 2 else "no"), 2

    def set_vision_fp32(self, enabled: bool) -> None:
        pass

    def last_logits(self, inputs) -> np.ndarray:
        return np.zeros(8, dtype=np.float32)

    def visual_embeddings(self, inputs) -> np.ndarray:
        return np.random.default_rng(0).normal(size=(len(inputs.vis_pos), 8))

    def vision_cues(self, inputs) -> tuple[np.ndarray, np.ndarray]:
        n = len(inputs.vis_pos)
        return self._scores(inputs.model_kwargs.get("qid", ""), "vit", n), np.random.default_rng(1).normal(size=(n, 8))

    def vision_attention(self, inputs, blocks: tuple[int, ...] = ()) -> dict:
        n, qid = len(inputs.vis_pos), inputs.model_kwargs.get("qid", "")
        attn = {b: self._scores(qid, f"vit{b}", n) for b in set(blocks) | {31}}
        return {"attn": attn, "keys": np.random.default_rng(2).normal(size=(n, 8)),
                "feats": np.random.default_rng(1).normal(size=(n, 8)), "embeds": self.visual_embeddings(inputs)}

    def instruction_embeddings(self, inputs) -> np.ndarray:
        return np.random.default_rng(3).normal(size=(5, 8))


def bundle_for(bundles: Path | list[Path], dataset: str) -> Path:
    """First bundle root holding the dataset's manifest. Several roots let RadImageNet-VQA live in a session-only
    directory (its license forbids putting it in the shared bundle or in notebook outputs)."""
    roots = [bundles] if isinstance(bundles, (str, Path)) else list(bundles)
    for root in roots:
        if (Path(root) / "manifests" / f"{dataset}.jsonl").exists():
            return Path(root)
    raise FileNotFoundError(f"no manifests/{dataset}.jsonl under {[str(r) for r in roots]}")


def run_many(specs: list[RunSpec], bundle: Path | list[Path], out: Path, rates: dict, max_hours: float = 6.0,
             limit: int | None = None, dry_run: bool = False, vision_fp32: bool = False,
             max_new_tokens: int = 64, keep_going: bool = False) -> list[dict]:
    """Load each model once and run its pending runs until the next one would exceed `max_hours`. With
    `keep_going`, a run that raises is reported (traceback, `failed` summary) and the next run starts; its partial
    file stays and resumes next time."""
    from .env import detect_profile
    from .models import load_adapter

    profile = None if dry_run else detect_profile()
    profile_name = "dryrun" if dry_run else profile.name
    start, results = time.time(), []
    for model in dict.fromkeys(s.model for s in specs):
        adapter = FakeAdapter(model) if dry_run else load_adapter(model, profile, vision_fp32=vision_fp32)
        for spec in [s for s in specs if s.model == model]:
            try:
                root = bundle_for(bundle, spec.dataset)
            except FileNotFoundError:  # e.g. RadImageNet not built (secret missing): run the other datasets
                if not keep_going:
                    raise
                print(json.dumps({"run_id": spec.run_id(profile_name), "failed": "no manifest"}))
                continue
            samples = load_manifest(root, spec.dataset)
            n_total = min(len(samples), limit) if limit else len(samples)
            remaining = n_total - len(_done_qids(out / "runs" / f"{spec.run_id(profile_name)}.jsonl",
                                                {s.qid for s in samples}))
            if remaining <= 0:
                continue
            hours = estimate_hours(spec, remaining, rates)
            if results and (time.time() - start) / 3600 + hours > max_hours:
                print(f"stopping: next run needs {hours:.2f} h, session budget {max_hours} h reached")
                return results
            try:
                summary = run_one(adapter, spec, root, out, profile_name, limit=limit, max_new_tokens=max_new_tokens,
                                  strict_masks=not dry_run)
            except Exception as error:  # noqa: BLE001 - only with keep_going: report and continue
                if not keep_going:
                    raise
                import traceback

                traceback.print_exc()
                summary = {"run_id": spec.run_id(profile_name), "failed": f"{type(error).__name__}: {error}"}
            print(json.dumps(summary))
            results.append(summary)
        del adapter
        import gc

        gc.collect()  # hooks hold the adapter in a reference cycle: free the model before loading the next one
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass
    return results


def main() -> None:
    from .env import data_root, results_root

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, default=Path(__file__).resolve().parent.parent / "configs" / "runs.yaml")
    p.add_argument("--exp")
    p.add_argument("--model")
    p.add_argument("--account")
    p.add_argument("--dataset")
    p.add_argument("--method")
    p.add_argument("--bundle", type=Path, action="append", default=None,
                   help="bundle root; repeat to add a session-only root (e.g. RadImageNet)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--list", action="store_true", help="print runs, progress and pending hours; run nothing")
    p.add_argument("--profile", default=None,
                   help="--list only: profile whose results to count (default: this machine's; use t4x2 on a laptop)")
    p.add_argument("--limit", type=int, default=None, help="questions per run (smoke tests)")
    p.add_argument("--dry-run", action="store_true", help="fake CPU model; tests selection, logging and resume")
    p.add_argument("--max-hours", type=float, default=6.0)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--vision-fp32", action="store_true")
    p.add_argument("--keep-going", action="store_true", help="report a failing run and continue with the next")
    args = p.parse_args()

    bundles = args.bundle or [data_root()]
    out = args.out or results_root()
    specs, rates = load_runs(args.runs, args.exp, args.model, args.account, args.dataset, args.method)
    if args.list:
        from .env import detect_profile

        profile = args.profile or ("dryrun" if args.dry_run else detect_profile().name)
        pending_hours, n_missing, n_complete = 0.0, 0, 0
        for spec in specs:
            try:
                manifest_samples = load_manifest(bundle_for(bundles, spec.dataset), spec.dataset)
                n = len(manifest_samples)
                expected_ids = {s.qid for s in manifest_samples}
            except FileNotFoundError:
                n = 0
                expected_ids = None
            n = min(n, args.limit) if args.limit else n
            done = len(_done_qids(out / "runs" / f"{spec.run_id(profile)}.jsonl", expected_ids))
            hours = estimate_hours(spec, max(n - done, 0), rates) if n else float("nan")
            n_missing += n == 0
            n_complete += bool(n) and done >= n
            pending_hours += 0 if n == 0 else hours
            status = "no manifest" if n == 0 else ("complete" if done >= n else f"{done}/{n}")
            print(f"{spec.run_id(profile):<70} {','.join(spec.exps):<9} {status:>11} {hours:6.2f} h pending")
        print(f"{len(specs)} runs: {n_complete} complete, {n_missing} without manifest, {pending_hours:.2f} h pending")
        return
    run_many(specs, bundles, out, rates, max_hours=args.max_hours, limit=args.limit, dry_run=args.dry_run,
             vision_fp32=args.vision_fp32, max_new_tokens=args.max_new_tokens, keep_going=args.keep_going)


if __name__ == "__main__":
    main()
