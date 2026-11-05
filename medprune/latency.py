"""E9 latency (plan.md Stage 7, task T6): prefill time with visual tokens physically removed, and prefill FLOPs.

Accuracy runs drop tokens by masking attention, which saves no time. This module measures what physical removal would
save: Lingshu's decoder runs layer by layer and, at a switch layer, the hidden states, M-RoPE cos/sin and sequence keep
only the text tokens plus the kept visual tokens (original positions, as FastV's code does). One code path covers the
schedules that matter:

    full          no removal
    gate_input    background tokens (gate_frac < tau) removed before layer 0 — the gate alone
    prellm_10     10% of visual tokens kept from layer 0 (DivPrune, PruneSID, VisPruner, `random_l0`)
    fastv_10      all tokens in layers 0-1, 10% from layer 2 (FastV, Gate+FastV, Uniform, Gate+Uniform)
    k16_10        all tokens in layers 0-15, 10% from layer 16 (`fastv_k16`)

Time depends on how many tokens are kept, not which, so the 10% schedules keep evenly spaced tokens. Scoring overhead
(FastV's layer-1 attention, DivPrune's pairwise distances) is not included. Lingshu only: Qwen3-VL adds DeepStack
features inside its first layers, which this loop does not reproduce.

    python -m medprune.latency measure --model lingshu7b --bundle $DATA --out $RES --n 200      # GPU
    python -m medprune.latency prompt-tokens --model qwen3vl8b --bundle $DATA --bundle $RIN --out $RES   # CPU
    python -m medprune.latency flops --results results/all                                       # laptop

FLOPs per decoder layer follow FastV's formula, 4nd^2 + 2n^2d + 2ndm (n tokens, d hidden size, m FFN size), with n =
prompt text tokens + the visual tokens that layer would compute if removal were physical. The visual-token counts per
layer come from each run's own keep schedule (`keep_idx`, `keep_stages`), so multi-stage methods are counted exactly.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

# Text-decoder sizes from each checkpoint's config.json (hidden_size, intermediate_size, num_hidden_layers).
MODEL_DIMS = {
    "lingshu7b": (3584, 18944, 28),
    "qwen3vl8b": (4096, 12288, 36),
    "huatuo7b": (3584, 18944, 28),
}
SCHEDULES = {"full": None, "gate_input": 0, "prellm_10": 0, "fastv_10": 2, "k16_10": 16}
DEFAULT_DATASETS = "slake_en,slake_zh,rin_anat_mc,rin_abn_closed,omni_anat,omni_panel,natural"


# ------------------------------------------------------------------------------------------------ FLOPs

def layer_flops(n: np.ndarray | int, d: int, m: int) -> np.ndarray:
    """FastV's per-layer prefill FLOPs for n tokens."""
    n = np.asarray(n, dtype=np.float64)
    return 4 * n * d * d + 2 * n * n * d + 2 * n * d * m


def visual_tokens_per_layer(row: dict, mask_from: int, rule: str, n_layers: int) -> np.ndarray:
    """Visual tokens each decoder layer would compute if dropped tokens were removed, from one run row."""
    n_vis = int(row.get("n_vis") or 0)
    counts = np.full(n_layers, n_vis, dtype=np.int64)
    if rule == "none" or row.get("keep_idx") is None:
        return counts
    stages = row.get("keep_stages")
    if stages:  # multi-stage methods: [[first layer the keep-set applies to, kept indices], ...]
        for layer, kept in stages:
            counts[int(layer):] = len(kept)
        return counts
    counts[mask_from:] = int(row["k_keep"])
    return counts


def prefill_flops(n_text: int, vis_counts: np.ndarray, d: int, m: int) -> float:
    return float(layer_flops(n_text + np.asarray(vis_counts), d, m).sum())


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def flops_table(results: Path, prompt_tokens: dict[str, dict[str, int]], profile: str = "t4x2"):
    """Mean prefill FLOPs relative to Full on the same questions, per model x dataset x method x budget."""
    import pandas as pd

    from .selection import METHODS

    paths = []
    for path in sorted((results / "runs").glob(f"*__{profile}.jsonl")):
        model, dataset, method, budget = path.name.split("__")[:4]
        if method in METHODS and model in MODEL_DIMS and dataset in prompt_tokens.get(model, {}):
            paths.append((path, model, dataset, method, budget, _read_rows(path)))
    # The Full reference is the image's visual-token count, taken from any run that saw the image: a text-only row
    # records none, and using its own count would make text-only cost as much as Full.
    n_vis = {}
    for _, model, dataset, _, _, run in paths:
        for r in run:
            if r.get("n_vis"):
                n_vis[(model, dataset, r["qid"])] = int(r["n_vis"])
    rows = []
    for path, model, dataset, method, budget, run in paths:
        d, m, n_layers = MODEL_DIMS[model]
        spec = METHODS[method]
        text = prompt_tokens[model][dataset]
        flops, full, vis_mean = [], [], []
        for r in run:
            ref = n_vis.get((model, dataset, r["qid"]))
            if r["qid"] not in text or ref is None:
                continue
            counts = visual_tokens_per_layer(r, spec.mask_from, spec.rule, n_layers)
            if not spec.image:
                counts = np.zeros(n_layers, dtype=np.int64)
            flops.append(prefill_flops(text[r["qid"]], counts, d, m))
            full.append(prefill_flops(text[r["qid"]], np.full(n_layers, ref), d, m))
            vis_mean.append(counts.mean())
        if flops:
            rows.append({"model": model, "dataset": dataset, "method": method, "budget_pct": int(budget[1:]),
                         "questions": len(flops), "rel_flops": float(np.sum(flops) / np.sum(full)),
                         "vis_tokens_per_layer": float(np.mean(vis_mean)),
                         "tflops_per_question": float(np.mean(flops) / 1e12)})
    return pd.DataFrame(rows).sort_values(["model", "dataset", "budget_pct", "rel_flops"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------- prompt tokens

def text_token_count(processor, sample) -> int:
    """Prompt tokens other than the image's visual tokens: the image-free chat template plus <|vision_start|> and
    <|vision_end|> (Qwen-family templates wrap the expanded image pads in those two)."""
    from .data import make_prompt

    content = [{"type": "text", "text": make_prompt(sample)}]
    text = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                         add_generation_prompt=True)
    return len(processor.tokenizer(text)["input_ids"]) + 2


def prompt_tokens(model: str, bundles: list[Path], datasets: list[str]) -> dict[str, dict[str, int]]:
    from transformers import AutoProcessor

    from .data import load_manifest
    from .models import MODEL_IDS
    from .run_eval import bundle_for

    processor = AutoProcessor.from_pretrained(MODEL_IDS[model])
    out = {}
    for ds in datasets:
        try:
            samples = load_manifest(bundle_for(bundles, ds), ds)
        except FileNotFoundError:
            print(f"  {ds}: no manifest, skipped")
            continue
        out[ds] = {s.qid: text_token_count(processor, s) for s in samples}
        print(f"  {ds}: {len(samples)} questions, median {int(np.median(list(out[ds].values())))} text tokens")
    return out


# ---------------------------------------------------------------------------------------- measured prefill

def kept_sequence(seq_len: int, vis_pos: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Sequence indices that survive: every non-visual position plus the kept visual tokens, in order."""
    alive = np.ones(seq_len, dtype=bool)
    alive[np.asarray(vis_pos)] = False
    alive[np.asarray(vis_pos)[np.asarray(keep, dtype=np.int64)]] = True
    return np.flatnonzero(alive)


def prefill(lm, embeds, position_ids, keep_seq: np.ndarray | None = None, switch: int | None = None):
    """Run a Qwen2.5-VL text decoder layer by layer on `embeds` [1, L, d] with M-RoPE `position_ids` [3, 1, L]; from
    layer `switch` on, only the sequence positions `keep_seq` are computed. Returns the final normed hidden states."""
    import torch

    hidden = embeds
    cos, sin = lm.rotary_emb(hidden, position_ids)
    for i, layer in enumerate(lm.layers):
        if switch is not None and i == switch:
            idx = torch.as_tensor(keep_seq, dtype=torch.long)
            hidden = hidden[:, idx.to(hidden.device)]
            cos, sin = cos[..., idx.to(cos.device), :], sin[..., idx.to(sin.device), :]
        out = layer(hidden, attention_mask=None, position_ids=None, position_embeddings=(cos, sin), use_cache=False)
        hidden = out[0] if isinstance(out, tuple) else out
    return lm.norm(hidden)


def _embed(adapter, inputs):
    """LLM input embeddings with the image features scattered in, and the M-RoPE position ids (Qwen2.5-VL)."""
    import torch

    base, kw = adapter.base, inputs.model_kwargs
    ids = kw["input_ids"]
    embeds = base.get_input_embeddings()(ids)
    feats = base.get_image_features(kw["pixel_values"], kw["image_grid_thw"])
    feats = torch.cat(list(feats), dim=0) if isinstance(feats, (list, tuple)) else feats
    image_mask = (ids == adapter.image_token_id).unsqueeze(-1).expand_as(embeds)
    embeds = embeds.masked_scatter(image_mask, feats.to(embeds.device, embeds.dtype))
    position_ids, _ = base.get_rope_index(input_ids=ids, image_grid_thw=kw["image_grid_thw"],
                                          attention_mask=kw["attention_mask"])
    return embeds, position_ids


def _sync() -> None:
    import torch

    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


def measure(model: str, bundle: Path, out: Path, n: int = 200, warmup: int = 10, budget: float = 0.10,
            tau: float = 0.5, dataset: str = "slake_en") -> dict:
    """Median prefill time per schedule on the first n questions of `dataset` (after `warmup` untimed questions).
    Wall time between device-wide synchronizations, so it is correct when the model is split over two GPUs."""
    import torch
    import transformers
    from PIL import Image

    from .data import load_manifest
    from .env import detect_profile
    from .masks import MaskStore
    from .models import load_adapter

    if model != "lingshu7b":
        raise NotImplementedError("measured prefill is implemented for Lingshu (Qwen2.5-VL) only")
    adapter = load_adapter(model, detect_profile())
    lm = adapter.base.language_model
    samples = load_manifest(bundle, dataset)[: n + warmup]
    gates = MaskStore.load(bundle / "masks" / f"{dataset}__{model}.npz")
    times = {name: [] for name in SCHEDULES}
    tokens = {name: [] for name in SCHEDULES}
    vision_ms, text_check, equivalence = [], [], None
    rng = np.random.default_rng(0)
    with torch.inference_mode():
        for qi, s in enumerate(samples):
            image = Image.open(bundle / s.image_path).convert("RGB")
            inputs = adapter.prepare(s, image)
            _sync()
            t0 = time.perf_counter()
            embeds, position_ids = _embed(adapter, inputs)
            _sync()
            t_vision = (time.perf_counter() - t0) * 1e3
            vis = inputs.vis_pos.numpy()
            seq_len, n_vis = embeds.shape[1], len(vis)
            k = max(1, int(round(budget * n_vis)))
            even = np.unique(np.linspace(0, n_vis - 1, k).round().astype(np.int64))
            _, frac = gates.get(s.image_id)
            fg = np.flatnonzero(frac.reshape(-1)[inputs.cell_index] >= tau)
            fg = fg if fg.size else np.arange(n_vis)
            keeps = {"full": None, "gate_input": fg, "prellm_10": even, "fastv_10": even, "k16_10": even}
            if qi == 0:  # the layer loop must reproduce the model's own forward
                ref = lm(inputs_embeds=embeds, position_ids=position_ids, use_cache=False).last_hidden_state
                mine = prefill(lm, embeds, position_ids)
                equivalence = float((mine.float() - ref.float().to(mine.device)).abs().max()
                                    / ref.float().abs().max().clamp_min(1e-6))
                if equivalence > 1e-2:
                    raise RuntimeError(f"layer loop differs from the model forward (relative max diff {equivalence:.2e})")
            order = list(SCHEDULES)
            rng.shuffle(order)
            row = {}
            for name in order:
                keep = keeps[name]
                seq = None if keep is None else kept_sequence(seq_len, vis, keep)
                _sync()
                t0 = time.perf_counter()
                prefill(lm, embeds, position_ids, seq, SCHEDULES[name])
                _sync()
                row[name] = (time.perf_counter() - t0) * 1e3
            if qi < warmup:
                continue
            vision_ms.append(t_vision)
            for name in SCHEDULES:
                times[name].append(row[name])
                tokens[name].append(n_vis if keeps[name] is None else len(keeps[name]))
            text_check.append(seq_len - n_vis == text_token_count(adapter.processor, s))
    full = np.asarray(times["full"])
    report = {
        "model": model, "dataset": dataset, "n": len(full), "warmup": warmup, "budget": budget, "tau": tau,
        "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "torch": torch.__version__, "transformers": transformers.__version__, "python": platform.python_version(),
        "equivalence_rel_max_diff": equivalence, "text_token_count_matches": float(np.mean(text_check)),
        "vision_and_embed_ms": {"median": float(np.median(vision_ms)), "p25": float(np.percentile(vision_ms, 25)),
                                "p75": float(np.percentile(vision_ms, 75))},
        "schedules": {},
    }
    for name, switch in SCHEDULES.items():
        t = np.asarray(times[name])
        report["schedules"][name] = {
            "switch_layer": switch, "median_ms": float(np.median(t)), "p25_ms": float(np.percentile(t, 25)),
            "p75_ms": float(np.percentile(t, 75)), "mean_visual_tokens_kept": float(np.mean(tokens[name])),
            "speedup_vs_full": float(np.median(full) / np.median(t)),
            "median_paired_ratio": float(np.median(t / full)),
        }
    dest = out / "latency" / f"{model}__{dataset}__{detect_profile().name}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["schedules"], indent=2))
    print("wrote", dest)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["measure", "prompt-tokens", "flops"])
    p.add_argument("--model", default="lingshu7b")
    p.add_argument("--bundle", type=Path, action="append", default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--results", type=Path, default=Path("results/all"))
    p.add_argument("--datasets", default=DEFAULT_DATASETS)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--warmup", type=int, default=10)
    args = p.parse_args()
    if args.command == "measure":
        measure(args.model, args.bundle[0], args.out, n=args.n, warmup=args.warmup)
    elif args.command == "prompt-tokens":
        counts = prompt_tokens(args.model, args.bundle, args.datasets.split(","))
        dest = args.out / "latency" / f"prompt_tokens__{args.model}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(counts), encoding="utf-8")
        print("wrote", dest)
    else:
        counts = {}
        for path in sorted((args.results / "latency").glob("prompt_tokens__*.json")):
            counts[path.stem.split("__")[1]] = json.loads(path.read_text(encoding="utf-8"))
        table = flops_table(args.results, counts)
        (args.results / "tables").mkdir(parents=True, exist_ok=True)
        table.to_csv(args.results / "tables" / "flops.csv", index=False)
        print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
