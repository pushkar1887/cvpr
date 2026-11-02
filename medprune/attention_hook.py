"""Applies a visual-token keep-set inside the language model through one custom attention function (plan.md §D3).

Every method in this project is "choose which visual tokens to keep, then from some layer on stop every query from
attending to the dropped ones". Masking dropped keys while all tokens keep their original position ids is numerically
identical to deleting those tokens (tests/test_attention_hook.py), so FastV, random and uniform selection, RoPE-free
scoring, the foreground gate, DivPrune and PDrop all run through the code below.

FastV reference (third_party/FastV/src/FastV/llava-hf/.../modeling_llama.py L1038-1052): scores are the last prompt
token's attention averaged over heads, read at layer K-1; tokens are removed from layer K on.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np
import torch
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

ATTN_NAME = "medprune_sdpa"

# (scores [N], survivors [N] bool) -> kept token indices
Selector = Callable[[np.ndarray, np.ndarray], np.ndarray]


class StopForward(Exception):
    """Raised once every stage has selected, when a pass only needs the scores (two-pass methods)."""


@dataclass
class Stage:
    score_layer: int  # layer whose attention is read
    mask_from: int  # first layer at which this stage's dropped tokens are masked
    select: Selector
    score_kind: str = "attn"  # "attn" | "ropefree"
    done: bool = False


@dataclass
class PruneState:
    """Plan for one generate() call. Batch size 1."""

    vis_pos: torch.Tensor  # [N] int64 sequence positions of the visual tokens
    stages: list[Stage] = field(default_factory=list)
    log_layers: tuple[int, ...] = ()  # last-token attention logged here; outputs unchanged
    log_ropefree: bool = False  # also log RoPE-free scores at the pre-RoPE capture layer
    stop_after_scoring: bool = False
    logs: dict[str, np.ndarray] = field(default_factory=dict)
    keep: np.ndarray | None = None  # kept indices into 0..N-1 after the latest selection
    pre_rope: dict = field(default_factory=dict)  # {"layer": int, "q": tensor, "k": tensor}, filled by capture hooks
    _survivors: np.ndarray | None = None
    _drops: list[tuple[int, torch.Tensor]] = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(self.vis_pos.numel())

    def survivors(self) -> np.ndarray:
        return self._survivors if self._survivors is not None else np.ones(self.n, dtype=bool)

    def apply_keep(self, kept: np.ndarray, mask_from: int) -> None:
        """Keep `kept` (indices into the visual tokens); everything else is masked from layer `mask_from` on."""
        kept = np.unique(np.asarray(kept, dtype=np.int64))
        survivors = np.zeros(self.n, dtype=bool)
        survivors[kept] = True
        previous = self.survivors()
        if np.any(survivors & ~previous):
            raise ValueError("a later stage kept a token that an earlier stage dropped")
        newly_dropped = np.flatnonzero(previous & ~survivors)
        if newly_dropped.size:
            self._drops.append((mask_from, self.vis_pos[torch.from_numpy(newly_dropped)]))
        self._survivors, self.keep = survivors, kept

    def dropped_at(self, layer: int) -> torch.Tensor | None:
        parts = [pos for start, pos in self._drops if start <= layer]
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else torch.cat(parts)


ACTIVE: PruneState | None = None
CALLS = {"decoder": 0}  # hook invocations on decoder layers; the smoke test checks it is > 0


@contextmanager
def pruning(state: PruneState | None) -> Iterator[PruneState | None]:
    """Activate `state` for the enclosed forward/generate calls."""
    global ACTIVE
    previous, ACTIVE = ACTIVE, state
    try:
        yield state
    finally:
        ACTIVE = previous


def last_token_attention(query: torch.Tensor, key: torch.Tensor, scaling: float,
                         exclude: torch.Tensor | None = None) -> torch.Tensor:
    """Head-mean softmax attention of the last query over all keys, [Tk] float32.

    query [1, Hq, Tq, D], key [1, Hk, Tk, D] (after RoPE). `exclude` key positions get zero weight (keys already
    masked by an earlier stage), so later stages score exactly the attention the model computes.
    """
    n_rep = query.shape[1] // key.shape[1]
    k = key[0].repeat_interleave(n_rep, dim=0) if n_rep > 1 else key[0]  # [Hq, Tk, D], same order as repeat_kv
    logits = torch.einsum("hd,htd->ht", query[0, :, -1, :].float(), k.float()) * scaling
    if exclude is not None:
        logits[:, exclude.to(logits.device)] = float("-inf")
    return logits.softmax(dim=-1).mean(dim=0)


def ropefree_attention(state: PruneState, head_dim: int, scaling: float) -> torch.Tensor:
    """Same as last_token_attention but from the captured pre-RoPE q and k (FEATHER's criterion), [T] float32."""
    q, k = state.pre_rope["q"], state.pre_rope["k"]  # [1, T, Hq*D] or [1, T, Hq, D]
    q = q.reshape(q.shape[0], q.shape[1], -1, head_dim)[0, -1]  # [Hq, D]
    k = k.reshape(k.shape[0], k.shape[1], -1, head_dim)[0].transpose(0, 1)  # [Hk, T, D]
    n_rep = q.shape[0] // k.shape[0]
    k = k.repeat_interleave(n_rep, dim=0) if n_rep > 1 else k
    logits = torch.einsum("hd,htd->ht", q.float(), k.float()) * scaling
    return logits.softmax(dim=-1).mean(dim=0)


def install_pre_rope_capture(q_module: torch.nn.Module, k_module: torch.nn.Module, layer: int) -> list:
    """Record the outputs of the modules that produce q and k right before RoPE, on prefill only."""

    def recorder(name: str):
        def hook(_module, _inputs, output):
            state = ACTIVE
            if state is not None and output.shape[1] > 1:
                state.pre_rope["layer"] = layer
                state.pre_rope[name] = output.detach()

        return hook

    return [q_module.register_forward_hook(recorder("q")), k_module.register_forward_hook(recorder("k"))]


def allow_mask(attention_mask: torch.Tensor | None, query: torch.Tensor, key: torch.Tensor,
               dropped: torch.Tensor) -> torch.Tensor:
    """Boolean [1, 1, Tq, Tk] mask (True = attend): causal on prefill, dropped visual keys removed.

    A mask passed to SDPA disables its is_causal fast path, so causality is built here explicitly.
    """
    tq, tk, device = query.shape[2], key.shape[2], query.device
    keys_ok = torch.ones(tk, dtype=torch.bool, device=device)
    keys_ok[dropped.to(device)] = False
    if attention_mask is None:
        base = torch.ones(tq, tk, dtype=torch.bool, device=device)
        if tq > 1:
            base = base.tril(diagonal=tk - tq)
        base = base[None, None]
    else:
        m = attention_mask[:, :, :, :tk]
        base = m if m.dtype == torch.bool else (m == 0)
    return base & keys_ok


def _prefill_step(state: PruneState, module, layer: int, query: torch.Tensor, key: torch.Tensor,
                  scaling: float) -> None:
    vis = state.vis_pos.to(query.device)
    cache: dict[str, np.ndarray] = {}

    def attention_scores() -> np.ndarray:
        if "attn" not in cache:
            cache["attn"] = last_token_attention(query, key, scaling, state.dropped_at(layer))[vis].cpu().numpy()
        return cache["attn"]

    def ropefree_scores() -> np.ndarray:
        if state.pre_rope.get("layer") != layer:
            raise RuntimeError(f"RoPE-free scores requested at layer {layer}, captured at {state.pre_rope.get('layer')}")
        if "ropefree" not in cache:
            cache["ropefree"] = ropefree_attention(state, module.head_dim, scaling)[vis].cpu().numpy()
        return cache["ropefree"]

    if layer in state.log_layers:
        state.logs[f"scores_l{layer}"] = attention_scores()
    if state.log_ropefree and state.pre_rope.get("layer") == layer:
        state.logs[f"scores_ropefree_l{layer}"] = ropefree_scores()
    for stage in state.stages:
        if stage.done or stage.score_layer != layer:
            continue
        scores = ropefree_scores() if stage.score_kind == "ropefree" else attention_scores()
        state.logs.setdefault(f"scores_{stage.score_kind}_l{layer}", scores)
        state.apply_keep(stage.select(scores, state.survivors()), stage.mask_from)
        stage.done = True
    if state.stop_after_scoring and state.stages and all(s.done for s in state.stages):
        raise StopForward


def medprune_sdpa(module, query, key, value, attention_mask, **kwargs):
    """Drop-in replacement for transformers' SDPA attention (registered as ATTN_NAME)."""
    state, layer = ACTIVE, getattr(module, "layer_idx", None)
    if state is None or layer is None:  # vision tower, or no pruning active
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)
    CALLS["decoder"] += 1
    if query.shape[0] != 1:
        raise ValueError("medprune supports batch size 1 only")
    if query.shape[2] > 1:
        scaling = kwargs.get("scaling") or getattr(module, "scaling", None) or query.shape[-1] ** -0.5
        _prefill_step(state, module, layer, query, key, scaling)
    dropped = state.dropped_at(layer)
    if dropped is not None:
        attention_mask = allow_mask(attention_mask, query, key, dropped)
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)


def register() -> None:
    """Make ATTN_NAME usable as `attn_implementation`; masks are built exactly as for plain SDPA."""
    AttentionInterface.register(ATTN_NAME, medprune_sdpa)
    AttentionMaskInterface.register(ATTN_NAME, sdpa_mask)


register()
