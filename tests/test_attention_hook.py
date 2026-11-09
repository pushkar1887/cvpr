"""The proof that the hook implements token pruning exactly (plan.md §D5).

Tiny random Qwen2 (1D RoPE, like HuatuoGPT's LLM) and Qwen3 (q/k norms before RoPE, like Qwen3-VL's LLM) models in
fp32 on CPU. Positions 4..19 play the role of visual tokens.
"""

import numpy as np
import pytest
import torch
from transformers import DynamicCache, Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from medprune import attention_hook as hook
from medprune.attention_hook import PruneState, Stage, StopForward, install_pre_rope_capture, pruning

T = 24
VIS = torch.arange(4, 20)
IDS = torch.randint(0, 97, (T,), generator=torch.Generator().manual_seed(1))
KEEP = np.array([0, 3, 5, 6, 10, 15])  # indices into the 16 visual tokens


def build(kind: str, attn: str):
    torch.manual_seed(0)
    common = dict(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=128, attn_implementation=attn)
    model = Qwen2ForCausalLM(Qwen2Config(**common)) if kind == "qwen2" else Qwen3ForCausalLM(Qwen3Config(head_dim=16, **common))
    return model.eval()


@pytest.fixture(params=["qwen2", "qwen3"], scope="module")
def models(request):
    hooked = build(request.param, hook.ATTN_NAME)
    plain, eager = build(request.param, "sdpa"), build(request.param, "eager")
    plain.load_state_dict(hooked.state_dict())
    eager.load_state_dict(hooked.state_dict())
    return request.param, hooked, plain, eager


@torch.no_grad()
def last_logits(model, ids, pos=None, state=None):
    with pruning(state):
        out = model(input_ids=ids[None], position_ids=None if pos is None else pos[None])
    return out.logits[0, -1]


def kept_positions() -> torch.Tensor:
    others = torch.cat([torch.arange(0, 4), torch.arange(20, T)])
    return torch.sort(torch.cat([others, VIS[torch.from_numpy(KEEP)]])).values


def preset(keep: np.ndarray, mask_from: int, **kw) -> PruneState:
    state = PruneState(vis_pos=VIS, **kw)
    state.apply_keep(keep, mask_from)
    return state


@torch.no_grad()
def fastv_reference(model, ids, keep_pos, k):
    """FastV's own implementation: physically drop the pruned tokens' hidden states before layer k."""
    m = model.model
    h = m.embed_tokens(ids[None])
    pos = torch.arange(ids.numel())[None]
    emb = m.rotary_emb(h, pos)
    for i, layer in enumerate(m.layers):
        if i == k:
            h, pos = h[:, keep_pos], pos[:, keep_pos]
            emb = m.rotary_emb(h, pos)
        out = layer(h, attention_mask=None, position_ids=pos, position_embeddings=emb)
        h = out[0] if isinstance(out, tuple) else out
    return model.lm_head(m.norm(h))[0, -1]


def test_no_state_is_plain_sdpa(models):
    _, hooked, plain, _ = models
    torch.testing.assert_close(last_logits(hooked, IDS), last_logits(plain, IDS), atol=1e-6, rtol=0)


def test_keep_all_is_plain_sdpa(models):
    _, hooked, plain, _ = models
    torch.testing.assert_close(last_logits(hooked, IDS, state=preset(np.arange(16), 0)), last_logits(plain, IDS),
                               atol=1e-6, rtol=0)


def test_masking_from_layer_0_equals_deleting_tokens(models):
    _, hooked, plain, _ = models
    pos = kept_positions()
    torch.testing.assert_close(last_logits(hooked, IDS, state=preset(KEEP, 0)), last_logits(plain, IDS[pos], pos=pos),
                               atol=1e-5, rtol=0)


@pytest.mark.parametrize("k", [1, 2])
def test_masking_from_layer_k_equals_fastv_removal(models, k):
    _, hooked, plain, _ = models
    torch.testing.assert_close(last_logits(hooked, IDS, state=preset(KEEP, k)),
                               fastv_reference(plain, IDS, kept_positions(), k), atol=1e-5, rtol=0)


@torch.no_grad()
def greedy(model, ids, pos, steps, state=None):
    tokens, cache = [], DynamicCache()
    with pruning(state):
        out = model(input_ids=ids[None], position_ids=pos[None], past_key_values=cache, use_cache=True)
        for step in range(steps):
            token = out.logits[0, -1].argmax()
            tokens.append(int(token))
            out = model(input_ids=token.view(1, 1), position_ids=torch.tensor([[T + step]]),
                        past_key_values=out.past_key_values, use_cache=True)
    return tokens, out.logits[0, -1]


def test_cached_decoding_equals_deleting_tokens(models):
    _, hooked, plain, _ = models
    pos = kept_positions()
    masked_tokens, masked_logits = greedy(hooked, IDS, torch.arange(T), 5, state=preset(KEEP, 0))
    ref_tokens, ref_logits = greedy(plain, IDS[pos], pos, 5)
    assert masked_tokens == ref_tokens
    torch.testing.assert_close(masked_logits, ref_logits, atol=1e-5, rtol=0)


def test_logged_scores_equal_eager_attention(models):
    _, hooked, _, eager = models
    state = PruneState(vis_pos=VIS, log_layers=(1,))
    last_logits(hooked, IDS, state=state)
    with torch.no_grad():
        attention = eager(input_ids=IDS[None], output_attentions=True).attentions[1]
    expected = attention[0, :, -1, :].mean(0)[VIS].numpy()
    np.testing.assert_allclose(state.logs["scores_l1"], expected, atol=1e-6)


TEXT = torch.arange(20, T)  # positions after the "image"


def test_text_readout_equals_eager_attention_of_the_tokens_after_the_image(models):
    _, hooked, _, eager = models
    state = PruneState(vis_pos=VIS, text_pos=TEXT, stages=[Stage(1, 2, lambda s, _surv: np.sort(np.argsort(-s)[:5]),
                                                               score_kind="text")])
    last_logits(hooked, IDS, state=state)
    with torch.no_grad():
        attention = eager(input_ids=IDS[None], output_attentions=True).attentions[1]
    expected = attention[0][:, TEXT, :].mean(dim=(0, 1))[VIS].numpy()
    np.testing.assert_allclose(state.logs["scores_text_l1"], expected, atol=1e-6)
    np.testing.assert_array_equal(state.keep, np.sort(np.argsort(-expected)[:5]))


def test_profile_logs_both_readouts_and_sink_scores_at_every_layer(models):
    _, hooked, _, eager = models
    handles = hook.install_profile_hooks(hooked.model.layers)
    try:
        state = PruneState(vis_pos=VIS, text_pos=TEXT, profile=True)
        last_logits(hooked, IDS, state=state)
    finally:
        for handle in handles:
            handle.remove()
    n_layers = len(hooked.model.layers)
    assert {f"{k}_l{i}" for k in ("lp_last", "lp_text", "sink") for i in range(n_layers)} == set(state.logs)
    with torch.no_grad():
        attention = eager(input_ids=IDS[None], output_attentions=True).attentions
    for i in range(n_layers):
        np.testing.assert_allclose(np.exp(state.logs[f"lp_last_l{i}"]), attention[i][0, :, -1, :].mean(0)[VIS].numpy(),
                                   atol=1e-6)
    assert all(state.logs[f"sink_l{i}"].shape == (16,) for i in range(n_layers))


def test_sink_score_flags_tokens_sharing_the_first_tokens_massive_dimension():
    h = torch.randn(6, 4096, generator=torch.Generator().manual_seed(0))  # RMS-normalized values reach sqrt(4096)
    h[0, 7] = 400.0  # token 0: massive activation in dimension 7
    h[3, 7] = 400.0  # a visual token with the same massive dimension
    h[4, 9] = 400.0  # massive, but in another dimension: not a sink by this definition
    scores = hook.sink_scores(h, torch.tensor([2, 3, 4]))
    assert scores[1] > hook.SINK_TAU and scores[0] < 5 and scores[2] < 5


def test_fastv_stage_keeps_the_top_scored_tokens(models):
    _, hooked, _, eager = models
    k = 5
    state = PruneState(vis_pos=VIS, stages=[Stage(1, 2, lambda s, _surv: np.sort(np.argsort(-s)[:k]))])
    last_logits(hooked, IDS, state=state)
    with torch.no_grad():
        attention = eager(input_ids=IDS[None], output_attentions=True).attentions[1]
    expected = np.sort(np.argsort(-attention[0, :, -1, :].mean(0)[VIS].numpy())[:k])
    np.testing.assert_array_equal(state.keep, expected)


def test_ropefree_scores_come_from_pre_rope_projections(models):
    kind, hooked, _, _ = models
    attn = hooked.model.layers[1].self_attn
    q_mod, k_mod = (attn.q_norm, attn.k_norm) if kind == "qwen3" else (attn.q_proj, attn.k_proj)
    handles = install_pre_rope_capture(q_mod, k_mod, layer=1)
    try:
        state = PruneState(vis_pos=VIS, log_ropefree=True)
        last_logits(hooked, IDS, state=state)
        q = state.pre_rope["q"].reshape(1, T, -1, attn.head_dim)[0]
        k = state.pre_rope["k"].reshape(1, T, -1, attn.head_dim)[0]
        rep = q.shape[1] // k.shape[1]
        heads = [((k[:, h // rep] @ q[-1, h]) * attn.scaling).softmax(0) for h in range(q.shape[1])]
        expected = torch.stack(heads).mean(0)[VIS].numpy()
        np.testing.assert_allclose(state.logs["scores_ropefree_l1"], expected, atol=1e-6)
    finally:
        for handle in handles:
            handle.remove()


def test_multistage_pruning_masks_progressively(models):
    _, hooked, _, _ = models

    def top(n):
        return lambda s, surv: np.sort(np.argsort(-np.where(surv, s, -np.inf))[:n])

    state = PruneState(vis_pos=VIS, stages=[Stage(1, 2, top(8)), Stage(2, 3, top(4))])
    last_logits(hooked, IDS, state=state)
    assert len(state.keep) == 4
    assert state.dropped_at(1) is None and len(state.dropped_at(2)) == 8 and len(state.dropped_at(3)) == 12


def test_stop_after_scoring(models):
    _, hooked, _, _ = models
    state = PruneState(vis_pos=VIS, stages=[Stage(1, 0, lambda s, surv: np.arange(3))], stop_after_scoring=True)
    with pytest.raises(StopForward):
        last_logits(hooked, IDS, state=state)
    np.testing.assert_array_equal(state.keep, np.arange(3))


def test_allow_mask_is_causal_with_cache_offset():
    q, k = torch.zeros(1, 2, 3, 4), torch.zeros(1, 2, 5, 4)
    mask = hook.allow_mask(None, q, k, torch.tensor([1]))[0, 0]
    expected = torch.tensor([[1, 0, 1, 0, 0], [1, 0, 1, 1, 0], [1, 0, 1, 1, 1]], dtype=torch.bool)
    assert torch.equal(mask, expected)


def test_modules_without_layer_idx_pass_through():
    class Vision(torch.nn.Module):
        is_causal = False
        num_key_value_groups = 1

    q, k, v = (torch.randn(1, 2, 6, 8) for _ in range(3))
    with pruning(preset(np.array([0]), 0)):
        out, _ = hook.medprune_sdpa(Vision(), q, k, v, None)
    ref, _ = sdpa_attention_forward(Vision(), q, k, v, None)
    torch.testing.assert_close(out, ref)


# ------------------------------------------------------------------------------------- E42a: the size term

class _Decoder(torch.nn.Module):
    layer_idx, is_causal, num_key_value_groups = 0, True, 1


def test_size_term_reproduces_full_attention_over_duplicate_clusters():
    """E42a's identity (plan.md): if each kept visual token stands for s identical tokens, + log s on its logit gives
    exactly the attention over all of them (proportional attention); merging alone weights each cluster as one."""
    torch.manual_seed(0)
    n_tok, heads, dim = 14, 2, 8
    vis = torch.arange(2, 10)  # 8 visual tokens in clusters of 3, 2, 1, 2
    clusters = [[0, 1, 2], [3, 4], [5], [6, 7]]
    q, k, v = (torch.randn(1, heads, n_tok, dim) for _ in range(3))
    for members in clusters:  # identical keys and values inside a cluster (what a merge of duplicates produces)
        for j in members[1:]:
            k[:, :, vis[j]], v[:, :, vis[j]] = k[:, :, vis[members[0]]], v[:, :, vis[members[0]]]
    full, _ = sdpa_attention_forward(_Decoder(), q, k, v, None)  # causal attention over every token
    keep = np.array([members[0] for members in clusters])
    after = slice(int(vis[-1]) + 1, n_tok)  # queries after the image see every cluster in full

    def pruned(with_term: bool):
        state = PruneState(vis_pos=vis)
        state.apply_keep(keep, 0)
        if with_term:
            state.key_bias = (keep, np.log([len(m) for m in clusters]).astype(np.float32))
        with pruning(state):
            out, _ = hook.medprune_sdpa(_Decoder(), q, k, v, None)
        return out

    torch.testing.assert_close(pruned(True)[:, after], full[:, after], atol=1e-6, rtol=0)
    assert (pruned(False)[:, after] - full[:, after]).abs().max() > 1e-2  # without the term the clusters shrink


def test_zero_size_term_equals_boolean_masking(models):
    """The float mask with a zero bias is the boolean mask: the numeric path changes nothing else."""
    _, hooked, _, _ = models
    state = preset(KEEP, 0)
    state.key_bias = (KEEP, np.zeros(len(KEEP), dtype=np.float32))
    torch.testing.assert_close(last_logits(hooked, IDS, state=state), last_logits(hooked, IDS, state=preset(KEEP, 0)),
                               atol=1e-6, rtol=0)


def _with_sizes() -> PruneState:
    state = preset(KEEP, 0)
    state.key_bias = (KEEP, np.log(np.arange(2, 2 + len(KEEP))).astype(np.float32))
    state.log_share = True
    return state


def test_size_term_is_identical_in_cached_decoding(models):
    """Prefill + cached decoding with the term equals a prefill of the whole sequence with it (the bias is zero-padded
    past the prompt), and the term changes the output."""
    _, hooked, _, _ = models
    tokens, cached = greedy(hooked, IDS, torch.arange(T), 5, state=_with_sizes())
    ids = torch.cat([IDS, torch.tensor(tokens)])
    scratch = last_logits(hooked, ids, pos=torch.arange(ids.numel()), state=_with_sizes())
    torch.testing.assert_close(cached, scratch, atol=1e-5, rtol=0)
    _, without = greedy(hooked, IDS, torch.arange(T), 5, state=preset(KEEP, 0))
    assert (cached - without).abs().max() > 1e-4


def test_image_share_is_logged_with_and_without_the_size_term(models):
    _, hooked, _, eager = models
    state = _with_sizes()
    last_logits(hooked, IDS, state=state)
    shares = np.stack([state.logs[f"ishare_l{i}"] for i in range(4)])  # [layer, (with, without)]
    assert shares.shape == (4, 2) and np.all(shares > 0) and np.all(shares < 1)
    assert np.all(shares[:, 0] > shares[:, 1])  # positive log sizes raise the image's share in every layer
    plain = preset(KEEP, 0)
    plain.log_share = True
    last_logits(hooked, IDS, state=plain)
    with torch.no_grad():  # without a term both columns equal the eager image share of the masked model
        attention = eager(input_ids=IDS[kept_positions()][None], position_ids=kept_positions()[None],
                          output_attentions=True).attentions
    for i in range(4):
        vis_cols = torch.isin(kept_positions(), VIS)
        expected = attention[i][0, :, -1, :].mean(0)[vis_cols].sum().item()
        np.testing.assert_allclose(plain.logs[f"ishare_l{i}"], [expected, expected], atol=1e-5)
