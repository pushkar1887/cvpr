import numpy as np
import torch

from medprune.latency import kept_sequence, layer_flops, prefill, prefill_flops, visual_tokens_per_layer


def test_visual_tokens_per_layer():
    row = {"n_vis": 100, "k_keep": 10, "keep_idx": list(range(10)), "keep_stages": None}
    assert visual_tokens_per_layer(row, 2, "topk", 6).tolist() == [100, 100, 10, 10, 10, 10]
    assert visual_tokens_per_layer(row, 0, "divprune", 4).tolist() == [10, 10, 10, 10]
    staged = dict(row, keep_stages=[[2, list(range(40))], [4, list(range(5))]])
    assert visual_tokens_per_layer(staged, 2, "pdrop", 6).tolist() == [100, 100, 40, 40, 5, 5]
    assert visual_tokens_per_layer({"n_vis": 64, "keep_idx": None}, 0, "none", 3).tolist() == [64, 64, 64]


def test_prefill_flops():
    d, m = 8, 16
    assert layer_flops(3, d, m) == 4 * 3 * 64 + 2 * 9 * 8 + 2 * 3 * 8 * 16
    full = prefill_flops(20, np.full(4, 100), d, m)
    fastv = prefill_flops(20, np.array([100, 100, 10, 10]), d, m)
    assert fastv < full and prefill_flops(20, np.full(4, 10), d, m) < fastv


def test_flops_table_text_only_uses_the_image_token_count_as_reference(tmp_path):
    import json

    from medprune.latency import flops_table

    runs = tmp_path / "runs"
    runs.mkdir()
    rows = {"full__r100": {"n_vis": 100, "keep_idx": None}, "text_only__r100": {"n_vis": None, "keep_idx": None},
            "fastv__r010": {"n_vis": 100, "k_keep": 10, "keep_idx": list(range(10))}}
    for name, row in rows.items():
        (runs / f"lingshu7b__slake_en__{name}__s0__t4x2.jsonl").write_text(json.dumps({"qid": "q1", **row}) + "\n")
    t = flops_table(tmp_path, {"lingshu7b": {"slake_en": {"q1": 30}}}).set_index("method")["rel_flops"]
    assert t["full"] == 1.0
    assert 0 < t["text_only"] < t["fastv"] < 1.0


def test_kept_sequence():
    vis = np.arange(3, 9)  # 6 visual tokens at positions 3..8 of a 12-token sequence
    assert kept_sequence(12, vis, np.array([0, 5])).tolist() == [0, 1, 2, 3, 8, 9, 10, 11]


def _tiny_qwen25_text_model():
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel

    cfg = Qwen2_5_VLTextConfig(vocab_size=50, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                               num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                               rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]})
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    return Qwen2_5_VLTextModel(cfg).eval()


def test_layer_loop_matches_model_forward_with_and_without_removal():
    lm = _tiny_qwen25_text_model()
    L = 14
    embeds = torch.randn(1, L, 32)
    pos = torch.arange(L).view(1, 1, L).expand(3, 1, L).clone()
    pos[1, 0, 3:9] = torch.tensor([3, 3, 3, 4, 4, 4])  # an image-like block: distinct h / w components
    pos[2, 0, 3:9] = torch.tensor([3, 4, 5, 3, 4, 5])
    with torch.inference_mode():
        ref = lm(inputs_embeds=embeds, position_ids=pos, use_cache=False).last_hidden_state
        assert torch.allclose(prefill(lm, embeds, pos), ref, atol=1e-5)
        seq = kept_sequence(L, np.arange(3, 9), np.array([1, 4]))
        idx = torch.as_tensor(seq)
        ref0 = lm(inputs_embeds=embeds[:, idx], position_ids=pos[..., idx], use_cache=False).last_hidden_state
        assert torch.allclose(prefill(lm, embeds, pos, seq, switch=0), ref0, atol=1e-5)
        late = prefill(lm, embeds, pos, seq, switch=2)  # layers 0-1 see every token, so it is neither of the above
        assert late.shape == ref0.shape
        assert not torch.allclose(late, ref0, atol=1e-3) and not torch.allclose(late, ref[:, idx], atol=1e-3)


# ------------------------------------------------------------------------------------- T11: ScanPrune's own path

def test_masked_layer_loop_equals_the_causal_loop_and_the_bias_matters():
    """A zero key bias through the explicit causal mask reproduces the is_causal loop (with and without removal and
    with two stages); a nonzero bias on kept visual keys changes the output; a later stage cannot re-add a token."""
    import pytest

    lm = _tiny_qwen25_text_model()
    L = 14
    embeds = torch.randn(1, L, 32)
    pos = torch.arange(L).view(1, 1, L).expand(3, 1, L).clone()
    with torch.inference_mode():
        assert torch.allclose(prefill(lm, embeds, pos, key_bias=torch.zeros(L)), prefill(lm, embeds, pos), atol=1e-5)
        s1 = kept_sequence(L, np.arange(3, 9), np.array([0, 2, 4, 5]))
        s2 = kept_sequence(L, np.arange(3, 9), np.array([2, 5]))
        two = [(0, s1), (2, s2)]
        assert torch.allclose(prefill(lm, embeds, pos, stages=two, key_bias=torch.zeros(L)),
                              prefill(lm, embeds, pos, stages=two), atol=1e-5)
        bias = torch.zeros(L)
        bias[torch.as_tensor([3, 5, 7, 8])] = torch.log(torch.tensor([3.0, 2.0, 4.0, 1.0]))
        diff = (prefill(lm, embeds, pos, stages=two, key_bias=bias) - prefill(lm, embeds, pos, stages=two)).abs().max()
        assert diff > 1e-4
        with pytest.raises(ValueError):
            prefill(lm, embeds, pos, stages=[(0, s2), (2, s1)])


def test_scanprune_plan_matches_the_run_rows():
    """The timed selection is the accuracy runs' selection: same counts as `run_eval` at both operating points, nested
    stages, sizes summing to the foreground."""
    from medprune.baselines.vscan import vscan_schedule
    from medprune.latency import scanprune_plan
    from medprune.remedy import fastv_matched_count, flat_sched_counts
    from medprune.selection import keep_count

    side = 18
    rc = np.stack(np.divmod(np.arange(side * side), side), axis=1)
    allowed = (np.abs(rc - side / 2) < 6).all(axis=1)
    emb = np.random.default_rng(0).normal(size=(side * side, 16))
    n, L = side * side, 28
    k = keep_count(n, 0.10)
    for point, branch in (("10", "flat"), ("10", "sched"), ("eq", "flat"), ("eq", "sched")):
        stages, rows, w, log_s = scanprune_plan(emb, rc, allowed, L, point, branch, np.random.default_rng(1))
        if branch == "flat":
            want = [k if point == "10" else fastv_matched_count(n, k, L)]
        elif point == "10":
            want = list(flat_sched_counts(k, n, L, 20))
        else:
            f1, _, f2 = vscan_schedule(0.10, L, layer=20)
            want = [keep_count(n, f1), keep_count(n, f2)]
        assert [len(kept) for _, kept in stages] == want, (point, branch)
        if branch == "sched":
            assert stages[1][0] == 20 and set(stages[1][1]) <= set(stages[0][1])
        assert np.array_equal(rows, stages[0][1]) and round(float(np.exp(log_s).sum())) == int(allowed.sum())
