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
