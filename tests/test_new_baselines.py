"""The pruners added Nov 7 (plan.md E14) select and merge exactly what their official code does.

Official code runs from the clones in third_party/ (git-ignored): functions are extracted with `ast` (as in
test_baselines.py), and the selection blocks that VisionZip and HiPrune inline in their model forward are extracted
as source and executed on the same tensors as the ports. Tests that need a clone are skipped when it is absent.
"""

import ast
import math
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from medprune import attention_hook as hook
from medprune.attention_hook import PruneState, Stage, install_pre_rope_capture, pruning
from medprune.baselines.cdpruner import instruction_relevance, select_cdpruner
from medprune.baselines.feather import feather_fractions, stage1_keep, uniform_grid
from medprune.baselines.hiprune import select_hiprune
from medprune.baselines.transprune import compute_transf_weight, stage_keep, transition_weight
from medprune.baselines.visionzip import select_visionzip
from medprune.baselines.vscan import merge_weights, stage1, token_merging, vscan_schedule, window_selection

ROOT = Path(__file__).resolve().parent.parent / "third_party"


def _text(path: Path) -> str:
    if not path.exists():
        pytest.skip(f"official code not cloned: {path}")
    return path.read_text(encoding="utf-8")


def _function(path: Path, name: str, cls: str | None = None):
    tree = ast.parse(_text(path))
    scope = tree.body
    if cls is not None:
        scope = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    node = next(n for n in scope if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    ns = {"torch": torch, "F": F, "np": np, "math": math}
    exec(compile(ast.Module([node], []), str(path), "exec"), ns)
    return ns[name]


def _block(path: Path, start: str, end: str) -> str:
    """Source lines from the line containing `start` up to (excluding) the next line containing `end`, dedented."""
    lines = _text(path).splitlines()
    i = next(n for n, line in enumerate(lines) if start in line)
    j = next(n for n in range(i + 1, len(lines)) if end in lines[n])
    return textwrap.dedent("\n".join(lines[i:j]))


def _sequence(n_vis: int, d: int, seed: int):
    """input_ids with n_vis image tokens (id 7) after 3 text tokens, and matching embeddings / positions."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.cat([torch.tensor([1, 2, 3]), torch.full((n_vis,), 7), torch.tensor([4, 5])])[None]
    t = ids.shape[1]
    return ids, torch.randn(1, t, d, generator=g), torch.arange(t).expand(3, 1, t).clone(), torch.ones(1, t)


# ------------------------------------------------------------------------------------------------ VisionZip

@pytest.mark.parametrize("n,k", [(64, 6), (64, 12), (300, 30), (1200, 120)])
def test_visionzip_matches_the_official_qwen25_block(n, k):
    src = _block(ROOT / "VisionZip" / "Qwen2_5_VL" / "qwen2_5vl_visionzip.py", "if select_pixel:", "torch.cuda.empty_cache()")
    g = torch.Generator().manual_seed(n + k)
    attn, keys = torch.rand(n, generator=g), torch.randn(1, n, 16, generator=g)
    ids, embeds, pos, mask = _sequence(n, 24, n)
    kept, merge = select_visionzip(attn.numpy(), keys[0].numpy(), k)
    contextual = merge[1].shape[0]
    src = src.replace("int(0.65 * attn_logits.size(0))", str(k - contextual)) \
             .replace("max(int(0.05 * attn_logits.size(0)),1)", str(contextual))
    assert str(k - contextual) in src and f"contextual_num = {contextual}" in src
    ns = {"torch": torch, "select_pixel": True, "attn_logits": attn.clone(), "attn_key": keys.clone(),
          "inputs_embeds": embeds.clone(), "input_ids": ids, "position_ids": pos, "attention_mask": mask,
          "self": SimpleNamespace(config=SimpleNamespace(image_token_id=7))}
    exec(compile(src, "visionzip_block", "exec"), ns)
    official = ns["inputs_embeds"][0, 3:3 + k]
    vis = embeds[0, 3:3 + n].double().numpy()
    ours = vis.copy()
    ours[merge[0]] = merge[1] @ vis
    assert len(kept) == k and ns["inputs_embeds"].shape[1] == ids.shape[1] - n + k
    np.testing.assert_allclose(ours[kept], official.double().numpy(), atol=1e-5)
    assert np.array_equal(ns["position_ids"][0, 0, 3:3 + k].numpy() - 3, kept)


def test_gated_visionzip_selects_and_merges_inside_the_foreground_only():
    rng = np.random.default_rng(0)
    attn, keys = rng.random(100), rng.normal(size=(100, 8))
    allowed = np.zeros(100, dtype=bool)
    allowed[20:70] = True
    kept, (rows, weights) = select_visionzip(attn, keys, 10, allowed)
    assert len(kept) == 10 and allowed[kept].all() and allowed[rows].all()
    assert not weights[:, ~allowed].any()


# -------------------------------------------------------------------------------------------------- HiPrune

@pytest.mark.parametrize("h,w,k", [(8, 8, 6), (18, 24, 43), (30, 40, 120)])
def test_hiprune_matches_the_official_qwen25_block(h, w, k):
    src = _block(ROOT / "HiPrune" / "Qwen2_5_VL" / "qwen2_5_vl_HiPrune.py", "##### HiPrune #####", "###################")
    n = h * w
    g = torch.Generator().manual_seed(n)
    attn_list = [torch.rand(n, generator=g) for _ in range(32)]
    ids, embeds, pos, mask = _sequence(n, 8, n)
    ns = {"torch": torch, "inputs_embeds": embeds, "attn_list": attn_list, "OBJECT_LAYER": 16, "RETAIN": k / n,
          "ALPHA": 0.1, "n_image_tokens": n, "n_image_features": n, "input_ids": ids, "position_ids": pos,
          "attention_mask": mask, "image_grid_thw": torch.tensor([[1, 2 * h, 2 * w]]),
          "self": SimpleNamespace(config=SimpleNamespace(image_token_id=7,
                                                         vision_config=SimpleNamespace(spatial_merge_size=2)))}
    exec(compile(src, "hiprune_block", "exec"), ns)
    official = np.flatnonzero(ns["select_mask"].numpy())
    ours = select_hiprune(attn_list[15].numpy(), attn_list[-1].numpy(), (h, w), k)
    assert np.array_equal(ours, official) and len(ours) == k


# ------------------------------------------------------------------------------------------------- CDPruner

@pytest.mark.parametrize("n,k", [(64, 6), (400, 40)])
def test_cdpruner_matches_the_official_encode_images(n, k):
    encode = _function(ROOT / "CDPruner" / "llava" / "model" / "llava_arch.py", "encode_images", "LlavaMetaForCausalLM")
    g = torch.Generator().manual_seed(n)
    visual, text = torch.randn(n, 64, generator=g), torch.randn(9, 64, generator=g)  # rank 64 >= k: no ties
    # The LLaVA code negates the CLIP similarity; feeding it -mean(text) reproduces Eq. 5 for projector embeddings.
    q = -text.mean(0, keepdim=True)
    tower = lambda images, texts=None: (visual[None], visual[None], q)  # noqa: E731
    model = SimpleNamespace(get_vision_tower=lambda: tower, mm_projector=lambda x: x)
    fake = SimpleNamespace(get_model=lambda: model, visual_token_num=k)
    _, masks = encode(fake, None, texts=["question"])
    official = np.flatnonzero(masks[0].numpy())
    assert np.array_equal(select_cdpruner(visual.numpy(), text.numpy(), k), official)


def test_cdpruner_neg_matches_the_official_llava_sign():
    encode = _function(ROOT / "CDPruner" / "llava" / "model" / "llava_arch.py", "encode_images", "LlavaMetaForCausalLM")
    g = torch.Generator().manual_seed(7)
    visual, text = torch.randn(200, 64, generator=g), torch.randn(9, 64, generator=g)
    tower = lambda images, texts=None: (visual[None], visual[None], text.mean(0, keepdim=True))  # noqa: E731
    model = SimpleNamespace(get_vision_tower=lambda: tower, mm_projector=lambda x: x)
    _, masks = encode(SimpleNamespace(get_model=lambda: model, visual_token_num=20), None, texts=["question"])
    assert np.array_equal(select_cdpruner(visual.numpy(), text.numpy(), 20, negate=True),
                          np.flatnonzero(masks[0].numpy()))


def test_cdpruner_greedy_map_maximizes_the_log_determinant_step_by_step():
    rng = np.random.default_rng(1)
    visual, text = rng.normal(size=(30, 6)), rng.normal(size=(4, 6))
    kept = select_cdpruner(visual, text, 5)
    v = visual / np.linalg.norm(visual, axis=1, keepdims=True)
    r = instruction_relevance(visual, text)
    r = (r - r.min() + 1e-6) / (r.max() - r.min())
    kernel = r[:, None] * (v @ v.T) * r[None]
    chosen: list[int] = []
    for _ in range(5):
        gains = [np.linalg.slogdet(kernel[np.ix_(chosen + [j], chosen + [j])])[1] if j not in chosen else -np.inf
                 for j in range(30)]
        chosen.append(int(np.argmax(gains)))
    assert np.array_equal(kept, np.sort(chosen))


# ---------------------------------------------------------------------------------------------------- VScan

def test_vscan_merge_matches_the_official_token_merging():
    merge = _function(ROOT / "VScan_repo" / "VScan" / "qwen" / "model" / "qwen2_5_vl_utils.py", "token_merging")
    g = torch.Generator().manual_seed(3)
    embeds = torch.randn(80, 16, generator=g)
    keep = torch.sort(torch.randperm(80, generator=g)[:12]).values
    official = merge(embeds, keep, scaling=1)
    np.testing.assert_allclose(merge_weights(embeds, keep) @ embeds.double().numpy(), official.double().numpy(),
                               atol=1e-5)
    np.testing.assert_allclose(token_merging(embeds, keep), official)


@pytest.mark.parametrize("h,w,k", [(8, 8, 8), (18, 26, 20), (9, 13, 7)])
def test_vscan_window_selection_matches_the_official_function(h, w, k):
    official = _function(ROOT / "VScan_repo" / "VScan" / "qwen" / "model" / "qwen2_5_vl_utils.py", "window_selection")
    attn = torch.rand(h * w, generator=torch.Generator().manual_seed(h * w))
    expected = official(attn, k, torch.tensor([[1, 2 * h, 2 * w]]), window_size=4)
    assert torch.equal(window_selection(attn, k, h, w), expected)


def test_vscan_stage1_keeps_local_then_global_and_merges_the_rest():
    rng = np.random.default_rng(4)
    local, glob, embeds = rng.random(64), rng.random(64), rng.normal(size=(64, 8))
    kept, (rows, weights) = stage1(local, glob, embeds, (8, 8), 16)
    assert len(kept) == 16 and np.array_equal(rows, kept)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    assert (weights > 0).sum() == 64  # every token lands in exactly one merged token


@pytest.mark.parametrize("layers", [28, 36])
def test_vscan_schedule_is_token_matched_to_fastv(layers):
    a, layer, b = vscan_schedule(0.10, layers)
    assert layer == layers // 2 and b == pytest.approx(a * 0.333)
    mean = (layer * a + (layers - layer) * b) / layers
    assert mean == pytest.approx((2 + (layers - 2) * 0.10) / layers)


# ----------------------------------------------------------------------------------------------- TransPrune

def test_transprune_weight_matches_the_official_function_on_the_live_tokens():
    official = _function(ROOT / "TransPrune" / "llava" / "model" / "modeling_llama_transprune.py",
                         "compute_transf_weight", "LlamaDecoderLayer_TransPrune")
    g = torch.Generator().manual_seed(5)
    before, after = torch.randn(30, 16, generator=g), torch.randn(30, 16, generator=g)
    live = torch.ones(30, dtype=torch.bool)
    torch.testing.assert_close(transition_weight(before, after, live), official(None, before[None], after[None])[0])
    live[[3, 7, 11, 12]] = False  # masked tokens: the official code would have removed them
    ours = transition_weight(before, after, live)
    torch.testing.assert_close(ours[live], official(None, before[live][None], after[live][None])[0])
    assert not ours[~live].any()
    torch.testing.assert_close(compute_transf_weight(before[None], after[None]), official(None, before[None], after[None]))


def test_transprune_stage_adds_normalized_ttv_and_attention():
    survivors = np.ones(6, dtype=bool)
    survivors[0] = False
    iga, ttv = np.array([9, 1, 2, 3, 4, 8.0]), np.array([9, 5, 4, 3, 2, 1.0])
    # over the survivors: ttv / 15 + iga / 18 = 0.389, 0.378, 0.367, 0.356, 0.511 -> tokens 5 and 1
    assert np.array_equal(stage_keep(iga, ttv, survivors, 2), [1, 5])


# -------------------------------------------------------------------------------------------------- FEATHER

def test_feather_uses_a_stride_three_grid_and_a_cubic_schedule():
    rc = np.array([(r, c) for r in range(8) for c in range(8)])
    uni = uniform_grid(rc)
    assert len(uni) == 9 and set(rc[uni, 0]) == {0, 3, 6}
    a, b = feather_fractions(0.10)
    assert a ** 3 == pytest.approx(b) and b == 0.10
    scores = np.arange(64, dtype=float)
    kept = stage1_keep(scores, np.ones(64, dtype=bool), uni, 30)
    assert len(kept) == 30 and set(uni) <= set(kept)
    assert set(kept) - set(uni) == set([i for i in range(63, -1, -1) if i not in set(uni)][:21])


# ------------------------------------------------------------------------------- hooks: RoPE-free and TTV

@pytest.fixture(scope="module")
def tiny():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                                         attn_implementation=hook.ATTN_NAME)).eval()
    return model


VIS = torch.arange(4, 20)
IDS = torch.randint(0, 97, (24,), generator=torch.Generator().manual_seed(1))


def test_ropefree_scores_at_a_later_stage_exclude_dropped_keys(tiny):
    attn = tiny.model.layers[2].self_attn
    handles = install_pre_rope_capture(attn.q_proj, attn.k_proj, layer=2)
    try:
        state = PruneState(vis_pos=VIS)
        state.apply_keep(np.arange(0, 16, 2), 0)
        state.stages.append(Stage(2, 3, lambda s, surv: np.flatnonzero(surv)[:4], score_kind="ropefree"))
        with torch.no_grad(), pruning(state):
            tiny(input_ids=IDS[None])
        q = state.pre_rope["q"].reshape(1, 24, -1, attn.head_dim)[0]
        k = state.pre_rope["k"].reshape(1, 24, -1, attn.head_dim)[0]
        rep = q.shape[1] // k.shape[1]
        dropped = VIS[1::2]
        heads = []
        for h in range(q.shape[1]):
            logits = (k[:, h // rep] @ q[-1, h]) * attn.scaling
            logits[dropped] = float("-inf")
            heads.append(logits.softmax(0))
        expected = torch.stack(heads).mean(0)[VIS].numpy()
        np.testing.assert_allclose(state.logs["scores_ropefree_l2"], expected, atol=1e-6)
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def test_ttv_under_masking_equals_the_official_weights_on_the_pruned_sequence(tiny):
    """Tokens masked from layer 0 = removed: the kept tokens' TTV at layer 1 equals the official computation on the
    physically shortened sequence."""
    official = _function(ROOT / "TransPrune" / "llava" / "model" / "modeling_llama_transprune.py",
                         "compute_transf_weight", "LlamaDecoderLayer_TransPrune")
    keep = np.array([0, 3, 5, 6, 10, 15])
    handles = hook.install_ttv_hooks(tiny.model.layers)
    try:
        state = PruneState(vis_pos=VIS, ttv_layers=(1,))
        state.apply_keep(keep, 0)
        with pruning(state):
            tiny(input_ids=IDS[None])
    finally:
        for h in handles:
            h.remove()
    pos = torch.sort(torch.cat([torch.arange(0, 4), torch.arange(20, 24), VIS[torch.from_numpy(keep)]])).values
    captured = {}
    layer = tiny.model.layers[1]
    hs = [layer.self_attn.register_forward_hook(lambda m, a, kw, o: captured.update(attn=(kw["hidden_states"], o[0])),
                                                with_kwargs=True),
          layer.mlp.register_forward_hook(lambda m, a, o: captured.update(mlp=(a[0], o)))]
    try:
        tiny(input_ids=IDS[pos][None], position_ids=pos[None])
    finally:
        for h in hs:
            h.remove()
    ref = sum(official(None, b, a)[0] for b, a in captured.values())
    vis_in_short = torch.isin(pos, VIS[torch.from_numpy(keep)]).nonzero().squeeze(1)
    np.testing.assert_allclose(state.ttv[1][keep], ref[vis_in_short].numpy(), rtol=1e-4, atol=1e-7)


def test_merge_rows_rewrites_only_the_merged_rows():
    x = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    w = np.array([[0.5, 0.5, 0, 0]])
    out = hook.merge_rows(x, (np.array([1]), w))
    torch.testing.assert_close(out[1], (x[0] + x[1]) / 2)
    torch.testing.assert_close(out[[0, 2, 3]], x[[0, 2, 3]])


# ------------------------------------------------------------------------------ vision cues on tiny towers

def _old_vision_cues(adapter, inputs):
    """medprune.models.QwenVLAdapter.vision_cues before Nov 7 (the code VisPruner's E13 runs used)."""
    visual = adapter.base.visual
    block = visual.blocks[-1]
    captured: dict = {}

    def take_input(_module, args, kwargs):
        captured["x"] = kwargs.get("hidden_states", args[0] if args else None)
        captured["pe"] = kwargs.get("position_embeddings")

    def take_output(_module, _args, output):
        captured["out"] = output[0] if isinstance(output, tuple) else output

    handles = [block.attn.register_forward_pre_hook(take_input, with_kwargs=True), block.register_forward_hook(take_output)]
    try:
        kw = inputs.model_kwargs
        grid_thw = kw["image_grid_thw"]
        visual(kw["pixel_values"].type(visual.dtype), grid_thw=grid_thw)
    finally:
        for h in handles:
            h.remove()
    attn, x = block.attn, captured["x"]
    p = x.shape[0]
    q, k, _ = attn.qkv(x).reshape(p, 3, attn.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    cos, sin = captured["pe"]
    rope = __import__(type(attn).__module__, fromlist=["apply_rotary_pos_emb_vision"]).apply_rotary_pos_emb_vision
    q, k = rope(q, k, cos, sin)
    received = torch.zeros(p, dtype=torch.float32)
    for h in range(attn.num_heads):
        received += (q[:, h].float() @ k[:, h].float().T * attn.scaling).softmax(dim=-1).mean(dim=0)
    received /= attn.num_heads
    unit = adapter.merge ** 2
    scores = received.reshape(p // unit, unit).mean(dim=1)
    feats = captured["out"].float().reshape(p // unit, -1)
    if hasattr(visual, "get_window_index"):
        order = torch.argsort(visual.get_window_index(grid_thw)[0])
        scores, feats = scores[order], feats[order]
    return scores.numpy(), feats.numpy()


def _tower(kind: str):
    torch.manual_seed(0)
    if kind == "qwen25":
        from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel

        cfg = Qwen2_5_VLVisionConfig(depth=4, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=48,
                                     patch_size=14, spatial_merge_size=2, temporal_patch_size=2, window_size=56,
                                     fullatt_block_indexes=[1, 3], in_channels=3)
        cfg._attn_implementation = "sdpa"
        return Qwen2_5_VisionTransformerPretrainedModel(cfg).eval(), 3 * 2 * 14 * 14
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    cfg = Qwen3VLVisionConfig(depth=4, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=48,
                              patch_size=16, spatial_merge_size=2, temporal_patch_size=2, num_position_embeddings=64,
                              deepstack_visual_indexes=[1, 2], in_channels=3)
    cfg._attn_implementation = "sdpa"
    return Qwen3VLVisionModel(cfg).eval(), 3 * 2 * 16 * 16


@pytest.mark.parametrize("kind", ["qwen25", "qwen3"])
@torch.no_grad()
def test_vision_attention_keeps_the_vispruner_cues_and_adds_blocks_and_keys(kind):
    from medprune.models import PreparedInputs, QwenVLAdapter

    tower, dim = _tower(kind)
    adapter = object.__new__(QwenVLAdapter)
    adapter.base, adapter.merge = SimpleNamespace(visual=tower), 2
    grid = torch.tensor([[1, 8, 12]])
    pixel = torch.randn(96, dim, generator=torch.Generator().manual_seed(2))
    inputs = PreparedInputs({"pixel_values": pixel, "image_grid_thw": grid}, torch.arange(24), (4, 6),
                            np.array([(r, c) for r in range(4) for c in range(6)]), 30)
    old_scores, old_feats = _old_vision_cues(adapter, inputs)
    new_scores, new_feats = adapter.vision_cues(inputs)
    np.testing.assert_array_equal(new_scores, old_scores)
    np.testing.assert_array_equal(new_feats, old_feats)
    cues = adapter.vision_attention(inputs, (1,))
    assert set(cues["attn"]) == {1, 3} and cues["keys"].shape == (24, 16)
    np.testing.assert_allclose(cues["attn"][1].sum(), 0.25, rtol=1e-5)  # query mean sums to 1; merge-unit mean / 4
    if kind == "qwen25":
        with pytest.raises(RuntimeError, match="window attention"):
            adapter.vision_attention(inputs, (0,))


@torch.no_grad()
def test_merge_hook_rewrites_the_tower_output_and_qwen3_deepstack():
    from medprune.models import QwenVLAdapter

    tower, dim = _tower("qwen3")
    grid = torch.tensor([[1, 8, 12]])
    pixel = torch.randn(96, dim, generator=torch.Generator().manual_seed(2))
    plain = tower(pixel, grid_thw=grid)
    handle = tower.register_forward_hook(QwenVLAdapter._apply_merge)
    try:
        state = PruneState(vis_pos=torch.arange(24))
        state.merge = (np.array([5]), np.eye(24)[[5, 6]].mean(0, keepdims=True))
        with pruning(state):
            merged = tower(pixel, grid_thw=grid)
        unmerged = tower(pixel, grid_thw=grid)  # no active state: untouched
    finally:
        handle.remove()
    for a, b in zip([plain[0], *plain[1]], [merged[0], *merged[1]]):
        torch.testing.assert_close(b[5], (a[5] + a[6]) / 2)
        torch.testing.assert_close(b[[0, 6, 23]], a[[0, 6, 23]])
    torch.testing.assert_close(unmerged[0], plain[0])


def test_merge_ablation_runs_drop_or_merge_on_the_fake_model(tmp_path):
    import json

    from medprune.data import Sample, write_manifest
    from medprune.run_eval import RunSpec, run_many
    from PIL import Image

    bundle, out = tmp_path / "bundle", tmp_path / "results"
    (bundle / "images").mkdir(parents=True)
    samples = []
    for i in range(3):
        Image.fromarray(np.full((32, 32, 3), 60 * i, np.uint8)).save(bundle / "images" / f"{i}.png")
        samples.append(Sample(qid=f"toy/{i}", dataset="slake_en", image_id=f"toy/{i}", image_path=f"images/{i}.png",
                              question="Is it?", answer="yes", answer_type="yesno", prompt_style="close"))
    write_manifest(samples, bundle, "slake_en")
    names = ("visionzip", "visionzip_nomerge", "vscan", "vscan_nomerge", "fastv_merge", "fastv_prellm")
    specs = [RunSpec("lingshu7b", "slake_en", m, 0.25) for m in names]
    run_many(specs, bundle, out, {"lingshu7b": {"slake": 1.0, "text_only": 0.5}}, dry_run=True)
    rows = {s.method: [json.loads(l) for l in open(out / "runs" / f"{s.run_id('dryrun')}.jsonl")] for s in specs}
    for a, b in (("visionzip", "visionzip_nomerge"), ("vscan", "vscan_nomerge"), ("fastv_merge", "fastv_prellm")):
        assert [r["keep_idx"] for r in rows[a]] == [r["keep_idx"] for r in rows[b]], (a, b)  # same selection


def test_depth_matched_random_keeps_random_l2s_tokens(tmp_path):
    import json

    from medprune.data import Sample, write_manifest
    from medprune.run_eval import RunSpec, run_many
    from PIL import Image

    bundle, out = tmp_path / "bundle", tmp_path / "results"
    (bundle / "images").mkdir(parents=True)
    samples = []
    for i in range(3):
        Image.fromarray(np.full((32, 32, 3), 60 * i, np.uint8)).save(bundle / "images" / f"{i}.png")
        samples.append(Sample(qid=f"toy/{i}", dataset="slake_en", image_id=f"toy/{i}", image_path=f"images/{i}.png",
                              question="Is it?", answer="yes", answer_type="yesno", prompt_style="close"))
    write_manifest(samples, bundle, "slake_en")
    specs = [RunSpec("lingshu7b", "slake_en", m, 0.25) for m in ("random_l2", "random_l8", "random_l16")]
    run_many(specs, bundle, out, {"lingshu7b": {"slake": 1.0, "text_only": 0.5}}, dry_run=True)
    keeps = [[json.loads(l)["keep_idx"] for l in open(out / "runs" / f"{s.run_id('dryrun')}.jsonl")] for s in specs]
    assert keeps[0] == keeps[1] == keeps[2]  # same tokens, only the layer the drop starts at differs


def test_random_merge_keeps_random_l0s_tokens(tmp_path):
    import json

    from medprune.data import Sample, write_manifest
    from medprune.run_eval import RunSpec, run_many
    from PIL import Image

    bundle, out = tmp_path / "bundle", tmp_path / "results"
    (bundle / "images").mkdir(parents=True)
    samples = []
    for i in range(3):
        Image.fromarray(np.full((32, 32, 3), 60 * i, np.uint8)).save(bundle / "images" / f"{i}.png")
        samples.append(Sample(qid=f"toy/{i}", dataset="slake_en", image_id=f"toy/{i}", image_path=f"images/{i}.png",
                              question="Is it?", answer="yes", answer_type="yesno", prompt_style="close"))
    write_manifest(samples, bundle, "slake_en")
    specs = [RunSpec("lingshu7b", "slake_en", m, 0.25) for m in ("random_l0", "random_merge")]
    run_many(specs, bundle, out, {"lingshu7b": {"slake": 1.0, "text_only": 0.5}}, dry_run=True)
    keeps = [[json.loads(l)["keep_idx"] for l in open(out / "runs" / f"{s.run_id('dryrun')}.jsonl")] for s in specs]
    assert keeps[0] == keeps[1]  # the merge changes what the kept tokens carry, not which tokens are kept


def test_random_vscan_follows_vscans_schedule(tmp_path):
    import json

    from medprune.baselines.vscan import vscan_schedule
    from medprune.data import Sample, write_manifest
    from medprune.run_eval import RunSpec, run_many
    from medprune.selection import keep_count
    from PIL import Image

    bundle, out = tmp_path / "bundle", tmp_path / "results"
    (bundle / "images").mkdir(parents=True)
    samples = []
    for i in range(3):
        Image.fromarray(np.full((64, 64, 3), 60 * i, np.uint8)).save(bundle / "images" / f"{i}.png")
        samples.append(Sample(qid=f"toy/{i}", dataset="slake_en", image_id=f"toy/{i}", image_path=f"images/{i}.png",
                              question="Is it?", answer="yes", answer_type="yesno", prompt_style="close"))
    write_manifest(samples, bundle, "slake_en")
    specs = [RunSpec("lingshu7b", "slake_en", m, 0.25) for m in ("random_vscan", "vscan")]
    run_many(specs, bundle, out, {"lingshu7b": {"slake": 1.0, "text_only": 0.5}}, dry_run=True)
    rows = {s.method: [json.loads(l) for l in open(out / "runs" / f"{s.run_id('dryrun')}.jsonl")] for s in specs}
    f1, _, f2 = vscan_schedule(0.25, 28)
    for r, v in zip(rows["random_vscan"], rows["vscan"]):
        n = r["n_vis"]
        assert r["k_keep"] == v["k_keep"] == keep_count(n, f2)  # same final count as VScan
        first, last = r["keep_stages"][0][1], r["keep_stages"][-1][1]
        assert len(first) == keep_count(n, f1) and set(last) <= set(first)


def test_random_sched20_moves_the_cut_and_keeps_the_token_budget(tmp_path):
    import json

    from medprune.baselines.vscan import vscan_schedule
    from medprune.data import Sample, write_manifest
    from medprune.run_eval import RunSpec, run_many
    from medprune.selection import keep_count
    from PIL import Image

    f1, layer, f2 = vscan_schedule(0.10, 28, layer=20)
    target = (2 + 26 * 0.10) / 28
    assert layer == 20 and abs((20 * f1 + 8 * f2) / 28 - target) < 1e-9  # same mean tokens per layer as FastV
    assert vscan_schedule(0.10, 28) == vscan_schedule(0.10, 28, layer=14)
    bundle, out = tmp_path / "bundle", tmp_path / "results"
    (bundle / "images").mkdir(parents=True)
    samples = []
    for i in range(2):
        Image.fromarray(np.full((64, 64, 3), 60 * i, np.uint8)).save(bundle / "images" / f"{i}.png")
        samples.append(Sample(qid=f"toy/{i}", dataset="slake_en", image_id=f"toy/{i}", image_path=f"images/{i}.png",
                              question="Is it?", answer="yes", answer_type="yesno", prompt_style="close"))
    write_manifest(samples, bundle, "slake_en")
    spec = RunSpec("lingshu7b", "slake_en", "random_sched20", 0.25)
    run_many([spec], bundle, out, {"lingshu7b": {"slake": 1.0, "text_only": 0.5}}, dry_run=True)
    f1, _, f2 = vscan_schedule(0.25, 28, layer=20)
    for r in (json.loads(l) for l in open(out / "runs" / f"{spec.run_id('dryrun')}.jsonl")):
        assert [s[0] for s in r["keep_stages"]] == [0, 20]
        assert len(r["keep_stages"][0][1]) == keep_count(r["n_vis"], f1) and r["k_keep"] == keep_count(r["n_vis"], f2)
