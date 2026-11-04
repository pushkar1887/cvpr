"""The published pruners (medprune/baselines/) select exactly the tokens their official code selects.

The official functions are loaded from the clones in third_party/ (git-ignored; see plan.md Appendix for the clone
commands) and run on the same tensors as the ports. Their repositories pin old transformers versions, so the
functions are extracted with `ast` and executed in a namespace with torch only, instead of importing the modules.
Tests that need a clone are skipped when it is absent.
"""

import ast
import importlib.util
import math
import os
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from medprune.baselines.btp import btp_first_fraction, btp_layers, div_prune, stage_attention, stage_balanced
from medprune.baselines.prunesid import select_prunesid
from medprune.baselines.sparsevlm import RETAIN_64, select_raters, sparsevlm_schedule, stage_keep
from medprune.baselines.vispruner import select_vispruner

ROOT = Path(__file__).resolve().parent.parent / "third_party"


def _source(path: Path):
    if not path.exists():
        pytest.skip(f"official code not cloned: {path}")
    text = path.read_text(encoding="utf-8")
    return text, ast.parse(text)


def _function(path: Path, name: str, cls: str | None = None, namespace: dict | None = None):
    """Compile the official function `cls.name` (or module-level `name`) in a torch-only namespace."""
    text, tree = _source(path)
    scope = tree.body
    if cls is not None:
        scope = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    node = next(n for n in scope if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    ns = {"torch": torch, "F": F, "np": np, "math": math, **(namespace or {})}
    exec(compile(ast.Module([node], []), str(path), "exec"), ns)
    return ns[name]


def _features(n: int, d: int = 64, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(n // 8 + 1, d, generator=g)  # clustered, so redundancy removal has something to remove
    return (base.repeat_interleave(8, 0)[:n] + 0.3 * torch.randn(n, d, generator=g)).float()


# ------------------------------------------------------------------------------------------------- PruneSID

def test_prunesid_matches_the_official_qwen2_plugin():
    path = ROOT / "prunesid" / "prunesid" / "prunesid_qwen" / "modeling_qwen2_vl.py"
    ns = {"pca_group": _function(path, "pca_group"), "nms": _function(path, "nms")}
    forward = _function(path, "forward", "Qwen2VisionTransformerPretrainedModel_prunesid", ns)
    for n, k, seed in [(400, 40, 1), (576, 58, 2), (1296, 130, 3)]:
        feats = _features(n, seed=seed)
        fake = SimpleNamespace(patch_embed=lambda x: x, rot_pos_emb=lambda g: None, blocks=[], merger=lambda x: x,
                               need_token_num=k * 576 / n)  # official count: round(n x need / 576) = k
        torch.manual_seed(seed)
        _, official = forward(fake, feats.clone(), torch.tensor([[1, 2, n // 2]]))
        ours = select_prunesid(feats.numpy(), k, seed)
        assert np.array_equal(np.sort(official.numpy()), ours), (n, k)


def test_prunesid_keeps_k_and_is_reproducible():
    feats = _features(64).numpy()
    a, b = select_prunesid(feats, 6, seed=5), select_prunesid(feats, 6, seed=5)
    assert len(a) == 6 and np.array_equal(a, b)  # no 16-token floor (matched budget), same seed -> same tokens


# ------------------------------------------------------------------------------------------------ VisPruner

def test_vispruner_matches_the_official_encode_images():
    path = ROOT / "vispruner" / "llava" / "model" / "llava_arch.py"
    encode = _function(path, "encode_images", "LlavaMetaForCausalLM")
    for n, k in [(576, 64), (300, 32), (64, 32)]:  # official domain: even residual count, >= 8 diverse tokens
        feats, attn = _features(n, seed=n), torch.rand(1, 1, n, generator=torch.Generator().manual_seed(n))
        tower = SimpleNamespace(get_vision_tower=lambda: (lambda images, output_attentions: (feats[None], attn)),
                                mm_projector=lambda x: x)
        fake = SimpleNamespace(get_model=lambda: tower, get_visual_token_num=lambda: k,
                               get_important_ratio=lambda: 0.5)
        _, masks = encode(fake, None)
        assert np.array_equal(np.flatnonzero(masks[0].numpy()), select_vispruner(attn[0, 0].numpy(), feats.numpy(), k))
    for n, k in [(301, 30), (64, 6), (16, 4)]:  # odd residual counts: the official loop raises, the port keeps k
        assert len(select_vispruner(np.random.rand(n), _features(n).numpy(), k)) == k


# ------------------------------------------------------------------------------------------------ SparseVLM

def test_sparsevlm_stage_matches_attn_postprocess_topk():
    path = ROOT / "sparsevlm" / "llava" / "model" / "language_model" / "score.py"
    if not path.exists():
        pytest.skip("SparseVLM not cloned")
    os.environ["RETAIN_TOKN"] = "64"
    spec = importlib.util.spec_from_file_location("sparsevlm_score", path)
    score = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score)
    assert tuple(score.sparse_token_dict[64]) == RETAIN_64
    v_start, n, t_start, total = 5, 120, 125, 150
    weights = torch.rand(1, 4, total, total, generator=torch.Generator().manual_seed(0)).softmax(-1)
    raters = (torch.zeros(9, dtype=torch.long), torch.arange(0, 25, 3))  # torch.where on [1, T]
    for layer, count in zip((2, 6, 15), RETAIN_64):
        mask, _, relation = score.attn_postprocess_topk(weights, v_start, n, t_start, raters, layer)
        ours = stage_keep(relation[0].numpy(), np.ones(n, bool), count)
        assert np.array_equal(np.flatnonzero(mask[0].numpy()), ours)


def test_sparsevlm_raters_follow_the_official_expression():
    g = torch.Generator().manual_seed(1)
    v_t, t_t = torch.randn(1, 50, 16, generator=g), torch.randn(1, 12, 16, generator=g)
    m_v_t = (v_t @ t_t.transpose(1, 2)).softmax(2).mean(1)  # modelling_sparse_llama.py L202-204
    official = torch.where(m_v_t > m_v_t.mean())[1].numpy()
    assert np.array_equal(select_raters(v_t[0], t_t[0]), official)


@pytest.mark.parametrize("layers", [28, 36])
def test_sparsevlm_schedule_is_token_matched_to_fastv(layers):
    stages = sparsevlm_schedule(0.10, layers)
    assert [s[:2] for s in stages] == [(2, 3), (6, 7), (15, 16)]
    fractions = [s[2] for s in stages]
    assert fractions[0] / fractions[1] == pytest.approx(66 / 30) and fractions[1] / fractions[2] == pytest.approx(30 / 17)
    mean = (3 + 4 * fractions[0] + 9 * fractions[1] + (layers - 16) * fractions[2]) / layers
    assert mean == pytest.approx((2 + (layers - 2) * 0.10) / layers)


# ------------------------------------------------------------------------------------------------------ BTP

def _btp_official():
    path = ROOT / "btp" / "qwen-2.5-vl" / "modeling_qwen2_5_vl.py"
    text, tree = _source(path)
    model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen2_5_VLModel")
    methods = {name: _function(path, name, "Qwen2_5_VLModel") for name in ("div_prune", "attn_prune")}
    forward = next(n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    blocks = {}
    for node in ast.walk(forward):  # the `if layer_index == self.<stage>` blocks, executed on their own
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and \
                isinstance(node.test.comparators[0], ast.Attribute):
            attr = node.test.comparators[0].attr
            if attr in ("start_layer", "img_sense_layer", "rel_start_layer"):
                body = "\n".join(textwrap.dedent(ast.get_source_segment(text, s, padded=True)) for s in node.body)
                blocks[attr] = compile(textwrap.dedent(body), str(path), "exec")
    return methods, blocks


def _run_btp_block(code, methods, hidden, attn_full, img_start, img_num, hidden_length):
    self = SimpleNamespace(img_start_idx=torch.tensor(img_start), img_num=img_num, hidden_length=hidden_length)
    self.div_prune = lambda *a: methods["div_prune"](self, *a)
    self.attn_prune = lambda *a: methods["attn_prune"](self, *a)
    length = hidden.shape[0]
    ns = {"torch": torch, "self": self, "hidden_states": hidden[None].clone(), "pre_layer_atten": attn_full,
          "position_embeddings": (torch.zeros(3, 1, length, 2), torch.zeros(3, 1, length, 2))}
    exec(code, ns)
    kept = ns["all_remained_index"].cpu().numpy()
    return kept[(kept >= img_start) & (kept < img_start + (hidden_length if hidden_length else img_num))] - img_start


def test_btp_stages_match_the_official_qwen25_code():
    methods, blocks = _btp_official()
    g = torch.Generator().manual_seed(3)
    n, start, d = 200, 7, 32
    seq = start + n + 15
    hidden = torch.randn(seq, d, generator=g)
    attn = torch.rand(1, 4, 1, seq, generator=g)
    head_mean = attn[0, :, -1, start:start + n].mean(0).numpy()
    vis = hidden[start:start + n].numpy()
    # stage 1 (before layer 3): k = N // 2
    official1 = _run_btp_block(blocks["start_layer"], methods, hidden, attn, start, n, None)
    assert np.array_equal(official1, stage_balanced(head_mean, vis, n // 2, attn_share=0.1, extend=1.2))
    # stage 2 (before layer 6): survivors compacted in raster order, k = survivors // 2
    m = len(official1)
    hidden2 = torch.cat([hidden[:start], hidden[start + official1], hidden[start + n:]])
    attn2 = torch.rand(1, 4, 1, hidden2.shape[0], generator=g)
    a2 = attn2[0, :, -1, start:start + m].mean(0).numpy()
    official2 = _run_btp_block(blocks["img_sense_layer"], methods, hidden2, attn2, start, n, m)
    assert np.array_equal(official2, stage_balanced(a2, hidden2[start:start + m].numpy(), m // 2, 0.4, 1.0))
    # stage 3 (before layer 15): attention only, the official `indices < k` preference
    m3 = len(official2)
    hidden3 = torch.cat([hidden2[:start], hidden2[start + official2], hidden2[start + m:]])
    attn3 = torch.rand(1, 4, 1, hidden3.shape[0], generator=g)
    a3 = attn3[0, :, -1, start:start + m3].mean(0).numpy()
    official3 = _run_btp_block(blocks["rel_start_layer"], methods, hidden3, attn3, start, n, m3)
    assert np.array_equal(official3, stage_attention(a3, m3 // 2))


def test_btp_div_prune_matches_the_official_method():
    methods, _ = _btp_official()
    feats = _features(90, seed=4)
    official = methods["div_prune"](None, feats.clone(), 20, [3, 50]).tolist()
    assert div_prune(feats, 20, [3, 50]) == [int(i) for i in official]


@pytest.mark.parametrize("layers", [28, 36])
def test_btp_schedule_is_token_matched_to_fastv(layers):
    l1, l2, l3, l4 = btp_layers(layers)
    assert (l1, l2, l3, l4) == ((3, 6, 15, 22) if layers == 28 else (4, 8, 19, 28))
    f = btp_first_fraction(0.10, layers)
    mean = (l1 + (l2 - l1) * f + (l3 - l2) * f / 2 + (l4 - l3) * f / 4) / layers
    assert mean == pytest.approx((2 + (layers - 2) * 0.10) / layers)


# ---------------------------------------------------------------------------------------------- end to end

def test_published_pruners_run_on_the_fake_model(tmp_path):
    from medprune.data import Sample, write_manifest
    from medprune.run_eval import RunSpec, run_many
    from PIL import Image
    import json

    bundle, out = tmp_path / "bundle", tmp_path / "results"
    (bundle / "images").mkdir(parents=True)
    samples = []
    for i in range(4):
        Image.fromarray(np.full((32, 32, 3), 50 * i, np.uint8)).save(bundle / "images" / f"{i}.png")
        samples.append(Sample(qid=f"toy/{i}", dataset="slake_en", image_id=f"toy/{i}", image_path=f"images/{i}.png",
                              question="Is it?", answer="yes", answer_type="yesno", prompt_style="close"))
    write_manifest(samples, bundle, "slake_en")
    specs = [RunSpec("lingshu7b", "slake_en", m, 0.25) for m in ("sparsevlm", "vispruner", "prunesid", "btp")]
    run_many(specs, bundle, out, {"lingshu7b": {"slake": 1.0, "text_only": 0.5}}, dry_run=True)
    rows = {s.method: [json.loads(l) for l in open(out / "runs" / f"{s.run_id('dryrun')}.jsonl")] for s in specs}
    assert all(len(r) == 4 for r in rows.values())
    assert all(r["k_keep"] == 4 for r in rows["vispruner"])  # 25% of the fake 4 x 4 grid
    assert all(len(r["keep_stages"]) == 3 for r in rows["sparsevlm"])
    btp = rows["btp"][0]["keep_stages"]
    assert [s[0] for s in btp] == [3, 6, 15, 22] and btp[-1][1] == []  # every visual token leaves at layer 22
