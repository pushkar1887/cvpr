"""HuatuoGPT-Vision adapter on tiny random weights (CPU, fp32): template, splicing, grid, pruning equivalence."""

from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel, Qwen2Config, Qwen2ForCausalLM

from medprune import attention_hook as hook
from medprune.attention_hook import PruneState, Stage, pruning
from medprune.data import Sample, make_prompt
from medprune.huatuo_llava import HuatuoAdapter, build_projector, conversation, split_checkpoint
from medprune.masks import model_view
from medprune.selection import select_topk


class CharTokenizer:
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(c) % 250 + 1 for c in text])

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(96 + int(i) % 26) for i in ids if int(i) != self.eos_token_id)


def tiny_adapter() -> HuatuoAdapter:
    torch.manual_seed(0)
    lm = Qwen2ForCausalLM(Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                                      attn_implementation=hook.ATTN_NAME))
    vision = CLIPVisionModel._from_config(CLIPVisionConfig(image_size=336, patch_size=14, hidden_size=16,
                                                           intermediate_size=32, num_hidden_layers=2,
                                                           num_attention_heads=2), attn_implementation="sdpa")
    processor = CLIPImageProcessor(size={"shortest_edge": 336}, crop_size={"height": 336, "width": 336},
                                   image_mean=[0.48145466, 0.4578275, 0.40821073],
                                   image_std=[0.26862954, 0.26130258, 0.27577711])
    return HuatuoAdapter.from_parts(lm, vision, build_projector(16, 64), CharTokenizer(), processor)


SAMPLE = Sample(qid="toy/0", dataset="slake_en", image_id="toy", image_path="toy.png", question="Is the liver normal?",
                answer="yes", answer_type="yesno", prompt_style="close")
IMAGE = Image.fromarray(np.random.default_rng(0).integers(0, 255, (200, 300, 3), dtype=np.uint8))  # padded to square


def test_conversation_is_the_authors_template():
    assert conversation("Is it?", True) == "<|user|>\n<image>\nIs it?\n<|assistant|>\n"
    assert conversation("Is it?", False) == "<|user|>\nIs it?\n<|assistant|>\n"


def test_checkpoint_split():
    tensors = {"model.vision_tower.vision_tower.vision_model.post_layernorm.weight": torch.ones(2),
               "model.mm_projector.0.weight": torch.ones(2, 2), "model.layers.0.mlp.up_proj.weight": torch.ones(1)}
    vision, projector = split_checkpoint(tensors)
    assert list(vision) == ["vision_model.post_layernorm.weight"] and list(projector) == ["0.weight"]


def test_prepare_splices_576_raster_tokens():
    adapter = tiny_adapter()
    inputs = adapter.prepare(SAMPLE, IMAGE)
    before = len(CharTokenizer()("<|user|>\n").input_ids)
    assert inputs.grid_hw == (24, 24) and torch.equal(inputs.vis_pos, torch.arange(before, before + 576))
    assert inputs.rc[25].tolist() == [1, 1] and inputs.cell_index.tolist() == list(range(576))
    view, box = model_view(IMAGE, "huatuo7b")
    assert view.size == (336, 336) and box == (0, 56, 336, 280)  # 300x200 padded top and bottom
    with torch.no_grad():
        pixels = adapter.processor(images=view, return_tensors="pt")["pixel_values"]
        expected = adapter.projector(adapter.vision(pixel_values=pixels, output_hidden_states=True).hidden_states[-2][0, 1:])
    torch.testing.assert_close(torch.from_numpy(adapter.visual_embeddings(inputs)), expected)
    text_only = adapter.prepare(SAMPLE, None)
    assert text_only.vis_pos.numel() == 0
    assert text_only.input_len == len(CharTokenizer()(conversation(make_prompt(SAMPLE), False)).input_ids)
    after = conversation(make_prompt(SAMPLE), True).split("<image>")[1]
    assert inputs.input_len == before + 576 + len(CharTokenizer()(after).input_ids)
    assert torch.equal(inputs.text_pos, torch.arange(before + 576, inputs.input_len))


@torch.no_grad()
def test_masking_equals_removing_tokens_with_original_positions():
    adapter = tiny_adapter()
    inputs = adapter.prepare(SAMPLE, IMAGE)
    keep = np.arange(0, 576, 9)  # 64 of 576 tokens
    state = PruneState(vis_pos=inputs.vis_pos)
    state.apply_keep(keep, 0)
    with pruning(state):
        masked = adapter.lm(**inputs.model_kwargs).logits[0, -1]
    embeds = inputs.model_kwargs["inputs_embeds"][0]
    dropped = set(inputs.vis_pos[np.setdiff1d(np.arange(576), keep)].tolist())
    kept_pos = torch.tensor([i for i in range(embeds.shape[0]) if i not in dropped])
    removed = adapter.lm(inputs_embeds=embeds[kept_pos][None], position_ids=kept_pos[None]).logits[0, -1]
    torch.testing.assert_close(masked, removed, atol=1e-5, rtol=1e-4)


def test_fastv_state_selects_k_tokens_and_generates():
    adapter = tiny_adapter()
    inputs = adapter.prepare(SAMPLE, IMAGE)
    state = PruneState(vis_pos=inputs.vis_pos, log_layers=(1,), log_ropefree=True)
    state.stages.append(Stage(1, 2, lambda scores, survivors: select_topk(scores, 58, survivors)))
    calls = hook.CALLS["decoder"]
    with pruning(state):
        text, n_new = adapter.generate(inputs, max_new_tokens=4)
    assert hook.CALLS["decoder"] > calls and len(state.keep) == 58 and 1 <= n_new <= 4 and isinstance(text, str)
    assert state.logs["scores_l1"].shape == (576,) and state.logs["scores_ropefree_l1"].shape == (576,)
    assert state.logs["llm_norm_l1"].shape == (576,) and state.logs["vit_norm"].shape == (576,)
