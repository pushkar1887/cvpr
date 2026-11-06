"""Model adapters: turn a Sample into model inputs, locate the visual tokens, generate (plan.md §D3).

Lingshu-7B is a Qwen2.5-VL checkpoint and Qwen3-VL-8B is Qwen3-VL; both go through QwenVLAdapter. The custom
attention is installed on the text decoder only — the vision tower keeps plain SDPA.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from . import attention_hook as hook
from .data import Sample, make_prompt
from .env import Profile, torch_dtype
from .masks import MODEL_VIEWS

MODEL_IDS = {
    "lingshu7b": "lingshu-medical-mllm/Lingshu-7B",
    "qwen3vl8b": "Qwen/Qwen3-VL-8B-Instruct",
    "huatuo7b": "FreedomIntelligence/HuatuoGPT-Vision-7B",
}


@dataclass
class PreparedInputs:
    model_kwargs: dict  # passed to model.generate / model.forward
    vis_pos: torch.Tensor  # [N] int64 (CPU), sequence positions of the visual tokens
    grid_hw: tuple[int, int]  # visual-token grid (H, W); N == H * W
    rc: np.ndarray  # [N, 2] (row, col) of each visual token
    input_len: int
    text_pos: torch.Tensor | None = None  # sequence positions after the image: question and chat-template suffix

    @property
    def cell_index(self) -> np.ndarray:
        """Raster index of each visual token's grid cell (to look up per-cell masks)."""
        return self.rc[:, 0] * self.grid_hw[1] + self.rc[:, 1]


class QwenVLAdapter:
    def __init__(self, model: str, profile: Profile, vision_fp32: bool = False, revision: str | None = None):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration

        self.name, self.repo = model, MODEL_IDS[model]
        view = MODEL_VIEWS[model]
        self.processor = AutoProcessor.from_pretrained(self.repo, revision=revision)
        image_processor = self.processor.image_processor
        image_processor.min_pixels, image_processor.max_pixels = view["min_pixels"], view["max_pixels"]
        image_processor.size = {"shortest_edge": view["min_pixels"], "longest_edge": view["max_pixels"]}
        self.merge = image_processor.merge_size

        cls = Qwen2_5_VLForConditionalGeneration if model == "lingshu7b" else Qwen3VLForConditionalGeneration
        self.dtype = torch_dtype(profile)
        self.model = cls.from_pretrained(self.repo, revision=revision, dtype=self.dtype,
                                         device_map=profile.device_map, max_memory=profile.max_memory,
                                         attn_implementation="sdpa").eval()
        self.base = self.model.model
        if vision_fp32:
            self.base.visual.float()
        self._install_text_hook()

        self.layers = self.base.language_model.layers
        self.num_layers = len(self.layers)
        self.image_token_id = self.model.config.image_token_id
        # RoPE-free scores (Qwen3: q/k norms precede RoPE): layer 1 (FastV K=2), 7 and 15 (FEATHER), L/2 (VScan)
        for layer in sorted({1, 7, 15, self.num_layers // 2} & set(range(self.num_layers))):
            attn = self.layers[layer].self_attn
            q_mod, k_mod = (attn.q_norm, attn.k_norm) if hasattr(attn, "q_norm") else (attn.q_proj, attn.k_proj)
            hook.install_pre_rope_capture(q_mod, k_mod, layer=layer)
        hook.install_profile_hooks(self.layers)  # visual-sink scores per layer, only under a profile state
        hook.install_layer_hooks(self.layers)  # raters (SparseVLM) and deferred stages (BTP)
        hook.install_ttv_hooks(self.layers)  # token transition variation (TransPrune), only when requested
        self._visual_out: torch.Tensor | None = None
        self.base.visual.register_forward_hook(self._record_visual)
        self.base.visual.register_forward_hook(self._apply_merge)  # VisionZip / VScan merged tokens
        self.im_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.layers[1].register_forward_pre_hook(self._record_layer1_norms, with_kwargs=True)

    def _install_text_hook(self) -> None:
        self.model.set_attn_implementation({"text_config": hook.ATTN_NAME, "vision_config": "sdpa"})
        text_config = self.base.language_model.config
        if text_config._attn_implementation != hook.ATTN_NAME:  # sub-config identity not preserved: set directly
            text_config._attn_implementation_internal = hook.ATTN_NAME
        if text_config._attn_implementation != hook.ATTN_NAME:
            raise RuntimeError("could not install the pruning attention on the text decoder")
        if self.base.visual.config._attn_implementation == hook.ATTN_NAME:
            raise RuntimeError("pruning attention must not run in the vision tower")

    def _record_visual(self, _module, _inputs, output) -> None:
        self._visual_out = (output[0] if isinstance(output, tuple) else output).detach()

    @staticmethod
    def _apply_merge(_module, _inputs, output):
        """Write merged tokens into the vision tower's output (and Qwen3-VL's deepstack features) under a state with a
        merge plan; the LLM then sees them at the merged tokens' own positions."""
        state = hook.ACTIVE
        if state is None or state.merge is None:
            return None
        if isinstance(output, tuple):
            main, deep = output[0], output[1]
            deep = [hook.merge_rows(x, state.merge) for x in deep] if isinstance(deep, (list, tuple)) else deep
            return (hook.merge_rows(main, state.merge), deep, *output[2:])
        return hook.merge_rows(output, state.merge)

    def _record_layer1_norms(self, _module, args, kwargs) -> None:
        state = hook.ACTIVE
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        if state is not None and state.log_layers and hidden is not None and hidden.shape[1] > 1:
            vis = state.vis_pos.to(hidden.device)
            state.logs["llm_norm_l1"] = hidden[0, vis].float().norm(dim=-1).cpu().numpy()
            if self._visual_out is not None:
                state.logs["vit_norm"] = self._visual_out.float().norm(dim=-1).cpu().numpy()

    def prepare(self, sample: Sample, image: Image.Image | None) -> PreparedInputs:
        content = ([{"type": "image"}] if image is not None else []) + [{"type": "text", "text": make_prompt(sample)}]
        text = self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                  add_generation_prompt=True)
        enc = self.processor(text=[text], images=[image] if image is not None else None, return_tensors="pt")
        enc = enc.to(self.model.device)
        ids = enc["input_ids"][0]
        if image is None:
            return PreparedInputs(dict(enc), torch.empty(0, dtype=torch.long), (0, 0), np.zeros((0, 2), int), ids.numel())
        vis_pos = (ids == self.image_token_id).nonzero().squeeze(1).cpu()
        _, h, w = (int(v) for v in enc["image_grid_thw"][0])
        grid = (h // self.merge, w // self.merge)
        position_ids, _ = self.base.get_rope_index(input_ids=enc["input_ids"], image_grid_thw=enc["image_grid_thw"],
                                                   attention_mask=enc["attention_mask"])
        pos = position_ids[:, 0, vis_pos.to(position_ids.device)].cpu().numpy()  # [3, N]: (t, h, w)
        rc = np.stack([pos[1] - pos[1].min(), pos[2] - pos[2].min()], axis=1)
        if len(vis_pos) != grid[0] * grid[1] or rc[:, 0].max() + 1 != grid[0] or rc[:, 1].max() + 1 != grid[1]:
            raise RuntimeError(f"visual tokens ({len(vis_pos)}) do not match grid {grid}")
        text_pos = torch.arange(int(vis_pos[-1]) + 1, ids.numel())  # <|vision_end|>, question, template suffix
        return PreparedInputs(dict(enc), vis_pos, grid, rc, ids.numel(), text_pos)

    @torch.inference_mode()
    def generate(self, inputs: PreparedInputs, max_new_tokens: int) -> tuple[str, int]:
        out = self.model.generate(**inputs.model_kwargs, max_new_tokens=max_new_tokens, do_sample=False,
                                  use_cache=True, temperature=None, top_p=None, top_k=None)
        new = out[0, inputs.input_len:]
        return self.processor.decode(new, skip_special_tokens=True), int(new.numel())

    def set_vision_fp32(self, enabled: bool) -> None:
        """Run the vision tower in fp32 (Stage 3 fp16 check); embeddings are cast back to the LLM dtype."""
        self.base.visual.to(torch.float32 if enabled else self.dtype)

    @torch.inference_mode()
    def last_logits(self, inputs: PreparedInputs) -> np.ndarray:
        """Next-token logits after the prompt (Stage 3: are they finite in fp16?)."""
        return self.model(**inputs.model_kwargs, use_cache=False).logits[0, -1].float().cpu().numpy()

    @torch.inference_mode()
    def score_pass(self, inputs: PreparedInputs) -> None:
        """Forward until the active state's stages have selected (two-pass methods); output is discarded."""
        try:
            self.model(**inputs.model_kwargs, use_cache=False)
        except hook.StopForward:
            pass

    @torch.inference_mode()
    def visual_embeddings(self, inputs: PreparedInputs) -> np.ndarray:
        """LLM-input visual embeddings [N, d] (DivPrune features, artifact norms)."""
        kw = inputs.model_kwargs
        out = self.base.get_image_features(kw["pixel_values"], kw["image_grid_thw"])
        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], (list, tuple)):  # Qwen3-VL: (embeds, deepstack)
            out = out[0]
        emb = out[0] if isinstance(out, (list, tuple)) else out
        return emb.float().cpu().numpy()

    @torch.inference_mode()
    def instruction_embeddings(self, inputs: PreparedInputs) -> np.ndarray:
        """LLM input embeddings [M, d] of the user turn's text after the image (CDPruner's instruction): from the token
        after <|vision_end|> up to the turn's <|im_end|>."""
        ids = inputs.model_kwargs["input_ids"][0]
        start = int(inputs.vis_pos[-1]) + 2
        ends = (ids == self.im_end_id).nonzero().squeeze(1).cpu().tolist()
        end = next((e for e in ends if e > start), None)
        if end is None:
            raise RuntimeError("no <|im_end|> after the image: cannot locate the instruction")
        return self.base.get_input_embeddings()(ids[start:end]).float().cpu().numpy()

    @torch.inference_mode()
    def vision_attention(self, inputs: PreparedInputs, blocks: tuple[int, ...] = ()) -> dict:
        """Vision-tower cues per LLM visual token, in raster order:
        attn  {block: [N]}  attention each token's patches receive at that block (head and query mean, merge-unit
                            mean), for every block in `blocks` and the last block;
        keys  [N, D]        the last block's keys after RoPE, head and merge-unit mean (VisionZip's metric);
        feats [N, unit x C] the last block's output concatenated over the merge unit (VisPruner's features);
        embeds [N, d]       the tower's output = the LLM-input visual embeddings (= visual_embeddings)."""
        visual = self.base.visual
        last = len(visual.blocks) - 1
        blocks = tuple(sorted(set(blocks) | {last}))
        if hasattr(visual, "fullatt_block_indexes"):
            windowed = [b for b in blocks if b not in visual.fullatt_block_indexes]
            if windowed:
                raise RuntimeError(f"vision blocks {windowed} use window attention; the cues need full attention")
        captured: dict = {}

        def take_input(index):
            def hook_fn(_module, args, kwargs):
                captured[index] = (kwargs.get("hidden_states", args[0] if args else None),
                                   kwargs.get("position_embeddings"))
            return hook_fn

        def take_output(_module, _args, output):
            captured["out"] = output[0] if isinstance(output, tuple) else output

        handles = [visual.blocks[b].attn.register_forward_pre_hook(take_input(b), with_kwargs=True) for b in blocks]
        handles.append(visual.blocks[last].register_forward_hook(take_output))
        try:
            kw = inputs.model_kwargs
            grid_thw = kw["image_grid_thw"]
            tower_out = visual(kw["pixel_values"].type(visual.dtype), grid_thw=grid_thw)
        finally:
            for h in handles:
                h.remove()
        unit = self.merge ** 2
        order = None
        if hasattr(visual, "get_window_index"):  # Qwen2.5-VL: blocks run in window order; tokens come out reversed
            window_index, _ = visual.get_window_index(grid_thw)
            order = torch.argsort(window_index.to(captured["out"].device))
        attn_out, keys = {}, None
        for b in blocks:
            attn, (x, (cos, sin)) = visual.blocks[b].attn, captured[b]
            p = x.shape[0]
            q, k, _ = attn.qkv(x).reshape(p, 3, attn.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
            rope = __import__(type(attn).__module__, fromlist=["apply_rotary_pos_emb_vision"]).apply_rotary_pos_emb_vision
            q, k = rope(q, k, cos, sin)
            received = torch.zeros(p, dtype=torch.float32, device=x.device)
            for h in range(attn.num_heads):  # one head at a time: P x P in float32 stays ~100 MB
                weights = (q[:, h].float() @ k[:, h].float().T * attn.scaling).softmax(dim=-1)
                received += weights.mean(dim=0)
            received /= attn.num_heads
            scores = received.reshape(p // unit, unit).mean(dim=1)
            attn_out[b] = scores if order is None else scores[order.to(scores.device)]
            if b == last:
                keys = k.float().mean(dim=1).reshape(p // unit, unit, -1).mean(dim=1)
                keys = keys if order is None else keys[order.to(keys.device)]
        feats = captured["out"].float().reshape(captured["out"].shape[0] // unit, -1)
        merged = visual.merger(captured["out"])
        if order is not None:
            feats, merged = feats[order.to(feats.device)], merged[order.to(merged.device)]
        tower_out = tower_out[0] if isinstance(tower_out, tuple) else tower_out
        if not torch.allclose(merged.float(), tower_out.float(), atol=1e-2, rtol=1e-2):
            raise RuntimeError("vision cues are not aligned with the LLM's visual tokens")
        n = len(inputs.vis_pos)
        if feats.shape[0] != n:
            raise RuntimeError(f"vision cues for {feats.shape[0]} tokens, prompt has {n}")
        return {"attn": {b: a.cpu().numpy() for b, a in attn_out.items()}, "keys": keys.cpu().numpy(),
                "feats": feats.cpu().numpy(), "embeds": tower_out.float().cpu().numpy()}

    def vision_cues(self, inputs: PreparedInputs) -> tuple[np.ndarray, np.ndarray]:
        """VisPruner's visual cues per LLM visual token (medprune/baselines/vispruner.py): the attention each token's
        patches receive in the vision tower's last block (head and query mean, averaged over the merge unit) [N],
        and that block's output concatenated over the merge unit [N, unit x C] (pre-projector features)."""
        cues = self.vision_attention(inputs)
        return cues["attn"][len(self.base.visual.blocks) - 1], cues["feats"]


def load_adapter(model: str, profile: Profile, vision_fp32: bool = False):
    if model == "huatuo7b":
        from .huatuo_llava import HuatuoAdapter  # LLaVA-1.5 on Qwen2, without the authors' code (task T5)

        return HuatuoAdapter(profile, vision_fp32=vision_fp32)
    return QwenVLAdapter(model, profile, vision_fp32=vision_fp32)
