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
        self.model = cls.from_pretrained(self.repo, revision=revision, dtype=torch_dtype(profile),
                                         device_map=profile.device_map, max_memory=profile.max_memory,
                                         attn_implementation="sdpa").eval()
        self.base = self.model.model
        if vision_fp32:
            self.base.visual.float()
        self._install_text_hook()

        self.layers = self.base.language_model.layers
        self.num_layers = len(self.layers)
        self.image_token_id = self.model.config.image_token_id
        attn = self.layers[1].self_attn
        q_mod, k_mod = (attn.q_norm, attn.k_norm) if hasattr(attn, "q_norm") else (attn.q_proj, attn.k_proj)
        hook.install_pre_rope_capture(q_mod, k_mod, layer=1)  # RoPE-free scores (Qwen3: q/k norms precede RoPE)
        self._visual_out: torch.Tensor | None = None
        self.base.visual.register_forward_hook(self._record_visual)
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
        return PreparedInputs(dict(enc), vis_pos, grid, rc, ids.numel())

    @torch.inference_mode()
    def generate(self, inputs: PreparedInputs, max_new_tokens: int) -> tuple[str, int]:
        out = self.model.generate(**inputs.model_kwargs, max_new_tokens=max_new_tokens, do_sample=False,
                                  use_cache=True, temperature=None, top_p=None, top_k=None)
        new = out[0, inputs.input_len:]
        return self.processor.decode(new, skip_special_tokens=True), int(new.numel())

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


def load_adapter(model: str, profile: Profile, vision_fp32: bool = False):
    if model == "huatuo7b":
        raise NotImplementedError("HuatuoGPT-Vision port (medprune/huatuo_llava.py) is written at Stage 7, plan.md")
    return QwenVLAdapter(model, profile, vision_fp32=vision_fp32)
