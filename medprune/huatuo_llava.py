"""HuatuoGPT-Vision-7B without the authors' LLaVA code (plan.md §D3, task T5; E8, gap G6).

The checkpoint (FreedomIntelligence/HuatuoGPT-Vision-7B, `LlavaQwen2ForCausalLM`, checked 2026-11-03) is LLaVA-1.5 on
Qwen2-7B:
- vision: CLIP ViT-L/14-336, fine-tuned. Its 391 tensors are inside the main checkpoint under
  `model.vision_tower.vision_tower.*` (the authors load them with init_vision_encoder_from_ckpt=True); the stock CLIP
  in `vit/clip_vit_large_patch14_336` is read only for its config and preprocessor. Features from hidden layer -2,
  CLS dropped: 576 x 1024 on a 24 x 24 raster grid.
- projector `model.mm_projector.{0,2}`: Linear(1024, 3584) -> GELU -> Linear(3584, 3584) (mlp2x_gelu).
- language model: the keys `model.embed_tokens/layers/norm` and `lm_head` are exactly Qwen2ForCausalLM's, so the LLM is
  a stock transformers Qwen2ForCausalLM (1D RoPE, theta 1e6, 28 layers, 28 heads, 4 KV heads) running the pruning
  attention; the vision keys are ignored when it loads and are read from the same shards afterwards.
Prompt (third_party/HuatuoGPT-Vision/cli.py, preprocess_huatuo): "<|user|>\\n<image>\\n{prompt}\\n<|assistant|>\\n" with
no system prompt and no BOS; <|user|> and <|assistant|> are plain text, and the 576 projected features replace the
single <image> placeholder. Image: expand2square with the CLIP mean colour, then CLIP preprocessing to 336 x 336 —
masks.model_view builds the same view, so gate cells line up with the tokens. Generation stops at <|endoftext|>.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from PIL import Image

from . import attention_hook as hook
from .data import Sample, make_prompt
from .env import Profile, torch_dtype
from .masks import model_view
from .models import MODEL_IDS, PreparedInputs, QwenVLAdapter

VISION_PREFIX = "model.vision_tower.vision_tower."
PROJECTOR_PREFIX = "model.mm_projector."
VIT_SUBFOLDER = "vit/clip_vit_large_patch14_336"
IMAGE_PLACEHOLDER = "<image>"
QWEN2_FIELDS = ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "hidden_act", "max_position_embeddings", "rms_norm_eps", "rope_theta",
                "tie_word_embeddings", "use_sliding_window", "sliding_window", "max_window_layers",
                "attention_dropout", "bos_token_id", "eos_token_id")


def conversation(prompt: str, with_image: bool) -> str:
    """The single-turn HuatuoGPT conversation (cli.py: insert_image_placeholder, then preprocess_huatuo)."""
    text = ((IMAGE_PLACEHOLDER + "\n") if with_image else "") + prompt
    return f"<|user|>\n{text.strip()}\n<|assistant|>\n"


def split_checkpoint(tensors: dict[str, torch.Tensor]) -> tuple[dict, dict]:
    """Checkpoint tensors -> (CLIPVisionModel state dict, projector state dict)."""
    vision = {k[len(VISION_PREFIX):]: v for k, v in tensors.items() if k.startswith(VISION_PREFIX)}
    projector = {k[len(PROJECTOR_PREFIX):]: v for k, v in tensors.items() if k.startswith(PROJECTOR_PREFIX)}
    return vision, projector


def build_projector(mm_hidden: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(mm_hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))


def load_strict(module: nn.Module, state: dict, what: str, ignorable: tuple[str, ...] = ()) -> None:
    """load_state_dict that fails on any missing weight and on unexpected keys other than `ignorable` buffers."""
    result = module.load_state_dict(state, strict=False)
    unexpected = [k for k in result.unexpected_keys if not k.endswith(ignorable)] if ignorable else \
        list(result.unexpected_keys)
    if result.missing_keys or unexpected:
        raise RuntimeError(f"{what}: missing {result.missing_keys[:5]}, unexpected {unexpected[:5]}")


class HuatuoAdapter:
    """Same interface as models.QwenVLAdapter (prepare, generate, score_pass, visual_embeddings, ...)."""

    name = "huatuo7b"
    _record_layer1_norms = QwenVLAdapter._record_layer1_norms  # logs the layer-1 input norms of visual tokens

    def __init__(self, profile: Profile, vision_fp32: bool = False, revision: str | None = None):
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        from transformers import (AutoTokenizer, CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel, Qwen2Config,
                                  Qwen2ForCausalLM)

        repo = MODEL_IDS[self.name]
        dtype = torch_dtype(profile)
        config = json.loads(Path(hf_hub_download(repo, "config.json", revision=revision)).read_text())
        text_config = Qwen2Config(**{k: config[k] for k in QWEN2_FIELDS if k in config})
        lm = Qwen2ForCausalLM.from_pretrained(repo, revision=revision, config=text_config, dtype=dtype,
                                              device_map=profile.device_map, max_memory=profile.max_memory,
                                              attn_implementation=hook.ATTN_NAME)
        index = json.loads(Path(hf_hub_download(repo, "model.safetensors.index.json", revision=revision)).read_text())
        wanted = {k: shard for k, shard in index["weight_map"].items()
                  if k.startswith((VISION_PREFIX, PROJECTOR_PREFIX))}
        tensors = {}
        for shard in sorted(set(wanted.values())):
            with safe_open(hf_hub_download(repo, shard, revision=revision), framework="pt") as f:
                tensors.update({k: f.get_tensor(k) for k in wanted if wanted[k] == shard})
        vision_state, projector_state = split_checkpoint(tensors)
        vision = CLIPVisionModel._from_config(CLIPVisionConfig.from_pretrained(repo, subfolder=VIT_SUBFOLDER,
                                                                               revision=revision),
                                             attn_implementation="sdpa")
        load_strict(vision, vision_state, "vision tower", ignorable=("position_ids",))
        projector = build_projector(config["mm_hidden_size"], config["hidden_size"])
        load_strict(projector, projector_state, "projector")
        tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
        processor = CLIPImageProcessor.from_pretrained(repo, subfolder=VIT_SUBFOLDER, revision=revision)
        device = lm.get_input_embeddings().weight.device
        self._setup(lm, vision.to(device, torch.float32 if vision_fp32 else dtype), projector.to(device, dtype),
                    tokenizer, processor, dtype)

    @classmethod
    def from_parts(cls, lm, vision, projector, tokenizer, processor, dtype=torch.float32) -> "HuatuoAdapter":
        """Assemble from loaded modules (CPU tests with tiny random weights)."""
        adapter = cls.__new__(cls)
        adapter._setup(lm, vision, projector, tokenizer, processor, dtype)
        return adapter

    def _setup(self, lm, vision, projector, tokenizer, processor, dtype) -> None:
        self.lm, self.vision, self.projector = lm.eval(), vision.eval(), projector.eval()
        self.tokenizer, self.processor, self.dtype = tokenizer, processor, dtype
        if lm.config._attn_implementation != hook.ATTN_NAME:
            raise RuntimeError("could not install the pruning attention on the language model")
        if vision.config._attn_implementation == hook.ATTN_NAME:
            raise RuntimeError("pruning attention must not run in the vision tower")
        self.layers = lm.model.layers
        self.num_layers = len(self.layers)
        self.device = lm.get_input_embeddings().weight.device
        self.eos_token_id = tokenizer.eos_token_id
        attn = self.layers[1].self_attn
        hook.install_pre_rope_capture(attn.q_proj, attn.k_proj, layer=1)  # 1D RoPE: q_proj/k_proj precede it
        hook.install_profile_hooks(self.layers)
        hook.install_layer_hooks(self.layers)  # raters (SparseVLM) and deferred stages (BTP)
        self._visual_out: torch.Tensor | None = None
        self.layers[1].register_forward_pre_hook(self._record_layer1_norms, with_kwargs=True)

    def _ids(self, text: str) -> torch.Tensor:
        return torch.tensor(self.tokenizer(text, add_special_tokens=False).input_ids, dtype=torch.long,
                            device=self.device)

    @torch.inference_mode()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """LLM-input visual embeddings [576, hidden] of the model's view of `image`."""
        view, _ = model_view(image, self.name)  # expand2square + 336² bicubic, the view the masks are built on
        pixels = self.processor(images=view, return_tensors="pt")["pixel_values"]
        pixels = pixels.to(self.vision.device, next(self.vision.parameters()).dtype)
        features = self.vision(pixel_values=pixels, output_hidden_states=True).hidden_states[-2][0, 1:]
        features = self.projector(features.to(self.projector[0].weight.dtype))
        self._visual_out = features.detach()
        return features.to(self.device)

    @torch.inference_mode()
    def prepare(self, sample: Sample, image: Image.Image | None) -> PreparedInputs:
        embed = self.lm.get_input_embeddings()
        text = conversation(make_prompt(sample), with_image=image is not None)
        if image is None:
            embeds = embed(self._ids(text))[None]
            mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=self.device)
            return PreparedInputs({"inputs_embeds": embeds, "attention_mask": mask}, torch.empty(0, dtype=torch.long),
                                  (0, 0), np.zeros((0, 2), int), embeds.shape[1])
        before, after = text.split(IMAGE_PLACEHOLDER)
        features = self.encode_image(image)
        ids_before = self._ids(before)
        embeds = torch.cat([embed(ids_before), features.to(embed.weight.dtype), embed(self._ids(after))])[None]
        n = features.shape[0]
        side = int(round(n ** 0.5))
        if side * side != n:
            raise RuntimeError(f"{n} visual tokens do not form a square grid")
        vis_pos = torch.arange(len(ids_before), len(ids_before) + n)
        rc = np.stack(np.divmod(np.arange(n), side), axis=1)  # raster order, row-major
        mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=self.device)
        return PreparedInputs({"inputs_embeds": embeds, "attention_mask": mask}, vis_pos, (side, side), rc,
                              embeds.shape[1], torch.arange(len(ids_before) + n, embeds.shape[1]))

    @torch.inference_mode()
    def generate(self, inputs: PreparedInputs, max_new_tokens: int) -> tuple[str, int]:
        # with inputs_embeds and no input_ids, generate() returns only the new tokens
        out = self.lm.generate(**inputs.model_kwargs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                               temperature=None, top_p=None, top_k=None, eos_token_id=self.eos_token_id,
                               pad_token_id=self.eos_token_id)
        new = out[0]
        return self.tokenizer.decode(new, skip_special_tokens=True).strip(), int(new.numel())

    def set_vision_fp32(self, enabled: bool) -> None:
        """Run CLIP in fp32 (Stage 3 fp16 check); the projector stays in the LLM dtype."""
        self.vision.to(torch.float32 if enabled else self.dtype)

    @torch.inference_mode()
    def last_logits(self, inputs: PreparedInputs) -> np.ndarray:
        return self.lm(**inputs.model_kwargs, use_cache=False, logits_to_keep=1).logits[0, -1].float().cpu().numpy()

    @torch.inference_mode()
    def score_pass(self, inputs: PreparedInputs) -> None:
        """Forward until the active state's stages have selected (two-pass methods); output is discarded."""
        try:
            self.lm(**inputs.model_kwargs, use_cache=False)
        except hook.StopForward:
            pass

    @torch.inference_mode()
    def visual_embeddings(self, inputs: PreparedInputs) -> np.ndarray:
        """LLM-input visual embeddings [N, d] (DivPrune features, artifact norms): the spliced projector outputs."""
        return inputs.model_kwargs["inputs_embeds"][0, inputs.vis_pos.to(self.device)].float().cpu().numpy()
