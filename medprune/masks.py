"""What counts as background: body masks and per-token foreground fractions (plan.md §A5, §D3).

The mask is computed on the exact image each model sees (after its own resize or padding), so token cells line up
with the model's visual-token grid. Pure NumPy/scikit-image: no torch, no transformers.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.measure import label

# Image views of each model (checked against the checkpoints' preprocessor configs, plan.md §C1).
MODEL_VIEWS: dict[str, dict] = {
    "lingshu7b": {"kind": "qwen", "cell": 28, "min_pixels": 3136, "max_pixels": 802_816},
    "qwen3vl8b": {"kind": "qwen", "cell": 32, "min_pixels": 65_536, "max_pixels": 1_048_576},
    "huatuo7b": {"kind": "llava_pad", "cell": 14, "size": 336, "pad_rgb": (122, 116, 104)},  # CLIP mean x 255
}

BORDER_FRACTION = 0.05  # width of the border ring used to decide polarity and coverage
MIN_COMPONENT_FRACTION = 0.10  # keep components at least this large relative to the largest
MIN_BORDER_BACKGROUND = 0.6  # at least this share of the border ring must be background, else the gate is a no-op
MAX_BACKGROUND_OFFSET = 0.2  # background pixels lie within this share of the range from the dark/bright end
DARK_BACKGROUND_MAX = 80  # ...and are absolutely dark (air, empty field) or bright (glass), in 0-255 luminance:
BRIGHT_BACKGROUND_MIN = 225  # light skin in a low-contrast dermoscopy image is "the bright end" but sits at ~185-210


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Qwen2-VL/Qwen3-VL resize rule (transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize)."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"aspect ratio too large: {height}x{width}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def model_view(image: Image.Image, model: str) -> tuple[Image.Image, tuple[int, int, int, int] | None]:
    """The image the model actually encodes, plus the content box (x0, y0, x1, y1) if the model pads."""
    view = MODEL_VIEWS[model]
    image = image.convert("RGB")
    if view["kind"] == "qwen":
        h, w = smart_resize(image.height, image.width, view["cell"], view["min_pixels"], view["max_pixels"])
        return image.resize((w, h), Image.BICUBIC), None
    # LLaVA "pad": expand to a square with the CLIP mean colour, then resize to the CLIP input size.
    side = max(image.width, image.height)
    canvas = Image.new("RGB", (side, side), view["pad_rgb"])
    x0, y0 = (side - image.width) // 2, (side - image.height) // 2
    canvas.paste(image, (x0, y0))
    scale = view["size"] / side
    box = (round(x0 * scale), round(y0 * scale), round((x0 + image.width) * scale), round((y0 + image.height) * scale))
    return canvas.resize((view["size"], view["size"]), Image.BICUBIC), box


def label_view(mask: Image.Image, model: str) -> np.ndarray:
    """A label image (SLAKE `mask.png`) in the model's view: model_view's geometry, nearest-neighbour resampling,
    padding = 0 (no structure). Cells of the result line up with the model's visual tokens."""
    view = MODEL_VIEWS[model]
    mask = mask.convert("L")
    if view["kind"] == "qwen":
        h, w = smart_resize(mask.height, mask.width, view["cell"], view["min_pixels"], view["max_pixels"])
        return np.asarray(mask.resize((w, h), Image.NEAREST))
    side = max(mask.width, mask.height)
    canvas = Image.new("L", (side, side), 0)
    canvas.paste(mask, ((side - mask.width) // 2, (side - mask.height) // 2))
    return np.asarray(canvas.resize((view["size"], view["size"]), Image.NEAREST))


def _luminance(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    if img.ndim == 2:
        return img
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def body_mask(img: np.ndarray, cell: int, pad_box: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Foreground (body/tissue) mask, [H, W] bool. Steps of plan.md §A5.

    Acquisition background is defined spatially: near-extreme pixels (black air, or a white slide)
    connected to the image border. Dark structures enclosed by the body (lungs, bowel gas, fat) never touch the border and stay
    foreground, however large the empty field around a small head or how much gas an abdomen holds.
    Returns all-True when background does not cover most of the border ring (dermoscopy, pathology crops, a body
    that fills the frame), so the gate is a no-op there.
    """
    y_full = _luminance(img)
    h, w = y_full.shape
    x0, y0, x1, y1 = pad_box if pad_box is not None else (0, 0, w, h)
    y = y_full[y0:y1, x0:x1]
    full = np.zeros((h, w), dtype=bool)
    full[y0:y1, x0:x1] = True

    ch, cw = y.shape
    ring = max(1, round(BORDER_FRACTION * min(ch, cw)))
    on_ring = np.zeros((ch, cw), dtype=bool)
    on_ring[:ring], on_ring[-ring:], on_ring[:, :ring], on_ring[:, -ring:] = True, True, True, True
    lo, hi = np.percentile(y, [1, 99])
    if hi - lo < 1e-6:
        return full

    dark_background = np.median(y[on_ring]) <= (lo + hi) / 2
    margin = MAX_BACKGROUND_OFFSET * (hi - lo)
    if dark_background:
        extreme = y <= min(lo + margin, DARK_BACKGROUND_MAX)
    else:
        extreme = y >= max(hi - margin, BRIGHT_BACKGROUND_MIN)
    labels = label(extreme, connectivity=1)
    edge = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    background = np.isin(labels, np.unique(edge[edge > 0]))
    if background[on_ring].mean() < MIN_BORDER_BACKGROUND:
        return full

    fg = ~background
    labels = label(fg, connectivity=2)
    if labels.max() > 0:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        keep = np.flatnonzero(sizes >= MIN_COMPONENT_FRACTION * sizes.max())
        fg = np.isin(labels, keep)
    fg = ndimage.binary_fill_holes(fg)
    fg = ndimage.binary_dilation(fg, structure=np.ones((cell, cell), dtype=bool))

    out = np.zeros((h, w), dtype=bool)
    out[y0:y1, x0:x1] = fg
    return out


def token_foreground_fraction(mask: np.ndarray, cell: int) -> np.ndarray:
    """Fraction of each cell x cell token patch inside the mask, [H/cell, W/cell] float32."""
    h, w = mask.shape
    if h % cell or w % cell:
        raise ValueError(f"mask {h}x{w} is not a multiple of the token cell {cell}")
    return mask.reshape(h // cell, cell, w // cell, cell).mean(axis=(1, 3)).astype(np.float32)


def background_fraction(fg_frac: np.ndarray, tau: float = 0.5) -> float:
    return float(np.mean(np.asarray(fg_frac) < tau))


def slake_oracle_mask(mask_png: np.ndarray) -> np.ndarray:
    """Union of every annotated SLAKE structure (non-zero labels in mask.png), [H, W] bool."""
    arr = np.asarray(mask_png)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0


def image_gate(image: Image.Image, model: str) -> tuple[tuple[int, int], np.ndarray]:
    """Grid shape and per-token foreground fraction of one image for one model."""
    view, box = model_view(image, model)
    cell = MODEL_VIEWS[model]["cell"]
    fg = token_foreground_fraction(body_mask(np.asarray(view), cell, box), cell)
    return fg.shape, fg


def oracle_gate(image: Image.Image, mask_png: Image.Image, model: str) -> np.ndarray:
    """Per-token foreground fraction using SLAKE's annotation mask instead of the heuristic body mask."""
    view, _ = model_view(image, model)
    cell = MODEL_VIEWS[model]["cell"]
    mask = mask_png.convert("L").resize(view.size, Image.NEAREST)
    m = ndimage.binary_dilation(slake_oracle_mask(np.asarray(mask)), structure=np.ones((cell, cell), dtype=bool))
    return token_foreground_fraction(m, cell)


class MaskStore:
    """Per-image gates for one dataset and model, saved as one .npz (plan.md §B Stage 2 outputs)."""

    def __init__(self, image_ids: list[str], grids: list[tuple[int, int]], fracs: list[np.ndarray]):
        self._index = {iid: i for i, iid in enumerate(image_ids)}
        self._grids = grids
        self._fracs = fracs

    def get(self, image_id: str) -> tuple[tuple[int, int], np.ndarray]:
        i = self._index[image_id]
        return self._grids[i], self._fracs[i]

    def __contains__(self, image_id: str) -> bool:
        return image_id in self._index

    def save(self, path: Path) -> None:
        offsets = np.cumsum([0] + [f.size for f in self._fracs])
        np.savez_compressed(
            path,
            image_ids=np.array(list(self._index), dtype=str),
            grids=np.array(self._grids, dtype=np.int32).reshape(-1, 2),
            offsets=offsets.astype(np.int64),
            flat=np.concatenate([f.ravel() for f in self._fracs]).astype(np.float16) if self._fracs else np.zeros(0),
        )

    @classmethod
    def load(cls, path: Path) -> "MaskStore":
        z = np.load(path)
        grids = [tuple(int(v) for v in g) for g in z["grids"]]
        fracs = [z["flat"][a:b].astype(np.float32).reshape(g) for a, b, g in zip(z["offsets"][:-1], z["offsets"][1:], grids)]
        return cls([str(i) for i in z["image_ids"]], grids, fracs)
