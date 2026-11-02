import numpy as np
import pytest
from PIL import Image

from medprune.masks import MODEL_VIEWS, MaskStore, body_mask, model_view, smart_resize, token_foreground_fraction


def ellipse(h=256, w=256, cy=128, cx=128, ry=80, rx=100) -> np.ndarray:
    yy, xx = np.mgrid[:h, :w]
    return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1


def rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(np.clip(gray, 0, 255)[..., None], 3, axis=2).astype(np.uint8)


def ct_like(rng) -> tuple[np.ndarray, np.ndarray]:
    body = ellipse()
    img = 5.0 + rng.normal(0, 2, body.shape)  # air: dark and flat
    img[body] = 120 + rng.normal(0, 25, body.sum())  # tissue: bright and textured
    return img, body


def test_ct_like_body_mask_matches_the_body():
    img, body = ct_like(np.random.default_rng(0))
    mask = body_mask(rgb(img), cell=1)
    assert abs(mask.sum() - body.sum()) / body.sum() < 0.02


def test_lungs_are_kept_by_hole_filling():
    rng = np.random.default_rng(0)
    img, _ = ct_like(rng)
    lungs = ellipse(cy=128, cx=90, ry=40, rx=25) | ellipse(cy=128, cx=166, ry=40, rx=25)
    img[lungs] = 5 + rng.normal(0, 2, lungs.sum())  # as dark as the air outside the body
    assert body_mask(rgb(img), cell=1)[lungs].mean() > 0.99


def test_scanner_table_is_removed():
    img, _ = ct_like(np.random.default_rng(0))
    table = np.zeros(img.shape, bool)
    table[230:236, 40:216] = True  # thin bright strip below the body
    img[table] = 200
    assert body_mask(rgb(img), cell=1)[table].mean() < 0.01


def test_white_background_is_handled_by_polarity():
    rng = np.random.default_rng(0)
    tissue = ellipse()
    img = 240.0 + rng.normal(0, 2, tissue.shape)  # glass slide
    img[tissue] = 90 + rng.normal(0, 25, tissue.sum())
    mask = body_mask(rgb(img), cell=1)
    assert abs(mask.sum() - tissue.sum()) / tissue.sum() < 0.02


def test_textured_image_without_background_is_left_alone():
    rng = np.random.default_rng(0)  # dermoscopy-like: textured skin around a darker lesion
    img = 180 + rng.normal(0, 20, (256, 256))
    lesion = ellipse(ry=50, rx=60)
    img[lesion] = 120 + rng.normal(0, 20, lesion.sum())
    assert body_mask(rgb(img), cell=1).all()


def test_padding_is_background():
    rng = np.random.default_rng(0)
    content = 5.0 + rng.normal(0, 2, (100, 300))
    content[20:80, 50:250] = 120 + rng.normal(0, 25, (60, 200))
    view, box = model_view(Image.fromarray(rgb(content)), "huatuo7b")
    assert view.size == (336, 336) and box == (0, 112, 336, 224)
    mask = body_mask(np.asarray(view), cell=14, pad_box=box)
    assert not mask[: box[1]].any() and not mask[box[3]:].any() and mask.any()


@pytest.mark.parametrize("model, grid", [("lingshu7b", (18, 18)), ("qwen3vl8b", (16, 16)), ("huatuo7b", (24, 24))])
def test_token_grid_matches_each_model(model, grid):
    img, _ = ct_like(np.random.default_rng(0))
    view, box = model_view(Image.fromarray(rgb(np.kron(img, np.ones((2, 2))))), model)  # 512 x 512
    cell = MODEL_VIEWS[model]["cell"]
    frac = token_foreground_fraction(body_mask(np.asarray(view), cell, box), cell)
    assert frac.shape == grid and 0.0 <= frac.min() and frac.max() <= 1.0
    assert 0.2 < (frac < 0.5).mean() < 0.7  # the ellipse leaves a sizeable background


def test_smart_resize_matches_transformers():
    tf = pytest.importorskip("transformers.models.qwen2_vl.image_processing_qwen2_vl")
    for h, w in [(512, 512), (224, 224), (2048, 1536), (100, 3000), (37, 41), (1000, 700)]:
        for factor, lo, hi in [(28, 3136, 802_816), (32, 65_536, 1_048_576)]:
            assert smart_resize(h, w, factor, lo, hi) == tf.smart_resize(h, w, factor=factor, min_pixels=lo,
                                                                          max_pixels=hi)


def test_mask_store_round_trip(tmp_path):
    a = np.random.default_rng(0).random((4, 5)).astype(np.float32)
    MaskStore(["x", "y"], [(4, 5), (2, 2)], [a, np.ones((2, 2), np.float32)]).save(tmp_path / "m.npz")
    grid, frac = MaskStore.load(tmp_path / "m.npz").get("x")
    assert grid == (4, 5)
    np.testing.assert_allclose(frac, a, atol=1e-3)
