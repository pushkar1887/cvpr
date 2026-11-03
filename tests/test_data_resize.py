import numpy as np
from PIL import Image

from medprune.data import Sample, build_variant, load_manifest, write_manifest


def test_resized_copy_keeps_questions_and_ids(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    (src / "images" / "toy").mkdir(parents=True)
    Image.fromarray(np.zeros((20, 30, 3), np.uint8)).save(src / "images" / "toy" / "a.jpg")
    samples = [Sample(qid=f"toy/{i}", dataset="toy", image_id="toy/a", image_path="images/toy/a.jpg", question=q,
                      answer="yes", answer_type="yesno", prompt_style="judgement") for i, q in enumerate("xy")]
    write_manifest(samples, src, "toy")
    assert build_variant(src, out, "toy", size=16) == {"toy_s16": 2}
    resized = load_manifest(out, "toy_s16")
    assert [s.qid for s in resized] == ["toy/0", "toy/1"] and {s.image_id for s in resized} == {"toy/a"}
    assert {s.dataset for s in resized} == {"toy_s16"} and resized[0].meta == {"variant_of": "toy", "size": 16}
    assert resized[0].image_path == "images/toy_s16/toy/a.png"
    assert Image.open(out / resized[0].image_path).size == (16, 16)
    assert len(list((out / "images").rglob("*.png"))) == 1  # one file per image, not per question


def test_flipped_copy_turns_the_image_upside_down(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    (src / "images").mkdir(parents=True)
    pixels = np.zeros((4, 4, 3), np.uint8)
    pixels[0] = 255  # bright top row
    Image.fromarray(pixels).save(src / "images" / "a.png")
    write_manifest([Sample(qid="q", dataset="toy", image_id="a", image_path="images/a.png", question="x", answer="a",
                           answer_type="open", prompt_style="open")], src, "toy")
    assert build_variant(src, out, "toy", vflip=True) == {"toy_vflip": 1}
    flipped = np.asarray(Image.open(out / load_manifest(out, "toy_vflip")[0].image_path))
    assert flipped[-1].min() == 255 and flipped[0].max() == 0
