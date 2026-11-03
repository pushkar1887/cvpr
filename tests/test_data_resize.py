import numpy as np
from PIL import Image

from medprune.data import Sample, build_resized, load_manifest, write_manifest


def test_resized_copy_keeps_questions_and_ids(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    (src / "images" / "toy").mkdir(parents=True)
    Image.fromarray(np.zeros((20, 30, 3), np.uint8)).save(src / "images" / "toy" / "a.jpg")
    samples = [Sample(qid=f"toy/{i}", dataset="toy", image_id="toy/a", image_path="images/toy/a.jpg", question=q,
                      answer="yes", answer_type="yesno", prompt_style="judgement") for i, q in enumerate("xy")]
    write_manifest(samples, src, "toy")
    assert build_resized(src, out, "toy", 16) == {"toy_s16": 2}
    resized = load_manifest(out, "toy_s16")
    assert [s.qid for s in resized] == ["toy/0", "toy/1"] and {s.image_id for s in resized} == {"toy/a"}
    assert {s.dataset for s in resized} == {"toy_s16"} and resized[0].meta == {"resized_from": "toy", "size": 16}
    assert resized[0].image_path == "images/toy_s16/toy/a.png"
    assert Image.open(out / resized[0].image_path).size == (16, 16)
    assert len(list((out / "images").rglob("*.png"))) == 1  # one file per image, not per question
