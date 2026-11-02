import json

import numpy as np
from PIL import Image

from medprune.checks import run_checks, timing_summary
from medprune.data import Sample, write_manifest
from medprune.run_eval import FakeAdapter


def _bundle(root):
    (root / "images").mkdir(parents=True)
    samples = []
    for i in range(5):
        Image.fromarray(np.full((16, 16, 3), 30 * i, np.uint8)).save(root / "images" / f"{i}.png")
        samples.append(Sample(qid=f"toy/{i}", dataset="slake_en", image_id=f"toy/{i}", image_path=f"images/{i}.png",
                              question="Is it?", answer="yes", answer_type="yesno", prompt_style="close"))
    write_manifest(samples, root, "slake_en")
    return samples


def test_checks_pass_on_a_deterministic_adapter(tmp_path):
    samples = _bundle(tmp_path / "bundle")
    report = run_checks(FakeAdapter("lingshu7b"), samples, tmp_path / "bundle", tmp_path / "results", "dryrun")
    assert report["pass"]
    assert report["E0d_offline_keepsets"]["keepsets_identical"] == 5
    assert (tmp_path / "results" / "e0_report__lingshu7b__dryrun.json").exists()


def test_timing_summary(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_text("\n".join(json.dumps({"elapsed_s": t}) for t in (1.0, 2.0, 3.0)) + "\n")
    summary = timing_summary(path)
    assert summary["mean_s"] == 2.0 and summary["n"] == 3
    assert timing_summary(tmp_path / "missing.jsonl") == {"run": "missing.jsonl", "available": False}
