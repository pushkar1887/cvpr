import json
from pathlib import Path

import numpy as np
from PIL import Image

from medprune.data import Sample, write_manifest
import pytest

from medprune.run_eval import RunSpec, bundle_for, estimate_hours, load_runs, run_many

RUNS = Path(__file__).resolve().parent.parent / "configs" / "runs.yaml"


def test_runs_yaml_expands_like_the_plan():
    e1, rates = load_runs(RUNS, exp="E1")
    assert len(e1) == 14  # 7 methods x 2 datasets; full on slake_en is shared with E0
    e3_lingshu, _ = load_runs(RUNS, exp="E3", model="lingshu7b")
    assert sum(r.method == "random_l2" and r.dataset == "rin_abn_closed" for r in e3_lingshu) == 3  # seeds 0-2
    everything, _ = load_runs(RUNS)
    shared = [r for r in everything if r.model == "lingshu7b" and r.dataset == "slake_en" and r.method == "fastv"
              and r.budget == 0.10]
    assert len(shared) == 1 and set(shared[0].exps) >= {"E1", "E3"}
    both, _ = load_runs(RUNS, exp="E6,E3", model="lingshu7b", account="A")  # several experiments, block order
    assert {r.exps for r in both} >= {("E6",), ("E3",)} and both[0].exps == ("E6",) and both[0].dataset == "slake_en"
    seeds = [r for r in both if r.method == "random_l2" and r.dataset == "rin_anat_mc"]
    assert sorted(r.seed for r in seeds) == [0, 1, 2] and next(r for r in seeds if r.seed == 1).exps == ("E3", "E6")
    assert all(r.budget == 1.0 for r in everything if r.method in ("full", "text_only"))
    assert estimate_hours(RunSpec("lingshu7b", "slake_en", "fastv", 0.10), 1061, rates) ==         pytest.approx(1061 * rates["lingshu7b"]["slake_en"] / 3600)
    flat = {"m": {"rin": 0.5, "text_only": 0.2}}  # family fallback and a scalar text-only rate still work
    assert estimate_hours(RunSpec("m", "rin_abn_closed", "full", 1.0), 3600, flat) == pytest.approx(0.5)
    assert estimate_hours(RunSpec("m", "rin_abn_closed", "text_only", 1.0), 3600, flat) == pytest.approx(0.2)


def test_run_id_has_no_experiment_and_encodes_budget():
    assert RunSpec("lingshu7b", "slake_en", "gate_fastv", 0.10, 0).run_id("t4x2") == \
        "lingshu7b__slake_en__gate_fastv__r010__s0__t4x2"


def test_bundle_for_searches_roots_in_order(tmp_path):
    shared, session = tmp_path / "shared", tmp_path / "session"
    write_manifest([], shared, "slake_en")
    write_manifest([], session, "rin_anat_mc")
    assert bundle_for([shared, session], "slake_en") == shared
    assert bundle_for([shared, session], "rin_anat_mc") == session
    with pytest.raises(FileNotFoundError):
        bundle_for([shared, session], "omni_anat")


def _tiny_bundle(root: Path) -> None:
    img_dir = root / "images"
    img_dir.mkdir(parents=True)
    samples = []
    for i in range(6):
        Image.fromarray(np.full((32, 32, 3), 40 * i, np.uint8)).save(img_dir / f"{i}.png")
        samples.append(Sample(qid=f"toy/{i}", dataset="slake_en", image_id=f"toy/{i // 2}",
                              image_path=f"images/{i}.png", question="Is it?", answer="yes" if i % 2 else "no",
                              answer_type="yesno", prompt_style="close"))
    write_manifest(samples, root, "slake_en")


def test_dry_run_writes_one_line_per_question_and_resumes(tmp_path):
    bundle, out = tmp_path / "bundle", tmp_path / "results"
    _tiny_bundle(bundle)
    rates = {"lingshu7b": {"slake": 1.0, "text_only": 0.5}}
    specs = [RunSpec("lingshu7b", "slake_en", m, 0.25 if m not in ("full", "full_profile") else 1.0) for m in
             ("full", "random_l2", "fastv", "gate_uniform", "fastv_prellm", "divprune", "pdrop", "fastv_bottom",
              "fastv_text", "full_profile")]
    run_many(specs, bundle, out, rates, limit=4, dry_run=True)
    for spec in specs:
        path = out / "runs" / f"{spec.run_id('dryrun')}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert [r["qid"] for r in rows] == [f"toy/{i}" for i in range(4)]
        if spec.method not in ("full", "pdrop", "full_profile"):
            assert all(r["k_keep"] == 4 for r in rows)  # 25% of 16 tokens
    from medprune.analysis import load_diag

    profile = load_diag(out, RunSpec("lingshu7b", "slake_en", "full_profile", 1.0).run_id("dryrun"))["toy/0"]
    assert {"lp_last_l0", "lp_text_l27", "sink_l13"} <= set(profile) and profile["sink_l13"].shape == (16,)
    text = load_diag(out, RunSpec("lingshu7b", "slake_en", "fastv_text", 0.25).run_id("dryrun"))["toy/0"]
    assert "scores_text_l1" in text
    keep = {m: [json.loads(line)["keep_idx"] for line in
                (out / "runs" / f"{RunSpec('lingshu7b', 'slake_en', m, 0.25).run_id('dryrun')}.jsonl").read_text().splitlines()]
            for m in ("fastv", "fastv_bottom")}
    assert all(not set(a) & set(b) for a, b in zip(keep["fastv"], keep["fastv_bottom"]))  # lowest attention only
    run_many(specs, bundle, out, rates, limit=6, dry_run=True)  # resumes: only the 2 new questions are added
    rows = (out / "runs" / f"{specs[1].run_id('dryrun')}.jsonl").read_text().splitlines()
    assert len(rows) == 6


def test_resume_rejects_duplicate_completed_ids(tmp_path):
    from medprune.run_eval import _done_qids

    path = tmp_path / "run.jsonl"
    path.write_text('{"qid":"template"}\n{"qid":"template"}\n')
    with pytest.raises(ValueError, match="[Dd]uplicate.*qid"):
        _done_qids(path)


def test_resume_rejects_results_from_a_different_manifest(tmp_path):
    from medprune.run_eval import FakeAdapter, run_one

    bundle, out = tmp_path / "bundle", tmp_path / "results"
    _tiny_bundle(bundle)
    spec = RunSpec("lingshu7b", "slake_en", "full")
    path = out / "runs" / f"{spec.run_id('dryrun')}.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"qid":"obsolete-template","correct":true}\n')
    with pytest.raises(ValueError, match="manifest"):
        run_one(FakeAdapter("lingshu7b"), spec, bundle, out, "dryrun", strict_masks=False)


def test_run_many_does_not_accept_obsolete_ids_as_a_complete_run(tmp_path):
    bundle, out = tmp_path / "bundle", tmp_path / "results"
    _tiny_bundle(bundle)
    spec = RunSpec("lingshu7b", "slake_en", "full")
    path = out / "runs" / f"{spec.run_id('dryrun')}.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps({"qid": f"obsolete/{i}", "correct": True}) + "\n" for i in range(6)))
    with pytest.raises(ValueError, match="manifest"):
        run_many([spec], bundle, out, {"lingshu7b": {"slake": 1.0}}, dry_run=True)


def test_list_does_not_report_obsolete_ids_as_complete(tmp_path, monkeypatch):
    import sys
    from medprune.run_eval import main

    bundle, out = tmp_path / "bundle", tmp_path / "results"
    _tiny_bundle(bundle)
    spec = RunSpec("lingshu7b", "slake_en", "full")
    path = out / "runs" / f"{spec.run_id('dryrun')}.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps({"qid": f"obsolete/{i}", "correct": True}) + "\n" for i in range(6)))
    monkeypatch.setattr(sys, "argv", ["run_eval", "--list", "--profile", "dryrun", "--exp", "E0", "--model",
                                     "lingshu7b", "--bundle", str(bundle), "--out", str(out)])
    with pytest.raises(ValueError, match="manifest"):
        main()
