import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from medprune.analysis import (accuracy_table, load_diag, load_runs, modality_group, paired_delta, parse_run_id,
                               query_dependence, stage4_report)
from medprune.data import Sample, write_manifest
from medprune.masks import MaskStore
from medprune.run_eval import DiagWriter, RunSpec, run_many


def test_parse_run_id_and_modality_groups():
    assert parse_run_id("lingshu7b__rin_anat_mc__fastv__r010__s0__t4x2") == {
        "model": "lingshu7b", "dataset": "rin_anat_mc", "method": "fastv", "budget_pct": 10, "seed": 0,
        "profile": "t4x2"}
    assert modality_group("CT") == modality_group("MRI") == modality_group("MR") == "CT/MRI"
    assert modality_group("X-Ray") == "X-ray" and modality_group("Dermoscopy") == "Dermoscopy"


def _frame(rows):
    df = pd.DataFrame(rows)
    df["modality_group"] = df["modality"].map(modality_group)
    df["image_dependent"] = False
    df["profile"] = "t4x2"
    return df


def test_accuracy_and_paired_delta_on_known_values():
    rows = []
    for i in range(20):
        common = dict(model="m", dataset="d", budget_pct=10, qid=f"q{i}", image_id=f"img{i // 2}", modality="CT",
                      lang="en", answer_type="yesno", bg_frac=0.4)
        rows.append({**common, "method": "fastv", "seed": 0, "correct": i < 5})
        for seed in (0, 1):  # Random averages two seeds: always right on seed 0, never on seed 1
            rows.append({**common, "method": "random_l2", "seed": seed, "correct": seed == 0})
    df = _frame(rows)
    table = accuracy_table(df, by=("method",)).set_index("method")
    assert table.loc["fastv", "acc"] == 25.0 and table.loc["random_l2", "acc"] == 50.0
    result = paired_delta(df, "m", "d", "fastv", "random_l2")
    assert result["delta"] == pytest.approx(-25.0) and result["n"] == 20


def test_query_dependence_is_one_for_identical_keep_sets():
    rows = [dict(model="m", dataset="d", method="fastv", budget_pct=10, qid=f"q{i}", image_id="img",
                 keep_idx=[1, 2, 3], modality="CT") for i in range(3)]
    result = query_dependence(_frame(rows), "m", "fastv")
    assert result.loc[0, "jaccard"] == 1.0 and result.loc[0, "images"] == 1


def test_load_diag_round_trip(tmp_path):
    writer = DiagWriter(tmp_path / "diag" / "run")
    writer.add("a", {"scores_l1": np.arange(4.0)})
    writer.add("b", {"scores_l1": np.arange(6.0)})
    writer.flush()
    diag = load_diag(tmp_path, "run")
    np.testing.assert_allclose(diag["b"]["scores_l1"], np.arange(6.0))


def test_load_runs_rejects_duplicate_question_ids_before_aggregation(tmp_path):
    path = tmp_path / "runs" / "m__d__full__r100__s0__t4x2.jsonl"
    path.parent.mkdir()
    records = [dict(qid="same-template", image_id=f"image{i}", modality="CT", correct=bool(i),
                    answer_type="closed", answer="yes", response="yes") for i in range(2)]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    with pytest.raises(ValueError, match="[Dd]uplicate.*qid"):
        load_runs(tmp_path)


def test_load_diag_rejects_duplicate_ids_instead_of_overwriting(tmp_path):
    writer = DiagWriter(tmp_path / "diag" / "run", shard_size=1)
    writer.add("same-template", {"scores_l1": np.array([1.0, 2.0])})
    writer.add("same-template", {"scores_l1": np.array([3.0, 4.0])})
    writer.flush()
    with pytest.raises(ValueError, match="[Dd]uplicate.*qid"):
        load_diag(tmp_path, "run")


def test_retention_uses_the_registered_background_fraction_range(tmp_path):
    from medprune.analysis import retention_table

    low, valid = np.ones((10, 10)), np.ones((10, 10))
    low.flat[0] = 0
    valid.flat[:50] = 0
    (tmp_path / "gates").mkdir()
    MaskStore(["low", "valid"], [(10, 10), (10, 10)], [low, valid]).save(
        tmp_path / "gates" / "d__m.npz")
    df = _frame([dict(model="m", dataset="d", method="fastv", budget_pct=10, qid=i,
                      image_id=i, modality="CT", keep_idx=[0]) for i in ("low", "valid")])
    row = retention_table(df, tmp_path, "m", "fastv").iloc[0]
    assert row["coverage"] == 0.5
    assert row["brr"] == 2.0


def test_stage4_report_end_to_end_on_dry_run(tmp_path):
    bundle, results = tmp_path / "bundle", tmp_path / "results"
    (bundle / "images").mkdir(parents=True)
    samples, ids = [], []
    for i in range(8):
        Image.fromarray(np.full((16, 16, 3), 20 * i, np.uint8)).save(bundle / "images" / f"{i}.png")
        ids.append(f"rin/{i // 2}")
        samples.append(Sample(qid=f"rin/{i}", dataset="rin_anat_mc", image_id=ids[-1], image_path=f"images/{i}.png",
                              question="Is it?", answer="yes" if i % 3 else "no", answer_type="yesno",
                              prompt_style="judgement", modality="CT"))
    write_manifest(samples, bundle, "rin_anat_mc")
    frac = np.ones((4, 4), np.float32)
    frac[:, :2] = 0.0  # left half is background
    unique = sorted(set(ids))
    (bundle / "masks").mkdir()
    MaskStore(unique, [(4, 4)] * len(unique), [frac] * len(unique)).save(bundle / "masks" / "rin_anat_mc__lingshu7b.npz")

    methods = ["full", "text_only", "random_l2", "random_l0", "fastv", "fastv_prellm", "gate_fastv"]
    specs = [RunSpec("lingshu7b", "rin_anat_mc", m, 1.0 if m in ("full", "text_only") else 0.25) for m in methods]
    run_many(specs, bundle, results, {"lingshu7b": {"rin": 0.6, "text_only": 0.5}}, dry_run=True)
    assert (results / "gates" / "rin_anat_mc__lingshu7b.npz").exists()

    report = stage4_report(results, model="lingshu7b", profile="dryrun", budget=25, write=True)
    retention = report["retention"].set_index("method")
    assert retention.loc["gate_fastv", "brr"] == 0.0  # the gate never keeps background
    assert set(report["decisions"]) >= {"K1", "K2"}
    assert (results / "tables" / "stage4_paired.csv").exists()
    assert (results / "figures" / "stage4_keepmaps.pdf").exists()
    decisions = json.loads((results / "tables" / "stage4_decisions.json").read_text())
    assert decisions["K1"].startswith(("GO", "REFRAME", "STOP", "undecided"))
