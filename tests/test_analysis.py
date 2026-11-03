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


def test_image_dependence_flags_a_fixed_keep_set():
    from medprune.analysis import image_dependence
    rows = [dict(model="m", dataset="d", method="fastv", budget_pct=10, qid=f"q{i}", image_id=f"img{i}",
                 keep_idx=[0, 1, 2], modality="CT") for i in range(6)]
    rows += [dict(model="m", dataset="d", method="random_l2", budget_pct=10, qid=f"q{i}", image_id=f"img{i}",
                  keep_idx=[3 * i, 3 * i + 1, 3 * i + 2], modality="CT") for i in range(6)]
    df = _frame(rows)
    fixed = image_dependence(df, "m", "fastv").iloc[0]
    varied = image_dependence(df, "m", "random_l2").iloc[0]
    assert fixed["distinct_keepsets"] == 1 and fixed["top_share"] == 1.0 and fixed["cross_image_jaccard"] == 1.0
    assert varied["distinct_keepsets"] == 6 and varied["cross_image_jaccard"] == 0.0


def test_static_map_comes_from_the_other_dataset(tmp_path):
    from medprune.analysis import build_static_maps

    run_id = "m__slake_en__full__r100__s0__t4x2"
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / f"{run_id}.jsonl").write_text(
        "\n".join(json.dumps({"qid": f"q{i}", "grid_hw": [2, 2]}) for i in range(3)), encoding="utf-8")
    writer = DiagWriter(tmp_path / "diag" / run_id)
    for i in range(3):
        writer.add(f"q{i}", {"scores_l1": np.array([0.0, 0.0, 0.0, 9.0])})  # bottom-right always ranks first
    writer.flush()
    written = {p.name for p in build_static_maps(tmp_path, "m")}
    assert written == {"m__for_rin_anat_mc.npy", "m__for_rin_abn_closed.npy"}  # no RadImageNet run: no SLAKE map
    smap = np.load(tmp_path / "static_maps" / "m__for_rin_anat_mc.npy")
    assert smap[-1, -1] == smap.max() and smap[0, 0] == smap.min()


def test_answer_key_reads_letters_and_normalizes_text():
    from medprune.analysis import answer_key

    assert answer_key("mc", "D.") == "D" and answer_key("mc", "(b) the liver") == "B" and answer_key("mc", "?") == "?"
    assert answer_key("open", "The CT.") == "ct"


def test_retained_gain_and_agreement_on_known_values():
    from medprune.analysis import retained_gain

    rows = []
    for i in range(4):
        common = dict(model="m", dataset="d", seed=0, qid=f"q{i}", answer_type="mc")
        rows += [{**common, "method": "full", "budget_pct": 100, "response": "A", "correct": True},
                 {**common, "method": "text_only", "budget_pct": 100, "response": "B", "correct": False},
                 {**common, "method": "fastv", "budget_pct": 10, "response": "A" if i < 2 else "B", "correct": i < 2}]
    row = retained_gain(pd.DataFrame(rows)).iloc[0]
    assert row["retained_gain"] == pytest.approx(0.5) and row["agree_text_only"] == 50 and row["agree_full"] == 50


def test_foreground_coverage_in_units_of_ideal_spacing(tmp_path):
    from medprune.analysis import foreground_coverage

    (tmp_path / "gates").mkdir()
    MaskStore(["img"], [(4, 4)], [np.ones((4, 4), np.float32)]).save(tmp_path / "gates" / "d__m.npz")
    df = pd.DataFrame([dict(model="m", dataset="d", method="fastv", seed=0, budget_pct=10, qid="q", image_id="img",
                            correct=True, keep_idx=[0], grid_hw=[4, 4])])
    row = foreground_coverage(df, tmp_path).iloc[0]
    yy, xx = np.divmod(np.arange(16), 4)
    assert row["fg_gap"] == pytest.approx(np.sqrt(yy ** 2 + xx ** 2).mean() / 4) and row["fg_share"] == 1.0


def test_attention_informativeness_detects_harmful_attention(tmp_path):
    from medprune.analysis import attention_informativeness

    rng = np.random.default_rng(0)
    writer = DiagWriter(tmp_path / "diag" / "m__d__full__r100__s0__t4x2")
    rows = []
    for i in range(300):
        scores = rng.permutation(16).astype(float)
        writer.add(f"q{i}", {"scores_l1": scores})
        keep = np.sort(rng.choice(16, 4, replace=False))
        rank = (scores[keep] + 1).mean() / 16
        correct = rank + rng.normal(0, 0.1) < 0.53  # keeping high-attention tokens hurts
        rows.append(dict(model="m", dataset="d", method="random_l2", seed=0, qid=f"q{i}", image_id=f"img{i}",
                         correct=correct, keep_idx=keep.tolist(), grid_hw=[4, 4]))
    writer.flush()
    result = attention_informativeness(pd.DataFrame(rows), tmp_path, "m", "d")
    assert result["n"] == 300 and result["coef_per_sd"] < -1 and result["p"] < 0.001


def test_slake_targets_name_sides_structures_and_lesions():
    from medprune.analysis import slake_targets

    assert slake_targets("Where is the liver? Left") == {"Liver"}
    assert slake_targets("Which organ is abnormal? Left Lung") == {"Left Lung"}
    assert slake_targets("Are the lungs healthy? no") == {"Left Lung", "Right Lung"}
    assert slake_targets("Where is the brain tumor?") >= {"Brain Enhancing Tumor", "Brain Edema"}
    assert slake_targets("What modality is used to take this image? CT") == set()


def test_label_view_follows_each_models_geometry():
    from medprune.masks import label_view

    mask = Image.fromarray(np.zeros((512, 512), np.uint8))
    assert label_view(mask, "lingshu7b").shape == (504, 504)  # 18 x 18 cells of 28 px
    wide = np.zeros((200, 300), np.uint8)
    wide[:, :] = 70
    view = label_view(Image.fromarray(wide), "huatuo7b")
    assert view.shape == (336, 336) and view[0].max() == 0 and view[168].min() == 70  # zero padding top and bottom


def test_evidence_retention_counts_the_named_structure(tmp_path):
    from medprune.analysis import evidence_fractions, evidence_report, evidence_retention

    mask = np.zeros((56, 56), np.uint8)
    mask[:28, :28] = 70  # the liver fills the top-left token of a 2 x 2 Lingshu grid
    (tmp_path / "xmlab1").mkdir()
    Image.fromarray(mask).save(tmp_path / "xmlab1" / "mask.png")
    base = {"model": "lingshu7b", "qid": "slake_en/1", "image_id": "slake/xmlab1", "question": "Where is the liver?",
            "answer": "Left", "grid_hw": [2, 2], "budget_pct": 25, "seed": 0, "correct": True}
    df = pd.DataFrame([{**base, "dataset": "slake_en", "method": "fastv", "keep_idx": [0]},
                       {**base, "dataset": "slake_en", "method": "random_l2", "keep_idx": [3], "correct": False},
                       {**base, "dataset": "slake_en_vflip", "method": "fastv", "keep_idx": [2]}])
    evidence = evidence_fractions(df, tmp_path, {"Liver": 70})
    assert evidence[("lingshu7b", "slake_en_vflip", "slake_en/1")]["frac"].tolist() == [0, 0, 1, 0]  # flipped
    ev = evidence_retention(df, evidence, budget=25).set_index(["dataset", "method"])
    assert ev.loc[("slake_en", "fastv"), "err"] == 4.0 and ev.loc[("slake_en", "random_l2"), "hit"] == 0.0
    assert ev.loc[("slake_en_vflip", "fastv"), "recall"] == 1.0 and ev["area"].iloc[0] == 0.25
    report = evidence_report(ev.reset_index())
    assert set(report["summary"]["method"]) == {"fastv", "random_l2"}


def test_rank_variance_shares_separate_position_image_and_question():
    from medprune.analysis import rank_variance_shares

    rng = np.random.default_rng(0)
    fixed = rng.random(64)
    qids = [f"q{i}" for i in range(40)]
    meta = pd.DataFrame({"qid": qids, "image_id": [f"img{i // 4}" for i in range(40)],
                         "grid_hw": [[8, 8]] * 40}).set_index("qid")
    position = rank_variance_shares({q: fixed for q in qids}, meta)
    assert position["position"] == pytest.approx(1.0) and position["question_beyond_image"] == pytest.approx(0.0)
    noise = rank_variance_shares({q: rng.random(64) for q in qids}, meta)
    assert noise["question_beyond_image"] > 0.6 and sum(noise[k] for k in ("position", "image_beyond_position",
                                                                           "question_beyond_image")) == pytest.approx(1.0)


def test_layer_profile_on_a_dry_run(tmp_path):
    from medprune.analysis import layer_profile, plot_layer_profile

    bundle, results = tmp_path / "bundle", tmp_path / "results"
    (bundle / "images").mkdir(parents=True)
    samples = []
    for i in range(6):
        Image.fromarray(np.full((16, 16, 3), 30 * i, np.uint8)).save(bundle / "images" / f"{i}.png")
        samples.append(Sample(qid=f"rin/{i}", dataset="rin_anat_mc", image_id=f"rin/{i // 2}",
                              image_path=f"images/{i}.png", question="Is it?", answer="yes", answer_type="yesno",
                              prompt_style="judgement", modality="CT"))
    write_manifest(samples, bundle, "rin_anat_mc")
    run_many([RunSpec("lingshu7b", "rin_anat_mc", "full_profile", 1.0)], bundle, results,
             {"lingshu7b": {"rin": 0.6}}, dry_run=True)
    df = load_runs(results, "dryrun")
    table = layer_profile(df, results, "lingshu7b", "rin_anat_mc", budget=0.25, profile="dryrun")
    assert len(table) == 2 * 28 and set(table["readout"]) == {"last", "text"}
    shares = table[["position", "image_beyond_position", "question_beyond_image"]].sum(axis=1)
    assert np.allclose(shares, 1.0) and table["sink_rate_all"].between(0, 1).all()
    plot_layer_profile(table, results / "figures" / "layer_profile.pdf")
    assert (results / "figures" / "layer_profile.pdf").exists()
