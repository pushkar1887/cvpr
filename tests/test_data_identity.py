"""Regression cases for image-independent question-template IDs in RadImageNet."""

import io
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from medprune.data import Sample, build_radimagenet, load_manifest, write_manifest
from medprune.selection import rng_for, select_random


def _sample(qid="q", image_id="image"):
    return Sample(qid=qid, dataset="toy", image_id=image_id, image_path="image.png", question="Is it?",
                  answer="yes", answer_type="yesno", prompt_style="judgement")


def test_radimagenet_template_ids_do_not_collapse_distinct_images(tmp_path):
    rows = []
    for image_number, value in enumerate((20, 180)):
        buffer = io.BytesIO()
        Image.fromarray(np.full((8, 8, 3), value, np.uint8)).save(buffer, format="PNG")
        image = {"bytes": buffer.getvalue(), "path": None}
        for content, qtype, template, answer in (
            ("anatomy", "multiple_choice", "anatomy_mc", "A"),
            ("pathology", "closed", "abnormal_closed", "yes"),
            ("anatomy", "open", "anatomy_open", "Chest"),
            ("pathology", "open", "pathology_open", "No abnormality"),
        ):
            rows.append({"image": image, "question": "Which region?", "choices": ["A. Chest", "B. Brain"],
                         "answer": answer, "question_type": qtype,
                         "metadata": {"content_type": content, "question_id": template,
                                      "modality": "CT" if image_number == 0 else "MRI"}})
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source)
    first, second = tmp_path / "first", tmp_path / "second"
    assert build_radimagenet(source, first) == {"rin_anat_mc": 2, "rin_abn_closed": 2, "rin_open": 4}
    samples = load_manifest(first, "rin_anat_mc") + load_manifest(first, "rin_abn_closed")
    assert len({s.qid for s in samples}) == 4
    anatomy = load_manifest(first, "rin_anat_mc")
    assert not np.array_equal(select_random(64, 6, rng_for(0, anatomy[0].qid)),
                              select_random(64, 6, rng_for(0, anatomy[1].qid)))
    pq.write_table(pa.Table.from_pylist(list(reversed(rows))), source)
    build_radimagenet(source, second)
    assert {(s.image_id, s.dataset): s.qid for s in samples} == {
        (s.image_id, s.dataset): s.qid
        for ds in ("rin_anat_mc", "rin_abn_closed") for s in load_manifest(second, ds)
    }


def test_radimagenet_open_keeps_one_question_per_image_and_template_and_samples_per_template(tmp_path, monkeypatch):
    import medprune.data as data

    monkeypatch.setattr(data, "RIN_OPEN_PER_TEMPLATE", 2)
    rows = []
    for value in (10, 60, 110, 160):
        buffer = io.BytesIO()
        Image.fromarray(np.full((8, 8, 3), value, np.uint8)).save(buffer, format="PNG")
        image = {"bytes": buffer.getvalue(), "path": None}
        for template, content, answer in (("anatomy_open", "anatomy", "Liver"), ("anatomy_open", "anatomy", "Liver"),
                                          ("pathology_open", "pathology", "Mass")):
            rows.append({"image": image, "question": "What is shown?", "choices": [], "answer": answer,
                         "question_type": "open", "metadata": {"content_type": content, "question_id": template}})
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source)
    assert build_radimagenet(source, tmp_path / "a", datasets=("rin_open",)) == {"rin_open": 4}
    first = load_manifest(tmp_path / "a", "rin_open")
    assert sorted(s.meta["question_id"] for s in first) == ["anatomy_open"] * 2 + ["pathology_open"] * 2
    assert all(s.answer_type == "open" and s.prompt_style == "open" for s in first)
    pq.write_table(pa.Table.from_pylist(list(reversed(rows))), source)
    build_radimagenet(source, tmp_path / "b", datasets=("rin_open",))
    assert [s.qid for s in load_manifest(tmp_path / "b", "rin_open")] == [s.qid for s in first]


def test_write_manifest_rejects_duplicate_ids_without_overwriting(tmp_path):
    path = write_manifest([_sample("original")], tmp_path, "toy")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="[Dd]uplicate.*qid"):
        write_manifest([_sample("same", "a"), _sample("same", "b")], tmp_path, "toy")
    assert path.read_bytes() == before


def test_load_manifest_rejects_duplicate_ids(tmp_path):
    from dataclasses import asdict

    path = tmp_path / "manifests" / "toy.jsonl"
    path.parent.mkdir()
    path.write_text("\n".join(json.dumps(asdict(_sample("same", image))) for image in ("a", "b")) + "\n")
    with pytest.raises(ValueError, match="[Dd]uplicate.*qid"):
        load_manifest(tmp_path, "toy")


def test_radimagenet_can_build_only_the_requested_anatomy_subset(tmp_path):
    buffer = io.BytesIO()
    Image.fromarray(np.full((8, 8, 3), 120, np.uint8)).save(buffer, format="PNG")
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist([{
        "image": {"bytes": buffer.getvalue(), "path": None}, "question": "Which region?",
        "choices": ["A. Chest", "B. Brain"], "answer": "A", "question_type": "multiple_choice",
        "metadata": {"content_type": "anatomy", "question_id": "anatomy_mc", "modality": "CT"},
    }]), source)
    out = tmp_path / "out"
    assert build_radimagenet(source, out, datasets=("rin_anat_mc",)) == {"rin_anat_mc": 1}
    assert not (out / "manifests" / "rin_abn_closed.jsonl").exists()


def test_radimagenet_abnormality_uses_observed_template_not_specific_pathology(tmp_path):
    """Source receipt: content_type=pathology, question_id=abnormal_closed (not pathology_closed_*)."""
    buffer = io.BytesIO()
    Image.fromarray(np.full((8, 8, 3), 100, np.uint8)).save(buffer, format="PNG")
    rows = []
    for content, template, question, answer in (
        ("pathology_specific", "pathology_closed_pos", "Is there evidence of a mass?", "yes"),
        ("pathology_specific", "pathology_closed_neg", "Is there evidence of a fracture?", "no"),
        ("pathology", "abnormal_closed", "Does the ankle foot look abnormal?", "yes"),
    ):
        rows.append({"image": {"bytes": buffer.getvalue(), "path": None}, "question": question,
                     "choices": [], "answer": answer, "question_type": "closed",
                     "metadata": {"content_type": content, "question_id": template, "modality": "MRI"}})
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source)
    out = tmp_path / "out"
    assert build_radimagenet(source, out, datasets=("rin_abn_closed",)) == {"rin_abn_closed": 1}
    sample, = load_manifest(out, "rin_abn_closed")
    assert sample.question == "Does the ankle foot look abnormal?"
    assert sample.qid.endswith("/abnormal_closed")
    assert sample.answer_type == "yesno" and sample.prompt_style == "judgement"
