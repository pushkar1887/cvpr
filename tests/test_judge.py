import json

import numpy as np
import pandas as pd
import pytest

from medprune.judge import (PROMPT_VERSION, apply_judge, clean_response, export_items, item_key, judge_prompt,
                            kappa_report, label_sheet, load_verdicts, open_rows, weighted_kappa)


def _write_run(results, run_id, rows):
    path = results / "runs" / f"{run_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _row(qid, question, answer, response, answer_type="open", correct=False):
    return {"qid": qid, "image_id": qid.split("/")[0], "question": question, "answer": answer, "response": response,
            "answer_type": answer_type, "lang": "en", "correct": correct}


def test_key_follows_the_scored_part_of_the_response():
    assert item_key("Which organ?", "Liver", r"\boxed{Liver}") == item_key("Which organ?", "Liver", " Liver ")
    assert item_key("Which organ?", "Liver", "Liver") != item_key("Which organ?", "Liver", "Spleen")
    assert clean_response("<answer>CT</answer>") == "CT"
    prompt = judge_prompt("What plane?", "Transverse Plane", "axial.")
    assert "Standard answer: Transverse Plane" in prompt and "Model's answer: axial." in prompt
    assert "You cannot see the image" in prompt and "less specific" in prompt
    assert prompt.endswith("Reply with one word: Yes or No.")


def test_export_is_unique_and_incremental(tmp_path):
    rows = [_row("i1/q1", "What plane?", "Transverse Plane", "axial"),
            _row("i1/q2", "Is it normal?", "yes", "yes", answer_type="yesno", correct=True)]
    _write_run(tmp_path, "lingshu7b__slake_en__full__r100__s0__t4x2", rows)
    _write_run(tmp_path, "lingshu7b__slake_en__fastv__r010__s0__t4x2", rows)  # same answer, second run
    assert len(open_rows(tmp_path)) == 2  # two open rows, one closed row skipped twice
    out = tmp_path / "judge" / "items.jsonl"
    assert export_items(tmp_path, out) == 1
    item = json.loads(out.read_text(encoding="utf-8"))
    verdicts = tmp_path / "judge" / "verdicts.jsonl"
    old = {"key": item["key"], "p_correct": 0.9, "prompt": "older/v2"}
    verdicts.write_text(json.dumps(old) + "\n" + json.dumps({**old, "p_correct": 0.8, "prompt": PROMPT_VERSION}) + "\n",
                        encoding="utf-8")
    assert export_items(tmp_path, out, verdicts) == 0  # already judged under the current prompt
    assert load_verdicts(verdicts) == {item["key"]: 0.8} and load_verdicts(verdicts, "older/v2") == {item["key"]: 0.9}


def test_apply_judge_scores_open_rows_only_and_refuses_gaps():
    df = pd.DataFrame([_row("i1/q1", "What plane?", "Transverse Plane", "axial"),
                       _row("i1/q2", "Is it normal?", "yes", "no", answer_type="yesno"),
                       _row("i1/q3", "Largest organ?", "Liver", "Liver.", correct=True)])
    key, exact = item_key("What plane?", "Transverse Plane", "axial"), item_key("Largest organ?", "Liver", "Liver.")
    judged = apply_judge(df, {key: 0.97, exact: 0.01})
    assert judged.loc[0, "correct"] and not judged.loc[0, "correct_rule"]
    assert not judged.loc[1, "correct"]  # closed rows keep their rule verdict
    assert judged.loc[2, "correct"]  # an exact match stays correct whatever the judge says
    with pytest.raises(ValueError, match="no judge verdict"):
        apply_judge(df, {})


def test_weighted_kappa():
    a = np.array([1, 1, 0, 0], bool)
    assert weighted_kappa(a, a, np.ones(4)) == pytest.approx(1.0)
    assert weighted_kappa(a, ~a, np.ones(4)) == pytest.approx(-1.0)
    b = np.array([1, 0, 0, 0], bool)  # p_o 0.75, p_e 0.5 x 0.25 + 0.5 x 0.75 = 0.5
    assert weighted_kappa(a, b, np.ones(4)) == pytest.approx(0.5)


def test_label_sheet_is_blind_and_kappa_round_trips(tmp_path):
    rows, verdicts = [], {}
    for i in range(30):
        answer, response = "Liver", ("Liver" if i % 3 else "hepatic organ")  # rule misses the paraphrase
        rows.append(_row(f"img{i}/q", f"Which organ is shown {i}?", answer, response))
        verdicts[item_key(f"Which organ is shown {i}?", answer, response)] = 0.9 if i % 5 else 0.2
    _write_run(tmp_path, "qwen3vl8b__slake_en__full__r100__s0__t4x2", rows)
    sheet_path = tmp_path / "judge" / "label_sheet.csv"
    sheet = label_sheet(tmp_path, verdicts, sheet_path, n=20)
    shown = pd.read_csv(sheet_path, encoding="utf-8-sig")
    assert list(shown.columns) == ["item", "question", "reference", "response", "label"]  # no verdicts, no methods
    assert set(sheet["stratum"]) == {"exact", "judge_yes", "judge_no"}
    assert "less specific" in sheet_path.with_name("label_sheet_README.txt").read_text(encoding="utf-8")
    key = pd.read_csv(sheet_path.with_name("label_sheet_key.csv"))
    shown["label"] = key.set_index("item").loc[shown["item"], "judge"].astype(int).to_numpy()  # human = judge
    shown.to_csv(sheet_path, index=False, encoding="utf-8-sig")
    report = kappa_report(sheet_path, B=50)
    assert report["kappa_judge"] == pytest.approx(1.0) and report["kappa_rule"] < 1.0
    assert report["judge_becomes_primary"] and report["judge_right_when_they_disagree"] == 1.0
