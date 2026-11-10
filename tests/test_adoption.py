"""The E42 adoption rule (medprune/adoption.py, plan.md): split, statistic, Holm, decision."""

import numpy as np

from medprune.adoption import CELLS, Cell, Result, balanced, decide, holm, image_half, nine_cell


def test_image_half_is_deterministic_and_balanced():
    ids = [f"slake/xmlab{i}" for i in range(4000)]
    halves = np.array([image_half(i) for i in ids])
    assert [image_half(i) for i in ids[:50]] == list(halves[:50])
    assert abs(halves.mean() - 0.5) < 3 * 0.5 / np.sqrt(len(ids))


def test_balanced_accuracy_by_hand():
    gold = np.array([True, True, True, False])
    yes = np.array([1.0, 0.0, 1.0, 0.0])  # TPR 2/3, TNR 1
    assert np.isclose(balanced(np.ones(4), gold, yes), 50 * (2 / 3 + 1))
    assert np.isclose(balanced(np.array([2.0, 0, 1, 1]), gold, yes), 50 * (1 + 1))  # weights move both rates


def _cells(shift: int, n: int = 400, rng=np.random.default_rng(0), loss_cell=None):
    """Nine synthetic cells, 4 questions per image; the candidate fixes `shift` reference errors in every cell."""
    cells = {}
    for key in CELLS:
        b = (rng.random(n) < 0.6).astype(float)
        a = b.copy()
        wrong = np.flatnonzero(b == 0)
        a[wrong[:shift]] = 1.0
        if key == loss_cell:
            right = np.flatnonzero(b == 1)
            a[right[:80]] = 0.0
        image = np.array([f"{key[1]}/img{i // 4}" for i in range(n)])
        if key == ("lingshu7b", "rin_abn_closed"):  # balanced cell: yes-rates from the same answers
            gold = rng.random(n) < 0.7
            cells[key] = Cell(a, b, image, gold, np.where(gold, a, 1 - a), np.where(gold, b, 1 - b))
        else:
            cells[key] = Cell(a, b, image)
    return cells


def test_identical_rows_are_never_adopted():
    cells = _cells(0)
    r = nine_cell(cells, draws=300)
    assert r.delta == 0 and r.se_boot == 0 and r.p == 1.0
    assert not decide({"same": r})["same"]["adopted"]


def test_a_real_gain_is_detected_and_its_size_is_exact():
    cells = _cells(12)
    r = nine_cell(cells, draws=600)
    expected = np.mean([c.delta() for c in cells.values()])
    assert np.isclose(r.delta, expected) and r.delta > 2
    assert r.se_boot > 0 and r.p < 1e-4 and all(h > 0 for h in r.halves)
    assert decide({"gain": r})["gain"]["adopted"]
    assert Result(r.delta, r.se_boot, r.halves, r.cells, seed_sd=0.28).se > r.se  # the seed term widens SE


def test_a_significant_cell_loss_blocks_adoption():
    cells = _cells(20, loss_cell=("qwen3vl8b", "rin_open"))
    r = nine_cell(cells, draws=600)
    assert r.delta > 0 and r.p < 0.025
    d = decide({"trade": r})["trade"]
    assert d["test"] and d["losses"] == [("qwen3vl8b", "rin_open")] and not d["adopted"]


def test_holm_step_down():
    assert holm({"a": 0.01, "b": 0.04}) == {"a": True, "b": True}
    assert holm({"a": 0.03, "b": 0.04}) == {"a": False, "b": False}  # 0.03 > 0.05 / 2: nothing is rejected
    assert holm({"a": 0.02, "b": 0.06}) == {"a": True, "b": False}


def test_prompt_style_matches_the_instructions():
    from medprune.adoption import SINGLE_TOKEN, prompt_style

    assert prompt_style("omni_anat", "mc") == "mc" and prompt_style("rin_abn_closed", "yesno") == "judgement"
    assert prompt_style("slake_en", "yesno") == "close" and prompt_style("slake_en", "open") == "open"
    assert prompt_style("natural", "yesno") == "judgement" and prompt_style("natural", "closed") == "close"
    assert {prompt_style(d, a) for d, a in (("rin_anat_mc", "mc"), ("rin_abn_closed", "yesno"))} <= SINGLE_TOKEN
    assert not {prompt_style(d, a) for d, a in (("slake_en", "closed"), ("rin_open", "open"))} & SINGLE_TOKEN


def test_route_takes_single_token_answers_from_the_first_row(tmp_path, monkeypatch):
    import pandas as pd

    from medprune import adoption

    frames = {
        "A": pd.DataFrame({"c": [1.0, 1.0, 1.0], "image": ["i0", "i1", "i2"], "gold": [True] * 3, "yes": [1.0] * 3,
                           "atype": ["mc", "open", "yesno"]}, index=["q0", "q1", "q2"]),
        "B": pd.DataFrame({"c": [0.0, 0.0, 0.0], "image": ["i0", "i1", "i2"], "gold": [True] * 3, "yes": [0.0] * 3,
                           "atype": ["mc", "open", "yesno"]}, index=["q0", "q1", "q2"]),
    }
    real = adoption.load_cell
    monkeypatch.setattr(adoption, "load_cell", lambda r, m, d, meth, v: frames[meth] if meth in frames
                        else real(r, m, d, meth, v))
    rin = adoption.load_cell(tmp_path, "lingshu7b", "rin_abn_closed", "route:A|B", {})
    assert list(rin["c"]) == [1.0, 0.0, 1.0]  # mc and judgement from A, open from B
    slake = adoption.load_cell(tmp_path, "lingshu7b", "slake_en", "route:A|B", {})
    assert list(slake["c"]) == [1.0, 0.0, 0.0]  # SLAKE's yes / no questions get "close": B


def test_route_on_an_all_single_token_cell_needs_only_the_first_branch(tmp_path, monkeypatch):
    import pandas as pd
    import pytest

    from medprune import adoption

    a = pd.DataFrame({"c": [1.0, 0.0], "image": ["i0", "i1"], "gold": [True, False], "yes": [1.0, 0.0],
                      "atype": ["mc", "mc"]}, index=["q0", "q1"])
    real = adoption.load_cell
    monkeypatch.setattr(adoption, "load_cell", lambda r, m, d, meth, v: a if meth == "A" else real(r, m, d, meth, v))
    out = adoption.load_cell(tmp_path, "qwen3vl8b", "rin_anat_mc", "route:A|missing", {})
    assert list(out["c"]) == [1.0, 0.0]
    mixed = a.assign(atype=["mc", "open"])
    monkeypatch.setattr(adoption, "load_cell", lambda r, m, d, meth, v: mixed if meth == "A" else real(r, m, d, meth, v))
    with pytest.raises(FileNotFoundError):  # generated-answer questions present: the missing branch is an error
        adoption.load_cell(tmp_path, "lingshu7b", "rin_open", "route:A|missing", {})
