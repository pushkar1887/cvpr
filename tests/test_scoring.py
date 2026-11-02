from medprune.data import Sample, make_prompt
from medprune.scoring import extract_boxed_content, is_correct, judge_closed, judge_mc, judge_open, judge_yesno


def sample(**kw) -> Sample:
    base = dict(qid="q", dataset="d", image_id="i", image_path="p", question="Is it?", answer="yes",
                answer_type="yesno", prompt_style="close")
    base.update(kw)
    return Sample(**base)


def test_prompts_match_medevalkit():
    assert make_prompt(sample(question="What organ?")) == \
        "What organ?\nAnswer the question using a single word or phrase."
    assert make_prompt(sample(question="Where?", prompt_style="open")) == "Where?\nPlease answer the question concisely."
    assert make_prompt(sample(question="Abnormal?", prompt_style="judgement")) == \
        "Abnormal?\nPlease output 'yes' or 'no'(no extra output)."
    mc = sample(question="Which region?", prompt_style="mc", answer_type="mc", answer="B",
                choices=("A. Chest", "B. Brain"))
    assert make_prompt(mc) == ("\nQuestion: Which region?\nOptions: \nA. Chest\nB. Brain\n"
                               "Answer with the option's letter from the given choices directly.")
    assert make_prompt(sample(question="什么器官?", lang="zh")) == "什么器官?\n请用一个单词或者短语回答该问题。"


def test_yesno_judge():
    assert judge_yesno("yes", "Yes.")
    assert judge_yesno("no", "No")
    assert judge_yesno("no", "The answer is no.")
    assert not judge_yesno("yes", "yes or no")  # both present
    assert not judge_yesno("yes", "No.")


def test_closed_judge_is_exact_after_stripping_periods():
    assert judge_closed("liver", "Liver.")
    assert not judge_closed("liver", "The liver")


def test_open_judge_is_exact_match_with_recall():
    assert judge_open("left lung", "Left lung") == (True, 1.0)
    assert judge_open("left lung", "lung") == (False, 0.5)
    assert judge_open("liver", "liver.")[0] is False  # MedEvalKit does not strip punctuation for open questions


def test_multiple_choice_judge():
    choices = ["A. Chest", "B. Brain"]
    assert judge_mc(choices, "A", "A")
    assert not judge_mc(choices, "A", "B")
    assert judge_mc(choices, "A", "A. Chest")
    assert judge_mc(choices, "B", "Brain")  # nearest choice by difflib


def test_boxed_and_dispatch():
    assert extract_boxed_content(r"so \boxed{B}") == "B"
    assert extract_boxed_content(r"\boxed{a{b}c}") == "a{b}c"
    assert extract_boxed_content("no box") == "None"
    mc = sample(answer_type="mc", prompt_style="mc", answer="B", choices=("A. Chest", "B. Brain"))
    assert is_correct(mc, r"\boxed{B}") == (True, None)
    assert is_correct(sample(answer="left lung", answer_type="open", prompt_style="open"), "left lung") == (True, 1.0)
