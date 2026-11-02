"""Answer correctness, ported from MedEvalKit (third_party/MedEvalKit/utils/utils.py L95-226), except open questions.

Yes/no, closed and multiple-choice judges are ported line by line, quirks included (the yes/no check is a substring
test). Open questions differ on purpose: MedEvalKit's published numbers (Lingshu, MedSTP) score them with a GPT-4.1
judge (eval.sh: USE_LLM_JUDGE="True"); its fallback is strict exact match, under which Lingshu's habitual trailing
period ("CT." vs "CT") counts as wrong (E0: 38% Full on slake_en). We use normalized exact match (case, punctuation,
articles, whitespace) as the primary open metric and keep strict exact match and token recall as secondary ones.
All claims are paired comparisons inside this pipeline, so absolute agreement with published numbers is not needed.
"""

from __future__ import annotations

import difflib
import re
import string
from typing import Sequence

from .data import Sample

_ANSWER_PATTERNS = ["**answer**:", "**answer**", "*answer*:", "**answer:**", "answer is", "answer:", "答案:",
                    "final answer", "final answer is"]


def extract_boxed_content(text: str) -> str:
    """Content of the last \\boxed{...} (mathruler.grader.extract_boxed_content); "None" if absent."""
    start = text.rfind(r"\boxed{")
    if start == -1:
        return "None"
    content = text[start + len(r"\boxed{"):]
    depth = 0
    for i, ch in enumerate(content):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if depth == -1:
            return content[:i].strip()
    return "None"


def extract_tag(text: str, tag: str) -> str:
    """Text between <tag> and </tag> (all occurrences, space-joined); the full text if there is none."""
    if not text:
        return ""
    start_tag, end_tag = f"<{tag}>", f"</{tag}>"
    pieces, start = [], text.find(start_tag)
    while start != -1:
        end = text.find(end_tag, start + len(start_tag))
        if end == -1:
            break
        pieces.append(text[start + len(start_tag): end])
        start = text.find(start_tag, end + len(end_tag))
    found = " ".join(pieces).strip()
    return found if found else text


def parse_response(response: str) -> str:
    response = response.lower()
    if "boxed" in response:
        response = extract_boxed_content(response)
    elif "<answer>" in response:
        response = extract_tag(response, "answer")
    for pattern in _ANSWER_PATTERNS:
        if pattern in response:
            response = response.split(pattern)[-1]
    return response


def judge_yesno(answer: str, response: str) -> bool:
    answer = answer.lower()
    response = parse_response(response).replace("\n", "").replace(".", "")
    if ("yes" in response) ^ ("no" in response):
        return answer in response
    return False


def judge_closed(answer: str, response: str) -> bool:
    answer = answer.lower()
    response = parse_response(response).replace("\n", "").replace(".", "")
    return response == answer


_PUNCT = re.compile("[" + re.escape(string.punctuation) + "　-〿！-／：-＠]")
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(text: str) -> str:
    """Lowercase, drop ASCII and CJK punctuation and English articles, collapse whitespace."""
    text = _PUNCT.sub(" ", text.lower())
    return " ".join(_ARTICLES.sub(" ", text).split())


def judge_open(answer: str, response: str) -> tuple[bool, float]:
    """(normalized exact match, token recall). Strict MedEvalKit exact match: `judge_open_strict`."""
    gold, pred = normalize_answer(answer), normalize_answer(parse_response(response))
    gt_tokens, pred_tokens = set(gold.split()), set(pred.split())
    recall = len(gt_tokens & pred_tokens) / len(gt_tokens) if gt_tokens else 0.0
    return pred == gold, recall


def judge_open_strict(answer: str, response: str) -> bool:
    return parse_response(response) == answer.lower()


def _most_similar_index(options: Sequence[str], target: str) -> int:
    best_index, best = 0, 0.0
    for i, option in enumerate(options):
        similarity = difflib.SequenceMatcher(None, option, target).ratio()
        if similarity > best:
            best_index, best = i, similarity
    return best_index


def judge_mc(choices: Sequence[str], answer: str, response: str) -> bool:
    """Multiple choice: letter, choice text, or nearest choice by difflib (MedEvalKit judge_multi_choice)."""
    letters = [chr(ord("a") + i) for i in range(len(choices))]
    response = response.lower()
    if response.split("\n\n")[0] in letters:
        response = response.split("\n\n")[0]
    elif response.split("\n\n")[-1].split(".")[0] in letters:
        response = response.split("\n\n")[-1].split(".")[0]
    response = parse_response(response)
    lowered = [c.lower() for c in choices]
    response = response.strip().lower().replace("\n", "")
    split_response = response.split(".")[0].split(":")[-1]
    answer = answer.strip().lower()
    if split_response == answer:
        return True
    if split_response in letters:
        return lowered[ord(split_response) - ord("a")] == answer
    if split_response in lowered:
        return answer in letters and split_response == lowered[ord(answer) - ord("a")]
    index = _most_similar_index(lowered, response)
    return letters[index] == answer or lowered[index] == answer


def is_correct(sample: Sample, response: str) -> tuple[bool, float | None]:
    """Correctness of one response and, for open questions, token recall."""
    boxed = extract_boxed_content(response)
    if boxed != "None":
        response = boxed
    elif "<answer>" in response:
        response = extract_tag(response, "answer")
    if sample.answer_type == "mc":
        return judge_mc(sample.choices or (), sample.answer, response), None
    if sample.answer_type == "yesno":
        return judge_yesno(sample.answer, response), None
    if sample.answer_type == "closed":
        return judge_closed(sample.answer, response), None
    if sample.answer_type == "open":
        exact, recall = judge_open(sample.answer, response)
        return exact, recall
    raise ValueError(f"unknown answer_type {sample.answer_type!r}")
