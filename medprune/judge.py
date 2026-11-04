"""Open-answer LLM judge (plan.md §A9, G9).

Normalized exact match (scoring.py) misses synonyms: "axial" vs "Transverse Plane", "lungs" vs "both lungs".
MedEvalKit's published SLAKE numbers instead ask GPT-4.1 whether the response means the same as the reference
(third_party/MedEvalKit/utils/utils.py, get_compare_messages). This module asks a local instruction-tuned LLM the same
question and reads the verdict as one token from the logits, P(Yes) / (P(Yes) + P(No)), so judging costs one forward
pass per unique (question, reference, response) and needs no API. MedEvalKit's own coding ("<judge>0" = correct) is
not used: without its reasoning step, a CPU smoke test (Qwen2.5-0.5B) read "0" as "incorrect" even for exact
matches. The judge sees no image, no model name and no method name.

Prompt v3 (Nov 3): pass 1 with MedEvalKit's wording (v2) let Qwen2.5-7B overrule the reference with its own knowledge
("largest organ in the picture? Liver" -> "No") and accept partial or refusing answers. v3 states the rubric — the
reference is ground truth, the judge cannot see the image, partial / less specific / refusing answers are wrong,
consistent extra detail is fine — and the scorer is `rule OR judge`: a normalized exact match is always correct, the
judge only rescues answers the rule misses. The hand-label sheet carries the same rubric.

Workflow (laptop -> Kaggle GPU -> laptop):
    python -m medprune.judge export --results results/all --out results/judge/items.jsonl
    python -m medprune.judge run --items items.jsonl --out verdicts.jsonl          # GPU, resumable
    python -m medprune.judge sheet --results results/all                           # blind hand-label sheet
    python -m medprune.judge kappa --results results/all                           # judge and rule vs hand labels
Decision rule (fixed before any label is read): the scorer rule OR judge becomes the primary open-question metric
(analysis.load_runs(open_scoring="judge")) if its kappa with the hand labels is >= 0.8 and above kappa(rule, human);
otherwise normalized exact match stays primary and the judge is reported as a robustness check.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .scoring import extract_boxed_content, extract_tag, judge_open

JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # Apache-2.0, ungated, bilingual (SLAKE-zh); fp16 on T4x2
PROMPT_VERSION = "rubric/yes-no/v3"
RUBRIC = (
    "The model's answer is correct if it means the same as the standard answer: synonyms, paraphrases, different "
    "formatting, or extra detail that is consistent with the standard answer count as correct. It is incorrect if it "
    "names a different thing, contradicts the standard answer, is less specific than the standard answer (for example "
    "'Left' for 'Upper Left'), gives only part of a standard answer that lists several items, or does not answer (for "
    "example, says it cannot see the image)."
)
VERDICTS = {"yes": ("Yes", "yes"), "no": ("No", "no")}  # first-token spellings summed per verdict
KAPPA_TARGET = 0.8


def clean_response(response: str) -> str:
    """The part of a response that is scored (the same extraction as scoring.is_correct), stripped."""
    response = str(response or "")
    boxed = extract_boxed_content(response)
    if boxed != "None":
        response = boxed
    elif "<answer>" in response:
        response = extract_tag(response, "answer")
    return response.strip()


def item_key(question: str, answer: str, response: str) -> str:
    """Identity of one judgement: the same question, reference and scored response get one verdict."""
    payload = json.dumps([question, answer, clean_response(response)], ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


def judge_prompt(question: str, answer: str, response: str) -> str:
    """Comparison against the reference under RUBRIC; the verdict is one word (Yes/No), read from the logits."""
    return (
        "You are grading a medical visual question answering system. You cannot see the image. The standard answer is "
        "the ground truth for this image; do not use your own medical knowledge to decide what the image shows.\n\n"
        f"Question: {question}\n"
        f"Standard answer: {answer}\n"
        f"Model's answer: {clean_response(response)}\n\n"
        f"{RUBRIC}\n\n"
        "Is the model's answer correct? Reply with one word: Yes or No."
    )


# ------------------------------------------------------------------------------------------- items, verdicts

def open_rows(results: Path) -> pd.DataFrame:
    """Every open-question row of every run under results/runs, with its judge key and rule verdict."""
    rows = []
    for path in sorted((Path(results) / "runs").glob("*.jsonl")):
        model, dataset, method, budget, seed, _ = path.stem.split("__")
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("answer_type") != "open":
                    continue
                rows.append({"run_id": path.stem, "model": model, "dataset": dataset, "method": method,
                             "budget_pct": int(budget[1:]), "seed": int(seed[1:]), "qid": r["qid"],
                             "lang": r.get("lang", ""), "question": r["question"], "answer": r["answer"],
                             "response": clean_response(r.get("response", "")),
                             "key": item_key(r["question"], r["answer"], r.get("response", "")),
                             "rule": bool(judge_open(r["answer"], r.get("response", "") or "")[0])})
    return pd.DataFrame(rows)


def export_items(results: Path, out: Path, verdicts: Path | None = None) -> int:
    """Write the unique (question, reference, response) items not judged yet; returns how many."""
    rows = open_rows(results)
    done = set(load_verdicts(verdicts)) if verdicts and Path(verdicts).exists() else set()
    items = rows.drop_duplicates("key")
    items = items[~items["key"].isin(done)]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in items.itertuples():
            f.write(json.dumps({"key": r.key, "question": r.question, "answer": r.answer, "response": r.response,
                                "lang": r.lang, "dataset": r.dataset}, ensure_ascii=False) + "\n")
    return len(items)


def load_verdicts(path: Path, prompt: str | None = PROMPT_VERSION) -> dict[str, float]:
    """{key: P(correct)} from a verdicts file, for one prompt version (None = any); a key judged twice under the same
    prompt must agree (resumed runs append)."""
    out: dict[str, float] = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                v = json.loads(line)
                if prompt is not None and v.get("prompt") != prompt:
                    continue
                if v["key"] in out and abs(out[v["key"]] - v["p_correct"]) > 1e-3:
                    raise ValueError(f"conflicting verdicts for {v['key']} in {path}")
                out[v["key"]] = float(v["p_correct"])
    return out


# Rubric guard (Nov 4, after the first label audit): the judge ignores the rubric's "less specific" clause for
# locations — of the answers it rescued, 17 of 90 were "Left Lung" for "Lower Left Lung" and alike — and rejects the
# one plane synonym SLAKE uses ("axial" for "Transverse Plane", 4 of 90 rejected items). Both are applied
# deterministically, so the scorer is: rule OR synonym OR (judge AND NOT less-specific-location).
LOCATION_WORDS = {"upper": "upper", "top": "upper", "apical": "upper", "superior": "upper",
                  "lower": "lower", "bottom": "lower", "basal": "lower", "inferior": "lower",
                  "left": "left", "right": "right", "center": "center", "central": "center", "middle": "center"}
SYNONYMS = (("transverse", "axial"),)
SCORER_VERSION = "rule|synonym|judge-v3&!less-specific-location"


def location_qualifiers(text: str) -> set[str]:
    return {LOCATION_WORDS[w] for w in re.findall(r"[a-z]+", str(text).lower()) if w in LOCATION_WORDS}


def less_specific_location(reference: str, response: str) -> bool:
    """True if the response's location qualifiers are a strict subset of the reference's ("left lung" for "Lower
    Left Lung"); more specific answers ("right lower lobe" for "Right Lung") pass."""
    ref = location_qualifiers(reference)
    return bool(ref) and location_qualifiers(response) < ref


def is_synonym(reference: str, response: str) -> bool:
    ref, resp = str(reference).lower(), clean_response(response).lower()
    return any((a in ref and b in resp) or (b in ref and a in resp) for a, b in SYNONYMS)


def apply_judge(df: pd.DataFrame, verdicts: dict[str, float], threshold: float = 0.5, guard: bool = True) -> pd.DataFrame:
    """Open rows' `correct` = rule OR judge (P(correct) >= threshold): a normalized exact match is always correct, the
    judge rescues the rest; `correct_rule` keeps the rule verdict. With `guard` (default; SCORER_VERSION), the plane
    synonym counts as correct and the judge cannot rescue a less specific location.

    Raises if an open row has no verdict: a table must never mix judged and rule-scored open questions.
    """
    df = df.copy()
    df["correct_rule"] = df["correct"]
    is_open = df["answer_type"] == "open"
    keys = [item_key(q, a, r) for q, a, r in zip(df.loc[is_open, "question"], df.loc[is_open, "answer"],
                                                   df.loc[is_open, "response"].fillna(""))]
    missing = [k for k in keys if k not in verdicts]
    if missing:
        raise ValueError(f"{len(missing)} open rows have no judge verdict (first key {missing[0]}); "
                         "export and judge them first")
    df.loc[is_open, "p_judge"] = [verdicts[k] for k in keys]
    judge_yes = df.loc[is_open, "p_judge"] >= threshold
    rule = df.loc[is_open, "correct_rule"].astype(bool)
    if guard:
        ref, resp = df.loc[is_open, "answer"], df.loc[is_open, "response"].fillna("")
        less = pd.Series([less_specific_location(a, r) for a, r in zip(ref, resp)], index=ref.index)
        syn = pd.Series([is_synonym(a, r) for a, r in zip(ref, resp)], index=ref.index)
        df.loc[is_open, "correct"] = rule | syn | (judge_yes & ~less)
    else:
        df.loc[is_open, "correct"] = rule | judge_yes
    df["correct"] = df["correct"].astype(bool)
    return df


# --------------------------------------------------------------------------------------------- the judge

def run_judge(items_path: Path, out_path: Path, model_id: str = JUDGE_MODEL, batch_size: int = 16,
              limit: int | None = None) -> int:
    """Judge every item not yet in `out_path` (appends, so a stopped session resumes). GPU; returns items judged."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .env import detect_profile, torch_dtype

    items = [json.loads(line) for line in Path(items_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    out_path = Path(out_path)
    done = set(load_verdicts(out_path)) if out_path.exists() else set()
    todo = [it for it in items if it["key"] not in done][:limit]
    if not todo:
        return 0
    profile = detect_profile()
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype(profile), device_map=profile.device_map,
                                                 max_memory=profile.max_memory).eval()
    verdict_ids = {}
    for verdict, spellings in VERDICTS.items():
        ids = [tok.encode(s, add_special_tokens=False) for s in spellings]
        if any(len(i) != 1 for i in ids):
            raise RuntimeError(f"verdict {verdict!r} is not one token for {model_id}: {ids}")
        verdict_ids[verdict] = [i[0] for i in ids]
    texts = [tok.apply_chat_template([{"role": "user", "content": judge_prompt(it["question"], it["answer"],
                                                                                it["response"])}],
                                     tokenize=False, add_generation_prompt=True) for it in todo]
    order = sorted(range(len(todo)), key=lambda i: len(texts[i]))  # similar lengths per batch, less padding
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f, torch.inference_mode():
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            enc = tok([texts[i] for i in batch], return_tensors="pt", padding=True).to(model.device)
            positions = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)  # left padding: real tokens start at 0
            logits = model(**enc, position_ids=positions, use_cache=False, logits_to_keep=1).logits[:, -1].float()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("non-finite judge logits (fp16 overflow?)")
            probs = logits.softmax(-1)
            p_yes = probs[:, verdict_ids["yes"]].sum(-1).tolist()
            p_no = probs[:, verdict_ids["no"]].sum(-1).tolist()
            for i, yes, no in zip(batch, p_yes, p_no):
                f.write(json.dumps({"key": todo[i]["key"], "p_correct": yes / (yes + no), "verdict_mass": yes + no,
                                    "judge": model_id, "prompt": PROMPT_VERSION}) + "\n")
            f.flush()
    return len(todo)


# ------------------------------------------------------------------------------------------ validation

SHEET_STRATA = {"exact": 0.1, "judge_yes": 0.45, "judge_no": 0.45}  # share of the n labelled items


def label_sheet(results: Path, verdicts: dict[str, float], out_csv: Path, n: int = 200, lang: str = "en",
                seed: int = 0) -> pd.DataFrame:
    """Blind hand-label sample, stratified where the scorer (rule OR judge) can be wrong: answers the rule already
    accepts ("exact", a small check), and rule-rejected answers the judge accepts / rejects ("judge_yes", "judge_no").
    Within a stratum, unique items are drawn with probability proportional to how many rows they stand for. The sheet
    shows only question, reference and response; strata and weights go to a key file for `kappa_report`, and the
    rubric the judge used to a README next to the sheet."""
    rows = open_rows(results)
    rows = rows[(rows["lang"] == lang) & rows["key"].isin(verdicts)].copy()
    rows["judge"] = rows["rule"] | (rows["key"].map(verdicts) >= 0.5)
    rows["stratum"] = np.where(rows["rule"], "exact", np.where(rows["judge"], "judge_yes", "judge_no"))
    rng = np.random.default_rng(seed)
    picked = []
    for stratum, g in rows.groupby("stratum"):
        unique = g.drop_duplicates("key")
        take = min(len(unique), int(round(n * SHEET_STRATA[stratum])))
        # one draw per unique item, chosen with probability proportional to how many rows it stands for
        weights = g["key"].value_counts().reindex(unique["key"]).to_numpy(float)
        chosen = rng.choice(len(unique), size=take, replace=False, p=weights / weights.sum())
        sample = unique.iloc[chosen].assign(stratum_rows=len(g), stratum_items=take)
        picked.append(sample)
    sheet = pd.concat(picked).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sheet["item"] = np.arange(1, len(sheet) + 1)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sheet[["item", "question", "answer", "response"]].rename(columns={"answer": "reference"}).assign(label="").to_csv(
        out_csv, index=False, encoding="utf-8-sig")
    sheet[["item", "key", "stratum", "stratum_rows", "stratum_items", "rule", "judge"]].to_csv(
        out_csv.with_name(out_csv.stem + "_key.csv"), index=False)
    out_csv.with_name(out_csv.stem + "_README.txt").write_text(
        "Label each row: 1 = the response is a correct answer, 0 = it is not. Judge only against the reference answer "
        "(treat it as the truth for that image); leave a cell empty to skip. Rubric:\n" + RUBRIC + "\n", encoding="utf-8")
    return sheet


def weighted_kappa(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    """Cohen's kappa for two binary raters with sampling weights."""
    a, b, w = np.asarray(a, bool), np.asarray(b, bool), np.asarray(w, float) / np.sum(w)
    p_o = np.sum(w * (a == b))
    p_a, p_b = np.sum(w * a), np.sum(w * b)
    p_e = p_a * p_b + (1 - p_a) * (1 - p_b)
    return float((p_o - p_e) / (1 - p_e)) if p_e < 1 else 1.0


def kappa_report(labels_csv: Path, B: int = 2000, seed: int = 0) -> dict:
    """Judge and rule against the hand labels, weighted back to the row population (stratified bootstrap CIs)."""
    labels = pd.read_csv(labels_csv, encoding="utf-8-sig")
    key = pd.read_csv(Path(labels_csv).with_name(Path(labels_csv).stem + "_key.csv"))
    d = labels.merge(key, on="item")
    d = d[d["label"].astype(str).str.strip().isin(["0", "1", "0.0", "1.0"])].copy()
    d["human"] = d["label"].astype(float) >= 0.5
    d["w"] = d["stratum_rows"] / d.groupby("stratum")["item"].transform("count")
    d["judge"], d["rule"] = d["judge"].astype(bool), d["rule"].astype(bool)
    less = np.array([less_specific_location(a, r) for a, r in zip(d["reference"].fillna(""), d["response"].fillna(""))])
    syn = np.array([is_synonym(a, r) for a, r in zip(d["reference"].fillna(""), d["response"].fillna(""))])
    d["guarded"] = d["rule"] | syn | (d["judge"] & ~less)

    def stats(x: pd.DataFrame) -> dict:
        return {"kappa_judge": weighted_kappa(x["judge"], x["human"], x["w"]),
                "kappa_guarded": weighted_kappa(x["guarded"], x["human"], x["w"]),
                "kappa_rule": weighted_kappa(x["rule"], x["human"], x["w"]),
                "acc_judge": float(np.average(x["judge"] == x["human"], weights=x["w"])),
                "acc_guarded": float(np.average(x["guarded"] == x["human"], weights=x["w"])),
                "acc_rule": float(np.average(x["rule"] == x["human"], weights=x["w"]))}

    point = stats(d)
    rng, boot = np.random.default_rng(seed), []
    groups = [g for _, g in d.groupby("stratum")]
    for _ in range(B):
        boot.append(stats(pd.concat([g.sample(len(g), replace=True, random_state=int(rng.integers(2**31)))
                                     for g in groups])))
    ci = {k: (float(np.percentile([b[k] for b in boot], 2.5)), float(np.percentile([b[k] for b in boot], 97.5)))
          for k in point}
    disagree = d[d["judge"] != d["rule"]]
    report = {"labelled": len(d), "by_stratum": d["stratum"].value_counts().to_dict(), **point, "ci": ci,
              "judge_right_when_they_disagree": float((disagree["judge"] == disagree["human"]).mean())
              if len(disagree) else np.nan}
    report["judge_becomes_primary"] = bool(point["kappa_judge"] >= KAPPA_TARGET
                                           and point["kappa_judge"] > point["kappa_rule"])
    return report


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Open-answer LLM judge (plan.md §A9 G9).")
    p.add_argument("command", choices=["export", "run", "sheet", "kappa"])
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--items", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--verdicts", type=Path)
    p.add_argument("--model", default=JUDGE_MODEL)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--lang", default="en")
    args = p.parse_args()
    judge_dir = args.results / "judge"
    verdicts = args.verdicts or judge_dir / "verdicts.jsonl"
    if args.command == "export":
        out = args.out or judge_dir / "items.jsonl"
        print(f"{export_items(args.results, out, verdicts)} new items -> {out}")
    elif args.command == "run":
        print(f"judged {run_judge(args.items, args.out or verdicts, args.model, args.batch_size, args.limit)} items")
    elif args.command == "sheet":
        out = args.out or judge_dir / "label_sheet.csv"
        sheet = label_sheet(args.results, load_verdicts(verdicts), out, args.n, args.lang)
        print(f"{len(sheet)} items -> {out} (strata: {sheet['stratum'].value_counts().to_dict()})")
    else:
        print(json.dumps(kappa_report(args.out or judge_dir / "label_sheet.csv"), indent=2, default=str))


if __name__ == "__main__":
    main()
