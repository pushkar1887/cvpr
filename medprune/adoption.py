"""The E42 adoption rule (plan.md, registered Nov 9 18:01, git c2199f7), as code.

Statistic: the mean over the nine informative cells (Lingshu: SLAKE, RIN anatomy, RIN abnormality, OmniMedVQA,
RIN open anatomy; Qwen3-VL: the same without RIN abnormality, its language-prior cell) of the paired per-question
accuracy difference candidate - reference, in points; Lingshu RIN abnormality in balanced accuracy (its gold answers
are 71.9% "yes"). Uncertainty: images resampled with replacement per dataset, jointly across both models (the same
image appears in both models' cells), B draws.

Rule, per candidate (Holm over the candidates, family-wise alpha 0.05):
  1. one-sided z = delta / SE with SE = sqrt(SE_boot^2 + seed_sd^2) (seed_sd = 0 when the candidate keeps the
     reference's tokens, 0.28 for a different keep-set) and p = 1 - Phi(z);
  2. delta > 0 on both image halves (dev: int(sha1(image_id)[:8], 16) even);
  3. no informative cell with a significant loss (two-sided p < alpha / 9).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CELLS: tuple[tuple[str, str], ...] = (
    ("lingshu7b", "slake_en"), ("lingshu7b", "rin_anat_mc"), ("lingshu7b", "rin_abn_closed"),
    ("lingshu7b", "omni_anat"), ("lingshu7b", "rin_open"),
    ("qwen3vl8b", "slake_en"), ("qwen3vl8b", "rin_anat_mc"), ("qwen3vl8b", "omni_anat"), ("qwen3vl8b", "rin_open"))
BALANCED: frozenset = frozenset({("lingshu7b", "rin_abn_closed")})
SINGLE_TOKEN: frozenset = frozenset({"mc", "judgement"})  # instructions whose answer is one token (E43 route)


def prompt_style(dataset: str, answer_type: str) -> str:
    """The MedEvalKit instruction a question gets (medprune.data.make_prompt): "mc" (answer with the option's letter),
    "judgement" (output 'yes' or 'no': RadImageNet abnormality and natural yes / no), "open" (answer concisely),
    "close" (a single word or phrase: SLAKE's closed questions, natural's closed ones)."""
    if answer_type == "mc":
        return "mc"
    if answer_type == "open":
        return "open"
    if answer_type == "yesno" and (dataset.startswith("rin_") or dataset == "natural"):
        return "judgement"
    return "close"


def image_half(image_id: str) -> int:
    """0 = dev, 1 = test; depends only on the image, so both models and every method share the split."""
    return int(hashlib.sha1(image_id.encode("utf-8")).hexdigest()[:8], 16) % 2


def normal_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


@dataclass
class Cell:
    """Paired per-question arrays of one (model, dataset) cell: candidate / reference correctness (seed means in
    [0, 1]), gold yes/no and predicted-yes rates (balanced cells), image ids."""
    a: np.ndarray
    b: np.ndarray
    image: np.ndarray
    gold: np.ndarray | None = None
    yes_a: np.ndarray | None = None
    yes_b: np.ndarray | None = None

    def subset(self, keep: np.ndarray) -> "Cell":
        pick = lambda x: None if x is None else x[keep]
        return Cell(self.a[keep], self.b[keep], self.image[keep], pick(self.gold), pick(self.yes_a), pick(self.yes_b))

    def delta(self, w: np.ndarray | None = None) -> float:
        """Points; w = per-question bootstrap weights (counts of each question's image in the draw)."""
        w = np.ones(len(self.a)) if w is None else w
        if self.gold is not None:
            return balanced(w, self.gold, self.yes_a) - balanced(w, self.gold, self.yes_b)
        return 100 * float((w * (self.a - self.b)).sum() / max(w.sum(), 1e-12))


def balanced(w: np.ndarray, gold: np.ndarray, yes: np.ndarray) -> float:
    """Balanced accuracy (points) of yes-rates `yes` against boolean `gold`, question weights w."""
    pos, neg = (w * gold).sum(), (w * ~gold).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    return 50 * float((w * gold * yes).sum() / pos + (w * ~gold * (1 - yes)).sum() / neg)


@dataclass
class Result:
    delta: float
    se_boot: float
    halves: tuple[float, float]
    cells: dict[tuple[str, str], tuple[float, float]]  # cell -> (delta, two-sided p)
    seed_sd: float = 0.0

    @property
    def se(self) -> float:
        return math.sqrt(self.se_boot ** 2 + self.seed_sd ** 2)

    @property
    def p(self) -> float:
        return normal_sf(self.delta / self.se) if self.se > 0 else (0.0 if self.delta > 0 else 1.0)


def nine_cell(cells: dict[tuple[str, str], Cell], draws: int = 4000, seed: int = 0, seed_sd: float = 0.0) -> Result:
    """Nine-cell mean difference with the joint image bootstrap; per-cell deltas with two-sided normal p-values from
    the same draws; deltas on the dev and test halves."""
    rng = np.random.default_rng(seed)
    datasets = sorted({ds for _, ds in cells})
    images = {ds: np.unique(np.concatenate([c.image for (m, d), c in cells.items() if d == ds])) for ds in datasets}
    index = {key: np.searchsorted(images[key[1]], c.image) for key, c in cells.items()}
    boot = np.empty((draws, len(cells)))
    for i in range(draws):
        counts = {ds: np.bincount(rng.integers(0, len(im), len(im)), minlength=len(im)) for ds, im in images.items()}
        boot[i] = [c.delta(counts[key[1]][index[key]].astype(float)) for key, c in cells.items()]
    point = np.array([c.delta() for c in cells.values()])
    per_cell = {}
    for j, key in enumerate(cells):
        sd = np.nanstd(boot[:, j], ddof=1)
        z = point[j] / sd if sd > 0 else 0.0
        per_cell[key] = (float(point[j]), 2 * normal_sf(abs(z)))
    halves = tuple(float(np.mean([c.subset(np.array([image_half(x) == h for x in c.image])).delta()
                                  for c in cells.values()])) for h in (0, 1))
    return Result(float(point.mean()), float(np.nanstd(boot.mean(axis=1), ddof=1)), halves, per_cell, seed_sd)


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm's step-down procedure: rejected[name]."""
    order = sorted(pvalues, key=pvalues.get)
    rejected, still = {}, True
    for rank, name in enumerate(order):
        still = still and pvalues[name] <= alpha / (len(order) - rank)
        rejected[name] = still
    return rejected


def decide(results: dict[str, Result], alpha: float = 0.05) -> dict[str, dict]:
    """Apply rules 1-3 to every candidate; adopted[name] with the reasons."""
    rejected = holm({name: r.p for name, r in results.items()}, alpha)
    out = {}
    for name, r in results.items():
        losses = [key for key, (d, p) in r.cells.items() if d < 0 and p < alpha / len(r.cells)]
        replicated = all(h > 0 for h in r.halves)
        out[name] = {"test": rejected[name], "replicated": replicated, "losses": losses,
                     "adopted": rejected[name] and replicated and not losses}
    return out


# ------------------------------------------------------------------------------------------------- loading

def load_cell(results: Path, model: str, dataset: str, method: str, verdicts: dict) -> pd.DataFrame:
    """Per-question judged correctness (seed mean), yes-rate and gold for one run family; rin_open = its anatomy
    questions (plan.md)."""
    from .analysis import parse_run_id, rescore_open
    from .judge import apply_judge

    if method.startswith("route:"):  # E43: "route:A|B" = A's answer where the instruction asks for one token, else B's
        single, generated = method[len("route:"):].split("|")
        try:
            out = load_cell(results, model, dataset, generated, verdicts)
        except FileNotFoundError:  # a cell whose questions are all single-token needs only the first branch
            a = load_cell(results, model, dataset, single, verdicts)
            if not a["atype"].map(lambda x: prompt_style(dataset, x)).isin(SINGLE_TOKEN).all():
                raise
            return a
        pick = out["atype"].map(lambda a: prompt_style(dataset, a)).isin(SINGLE_TOKEN)
        if pick.any():
            a = load_cell(results, model, dataset, single, verdicts)
            idx = pick.index[pick]
            if not idx.isin(a.index).all():
                raise ValueError(f"{single} lacks single-token questions of {model} {dataset}")
            out = out.copy()
            out.loc[idx, ["c", "yes"]] = a.loc[idx, ["c", "yes"]].to_numpy()
        return out
    method, _, seed = method.partition("@s")  # "name@s1" = that seed only; plain names average every seed on disk
    budget = "r100" if method == "full" else "r010"
    rows = []
    for path in sorted((Path(results) / "runs").glob(f"{model}__{dataset}__{method}__{budget}__s{seed or '*'}__t4x2.jsonl")):
        info = parse_run_id(path.stem)
        rows += [{**json.loads(line), **info} for line in path.open(encoding="utf-8") if line.strip()]
    if not rows:
        raise FileNotFoundError(f"no runs for {model} {dataset} {method}")
    df = apply_judge(rescore_open(pd.DataFrame(rows)), verdicts)
    if dataset == "rin_open":
        df = df[df["qid"].str.endswith("anatomy_open")]
    df["gold"] = df["answer"].str.strip().str.lower().str.startswith("yes")
    df["yes"] = df["response"].fillna("").str.strip().str.lower().str.match(r"\W*yes")
    return df.groupby("qid").agg(c=("correct", "mean"), image=("image_id", "first"), gold=("gold", "first"),
                                 yes=("yes", "mean"), atype=("answer_type", "first"))


def paired_cells(results: Path, candidate: str, reference: str, verdicts: dict) -> dict[tuple[str, str], Cell]:
    cells = {}
    for model, dataset in CELLS:
        a = load_cell(results, model, dataset, candidate, verdicts)
        b = load_cell(results, model, dataset, reference, verdicts)
        q = a.index.intersection(b.index)
        if len(q) < len(b):
            raise ValueError(f"{candidate} lacks {len(b) - len(q)} of {reference}'s questions in {model} {dataset}")
        bal = (model, dataset) in BALANCED
        cells[(model, dataset)] = Cell(a.loc[q, "c"].to_numpy(float), b.loc[q, "c"].to_numpy(float),
                                       a.loc[q, "image"].to_numpy(),
                                       a.loc[q, "gold"].to_numpy(bool) if bal else None,
                                       a.loc[q, "yes"].to_numpy(float) if bal else None,
                                       b.loc[q, "yes"].to_numpy(float) if bal else None)
    return cells


def main(argv: list[str] | None = None) -> None:
    from .judge import load_verdicts

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", type=Path, default=Path("results/all"))
    p.add_argument("--reference", default="gate_jitter_merge_eq")
    p.add_argument("--candidate", action="append", required=True,
                   help="method[:seed_sd]; seed_sd 0 when the candidate keeps the reference's tokens")
    p.add_argument("--draws", type=int, default=4000)
    args = p.parse_args(argv)
    verdicts = load_verdicts(args.results / "judge" / "verdicts.jsonl")
    results = {}
    for spec in args.candidate:
        name, _, sd = spec.partition(":")
        results[name] = nine_cell(paired_cells(args.results, name, args.reference, verdicts), args.draws,
                                  seed_sd=float(sd or 0))
    decisions = decide(results)
    for name, r in results.items():
        d = decisions[name]
        print(f"{name} - {args.reference}: nine-cell {r.delta:+.2f} (SE {r.se:.2f}; boot {r.se_boot:.2f}, seed "
              f"{r.seed_sd:.2f}), z {r.delta / r.se:.2f}, one-sided p {r.p:.4f}; halves dev {r.halves[0]:+.2f} / "
              f"test {r.halves[1]:+.2f}; Holm {'reject' if d['test'] else 'keep'} H0; replicated {d['replicated']}; "
              f"significant losses {d['losses'] or 'none'} -> {'ADOPT' if d['adopted'] else 'not adopted'}")
        print("   " + "  ".join(f"{m[:4]}:{ds} {dl:+.1f} (p {pv:.3f})" for (m, ds), (dl, pv) in r.cells.items()))


if __name__ == "__main__":
    main()
