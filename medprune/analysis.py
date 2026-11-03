"""Turn run files into tables, figures and decisions (plan.md Stage 4 E2, §C7). Laptop only, no GPU.

Inputs are the merged Kaggle outputs in results/: runs/*.jsonl (one line per question), diag/<run_id>/part-*.npz
(scores and norms of `full` and score-based runs) and gates/<model>__<dataset>.npz (per-token foreground fractions,
copied by run_eval so RadImageNet's session-only masks are available here too).

Visual-token index i is grid cell i in raster order for every model used here (Qwen's get_rope_index assigns
positions in raster order; HuatuoGPT's 24x24 grid is raster), so keep_idx indexes the gate arrays directly.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from PIL import Image

from .masks import MODEL_VIEWS, MaskStore, label_view, token_foreground_fraction
from .scoring import judge_open, normalize_answer, parse_response
from .stats import (background_retention_ratio, cluster_bootstrap_ci, jaccard, keep_map, mcnemar_exact,
                    paired_delta_ci, quadrant_concentration, static_map_from_scores)

TAU = 0.5  # background threshold on the per-token foreground fraction (plan.md §A5)


# --------------------------------------------------------------------------------------------- loading

def parse_run_id(run_id: str) -> dict:
    model, dataset, method, budget, seed, profile = run_id.split("__")
    return {"model": model, "dataset": dataset, "method": method, "budget_pct": int(budget[1:]),
            "seed": int(seed[1:]), "profile": profile}


def modality_group(modality: str) -> str:
    """CT and MRI together (the air-heavy scans), X-ray, everything else by name."""
    low = str(modality).lower()
    if low.startswith("ct") or "computed tomography" in low or low.startswith("mr") or "magnetic" in low:
        return "CT/MRI"
    if "x-ray" in low or "xray" in low:
        return "X-ray"
    return str(modality)


def load_runs(results: Path, profile: str | None = None, open_scoring: str = "rule") -> pd.DataFrame:
    """One row per (run, question): the run file's fields plus model/dataset/method/budget/seed/profile.

    `open_scoring="judge"` scores open questions with the LLM judge's verdicts (results/judge/verdicts.jsonl,
    medprune/judge.py) instead of normalized exact match; the rule verdict stays in `correct_rule`.
    """
    rows = []
    for path in sorted((Path(results) / "runs").glob("*.jsonl")):
        info = parse_run_id(path.stem)
        if profile and info["profile"] != profile:
            continue
        seen = set()
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["qid"] in seen:
                    raise ValueError(f"Duplicate qid {row['qid']!r} in {path}; repair legacy identities before analysis")
                seen.add(row["qid"])
                rows.append({**row, **info, "run_id": path.stem})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["modality_group"] = df["modality"].map(modality_group)
    df = rescore_open(df)
    if open_scoring == "judge":
        from .judge import apply_judge, load_verdicts

        df = apply_judge(df, load_verdicts(Path(results) / "judge" / "verdicts.jsonl"))
    elif open_scoring != "rule":
        raise ValueError(f"open_scoring must be 'rule' or 'judge', not {open_scoring!r}")
    return add_image_dependent(df)


def rescore_open(df: pd.DataFrame) -> pd.DataFrame:
    """Re-judge open questions from the saved responses, so a scoring change never needs a GPU rerun."""
    if "response" not in df or "answer_type" not in df:
        return df
    is_open = df["answer_type"] == "open"
    if is_open.any():
        judged = [judge_open(a, r) for a, r in zip(df.loc[is_open, "answer"], df.loc[is_open, "response"].fillna(""))]
        df.loc[is_open, "correct"] = [j[0] for j in judged]
        df.loc[is_open, "open_recall"] = [j[1] for j in judged]
        df["correct"] = df["correct"].astype(bool)
    return df


def add_image_dependent(df: pd.DataFrame) -> pd.DataFrame:
    """Column `image_dependent`: the model answers correctly with the image and wrongly without it."""
    base = df[df["method"].isin(["full", "text_only"])]
    if base.empty or base["method"].nunique() < 2:
        df["image_dependent"] = False
        return df
    wide = base.pivot_table(index=["model", "profile", "qid"], columns="method", values="correct", aggfunc="first")
    dependent = set(wide[(wide["full"] == True) & (wide["text_only"] == False)].index)  # noqa: E712
    df["image_dependent"] = [k in dependent for k in zip(df["model"], df["profile"], df["qid"])]
    return df


def load_diag(results: Path, run_id: str) -> dict[str, dict[str, np.ndarray]]:
    """{qid: {key: array}} from a run's diagnostic shards."""
    out: dict[str, dict[str, np.ndarray]] = {}
    seen = set()
    for part in sorted((Path(results) / "diag" / run_id).glob("part-*.npz")):
        z = np.load(part)
        qids = [str(q) for q in z["qid"]]
        for qid in qids:
            if qid in seen:
                raise ValueError(f"Duplicate qid {qid!r} in diagnostic shards for {run_id}")
            seen.add(qid)
        for key in z.files:
            if key == "qid" or key.endswith("__offsets"):
                continue
            offsets = z[f"{key}__offsets"]
            for i, qid in enumerate(qids):
                out.setdefault(qid, {})[key] = z[key][offsets[i]:offsets[i + 1]].astype(np.float32)
    return out


def load_gates(results: Path, model: str, dataset: str) -> MaskStore | None:
    path = Path(results) / "gates" / f"{dataset}__{model}.npz"
    return MaskStore.load(path) if path.exists() else None


# ------------------------------------------------------------------------------------------- accuracy

def collapse_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Average correctness over Random seeds, one row per (model, profile, dataset, method, budget, question)."""
    keys = ["model", "profile", "dataset", "method", "budget_pct", "qid"]
    first = ["image_id", "modality", "modality_group", "lang", "answer_type", "bg_frac", "image_dependent"]
    agg = {"correct": "mean", **{c: "first" for c in first if c in df.columns}}
    return df.groupby(keys, as_index=False).agg(agg)


def accuracy_table(df: pd.DataFrame, by: tuple[str, ...] = ("model", "dataset", "method", "budget_pct"),
                   image_dependent_only: bool = False) -> pd.DataFrame:
    """Accuracy (%) with 95% image-clustered bootstrap CIs; Random seeds averaged per question."""
    d = collapse_seeds(df)
    if image_dependent_only:
        d = d[d["image_dependent"]]
    rows = []
    for key, g in d.groupby(list(by)):
        mean, low, high = cluster_bootstrap_ci(g["correct"].to_numpy(float), g["image_id"].to_numpy())
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), "n": len(g), "images": g["image_id"].nunique(),
                     "acc": 100 * mean, "ci_low": 100 * low, "ci_high": 100 * high})
    return pd.DataFrame(rows)


def paired_delta(df: pd.DataFrame, model: str, dataset: str, method_a: str, method_b: str, budget_a: int = 10,
                 budget_b: int | None = None, modality_group_: str | None = None) -> dict:
    """Accuracy of A minus B (points) on the same questions, image-clustered CI and exact McNemar p."""
    d = collapse_seeds(df)
    d = d[(d["model"] == model) & (d["dataset"] == dataset)]
    a = d[(d["method"] == method_a) & (d["budget_pct"] == budget_a)].set_index("qid")
    b = d[(d["method"] == method_b) & (d["budget_pct"] == (budget_a if budget_b is None else budget_b))].set_index("qid")
    common = a.index.intersection(b.index)
    if modality_group_ is not None:
        common = common[a.loc[common, "modality_group"] == modality_group_]
    row = {"model": model, "dataset": dataset, "modality_group": modality_group_ or "all",
           "comparison": f"{method_a} - {method_b}", "budget_pct": budget_a, "n": len(common)}
    if len(common) == 0:
        return {**row, "delta": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_mcnemar": np.nan}
    delta, low, high = paired_delta_ci(a.loc[common, "correct"].to_numpy(float), b.loc[common, "correct"].to_numpy(float),
                                       a.loc[common, "image_id"].to_numpy())
    p = mcnemar_exact(a.loc[common, "correct"].to_numpy() >= 0.5, b.loc[common, "correct"].to_numpy() >= 0.5)
    return {**row, "delta": 100 * delta, "ci_low": 100 * low, "ci_high": 100 * high, "p_mcnemar": p}


# ----------------------------------------------------------------------------------------- mechanism

def _background(gates: MaskStore, image_id: str) -> np.ndarray | None:
    if gates is None or image_id not in gates:
        return None
    return gates.get(image_id)[1].reshape(-1) < TAU


def retention_table(df: pd.DataFrame, results: Path, model: str, method: str, budget: int = 10) -> pd.DataFrame:
    """Background retention ratio per dataset x modality group (Random ≈ 1; > 1 = prefers background)."""
    d = df[(df["model"] == model) & (df["method"] == method) & (df["budget_pct"] == budget) & df["keep_idx"].notna()]
    rows = []
    for (dataset, group), g in d.groupby(["dataset", "modality_group"]):
        gates = load_gates(results, model, dataset)
        values, images = [], []
        for keep, image_id in zip(g["keep_idx"], g["image_id"]):
            bg = _background(gates, image_id)
            eligible = bg is not None and 0.05 <= float(bg.mean()) <= 0.95
            ratio = background_retention_ratio(np.asarray(keep), bg) if eligible else np.nan
            if np.isfinite(ratio):
                values.append(ratio)
                images.append(image_id)
        row = {"model": model, "method": method, "dataset": dataset, "modality_group": group,
               "questions": len(g), "coverage": len(values) / max(len(g), 1)}
        if values:
            mean, low, high = cluster_bootstrap_ci(np.array(values), np.array(images))
            row.update(brr=mean, ci_low=low, ci_high=high)
        else:
            row.update(brr=np.nan, ci_low=np.nan, ci_high=np.nan)
        rows.append(row)
    return pd.DataFrame(rows, columns=["model", "method", "dataset", "modality_group", "questions", "coverage", "brr",
                                       "ci_low", "ci_high"])


def keep_maps(df: pd.DataFrame, model: str, method: str, budget: int = 10, res: int = 16) -> dict[str, np.ndarray]:
    """Average keep-probability maps per modality group, on a res x res normalized grid."""
    d = df[(df["model"] == model) & (df["method"] == method) & (df["budget_pct"] == budget) & df["keep_idx"].notna()]
    return {group: keep_map([np.asarray(k) for k in g["keep_idx"]], [tuple(x) for x in g["grid_hw"]], res=res)
            for group, g in d.groupby("modality_group")}


def query_dependence(df: pd.DataFrame, model: str, method: str, budget: int = 10) -> pd.DataFrame:
    """Mean Jaccard of keep-sets between different questions on the same image (1 = ignores the question)."""
    d = df[(df["model"] == model) & (df["method"] == method) & (df["budget_pct"] == budget) & df["keep_idx"].notna()]
    rows = []
    for dataset, g in d.groupby("dataset"):
        per_image, ids = [], []
        for image_id, h in g.groupby("image_id"):
            sets = [np.asarray(k) for k in h["keep_idx"]]
            if len(sets) >= 2:
                per_image.append(np.mean([jaccard(a, b) for a, b in combinations(sets, 2)]))
                ids.append(image_id)
        row = {"model": model, "method": method, "dataset": dataset, "images": len(per_image),
               "jaccard": np.nan, "ci_low": np.nan, "ci_high": np.nan}
        if per_image:
            mean, low, high = cluster_bootstrap_ci(np.array(per_image), np.array(ids))
            row.update(jaccard=mean, ci_low=low, ci_high=high)
        rows.append(row)
    return pd.DataFrame(rows, columns=["model", "method", "dataset", "images", "jaccard", "ci_low", "ci_high"])


def image_dependence(df: pd.DataFrame, model: str, method: str, budget: int = 10, pairs: int = 5000,
                     seed: int = 0) -> pd.DataFrame:
    """Does the selector look at the image? One keep-set per image (its first question): the number of distinct
    keep-sets, the largest share of images sharing one keep-set, and the mean Jaccard between keep-sets of random
    pairs of different images (Random is the reference; ~1 = the same tokens for every image)."""
    d = df[(df["model"] == model) & (df["method"] == method) & (df["budget_pct"] == budget) & df["keep_idx"].notna()]
    rows = []
    for dataset, g in d.groupby("dataset"):
        sets = [tuple(np.asarray(k).tolist()) for k in g.sort_values("qid").groupby("image_id")["keep_idx"].first()]
        counts = pd.Series(sets).value_counts()
        rng = np.random.default_rng(seed)
        i, j = rng.integers(0, len(sets), pairs), rng.integers(0, len(sets), pairs)
        jac = [jaccard(np.asarray(sets[a]), np.asarray(sets[b])) for a, b in zip(i, j) if a != b]
        rows.append({"model": model, "method": method, "dataset": dataset, "images": len(sets),
                     "distinct_keepsets": int(counts.size), "top_share": float(counts.iloc[0] / len(sets)),
                     "cross_image_jaccard": float(np.mean(jac)) if jac else np.nan})
    return pd.DataFrame(rows, columns=["model", "method", "dataset", "images", "distinct_keepsets", "top_share",
                                       "cross_image_jaccard"])


# target dataset -> anchor dataset whose `full` run supplies the map (never the target itself: no leakage)
STATIC_SOURCES = {"rin_anat_mc": "slake_en", "rin_abn_closed": "slake_en", "slake_en": "rin_anat_mc",
                  "slake_zh": "rin_anat_mc"}


def build_static_maps(results: Path, model: str, profile: str = "t4x2") -> list[Path]:
    """Position-only rankings for the `static_map` method (plan.md T4): the mean rank map of layer-1 attention
    (`scores_l1`) over the other anchor dataset's `full` run, saved as static_maps/<model>__for_<dataset>.npy."""
    results, written = Path(results), []
    for target, source in STATIC_SOURCES.items():
        run_id = f"{model}__{source}__full__r100__s0__{profile}"
        runs_file = results / "runs" / f"{run_id}.jsonl"
        diag = load_diag(results, run_id)
        if not diag or not runs_file.exists():
            continue
        grids = {}
        for line in runs_file.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            grids[record["qid"]] = tuple(record["grid_hw"])
        qids = [q for q in diag if q in grids and "scores_l1" in diag[q]]
        smap = static_map_from_scores([diag[q]["scores_l1"] for q in qids], [grids[q] for q in qids])
        path = results / "static_maps" / f"{model}__for_{target}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, smap.astype(np.float32))
        written.append(path)
    return written


def high_norm_rates(df: pd.DataFrame, results: Path, model: str, method: str = "fastv", budget: int = 10,
                    z_threshold: float = 5.0) -> pd.DataFrame:
    """How often high-norm (artifact) visual tokens are kept, relative to the budget, and where they sit."""
    rows = []
    d = df[(df["model"] == model) & (df["method"] == method) & (df["budget_pct"] == budget) & df["keep_idx"].notna()]
    for dataset, g in d.groupby("dataset"):
        full_runs = df[(df["model"] == model) & (df["dataset"] == dataset) & (df["method"] == "full")]["run_id"].unique()
        norms = load_diag(results, full_runs[0]) if len(full_runs) else {}
        gates = load_gates(results, model, dataset)
        ratios, in_background, with_outliers = [], [], 0
        for qid, keep, image_id, n_vis, k in zip(g["qid"], g["keep_idx"], g["image_id"], g["n_vis"], g["k_keep"]):
            norm = norms.get(qid, {}).get("vit_norm")
            if norm is None or len(norm) != n_vis:
                continue
            med = np.median(norm)
            z = (norm - med) / max(np.median(np.abs(norm - med)) * 1.4826, 1e-8)
            outliers = z > z_threshold
            if not outliers.any():
                continue
            with_outliers += 1
            kept = np.zeros(n_vis, dtype=bool)
            kept[np.asarray(keep)] = True
            ratios.append((kept & outliers).sum() / outliers.sum() / (k / n_vis))
            bg = _background(gates, image_id)
            if bg is not None:
                in_background.append(bg[outliers].mean())
        rows.append({"model": model, "method": method, "dataset": dataset, "questions": len(g),
                     "questions_with_outliers": with_outliers,
                     "selection_rate_vs_budget": float(np.mean(ratios)) if ratios else np.nan,
                     "outliers_in_background": float(np.mean(in_background)) if in_background else np.nan})
    return pd.DataFrame(rows, columns=["model", "method", "dataset", "questions", "questions_with_outliers",
                                       "selection_rate_vs_budget", "outliers_in_background"])


# --------------------------------------------------------------------------------------------- diagnosis
# Added 2026-11-03 (plan.md §A9). Each function answers one reviewer objection from the gap register.

def answer_key(answer_type: str, response: str) -> str:
    """The answer a response commits to: the option letter for multiple choice, else the normalized parsed text."""
    if answer_type == "mc":
        match = re.match(r"\s*\(?([A-Ea-e])(?:[.):\s]|$)", str(response))
        return match.group(1).upper() if match else "?"
    return normalize_answer(parse_response(str(response)))


def retained_gain(df: pd.DataFrame, budget: int = 10) -> pd.DataFrame:
    """How blind is a pruned model? (G14) Per model x dataset x method, seed 0, on questions answered by Full and
    text-only too: accuracy, the share of the image's contribution kept, (acc - text-only) / (Full - text-only), and how
    often the pruned answer equals the text-only answer and the Full answer to the same question."""
    d = df[(df["seed"] == 0) & df["budget_pct"].isin([budget, 100])].copy()
    d["pred"] = [answer_key(t, r) for t, r in zip(d["answer_type"], d["response"].fillna(""))]
    rows = []
    for (model, dataset), g in d.groupby(["model", "dataset"]):
        full, text = (g[g["method"] == m].set_index("qid") for m in ("full", "text_only"))
        if full.empty or text.empty:
            continue
        for method, r in g[~g["method"].isin(["full", "text_only"])].groupby("method"):
            r = r.set_index("qid")
            common = r.index.intersection(full.index).intersection(text.index)
            acc, acc_full, acc_text = (100 * x.loc[common, "correct"].mean() for x in (r, full, text))
            rows.append({"model": model, "dataset": dataset, "method": method, "n": len(common), "acc": acc,
                         "full": acc_full, "text_only": acc_text,
                         "retained_gain": (acc - acc_text) / (acc_full - acc_text) if acc_full > acc_text else np.nan,
                         "agree_text_only": 100 * (r.loc[common, "pred"] == text.loc[common, "pred"]).mean(),
                         "agree_full": 100 * (r.loc[common, "pred"] == full.loc[common, "pred"]).mean()})
    return pd.DataFrame(rows)


def _foreground(gates: MaskStore | None, image_id: str, n: int) -> np.ndarray:
    if gates is None or image_id not in gates:
        return np.ones(n, dtype=bool)
    fg = gates.get(image_id)[1].reshape(-1) >= TAU
    return fg if fg.any() else np.ones(n, dtype=bool)


def foreground_coverage(df: pd.DataFrame, results: Path, budget: int = 10) -> pd.DataFrame:
    """Per question of every pruned run: `fg_gap`, the mean distance from a foreground cell to its nearest kept token in
    units of the ideal spacing sqrt(n_foreground / k) (lower = better spread over the body), and `fg_share`, the share of
    kept tokens in the foreground (G11)."""
    d = df[(df["budget_pct"] == budget) & df["keep_idx"].notna()]
    rows = []
    for (model, dataset), g in d.groupby(["model", "dataset"]):
        gates = load_gates(results, model, dataset)
        for r in g.itertuples():
            h, w = r.grid_hw
            n, keep = h * w, np.asarray(r.keep_idx)
            fg = _foreground(gates, r.image_id, n)
            cy, cx = np.divmod(np.flatnonzero(fg), w)
            ky, kx = np.divmod(keep, w)
            gap = np.sqrt((cy[:, None] - ky[None]) ** 2 + (cx[:, None] - kx[None]) ** 2).min(axis=1).mean()
            rows.append({"model": model, "dataset": dataset, "method": r.method, "seed": r.seed, "qid": r.qid,
                         "image_id": r.image_id, "correct": float(r.correct),
                         "fg_gap": gap / np.sqrt(fg.sum() / len(keep)), "fg_share": fg[keep].mean()})
    return pd.DataFrame(rows)


def coverage_mediation(cov: pd.DataFrame, reference: str = "random_l2",
                       methods: tuple[str, ...] = ("fastv", "gate_fastv", "gate_uniform")) -> pd.DataFrame:
    """Does coverage explain a method's penalty? Logistic regressions on seed-0 questions, image-clustered SEs:
    correct ~ method versus correct ~ method + fg_gap + fg_share; `attenuation` = share of the method's log-odds
    coefficient (relative to `reference`) removed by the coverage terms."""
    import statsmodels.formula.api as smf

    rows = []
    for (model, dataset), g in cov[cov["seed"] == 0].groupby(["model", "dataset"]):
        present = [reference] + sorted(set(g["method"]) - {reference})
        g = g.assign(method=pd.Categorical(g["method"], categories=present))
        cluster = {"groups": pd.factorize(g["image_id"])[0]}
        terms = [t for t in ("fg_gap", "fg_share") if g[t].std() > 1e-9]
        raw = smf.logit("correct ~ C(method)", g).fit(disp=0, cov_type="cluster", cov_kwds=cluster)
        ctrl = smf.logit("correct ~ C(method) + " + " + ".join(terms), g).fit(disp=0, cov_type="cluster",
                                                                             cov_kwds=cluster)
        for method in methods:
            name = f"C(method)[T.{method}]"
            if name not in raw.params:
                continue
            significant = raw.pvalues[name] < 0.05  # attenuation of a null effect is noise
            rows.append({"model": model, "dataset": dataset, "method": method, "logodds_raw": raw.params[name],
                         "p_raw": raw.pvalues[name], "logodds_controlled": ctrl.params[name],
                         "p_controlled": ctrl.pvalues[name],
                         "attenuation": 1 - ctrl.params[name] / raw.params[name] if significant else np.nan})
    return pd.DataFrame(rows)


def attention_informativeness(df: pd.DataFrame, results: Path, model: str, dataset: str,
                              method: str = "random_l2", seed: int = 0, profile: str = "t4x2") -> dict:
    """Does layer-1 attention mark useful tokens? (G4) Natural experiment inside a random selector: the mean
    within-image rank of the kept tokens' layer-1 attention (the `full` run's `scores_l1` for the same question) is
    assigned at random. Logistic regression of correctness on its z-score, image-clustered SEs; a negative coefficient
    means high-attention tokens are less useful than random ones."""
    import scipy.stats as sps
    import statsmodels.formula.api as smf

    diag = load_diag(results, f"{model}__{dataset}__full__r100__s0__{profile}")
    gates = load_gates(results, model, dataset)
    runs = df[(df["model"] == model) & (df["dataset"] == dataset) & (df["method"] == method) & (df["seed"] == seed)]
    rows = []
    for r in runs.itertuples():
        if r.qid not in diag or "scores_l1" not in diag[r.qid] or r.keep_idx is None:
            continue
        scores, keep = diag[r.qid]["scores_l1"], np.asarray(r.keep_idx)
        h, w = r.grid_hw
        rows.append({"correct": float(r.correct), "image_id": r.image_id,
                     "attn_rank": (sps.rankdata(scores) / len(scores))[keep].mean(),
                     "top_rows": (keep // w < max(1, round(0.25 * h))).mean(),
                     "fg_share": _foreground(gates, r.image_id, h * w)[keep].mean()})
    t = pd.DataFrame(rows)
    out = {"model": model, "dataset": dataset, "method": method, "n": len(t), "coef_per_sd": np.nan, "p": np.nan}
    if len(t) < 30 or t["correct"].nunique() < 2:
        return out
    t["z"] = (t["attn_rank"] - t["attn_rank"].mean()) / t["attn_rank"].std()
    terms = ["z"] + [c for c in ("top_rows", "fg_share") if t[c].std() > 1e-9]
    fit = smf.logit("correct ~ " + " + ".join(terms), t).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(t["image_id"])[0]})
    quartile = pd.qcut(t["attn_rank"], 4, labels=False, duplicates="drop")
    out.update(coef_per_sd=fit.params["z"], p=fit.pvalues["z"],
               acc_by_attention_quartile=(100 * t.groupby(quartile)["correct"].mean()).round(1).tolist())
    return out


def score_selection(df: pd.DataFrame, results: Path, model: str, dataset: str, budget: float = 0.10,
                    keys: tuple[str, ...] = ("scores_l1", "scores_ropefree_l1", "scores_l7", "scores_l15"),
                    profile: str = "t4x2", pairs: int = 3000, seed: int = 0) -> pd.DataFrame:
    """What would top-k by each logged score of the `full` run keep? Mean row (0 = top), background retention ratio,
    overlap with layer-1 FastV, and cross-image Jaccard. Previews E6 (RoPE-free, deeper layers) without a GPU (G8)."""
    diag = load_diag(results, f"{model}__{dataset}__full__r100__s0__{profile}")
    gates = load_gates(results, model, dataset)
    full = df[(df["model"] == model) & (df["dataset"] == dataset) & (df["method"] == "full")].set_index("qid")
    picks: dict[str, list] = {k: [] for k in keys}
    for qid, logged in diag.items():
        if qid not in full.index or "scores_l1" not in logged:
            continue
        h, w = full.loc[qid, "grid_hw"]
        n = h * w
        k = max(1, min(n, int(round(budget * n))))
        background = ~_foreground(gates, full.loc[qid, "image_id"], n)
        layer1 = np.argsort(-logged["scores_l1"], kind="stable")[:k]
        for key in keys:
            if key in logged:
                top = np.sort(np.argsort(-logged[key], kind="stable")[:k])
                picks[key].append((full.loc[qid, "image_id"], top, (top // w).mean() / max(h - 1, 1),
                                   background_retention_ratio(top, background) if background.any() else np.nan,
                                   jaccard(top, layer1)))
    rows, rng = [], np.random.default_rng(seed)
    for key, items in picks.items():
        if not items:
            continue
        first = list({image_id: top for image_id, top, *_ in items}.values())
        i, j = rng.integers(0, len(first), pairs), rng.integers(0, len(first), pairs)
        cross = [jaccard(first[a], first[b]) for a, b in zip(i, j) if a != b]
        rows.append({"model": model, "dataset": dataset, "score": key, "questions": len(items),
                     "mean_row": np.mean([x[2] for x in items]), "brr": np.nanmean([x[3] for x in items]),
                     "jaccard_with_layer1": np.mean([x[4] for x in items]),
                     "cross_image_jaccard": np.mean(cross) if cross else np.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------- evidence grounding (§A10 D3)
# Does a keep-set contain the structure the question is about? SLAKE ships a label mask per image (mask.png, values
# in mask.txt); a question's evidence is the union of the structures its text or answer names.

PAIRED_STRUCTURES = {"lung": "Lung", "kidney": "Kidney", "eye": "Eye", "parotid": "Parotid",
                     "temporal lobe": "Temporal Lobe", "femoral head": "Femoral Head", "mandible": "Mandible",
                     "humerus head": "Humerus Head", "ear": "Ear"}
SINGLE_STRUCTURES = {r"liver": ["Liver"], r"heart": ["Heart"], r"spleen": ["Spleen"], r"stomach": ["Stomach"],
                     r"colon": ["Colon"], r"bladder": ["Bladder"], r"rectum": ["Rectum"], r"esophagus": ["Esophagus"],
                     r"trachea": ["Trachea"], r"duodenum": ["Duodenum"], r"small (?:bowel|intestine)": ["Small Bowel"],
                     r"spinal cord": ["Spinal Cord"], r"brain ?stem": ["Brain Stem"], r"larynx": ["Larynx"],
                     r"tooth|teeth": ["Tooth"],
                     r"tumou?rs?|edema": ["Brain Edema", "Brain Non-enhancing Tumor", "Brain Enhancing Tumor"],
                     r"cancers?|nodules?|mass": ["Lung Cancer", "Liver Cancer", "Kidney Cancer"]}


def read_mask_labels(path: Path) -> dict[str, int]:
    """SLAKE mask.txt ("value:Name" per line) -> {Name: value}."""
    labels = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if ":" in line:
            value, name = line.split(":", 1)
            labels[name.strip()] = int(value)
    return labels


def slake_targets(text: str) -> set[str]:
    """SLAKE mask labels named in a question/answer text; "left lung" names one side, "lungs" both."""
    text, found = text.lower(), set()
    for base, label in PAIRED_STRUCTURES.items():
        for side in ("left", "right"):
            if re.search(rf"\b{side} {base}s?\b", text):
                found.add(f"{side.title()} {label}")
        if re.search(rf"(?<!left )(?<!right )\b{base}s?\b", text):
            found.update({f"Left {label}", f"Right {label}"})
    for pattern, names in SINGLE_STRUCTURES.items():
        if re.search(rf"\b(?:{pattern})\b", text):
            found.update(names)
    return found


def _variant_mask(mask: Image.Image, dataset: str) -> Image.Image:
    """The label mask transformed like a variant dataset's images (data.build_variant)."""
    size = re.search(r"_s(\d+)$", dataset)
    if size:
        return mask.resize((int(size.group(1)),) * 2, Image.NEAREST)
    return mask.transpose(Image.FLIP_TOP_BOTTOM) if dataset.endswith("_vflip") else mask


def evidence_fractions(df: pd.DataFrame, slake_images: Path, labels: dict[str, int]) -> dict[tuple, dict]:
    """{(model, dataset, qid): {"frac": per-token share of the named structures [N], "area": share of the image}} for
    SLAKE questions (any variant) whose text or answer names a structure present in the image's mask."""
    out: dict[tuple, dict] = {}
    masks: dict[str, Image.Image] = {}
    d = df[df["dataset"].str.startswith("slake_en") & df["grid_hw"].map(lambda g: isinstance(g, list) and g[0] > 0)]
    for r in d.drop_duplicates(["model", "dataset", "qid"]).itertuples():
        folder = str(r.image_id).split("/")[-1]
        path = Path(slake_images) / folder / "mask.png"
        if not path.exists():
            continue
        if folder not in masks:
            masks[folder] = Image.open(path).convert("L")
        base = np.asarray(masks[folder])
        named = {n for n in slake_targets(f"{r.question} {r.answer}") if n in labels and (base == labels[n]).any()}
        if not named:
            continue
        values = [labels[n] for n in named]
        view = label_view(_variant_mask(masks[folder], r.dataset), r.model)
        cell = MODEL_VIEWS[r.model]["cell"]
        frac = token_foreground_fraction(np.isin(view, values), cell)
        if tuple(frac.shape) != tuple(r.grid_hw):
            raise RuntimeError(f"{r.qid}: evidence grid {frac.shape} != model grid {tuple(r.grid_hw)}")
        out[(r.model, r.dataset, r.qid)] = {"frac": frac.reshape(-1), "area": float(np.isin(base, values).mean()),
                                            "structures": "+".join(sorted(named))}
    return out


def evidence_retention(df: pd.DataFrame, evidence: dict[tuple, dict], budget: int = 10) -> pd.DataFrame:
    """Per pruned question: `recall` = share of evidence tokens (cells touching the structure) kept, `err` = recall /
    (k / N) (1 = chance, the evidence retention ratio), `hit` = at least one evidence token kept."""
    d = df[(df["budget_pct"] == budget) & df["keep_idx"].notna()]
    rows = []
    for r in d.itertuples():
        e = evidence.get((r.model, r.dataset, r.qid))
        if e is None:
            continue
        ev, keep = e["frac"] > 0, np.asarray(r.keep_idx)
        if not ev.any():
            continue
        recall = ev[keep].sum() / ev.sum()
        rows.append({"model": r.model, "dataset": r.dataset, "method": r.method, "seed": r.seed, "qid": r.qid,
                     "image_id": r.image_id, "structures": e["structures"], "area": e["area"], "n_ev": int(ev.sum()),
                     "n": len(ev), "k": len(keep), "recall": recall, "err": recall / (len(keep) / len(ev)),
                     "hit": float(ev[keep].any()), "correct": float(r.correct)})
    return pd.DataFrame(rows)


def evidence_report(ev: pd.DataFrame, reference: str = "random_l2") -> dict[str, pd.DataFrame]:
    """Evidence retention by method (image-clustered CIs), by structure-area tertile, and the paired accuracy loss of
    each method against `reference` per tertile."""
    if ev.empty:
        return {"summary": ev, "tertiles": ev, "paired_by_tertile": ev}
    first = ev.drop_duplicates(["model", "dataset", "qid"])[["model", "dataset", "qid", "area"]].copy()
    first["tertile"] = first.groupby(["model", "dataset"])["area"].transform(  # equal-count tertiles by area rank
        lambda a: (3 * (a.rank(method="first") - 1) // len(a)).astype(int)).map({0: "small", 1: "medium", 2: "large"})
    ev = ev.merge(first[["model", "dataset", "qid", "tertile"]], on=["model", "dataset", "qid"])
    summary = []
    for (model, dataset, method), g in ev.groupby(["model", "dataset", "method"]):
        mean, low, high = cluster_bootstrap_ci(g["err"].to_numpy(float), g["image_id"].to_numpy())
        summary.append({"model": model, "dataset": dataset, "method": method, "questions": g["qid"].nunique(),
                        "images": g["image_id"].nunique(), "err": mean, "err_low": low, "err_high": high,
                        "hit": g["hit"].mean(), "acc": 100 * g["correct"].mean()})
    tertiles = (ev.groupby(["model", "dataset", "tertile", "method"], observed=True)
                .agg(questions=("qid", "nunique"), median_area=("area", "median"), err=("err", "mean"),
                     hit=("hit", "mean"), acc=("correct", "mean")).reset_index())
    tertiles["acc"] *= 100
    paired = []
    seed0 = ev[ev["seed"] == 0]
    for (model, dataset, t), g in seed0.groupby(["model", "dataset", "tertile"], observed=True):
        wide = g.pivot_table(index=["qid", "image_id"], columns="method", values="correct").reset_index()
        if reference not in wide:
            continue
        for method in [m for m in wide.columns if m not in ("qid", "image_id", reference)]:
            both = wide.dropna(subset=[method, reference])
            if both.empty:
                continue
            delta, low, high = paired_delta_ci(both[method].to_numpy(float), both[reference].to_numpy(float),
                                               both["image_id"].to_numpy())
            paired.append({"model": model, "dataset": dataset, "tertile": t, "comparison": f"{method} - {reference}",
                           "n": len(both), "delta": 100 * delta, "ci_low": 100 * low, "ci_high": 100 * high})
    return {"summary": pd.DataFrame(summary), "tertiles": tertiles, "paired_by_tertile": pd.DataFrame(paired)}


# ------------------------------------------------------------------------------------ layer profile (§A10 D1)

def rank_variance_shares(scores: dict[str, np.ndarray], meta: pd.DataFrame, bins: int = 8,
                         position: str = "normalized", min_images: int = 5) -> dict:
    """Split the variance of within-question score ranks (u = rank / N) into grid position, the image beyond
    position, and the question beyond the image. Exactly additive. `meta`: indexed by qid with `image_id` and
    `grid_hw`. With one question per image the "image" share includes the question.

    position="normalized": bins x bins cells of normalized coordinates, pooled over grid sizes (a lower bound when a
    prior is tied to absolute token indices and the dataset mixes image sizes). position="grid": the exact token index
    within each grid shape, using only grid shapes with at least `min_images` images.
    """
    import scipy.stats as sps

    by_image: dict[str, list[np.ndarray]] = {}
    grid: dict[str, tuple[int, int]] = {}
    for qid, s in scores.items():
        if qid not in meta.index:
            continue
        image = meta.at[qid, "image_id"]
        by_image.setdefault(image, []).append(sps.rankdata(s) / len(s))
        grid[image] = tuple(meta.at[qid, "grid_hw"])
    if position == "grid":
        shapes = pd.Series(grid).value_counts()
        keep = set(shapes[shapes >= min_images].index)
        by_image = {i: u for i, u in by_image.items() if grid[i] in keep}
        cells = {shape: k for k, shape in enumerate(sorted(keep))}
        offsets = np.cumsum([0] + [h * w for h, w in sorted(keep)])
    n_pos = bins * bins if position == "normalized" else (int(offsets[-1]) if by_image else 0)
    pos_sum, pos_cnt, cached = np.zeros(n_pos), np.zeros(n_pos), []
    for image, us in by_image.items():
        h, w = grid[image]
        u = np.stack(us)
        if position == "normalized":
            r, c = np.divmod(np.arange(h * w), w)
            b = np.minimum(r * bins // h, bins - 1) * bins + np.minimum(c * bins // w, bins - 1)
        else:
            b = offsets[cells[(h, w)]] + np.arange(h * w)
        pos_sum += np.bincount(b, weights=u.sum(0), minlength=n_pos)
        pos_cnt += np.bincount(b, minlength=n_pos) * u.shape[0]
        cached.append((u, b))
    if not cached:
        return {"questions": 0}
    pos = pos_sum / np.maximum(pos_cnt, 1)
    grand = pos_sum.sum() / pos_cnt.sum()
    ss_pos = ss_img = ss_q = 0.0
    for u, b in cached:
        img = u.mean(0)
        ss_pos += u.shape[0] * float(np.sum((pos[b] - grand) ** 2))
        ss_img += u.shape[0] * float(np.sum((img - pos[b]) ** 2))
        ss_q += float(np.sum((u - img) ** 2))
    total = ss_pos + ss_img + ss_q
    return {"questions": sum(len(u) for u, _ in cached), "q_per_image": np.mean([len(u) for u, _ in cached]),
            "position": ss_pos / total, "image_beyond_position": ss_img / total, "question_beyond_image": ss_q / total}


def layer_profile(df: pd.DataFrame, results: Path, model: str, dataset: str, budget: float = 0.10,
                  evidence: dict[tuple, dict] | None = None, profile: str = "t4x2", sink_tau: float = 20.0) -> pd.DataFrame:
    """Per readout ("last" prompt token = FastV; "text" = every token after the image) and decoder layer, from a
    `full_profile` run: the position / image / question variance split of the token ranking, and what top-k by that
    layer would keep — background retention ratio, evidence retention ratio (SLAKE), share of visual-sink tokens
    (Kang et al.) — plus the attention mass on visual tokens and Spearman(score, sink score)."""
    import scipy.stats as sps

    diag = load_diag(results, f"{model}__{dataset}__full_profile__r100__s0__{profile}")
    meta = df[(df["model"] == model) & (df["dataset"] == dataset) & (df["method"] == "full_profile")] \
        .drop_duplicates("qid").set_index("qid")
    gates = load_gates(results, model, dataset)
    layers = sorted({int(k.rsplit("_l", 1)[1]) for x in diag.values() for k in x if k.startswith("lp_last_l")})
    rows = []
    for readout in ("last", "text"):
        for layer in layers:
            key, sink_key = f"lp_{readout}_l{layer}", f"sink_l{layer}"
            scores = {q: x[key] for q, x in diag.items() if key in x and q in meta.index}
            if not scores:
                continue
            stats: dict[str, list] = {k: [] for k in ("brr", "err", "hit", "mass", "sink_top", "sink_all", "rho")}
            for qid, s in scores.items():
                n = len(s)
                k = max(1, min(n, int(round(budget * n))))
                top = np.argsort(-s, kind="stable")[:k]
                stats["mass"].append(float(np.exp(s.astype(np.float64)).sum()))
                background = ~_foreground(gates, meta.at[qid, "image_id"], n)
                if background.any() and not background.all():
                    stats["brr"].append(background_retention_ratio(np.sort(top), background))
                e = (evidence or {}).get((model, dataset, qid))
                if e is not None and (e["frac"] > 0).any():
                    ev = e["frac"] > 0
                    stats["err"].append(ev[top].sum() / ev.sum() / (k / n))
                    stats["hit"].append(float(ev[top].any()))
                sink = diag[qid].get(sink_key)
                if sink is not None:
                    stats["sink_top"].append(float(np.mean(sink[top] > sink_tau)))
                    stats["sink_all"].append(float(np.mean(sink > sink_tau)))
                    if np.std(sink) > 0:
                        stats["rho"].append(sps.spearmanr(s, sink)[0])
            mean = {k: (float(np.nanmean(v)) if v else np.nan) for k, v in stats.items()}
            rows.append({"model": model, "dataset": dataset, "readout": readout, "layer": layer,
                         **rank_variance_shares(scores, meta), "brr_top": mean["brr"], "err_top": mean["err"],
                         "hit_top": mean["hit"], "visual_mass": mean["mass"], "sink_rate_top": mean["sink_top"],
                         "sink_rate_all": mean["sink_all"], "spearman_score_sink": mean["rho"]})
    return pd.DataFrame(rows)


def plot_layer_profile(table: pd.DataFrame, path: Path) -> None:
    """One row of panels per (model, dataset): variance shares, evidence/background retention, sink share by layer."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = list(table.groupby(["model", "dataset"]))
    fig, axes = plt.subplots(len(groups), 3, figsize=(12, 3 * len(groups)), squeeze=False)
    for i, ((model, dataset), g) in enumerate(groups):
        for readout, style in (("last", "-"), ("text", "--")):
            x = g[g["readout"] == readout]
            axes[i][0].plot(x["layer"], x["question_beyond_image"], style, color="C0", label=f"question ({readout})")
            axes[i][0].plot(x["layer"], x["position"], style, color="C1", label=f"position ({readout})")
            axes[i][1].plot(x["layer"], x["err_top"], style, color="C2", label=f"evidence ({readout})")
            axes[i][1].plot(x["layer"], x["brr_top"], style, color="C3", label=f"background ({readout})")
            axes[i][2].plot(x["layer"], x["sink_rate_top"], style, color="C4", label=f"sinks in top 10% ({readout})")
        axes[i][2].plot(g[g["readout"] == "last"]["layer"], g[g["readout"] == "last"]["sink_rate_all"], ":", color="k",
                        label="sinks, all tokens")
        axes[i][1].axhline(1.0, color="grey", lw=0.5)
        axes[i][0].set_ylabel(f"{model}\n{dataset}")
        for ax, title in zip(axes[i], ("variance share of the ranking", "retention ratio of top 10% (1 = chance)",
                                       "visual-sink share")):
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("layer")
            ax.legend(fontsize=6)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def diagnosis_report(results: Path, models: tuple[str, ...] = ("lingshu7b", "qwen3vl8b"),
                     datasets: tuple[str, ...] = ("rin_anat_mc", "slake_en"), budget: int = 10,
                     open_scoring: str = "rule") -> dict:
    """All §A9 diagnostics for the anchor datasets; writes results/tables/diagnosis_*.csv."""
    results = Path(results)
    df = load_runs(results, open_scoring=open_scoring)
    df = df[df["dataset"].isin(datasets) & df["model"].isin(models)]
    cov = foreground_coverage(df, results, budget)
    report = {
        "retained_gain": retained_gain(df, budget),
        "question_dependence": pd.concat([query_dependence(df, m, method, budget) for m in models
                                          for method in ("fastv", "gate_fastv", "random_l2")], ignore_index=True),
        "image_dependence": pd.concat([image_dependence(df, m, method, budget) for m in models
                                       for method in ("fastv", "gate_fastv", "uniform_l2", "random_l2")],
                                      ignore_index=True),
        "coverage": cov.groupby(["model", "dataset", "method"])[["correct", "fg_gap", "fg_share"]].mean().reset_index(),
        "coverage_mediation": coverage_mediation(cov),
        "attention_informativeness": pd.DataFrame([attention_informativeness(df, results, m, d, method)
                                                   for m in models for d in datasets
                                                   for method in ("random_l2", "gate_random")]),
        "score_selection": pd.concat([score_selection(df, results, m, d, budget / 100) for m in models
                                      for d in datasets], ignore_index=True),
    }
    tables = results / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    for name, table in report.items():
        table.to_csv(tables / f"diagnosis_{name}.csv", index=False)
    return report


# --------------------------------------------------------------------------------------------- Stage 4

STAGE4_PAIRS = [("fastv", "random_l2"), ("fastv_prellm", "random_l0"), ("gate_fastv", "fastv")]


def stage4_report(results: Path, model: str = "lingshu7b", profile: str = "t4x2", budget: int = 10,
                  write: bool = True) -> dict:
    """Everything Stage 4 needs: accuracy, paired deltas (K1), background retention (K2), keep-maps, query
    dependence, high-norm rates, and the two decisions. Writes results/tables/stage4_*.csv and the keep-map figure."""
    results = Path(results)
    df = load_runs(results, profile)
    if df.empty:
        raise FileNotFoundError(f"no runs for profile {profile} under {results / 'runs'}")
    df = df[df["model"] == model]
    report: dict = {
        "accuracy": accuracy_table(df, by=("dataset", "method", "budget_pct")),
        "accuracy_by_modality": accuracy_table(df, by=("dataset", "modality_group", "method", "budget_pct")),
        "accuracy_image_dependent": accuracy_table(df, by=("dataset", "method", "budget_pct"), image_dependent_only=True),
    }
    pairs = []
    for dataset in sorted(df["dataset"].unique()):
        groups = [None] + sorted(df[df["dataset"] == dataset]["modality_group"].unique())
        for a, b in STAGE4_PAIRS:
            for group in groups:
                pairs.append(paired_delta(df, model, dataset, a, b, budget_a=budget, modality_group_=group))
    report["paired"] = pd.DataFrame(pairs).dropna(subset=["delta"])
    report["retention"] = pd.concat([retention_table(df, results, model, m, budget)
                                     for m in ("fastv", "random_l2", "gate_fastv")], ignore_index=True)
    report["query_dependence"] = pd.concat([query_dependence(df, model, m, budget) for m in ("fastv", "random_l2")],
                                           ignore_index=True)
    report["image_dependence"] = pd.concat([image_dependence(df, model, m, budget)
                                            for m in ("fastv", "random_l2", "gate_fastv")], ignore_index=True)
    report["high_norm"] = high_norm_rates(df, results, model, budget=budget)
    report["keep_maps"] = {m: keep_maps(df, model, m, budget) for m in ("fastv", "random_l2", "gate_fastv")}
    report["decisions"] = stage4_decisions(report)
    if write:
        _write_stage4(report, results)
    return report


def stage4_decisions(report: dict) -> dict:
    """K1 and K2 exactly as plan.md §A6 defines them."""
    paired, retention = report["paired"], report["retention"]

    def pick(dataset, comparison):
        rows = paired[(paired["dataset"] == dataset) & (paired["comparison"] == comparison) &
                      (paired["modality_group"] == "all")] if not paired.empty else paired
        return rows.iloc[0].to_dict() if len(rows) else None

    std, pre = pick("rin_anat_mc", "fastv - random_l2"), pick("rin_anat_mc", "fastv_prellm - random_l0")
    if std is None:
        k1 = "undecided: no fastv/random_l2 pair on rin_anat_mc yet"
    elif std["ci_high"] < 0:
        k1 = "GO: standard FastV is significantly below Random@L2 on rin_anat_mc"
    elif pre is not None and pre["ci_high"] < 0:
        k1 = "REFRAME: only pre-LLM FastV is below Random@L0 — discuss with the mentor"
    else:
        k1 = "STOP (after the 5% check): neither FastV variant is below random on rin_anat_mc"

    ctmri = retention[(retention["method"] == "fastv") & (retention["modality_group"] == "CT/MRI")] \
        if not retention.empty else retention
    if ctmri.empty or ctmri["brr"].isna().all():
        k2 = "undecided: no FastV keep-sets with gates on CT/MRI yet"
    elif (ctmri["ci_low"] > 1).any():
        k2 = "H1 ALIVE: FastV keeps more background than its share on CT/MRI (BRR CI above 1)"
    else:
        k2 = "H1 OUT: FastV's BRR CI on CT/MRI includes 1 — lead with position (H2)"
    return {"K1": k1, "K2": k2, "K1_numbers": std, "K1_prellm_numbers": pre}


def _write_stage4(report: dict, results: Path) -> None:
    tables, figures = results / "tables", results / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    for name in ("accuracy", "accuracy_by_modality", "accuracy_image_dependent", "paired", "retention",
                 "query_dependence", "image_dependence", "high_norm"):
        report[name].to_csv(tables / f"stage4_{name}.csv", index=False)
    (tables / "stage4_decisions.json").write_text(json.dumps(report["decisions"], indent=2, default=str),
                                                  encoding="utf-8")
    maps = report["keep_maps"]
    groups = sorted({g for m in maps.values() for g in m})
    if not groups:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(maps), len(groups), figsize=(2.6 * len(groups), 2.6 * len(maps)), squeeze=False)
    for i, (method, by_group) in enumerate(maps.items()):
        for j, group in enumerate(groups):
            ax = axes[i][j]
            if group in by_group:
                ax.imshow(by_group[group], vmin=0, vmax=max(0.3, float(by_group[group].max())), cmap="magma")
                ax.set_title(f"{method} · {group}\nBR conc. {quadrant_concentration(by_group[group]):.2f}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(figures / "stage4_keepmaps.pdf")
    plt.close(fig)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Laptop-side analysis commands.")
    p.add_argument("command", choices=["static-maps", "diagnosis", "evidence", "profile"])
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--models", default="lingshu7b,qwen3vl8b")
    p.add_argument("--profile", default="t4x2")
    p.add_argument("--open-scoring", choices=["rule", "judge"], default="rule")
    p.add_argument("--slake-images", type=Path, help="folder with <xmlabN>/mask.png (the bundle's images/slake)")
    p.add_argument("--labels", type=Path, help="SLAKE mask.txt (the bundle's slake_mask_labels.txt)")
    p.add_argument("--datasets", default="slake_en,rin_anat_mc,natural", help="profile: datasets with full_profile runs")
    args = p.parse_args()
    tables = args.results / "tables"
    if args.command == "diagnosis":
        for name, table in diagnosis_report(args.results, tuple(args.models.split(",")),
                                            open_scoring=args.open_scoring).items():
            print(f"\n## {name}\n{table.round(3).to_string(index=False)}")
        return
    if args.command in ("evidence", "profile"):
        df = load_runs(args.results, args.profile, open_scoring=args.open_scoring)
        df = df[df["model"].isin(args.models.split(","))]
        evidence = evidence_fractions(df, args.slake_images, read_mask_labels(args.labels)) \
            if args.slake_images and args.labels else {}
        tables.mkdir(parents=True, exist_ok=True)
        if args.command == "evidence":
            ev = evidence_retention(df, evidence)
            ev.to_csv(tables / "evidence_rows.csv", index=False)
            for name, table in evidence_report(ev).items():
                table.to_csv(tables / f"evidence_{name}.csv", index=False)
                print(f"\n## evidence {name}\n{table.round(3).to_string(index=False)}")
            return
        present = set(zip(df["model"], df["dataset"], df["method"]))
        parts = [layer_profile(df, args.results, m, d, evidence=evidence, profile=args.profile)
                 for m in args.models.split(",") for d in args.datasets.split(",") if (m, d, "full_profile") in present]
        table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        table.to_csv(tables / "layer_profile.csv", index=False)
        if not table.empty:
            plot_layer_profile(table, args.results / "figures" / "layer_profile.pdf")
        print(table.round(3).to_string(index=False))
        return
    for model in args.models.split(","):
        for path in build_static_maps(args.results, model, args.profile):
            print("wrote", path)


if __name__ == "__main__":
    main()
