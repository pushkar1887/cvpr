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

import numpy as np
import pandas as pd

from .masks import MaskStore
from .scoring import judge_open
from .stats import (background_retention_ratio, cluster_bootstrap_ci, jaccard, keep_map, mcnemar_exact,
                    paired_delta_ci, quadrant_concentration)

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


def load_runs(results: Path, profile: str | None = None) -> pd.DataFrame:
    """One row per (run, question): the run file's fields plus model/dataset/method/budget/seed/profile."""
    rows = []
    for path in sorted((Path(results) / "runs").glob("*.jsonl")):
        info = parse_run_id(path.stem)
        if profile and info["profile"] != profile:
            continue
        with path.open(encoding="utf-8") as f:
            rows.extend({**json.loads(line), **info, "run_id": path.stem} for line in f if line.strip())
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["modality_group"] = df["modality"].map(modality_group)
    return add_image_dependent(rescore_open(df))


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
    for part in sorted((Path(results) / "diag" / run_id).glob("part-*.npz")):
        z = np.load(part)
        qids = [str(q) for q in z["qid"]]
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
            ratio = background_retention_ratio(np.asarray(keep), bg) if bg is not None else np.nan
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
                 "query_dependence", "high_norm"):
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
