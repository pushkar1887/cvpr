# Pruning the Air

**Why attention-guided visual token pruning fails on medical images — diagnosis and a training-free fix.**
CVPR 2027 · paper enrollment **Nov 10 AoE (Nov 11, 17:29 IST)** · paper **Nov 16 AoE** · supplementary **Nov 23 AoE**.

## 0. Framing

- **Title:** *Pruning the Air: Why Attention-Guided Token Pruning Fails on Medical Images*
  (alt.: *Air Is Not Evidence: Diagnosing Visual Token Pruning in Medical MLLMs*). Avoid "Worse Than Random" —
  ProbMed (Yan et al., 2024) and Wang et al. (arXiv 2512.07580) already use it.
- **Thesis:** on medical images, the attention FastV-style methods use to rank visual tokens points at scan background
  (air, empty field of view), not anatomy. At aggressive budgets they pick worse than chance. Removing background
  before scoring should recover most of the loss, at no cost.

## 1. Problem

- Medical MLLMs (Lingshu-7B, HuatuoGPT-Vision-7B, MedGemma) feed hundreds to thousands of visual tokens to the LLM;
  prefill dominates inference cost on clinical and edge hardware.
- The standard remedy is training-free pruning by the LLM's own text-to-image attention: FastV (ECCV 2024),
  SparseVLM (ICML 2025), PDrop (CVPR 2025).
- FastV's assumption "high attention = important token" was validated only on natural images at ~50% retention. On
  medical images it falls below random, and no paper explains why.

## 2. Evidence (published results)

MedSTP (arXiv 2610.22751, Oct 2026), 10% of tokens kept — **FastV / Random** accuracy (%); bold = FastV significantly
below Random (from MedSTP's bootstrap SEs):

| Model | OmniMedVQA | SLAKE | RadImageNet closed | RadImageNet open |
|---|---|---|---|---|
| Lingshu-7B | **32.3 / 38.7** | 55.0 / 56.9 | **37.5 / 55.6** | **28.7 / 43.1** |
| Qwen3-VL-8B | 51.5 / 49.9 | 66.9 / 49.3 | **52.1 / 64.3** | **37.7 / 47.1** |

ViToS (ICML 2026), 80% kept: FastV is the lowest of four training-free methods on Lingshu-7B (60.3 vs 62.9–63.3;
full 63.7) and HuatuoGPT-Vision-7B (55.3 vs 55.5–56.8); on OmniMedVQA FastV alone drops 3.8 points (others ≤ 0.3).

## 3. Prior work and the gap (all on natural images)

- **FEATHER** (ICCV 2025): early pruning keeps the image bottom (mean kept position 80.7% down); localization 5.9 vs
  28.0 for uniform sampling. Cause: 1D-RoPE decay.
- **PoRe** (arXiv 2508.17807): recency bias; padding tokens get inflated attention.
- **When Token Pruning is Worse than Random** (Wang et al., arXiv 2512.07580): deep-layer pruning ≈ random
  (vanishing token information) — does not cover FastV, which prunes at layer 2.
- **Registers** (ICLR 2024; test-time registers, arXiv 2506.08010): high-norm artifacts on flat background absorb
  attention.
- **VTC-Bench** (ACL 2026): downsampling beats many compressors; benchmarks are too easy.
- **Gap:** none covers medical images, below-random on ordinary VQA, M-RoPE models (Qwen2.5-VL/Qwen3-VL, the base of
  Lingshu), or scan background as the trigger. Medical work (MedSTP, ViToS, Photon, MedARC, EasyLens) proposes new
  methods without explaining the failure.

## 4. What the published numbers already tell us

- **D1** FastV is significantly below Random in 5 of 8 MedSTP cells (by 6.4–18.1 points); 2 ties; 1 above.
- **D2** It follows modality: only RadImageNet (pure CT/MRI) is below Random on both models, with the largest gaps.
- **D3** Not only an extreme-budget effect: the FastV-specific loss is visible at 80% kept.
- **D4** Existing explanations don't fit: FEATHER = 1D RoPE + localization; Wang et al. = deep layers. MedSTP reports no
  FastV layer and prunes every method *before* the LLM, so we test both insertion points.
- **D5** The weak point is early single-shot attention: PDrop (attention-based, multi-stage, later layers) holds at 80%.

## 5. Hypotheses (each with a kill condition)

| | Hypothesis | Dead if |
|---|---|---|
| **H1** | Background capture: CT/MRI air acts like padding and layer-2 attention over-selects it; the gap grows with background fraction (CT/MRI > X-ray > fundus > dermoscopy/pathology ≈ 0) | FastV keeps no more background than Random |
| **H2** | Positional bias: M-RoPE favors the bottom-right corner, 1D RoPE the bottom rows — exactly where scans hold air; the gap shrinks as K goes 2 → 8 → 16 | Kept positions are uniform (χ² n.s.) |
| **H3** | Artifact tokens: high-norm tokens on flat background attract the scoring attention | No outliers, or selected at base rate |
| **H4** | Small evidence: the remaining loss grows as the evidence region shrinks (SLAKE organ masks) | Flat across evidence sizes |

## 6. Method — ScanPrune: gate, coverage, merge, size-aware attention, routed schedule

**ScanPrune** (`medprune.selection.METHODS["scanprune"]` at 10% kept, `["scanprune_eq"]` at VScan's tokens per layer).
Training-free and without attention scores; selection happens before the LLM, plus one attention bias inside it.

1. **Foreground gate** — a zero-parameter body mask (near-black/white pixels connected to the image edge = background
   → keep large components → fill holes → dilate) gives each visual token a foreground fraction; tokens below 0.5 are
   background and are dropped and discarded.
2. **Coverage** — the foreground tokens are ordered along a Hilbert curve and cut into k equal stretches; one random
   token per stretch is kept.
3. **Merge** — every dropped foreground token is averaged into its most cosine-similar kept token (vision output and
   Qwen3-VL's DeepStack rows).
4. **Size-aware attention** — in every LLM layer, + log s on each kept token's attention logit (s = the tokens merged
   into it). For clusters of identical tokens this reproduces the attention over all of them exactly (proportional
   attention, Bolya et al., ICLR 2023); merging alone weighs each cluster as one token.
5. **Routed schedule** — when the instruction asks for a one-token answer (option letter, yes/no), more tokens are kept
   until 5/7 of the depth and cut after it (same mean tokens per layer: matched to FastV's at the equal budget, to the
   flat 10% count at 10%); generated answers keep a flat budget.

Every row is reported with its measured budget: mean visual tokens per decoder layer as % of N, from each question's
keep-set and schedule (Lingshu / Qwen3-VL, averaged over the model's cells). "10%" covers three budget classes, and a
comparison is only made within one. Nine-cell = raw accuracy; the per-model columns score Lingshu RadImageNet
abnormality in balanced accuracy (its gold answers are 71.9% "yes"). Judged open answers; plan.md E40-E47.

| Method | Selection signal | Budget rule | Tokens / layer (L / Q) | Nine-cell | Lingshu (5) | Qwen3-VL (4) |
|---|---|---|---|---|---|---|
| Full | — | all tokens | 100 / 100% | 70.27 | 73.32 | 69.40 |
| *10% kept before the LLM* | | | | | | |
| VisionZip | ViT attention + merge | 10% kept | 9.5 / 9.6% | 62.91 | 65.95 | 62.39 |
| PruneSID | similarity / diversity | 10% kept | 10.0 / 9.6% | 62.30 | 64.32 | 62.86 |
| CDPruner (released sign) | DPP with the instruction | 10% kept | 9.5 / 9.6% | 61.96 | 63.87 | 62.92 |
| VisPruner | ViT attention + diversity | 10% kept | 9.5 / 9.6% | 60.08 | 64.38 | 57.52 |
| DivPrune | max-min diversity | 10% kept | 9.5 / 9.6% | 57.76 | 57.00 | 61.46 |
| HiPrune | ViT hierarchical attention | 10% kept | 9.5 / 9.6% | 58.35 | 63.08 | 55.64 |
| Pool (VTC-Bench) | block average | 10% kept | 9.5 / 9.6% | 62.14 | 64.35 | 62.39 |
| Random + merge | none + merge | 10% kept | 9.5 / 9.6% | 64.66 | 68.25 | 63.11 |
| ScanPrune without the schedule (gate + coverage + merge + size term; 3 seeds) | gate, strata, log s | 10% kept | 9.5 / 9.6% | **66.17** | **69.43** | **64.53** |
| **ScanPrune, 10% (3 seeds)** | + read-then-prune | ≤ the 10% count | 9.4 / 9.5% | **66.41** | **69.71** | **64.79** |
| ScanPrune flat + sink token kept (BG-S ablation; seed 0, not adopted: +0.66, p 0.08) | + token 0 unmerged | 10% kept | 9.4 / 9.4% | 66.81 | 69.51 | 66.26 |
| *FastV's tokens per layer (all tokens to layer 2, then 10%)* | | | | | | |
| Random (K = 2) | none | FastV rule | 16.0 / 14.6% | 61.33 | 64.27 | 60.31 |
| FastV | LLM attention, layer 2 | FastV rule | 16.0 / 14.6% | 40.89 | 41.50 | 45.27 |
| VScan | ViT + LLM attention, merge | FastV-matched, 2 stages | 16.4 / 14.9% | 65.53 | 68.23 | 65.11 |
| SparseVLM ¹ | LLM attention, text raters | FastV-matched, 3 stages | 16.2 / 15.0% | 51.91 ¹ | | |
| BTP ¹ | LLM attention + spatial | FastV-matched, 4 stages | 16.0 / 15.1% | 63.02 ¹ | | |
| **ScanPrune, equal budget (3 seeds)** | gate, strata, log s, schedule | FastV-matched | 16.3 / 15.2% | **67.62** | **71.21** | **65.41** |
| *Larger budgets (own rules)* | | | | | | |
| FEATHER | RoPE-free LLM attention | 2 stages | 46.0 / 37.9% | 61.21 | 65.43 | 60.57 |
| TransPrune | token transition | multi-stage | 43.4 / 35.9% | 60.91 | 62.50 | 64.27 |

¹ No RadImageNet open-anatomy runs: mean over the seven other cells (same cells: Full 74.75, VScan 71.17, ours at the
equal budget 73.06, ours at 10% 71.89). PDrop (Lingshu only, three cells) is in plan.md.

Paired, images resampled jointly across models, three-seed means: ScanPrune (10%) − VisionZip **+3.16 (SE 0.42)**, −
PruneSID +3.86 (SE 0.43), − 2B +0.27 (SE 0.20 with the residual seed term, p 0.08; held-out `omni_panel` +1.08, p 0.009);
ScanPrune without the schedule − VisionZip +2.89 (SE 0.42); ScanPrune (equal budget) − VScan **+1.79 (SE 0.39)**. Qwen3-VL's equal-budget route spends 15.2% per layer against VScan's 14.9%
(integer rounding of the schedule on 64-token images), 2% more tokens for a +0.3 Qwen3-VL margin.

Per model at VScan's budget, three seeds: Lingshu +3.0, Qwen3-VL +0.3 (every seed above VScan).

Contemporaneous work: MedSTP (arXiv 2610.22751, Oct 15, 2026; 10% kept, pre-LLM) is reported as a separate reference block, not as a baseline, because its SLAKE set and open-answer metric differ from ours. On the two benchmarks where the protocols align (OmniMedVQA, RadImageNet anatomy MC), our 10% row gains more over PruneSID than MedSTP reports, on both models (Qwen3-VL margins within noise; plan.md §A3.1).

---

## Plan and status

**[plan.md](plan.md)** is the single plan: stage tracker (Stages 0–8 with checkboxes and exit gates), research,
protocol, code design and compute backends. Predictions are pre-registered in plan.md §A6 (2026-11-02).

Status (Nov 10, 16:00 IST): the diagnosis (Stages 0-7) is complete; the method above was built and tested in E40-E46,
each experiment registered in plan.md before its runs, adopted or rejected by a fixed one-sided test with Holm control,
two-halves replication and a per-cell loss guard (`medprune/adoption.py`). The final row beats VScan by +1.79 nine-cell
(SE 0.39) over three seeds (E45); the repairs of the background summary and k-means pieces (E46) were not adopted. All
registered runs are complete. Status file: [status_2026-11-07.md](status_2026-11-07.md) (Nov 10 update at the end);
live tracker: [plan.md](plan.md#stage-tracker).

**Other agents (Codex, Gemini):** start at plan.md Part F (rules, environment, task queue with definitions of done),
then read the Work log at the end of plan.md and add an entry when you finish.

Compute: six Kaggle accounts (A-F, T4 x2, 30 GPU-h per week each); the ledger is at the top of plan.md. Storage: ≤ 2 GB
persistent; one model (~17 GB) temporarily per GPU session.

## Repository

```
cvpr/
├── README.md               framing, status, quick start
├── plan.md                 the plan (stages, research, protocol, code design, backends)
├── mentor_brief.tex        two-page mentor brief (Overleaf)
├── requirements.txt        pinned (transformers==4.57.1); torch comes from the platform
├── pyproject.toml          package metadata + pytest config
├── scripts/setup_env.sh    identical setup on Kaggle and SSH; prints the compute profile
├── configs/runs.yaml       the run matrix (every experiment's runs); bundle_manifest.json added at Stage 2
├── medprune/               the package: env, data, masks, selection, attention_hook, models, scoring, stats,
│                           run_eval, checks, analysis, remedy (the method), adoption (the decision rule),
│                           baselines/ (14 published pruners), judge
├── notebooks/              00 data bundle (Kaggle CPU), 01 GPU runs (Kaggle), 02 analysis (laptop)
├── tests/                  218 CPU tests incl. the masking ≡ token-removal proof and the size-term identity
├── results/                runtime output (git-ignored)
└── third_party/            FastV, MedEvalKit, HuatuoGPT-Vision references (git-ignored)
```

## Where it runs

| Backend | When | How |
|---|---|---|
| Kaggle GPU T4×2 | Now | `notebooks/01_run_gpu.ipynb` (thin wrapper around the CLI), commit mode |
| SSH A100 | If the mentor provides one | Same CLI with `MEDPRUNE_PROFILE=a100` (bf16, one GPU) — plan.md Part E |
| Laptop CPU | Always | Tests and analysis |

Quick start on any Linux GPU machine (available after Stage 1):

```bash
bash scripts/setup_env.sh
python -m medprune.data build --out "$MEDPRUNE_DATA"
python -m medprune.run_eval --exp E1 --model lingshu7b
```

Reference repositories (read-only): see the Appendix of plan.md for the clone commands.
