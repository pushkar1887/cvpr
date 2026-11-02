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

## 6. Fix — Foreground Gate

A zero-parameter body mask on the image the model sees (Otsu → keep large components → fill holes → dilate) gives
each visual token a foreground fraction; any pruner then selects only among foreground tokens, within the same budget.
Under 5 ms per image, no training, composes with FastV, PDrop, DivPrune and uniform sampling.

---

## Plan and status

**[plan.md](plan.md)** is the single plan: stage tracker (Stages 0–8 with checkboxes and exit gates), research,
protocol, code design and compute backends. Current stage: **0–1** (accounts, code). Nothing has run yet; predictions
are pre-registered in plan.md §A6 (2026-11-02).

Compute: 66.2 Kaggle T4-hours (39.4 h must-have) across three accounts, or ≈ 14–17 h on an A100. Storage: ≤ 2 GB
persistent; one model (~17 GB) temporarily per GPU session.

## Repository

```
cvpr/
├── README.md               framing, status, quick start
├── plan.md                 the plan (stages, research, protocol, code design, backends)
├── mentor_brief.tex        two-page mentor brief (Overleaf)
├── requirements.txt        pinned (transformers==4.57.1); torch comes from the platform
├── scripts/setup_env.sh    identical setup on Kaggle and SSH; prints the compute profile
├── configs/                runs.yaml (every run), bundle_manifest.json (data provenance)   [Stage 1–2]
├── medprune/               the package: env, data, masks, selection, attention_hook, models,
│                           huatuo_llava, scoring, stats, run_eval                            [Stage 1]
├── notebooks/              00 data bundle (Kaggle CPU), 01 GPU runs (Kaggle), 02 analysis (laptop)
├── tests/                  CPU-only; masking ≡ token removal proof                          [Stage 1]
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
