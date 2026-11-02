# Plan — Pruning the Air

Single source of truth for the project: research (Part A), execution by stages (Part B), protocol (Part C), code
design (Part D), compute backends incl. SSH A100 (Part E). Version 2 (merged), 2026-11-02.

Conventions: ✓ = verified on 2026-11-02 against the named source. Stages run in order; a stage starts only when the
previous stage's **exit gate** is met (parallel work is marked). Tick boxes on GitHub as work completes. Predictions in
§A6 are frozen as of 2026-11-02 — record outcomes, never edit predictions.

## Stage tracker

| Stage | Work | Where | GPU-h | Dates (IST) | Exit gate |
|---|---|---|---|---|---|
| 0 | Accounts, access, repo | Laptop, Hugging Face, Kaggle, GitHub | 0 | Nov 2 | All three Kaggle accounts ready; RadImageNet access granted |
| 1 | Code and tests | Laptop CPU | 0 | Nov 2 | `pytest -q` green, incl. masking ≡ token removal |
| 2 | Data bundle | Kaggle CPU (or SSH) | 0 | Nov 2 | Counts match §C2; masks eyeballed |
| 3 | Validation and calibration (E0) | Kaggle GPU, account A | 3.0 | Nov 2–3 | fp16 stable; Full SLAKE-en within 2 pts; rates measured |
| 4 | Kill test (E1) + diagnostics (E2) | GPU A + CPU | 3.3 | Nov 3 | Decisions K1, K2 |
| 5 | Main grid (E3) + modality panel (E4) | GPU A, B | 33.1 | Nov 3–5 | Table 1, Fig. 2 drafts → **enrollment Nov 10 AoE** |
| 6 | Curves, ablations, natural control (E5, E6-P1, E7) | GPU A, B | 16.8 | Nov 6–8 | Decisions K3, K4 |
| 7 | Supplementary runs (E6-P2, E8, E9) | GPU C | 10.0 | Nov 7–12 | — |
| 8 | Analysis and writing | Laptop | 0 | Nov 9–23 | **Paper Nov 16 AoE**, supplementary Nov 23 AoE |

Totals: 66.2 T4-hours (≈ 79 h with 20% contingency), ≈ 256k question-evaluations. On an A100 (Part E) the same
work is ≈ 14–17 h.

Deadlines (CVPR 2027 CFP ✓): paper enrollment **Nov 10, 23:59 AoE = Nov 11, 17:29 IST**; submission **Nov 16 AoE =
Nov 17, 17:29 IST**; supplementary **Nov 23 AoE**. All authors must accept reviewing duties.

---

# Part A — Research

## A1. Thesis

FastV-style pruning keeps the top-k visual tokens by the last prompt token's attention at an early layer. That
attention has two content-independent components: a **positional** component from rotary decay (favoring the bottom
rows under 1D RoPE, the bottom-right corner under M-RoPE, §A4) and a **sink** component (flat, empty, padding-like and
high-norm tokens absorb attention). On photographs both land on content. On scans, acquisition fixes the geometry —
body in the center; corners, bottom rows and borders are air, table or empty field of view — so both point at
background and the keep-set becomes anti-informative. A geometric prior that removes background before scoring (the
**Foreground Gate**) should recover most of the loss at zero training cost.

Not another scorer: MedSTP, ViToS and others propose new pruners; none explains why the standard family fails.

## A2. Claims and required evidence

| Claim | Evidence (reviewer bar) | Experiments | Paper element |
|---|---|---|---|
| **C1 Phenomenon**: early attention-guided pruning selects worse than random on medical images | 2 models (medical + general) × 3 benchmarks; paired CIs excluding 0; several budgets; image-dependent subset; natural-image contrast | E1, E3, E5, E7 | Table 1, Fig. 4 |
| **C2 Predictability**: background fraction predicts the size of the failure | Monotone dose-response across 8 modalities and across images; none in background-free modalities | E3, E4 | Fig. 2 |
| **C3 Mechanism**: the keep-set follows position and background, not content | Keep-maps match the rotary prediction (1D vs M-RoPE); position-only static map ≈ FastV; background-only keep-set ≈ FastV; regression separates position from background; K-sweep shrinks the gap | E2, E6, E8 | Fig. 3, Tables 2–3 |
| **C4 Remedy**: a zero-parameter gate recovers the loss and composes with pruners | Gate+FastV ≫ FastV where background exists, ≈ where not; 2×2 factorial vs RoPE-free; DivPrune/PDrop ± gate; budget curves; latency | E3, E5, E6, E9 | Tables 3–4, Fig. 4 |
| **C5 Protocol**: report image-dependent accuracy, image-clustered CIs, background fraction | At least one conclusion that shrinks or flips under the stricter protocol | E3 analysis | §3 + supp |

If C3 fails but C1, C2, C4 hold, the paper stands as empirical + fix. If C1 fails in our pipeline, stop (K1).

## A3. Prior work and our delta

| Work | Venue | Established | Medical | Open (our delta) |
|---|---|---|---|---|
| FastV (Chen et al.) | ECCV 2024 | Prune by last-token attention at layer K | No | "High attention ⇒ important" untested on images with large empty regions |
| FEATHER (Endo et al.) | ICCV 2025 | Early pruning keeps the image bottom (80.7% down); localization 5.9 vs 28.0 uniform; 1D-RoPE decay | No | M-RoPE; ordinary VQA; acquisition geometry |
| Attention Debiasing / PoRe | arXiv 2508.17807 | Recency bias; padding tokens over-attended | No | Medical "padding-like" background |
| When Token Pruning is Worse than Random (Wang et al.) | arXiv 2512.07580 | Deep-layer pruning ≈ random; Qwen2.5-VL has a deeper information horizon | No | Predicts layer-2 tokens are informative → cannot explain FastV@K=2 < random |
| VTC-Bench | ACL 2026 | Downsampling beats many compressors; benchmarks too easy | No | Medical benchmarks + image-dependent subset |
| Registers (Darcet et al.); test-time registers (Jiang et al.) | ICLR 2024; arXiv 2506.08010 | High-norm artifacts on flat background absorb attention | No | Present in Qwen ViTs on scans? Selected by pruners? |
| MedSTP | arXiv 2610.22751 (Oct 15, 2026) | Spectral pruner; its table shows FastV < Random in 5/8 cells, unexplained | Yes | No diagnosis; FastV adapted to pre-LLM insertion; protocol under-specified |
| ViToS | ICML 2026 | RL-trained token sparsity; FastV weakest at 80% | Yes | Training-free regime |
| Photon; MedARC | arXiv 2603.25155; 2607.26554 | 3D token reduction | Yes | 2D; attention-family failure |
| EasyLens | arXiv 2606.06379 | Subtle-lesion feature amplification | Yes | No pruning |

CVPR 2027 rule ✓: work online after **Sept 15, 2026** is contemporaneous — MedSTP must be cited and discussed, not beaten.

## A4. Mechanism and hypotheses

**Notation.** N visual tokens on an H×W grid, cell c px (Lingshu 28, Qwen3-VL 32, HuatuoGPT 14 at 336 px → 24×24).
FastV score: s_i = mean_h softmax_j(q_last,h · k_j,h / √d)_i at layer K−1; keep top k = round(ρN) from layer K on;
positions unchanged ✓ (`third_party/FastV/.../modeling_llama.py` L1038–1052). Body mask M (§A5) → per-token
foreground fraction m_i; background: m_i < τ = 0.5; background fraction b = mean_i [m_i < τ].

**Why M-RoPE favors the bottom-right.** In Qwen2.5-VL/Qwen3-VL `get_rope_index` ✓, token (r, c) gets position ids
(β, β + r, β + c); text after the image starts at s = β + max(H−1, W−1) + 1 on all three axes. The last prompt token's
offsets are Δh = s + j − β − r, Δw = s + j − β − c. RoPE's long-term decay (RoFormer §3.4.3) favors small offsets →
large r **and** c → bottom-right corner. 1D RoPE (HuatuoGPT) favors the last rows (FEATHER). Axial CT/MRI put air or
table there.

| ID | Hypothesis | Prediction | Test | Rejected if |
|---|---|---|---|---|
| **H1** Background capture | Background tokens over-selected beyond position | BRR = P(bg \| kept)/P(bg) ≥ 1.3 on CT/MRI (Random ≈ 1); positive background coefficient after controlling for position | E2 | BRR CI includes 1 and coefficient ≈ 0 |
| **H2** Positional bias | Keep-set follows rotary geometry | Bottom-right concentration (M-RoPE), bottom rows (1D); cross-dataset static map ≥ 80% of FastV accuracy, Jaccard ≥ 0.4; gap shrinks as K: 2 → 8 → 16 | E2, E6, E8 | χ² n.s. or static map ≪ FastV |
| **H3** Artifact sinks | High-norm tokens on flat background attract scoring attention | Selected ≥ 2× base rate, located in background | E2, E6 | No outliers or base-rate selection |
| **H4** Small evidence | Residual loss sits on small findings | Largest residual in smallest SLAKE evidence tertile; abnormality > anatomy loss on the same RadImageNet images | E3, E6 | Flat across strata |
| **H5** Architecture | DeepStack makes Qwen3-VL's layer-1 attention more content-aware | Smaller gap than Lingshu at matched b; explains MedSTP's SLAKE inversion | E3, E4 | Equal gaps at matched b |

The 2×2 factorial {attention, RoPE-free} × {no gate, gate} separates H1 from H2; either outcome is publishable.

## A5. Method — Foreground Gate

Mask on the exact image the model sees (after its resize; HuatuoGPT after `expand2square`):
1. Luminance Y = 0.299R + 0.587G + 0.114B.
2. Polarity: border-ring median vs central-crop median; foreground is the side unlike the border.
3. Otsu threshold → F.
4. Keep components (8-connectivity) ≥ 10% of the largest → removes table, text, markers.
5. Fill holes → keeps lungs, airway, bowel gas.
6. Dilate by one cell.
7. Force known padding (HuatuoGPT) to background.
8. m_i = mean of F over cell i; gate g_i = [m_i ≥ τ], τ = 0.5.

Selection: score-based → top-k among g_i = 1, fill from best background if short; score-free → k evenly spaced (Gate+
Uniform) or max–min diversity (Gate+DivPrune) among g_i = 1. **Budget fairness:** k = round(ρN), N counts all visual
tokens. **Variants:** Gate-score (selection only; causal test of H1); Gate-input (drop background before the LLM; free
compression; latency only). Cost < 5 ms per 512² image, no parameters. Ablations: τ ∈ {0.25, 0.5, 0.75}; background
quota 0/3%; oracle gate (SLAKE masks) as upper bound; inverse gate (background only) as lower bound.

## A6. Pre-registered predictions (frozen 2026-11-02) and decision rules

| # | Experiment | Prediction | Confidence | Outcome |
|---|---|---|---|---|
| P1 | E1, RadImageNet anatomy MC, Lingshu, 10% | Standard FastV < Random@L2 by ≥ 8; pre-LLM FastV < Random@L0 by ≥ 10 (MedSTP: 37.5 vs 55.6) | 65% / 80% | |
| P2 | E1, SLAKE-en | Full 81–85; FastV − Random ≤ −4 on CT+MRI (66 images), ≥ 0 on X-ray (30) | 55% | |
| P3 | E2 | BRR(FastV) ≥ 1.3 on CT/MRI | 65% | |
| P4 | E2 | Keep-mass ≥ 1.5× uniform in bottom-right (Lingshu/Qwen3-VL), bottom rows (HuatuoGPT) | 60% | |
| P5 | E2/E6 | High-norm artifacts over-selected ≥ 2× | 35% | |
| P6 | E3 | Gate+FastV − FastV ≥ +8 on CT/MRI; \|Δ\| ≤ 1 on dermoscopy/microscopy | 55% | |
| P7 | E3 | Gate+Uniform within 3 of Gate+FastV on CT/MRI | 50% | |
| P8 | E3/E6 | Gate adds ≥ 3 on top of RoPE-free on CT/MRI | 50% | |
| P9 | E4 | Modality-level Spearman ρ ≥ 0.7; per-image ρ > 0 (CI excl. 0) | 55% | |
| P10 | E3 | Qwen3-VL same sign as Lingshu on CT/MRI, smaller magnitude | 60% | |
| P11 | E6 | Gap shrinks monotonically with K; K=16 ≈ Random | 65% | |
| P12 | E6 | Inverse gate within 5 of FastV on CT/MRI | 50% | |
| P13 | E7 | Natural images: FastV ≥ Random − 2 | 50% | |
| P14 | E6 | Residual largest in smallest evidence tertile | 55% | |

- **K1 (Stage 4):** GO if standard FastV < Random@L2 on RadImageNet anatomy MC, paired 95% CI excl. 0. If only pre-LLM
  FastV < Random@L0 → reframe as "pre-LLM attention selection fails" (discuss with mentor). Neither → stop.
- **K2 (Stage 4):** BRR CI includes 1 → H1 out; lead with position (H2) and the static map.
- **K3 (Stage 6):** Gate+FastV recovers < ⅓ of (Full − FastV) on CT/MRI → analysis-first paper (consider MIDL/MICCAI).
- **K4 (Stage 6):** RoPE-free alone recovers ≥ 90% of the gate's gain and the gate adds nothing → reframe as
  "positional bias meets acquisition geometry"; gate = cheap geometric alternative.

## A7. Paper outline (8 pages)

| Section | Content | Space |
|---|---|---|
| 1 Introduction | MedSTP's unexplained inversion, thesis, contributions; Fig. 1 | 1.0 |
| 2 Related work | Pruning families, positional bias, sinks/registers, medical VLM efficiency | 0.6 |
| 3 Setup and protocol | Models, benchmarks, budget convention, image-dependent subset, cluster bootstrap | 0.7 |
| 4 Phenomenon | Table 1, Fig. 2, natural control | 1.4 |
| 5 Diagnosis | Fig. 3, Table 2, K-sweep, 1D vs M-RoPE | 1.6 |
| 6 Foreground Gate | Method, Table 3, Fig. 4, Fig. 5, Table 4 | 1.8 |
| 7 Limitations, conclusion | 2D only, heuristic mask, closed-form benchmarks, no clinical claim | 0.4 |

- **Fig. 1** CT + chest X-ray with kept tokens (FastV, Random, Gate+FastV; Lingshu, 10%); bars of Random − FastV by modality.
- **Fig. 2** Background-fraction deciles vs Random − FastV (95% cluster CI), per model; modality means labeled.
- **Fig. 3** Keep-probability heatmaps (FastV) for Lingshu, Qwen3-VL, HuatuoGPT × {CT/MRI, X-ray, natural}; body contour; predicted region.
- **Fig. 4** Accuracy vs budget {5, 10, 25, 50}% for Full, Random, FastV, Gate+FastV, Gate+Uniform; SLAKE-en, RadImageNet MC; both models.
- **Fig. 5** Accuracy by SLAKE evidence tertile; RadImageNet anatomy vs abnormality.
- **Table 1** Methods × {SLAKE all/en, OmniMedVQA-anatomy, RadImageNet anatomy MC, RadImageNet abnormality, mean} × 2 models; MedSTP's numbers as a separate grey block.
- **Table 2** BRR, quadrant concentration, regression coefficients, high-norm rate, static-map accuracy/Jaccard, inverse/oracle gate.
- **Table 3** 2×2 factorial; K-sweep; DivPrune/PDrop ± gate; τ, quota.
- **Table 4** Tokens into the LLM, prefill TFLOPs, measured latency.

## A8. Risks

| Risk | Mitigation |
|---|---|
| Standard in-LLM FastV not below random (MedSTP pruned pre-LLM) | E1 runs both insertion points; K1 |
| fp16 overflow on T4 | Day-1 check; vision fp32; 8-bit fallback (2–3× slower) |
| Throughput half of plan | P0 = 39 h fits; cut Random seeds off S1; A100 (Part E) |
| Full accuracy ≠ MedSTP's | MedEvalKit prompts/scoring; self-contained tables; MedSTP only as reference |
| "Fix is trivial" / "bias is known" | Minimal causal intervention; FEATHER/PoRe as baselines; factorial |
| "Medical only" | Natural control + dose-response = statement about structured backgrounds |
| SLAKE has 96 test images | Image-clustered bootstrap; RadImageNet (1,000 images) and panel carry statistics |
| RadImageNet license forbids redistribution | Per-account download; aggregates only |
| Public repo before submission | Keep the GitHub repo **private** until the paper is submitted (scooping, double-blind) |

---

# Part B — Execution by stages

## Stage 0 — Accounts, access, repo (Nov 2, no GPU)

- [ ] GitHub repo `pushkar1887/cvpr` private; all teammates added as collaborators.
- [ ] Every teammate: accept RadImageNet-VQA terms on Hugging Face; create a read token.
- [ ] Every Kaggle account (A, B, C): phone-verified (internet), secrets `HF_TOKEN` and `GITHUB_TOKEN` added.
- [ ] Record each account's weekly GPU quota here: A ___ h, B ___ h, C ___ h (unpublished; reports 15–30 h).
- [ ] Laptop: clone references into `third_party/` (commands in the Appendix); they are git-ignored.

**Exit gate:** three accounts ready; RadImageNet access granted on at least accounts A and B.

## Stage 1 — Code and tests (Nov 2, laptop CPU)

- [ ] `requirements.txt`, `scripts/setup_env.sh` (done), `.gitignore` (done).
- [ ] `medprune/env.py`, `data.py`, `masks.py`, `selection.py`, `attention_hook.py`, `models.py`, `scoring.py`,
      `stats.py`, `run_eval.py` per Part D.
- [ ] `tests/` per §D5; `pytest -q` passes, including masking ≡ physical token removal on tiny Qwen2 and Qwen3.
- [ ] `configs/runs.yaml` generated from the run tables in Stages 3–7 (hours from §C4).

**Exit gate:** tests green; `runs.yaml` lists every run with an hour estimate.

## Stage 2 — Data bundle (Nov 2, Kaggle CPU; runs identically over SSH)

- [ ] `00_build_data_bundle.ipynb` (thin wrapper around `python -m medprune.data build --out <dir>`).
- [ ] Counts match §C2 (✓ rows exactly; record the others, especially `omni_anat` vs MedSTP's 3,329).
- [ ] Masks for the three model grids; 20 overlays per modality checked by eye.
- [ ] Publish as private Kaggle dataset `medprune-data` (~0.7 GB); commit `configs/bundle_manifest.json` (counts + hashes).

**Exit gate:** counts verified; overlays look right (table removed, lungs kept, padding background).

## Stage 3 — Validation and calibration, E0 (Nov 2–3, GPU A, 3.0 h)

- [ ] fp16 vs fp32-vision on 50 SLAKE questions: no NaN/inf, ≥ 96% identical answers → choose dtype, fix for all runs.
- [ ] Hook installed with no pruning ≡ vanilla outputs (50 questions); FastV at 50% → small drop only.
- [ ] Full SLAKE-en accuracy within 2 points of the published Lingshu reference (≈ 82–84 under MedEvalKit).
- [ ] Time 50 questions; scale all hours in `runs.yaml` by measured/planned.

**Exit gate:** dtype fixed; Full accuracy plausible; rates calibrated. If the factor > 1.5: Random 1 seed outside S1,
drop `uniform_l2` from E3, move E5's 50% points to Stage 7.

## Stage 4 — Kill test E1 + diagnostics E2 (Nov 3, GPU A 3.3 h + CPU)

E1: Lingshu, group K (`slake_en` + `rin_anat_mc`, 2,061 questions), 10%.

| Method | Seeds | Hours |
|---|---|---|
| `full`, `random_l2`, `random_l0`, `fastv`, `gate_fastv` | 0 | 5 × 0.462 = 2.31 |
| `fastv_prellm` | — | 0.60 |
| `text_only` | — | 0.29 |
| model load | — | 0.10 |

- [ ] E1 runs complete (14,427 question-evaluations).
- [ ] P1, P2 outcomes recorded in §A6. **Decision K1.**
- [ ] E2 on E1 logs: BRR, keep-maps, keep regression, high-norm rates. P3–P5 recorded. **Decision K2.**

**Exit gate:** GO (or reframe) agreed with the mentor.

## Stage 5 — Main grid E3 + modality panel E4 (Nov 3–5, GPU A and B, 33.1 h)

E3 at 10% (Lingshu on A: 12.6 h; Qwen3-VL on B: 16.3 h):

| Group | Methods | Image runs | Lingshu h | Qwen3-VL h |
|---|---|---|---|---|
| S1 | `full`, `random_l2` ×3, `uniform_l2`, `fastv`, `ropefree`, `gate_fastv`, `gate_ropefree`, `gate_uniform` | 10 | 4.44 (reuses E1's 4 runs) | 6.92 |
| S2 | `full`, `random_l2` s0, `fastv`, `gate_fastv` | 4 | 1.15 | 1.26 |
| S3 | `full`, `random_l2` s0, `fastv`, `ropefree`, `gate_fastv`, `gate_uniform` | 6 | 6.10 | 6.71 |
| all | `text_only` | — | 0.75 | 1.13 |
| | loads | — | 0.20 | 0.30 |

E4, group P: `full`, `random_l2` s0, `fastv`, `gate_fastv`, `gate_uniform` + `text_only` → Lingshu 2.0 h (A), Qwen3-VL 2.2 h (B).

- [ ] E3 complete (Lingshu 51,834; Qwen3-VL 62,139 question-evaluations). P6, P7, P10 recorded.
- [ ] E4 complete; dose-response computed. P9 recorded.
- [ ] Draft Table 1 and Fig. 2; abstract draft to mentor by Nov 9.
- [ ] **Paper enrollment on OpenReview by Nov 10 evening IST.**

**Exit gate:** Table 1 and Fig. 2 support C1–C2 (or the reframed claim from K1/K2).

## Stage 6 — Curves, ablations, natural control (Nov 6–8, GPU A and B, 16.8 h)

| Exp | Model | Group | Methods / budgets | Hours |
|---|---|---|---|---|
| E5 | Lingshu (A) | K | {5, 25}% × {`random_l2`, `fastv`, `gate_fastv`, `gate_uniform`} + 50% × {`fastv`, `gate_fastv`} | 4.62 |
| E5 | Qwen3-VL (B) | K | {5, 25}% × {`random_l2`, `fastv`, `gate_fastv`} | 3.05 |
| E6-P1 | Lingshu (A) | S1 | `fastv_k8`, `fastv_k16`, `static_map`, `inverse_gate`, `divprune`, `gate_divprune` | 3.77 |
| E6-P1 | Lingshu (A) | `slake_en` | `oracle_gate` | 0.30 |
| E6-P1 | Qwen3-VL (B) | K | `fastv_k16`, `static_map`, `inverse_gate`, `divprune`, `gate_divprune` | 2.54 |
| E7 | both (B) | N | `full`, `random_l2`, `uniform_l2`, `fastv`, `ropefree` (no gate) | 2.53 |

- [ ] Runs complete; P8, P11–P13 recorded. **Decisions K3, K4.**
- [ ] Fig. 3, Fig. 4, Table 3 drafts.

**Exit gate:** remedy and mechanism claims settled (or reframed per K3/K4).

## Stage 7 — Supplementary runs (Nov 7–12, GPU C, 10.0 h)

| Exp | Model | Group | Methods | Hours |
|---|---|---|---|---|
| E6-P2 | Lingshu | S1 | `artifact_excl`, `gate_quota3`, `gate_tau25`, `gate_tau75`, `pdrop`, `gate_pdrop` | 3.77 |
| E8 | HuatuoGPT-Vision-7B | S1 | `full`, `random_l2`, `fastv`, `ropefree`, `gate_fastv`, `gate_ropefree` + `text_only` + 1.0 h port validation | 5.71 |
| E9 | Lingshu | 200 × `slake_en` | `full` vs `gate_input_fastv`, CUDA-timed prefill; FLOPs for all methods | 0.50 |

- [ ] HuatuoGPT port validated (Full within 3 points of the authors' numbers) before E8 runs.
- [ ] P14 recorded; Table 4 and supplementary tables drafted.

## Stage 8 — Analysis and writing (Nov 9–23, laptop)

- [ ] `02_analyze_results.ipynb` regenerates every table and figure from `results/` (no hand-edited numbers).
- [ ] Paper: §A7 outline; limitations honest; MedSTP cited as contemporaneous.
- [ ] **Submit by Nov 16 AoE** (aim for Nov 16 IST evening). Supplementary by Nov 23 AoE.

---

# Part C — Protocol

## C1. Fixed settings

| Setting | Value |
|---|---|
| Models | `lingshu-medical-mllm/Lingshu-7B` (16.58 GB ✓), `Qwen/Qwen3-VL-8B-Instruct` (17.53 GB ✓), `FreedomIntelligence/HuatuoGPT-Vision-7B` (15.87 GB + 1.71 GB CLIP ✓) |
| Precision / placement | By compute profile (Part E): T4×2 fp16 split over both GPUs; A100 bf16 on one GPU. One profile per model across all its runs |
| Attention | Text decoder `medprune_sdpa` (hook around transformers' `sdpa_attention_forward`); vision `sdpa` |
| transformers | 4.57.1 (Qwen3-VL needs ≥ 4.57 ✓; `AttentionInterface`, `sdpa_attention_forward` verified ✓) |
| Resolution | Qwen: `min_pixels` default (Lingshu 3,136 ✓, Qwen3-VL 65,536 ✓); `max_pixels` = 1,024 tokens × cell² (802,816 / 1,048,576) — defaults are uncapped ✓. HuatuoGPT: `expand2square` (CLIP mean) → 336² → 576 tokens ✓ |
| Prompts | MedEvalKit `question_formats.py` ✓ (§C1.1) |
| Chat template | Model's own; HuatuoGPT `"<|user|>\n<image>\n{prompt}\n<|assistant|>\n"` ✓ |
| Decoding | Greedy, `max_new_tokens=64`, `use_cache=True` (Lingshu and HuatuoGPT configs set `use_cache: false` ✓); rerun truncated items with 512 if > 1% |
| Scoring | Port of MedEvalKit `utils.py` ✓ (§C1.2) |
| Budget | k = max(1, round(ρN)), N = all visual tokens; main ρ = 10% |
| FastV | Last prompt token, softmax over all keys, head mean, at layer K−1; mask from layer K; K = 2 ✓ |
| Seeds | Random seeds {0, 1, 2} per question via `sha256(f"{seed}:{qid}")`; torch seed 42 |

**C1.1 Prompts.** SLAKE closed: `{q}\nAnswer the question using a single word or phrase.` · SLAKE open:
`{q}\nPlease answer the question concisely.` · Multiple choice: `\nQuestion: {q}\nOptions: \nA. …\nAnswer with the
option's letter from the given choices directly.` · Yes/no (RadImageNet abnormality, POPE): `{q}\nPlease output 'yes'
or 'no'(no extra output).` · GQA: close-ended prompt. Chinese variants from the same file for SLAKE-zh.

**C1.2 Correctness.** Yes/no: exactly one of yes/no present and equal to the answer. Other closed: exact match after
lowercasing and stripping newlines/periods. Open: exact match (primary, MedEvalKit's "right"), token recall
(secondary). Multiple choice: letter, choice text, or difflib nearest choice. All after `parse_response`.

## C2. Data

| ID | Source | Size ✓ | Access | Subset rule | Questions | Images |
|---|---|---|---|---|---|---|
| `slake_en` | `BoKelvin/SLAKE` | `imgs.zip` 212 MB (masks + boxes per image), `test.json` 636 KB, `mask.txt` 39 labels | CC-BY-4.0 | test, `q_lang == "en"` | **1,061** ✓ (open 645, closed 416) | **96** ✓ (CT 40, MRI 26, X-ray 30) |
| `slake_zh` | same | same | same | test, `q_lang == "zh"` | **1,033** ✓ | 96 |
| `omni_anat` | `foreverbeliever/OmniMedVQA` | zip 10.70 GB | public, per-source licenses (bundle private) | open-access, `question_type == "Anatomy Identification"`; expect MedSTP's 3,329 — verify | ~3,329 MC | verify |
| `omni_panel` | same | same | same | open-access; 8 modalities (CT, MRI, X-Ray, Ultrasound, OCT, Fundus, Dermoscopy, Microscopy) × 150, seed 0, ≤ 3/image, disjoint from `omni_anat` | 1,200 | ≥ 400 |
| `rin_anat_mc` | `raidium/RadImageNet-VQA` | benchmark parquet 361 MB (9,000 QA / 1,000 images, CT/MRI) | gated, research-use, **no redistribution** | anatomy × multiple choice, one per image (verify field values) | ~1,000 | 1,000 |
| `rin_abn_closed` | same | same | same | abnormality × yes/no, one per image | ~1,000 | same 1,000 |
| `natural` | `lmms-lab/GQA`, `lmms-lab/POPE` | 2.1 + 65.6 MB; 85 MB | public | 500 GQA testdev-balanced + 500 POPE adversarial, seed 0 | 1,000 | ~900 |

SLAKE-en mix ✓: Organ 253, Position 186, Abnormality 150, KG 148, Modality 108, Size 65, Plane 58, Quantity 52, Color
34, Shape 7. MedSTP reports 2,394 SLAKE instances; SLAKE test has **2,094** (en + zh) ✓, which MedEvalKit loads in full
✓ — "SLAKE (all)" = `slake_en ∪ slake_zh`. MedSTP documents no prompts, subset rules or FastV layer, so its numbers
appear only as a reference block.

**Dataset groups** (used by the stage tables):

| Group | Datasets | Questions | Lingshu h/run | Qwen3-VL h/run |
|---|---|---|---|---|
| K | `slake_en` + `rin_anat_mc` | 2,061 | 0.462 | 0.508 |
| S1 | K + `rin_abn_closed` | 3,061 | 0.629 | 0.692 |
| S2 | `slake_zh` | 1,033 | 0.287 | 0.316 |
| S3 | `omni_anat` | 3,329 | 1.017 | 1.119 |
| P | `omni_panel` | 1,200 | 0.367 | 0.403 |
| N | `natural` | 1,000 | 0.222 | 0.244 |

## C3. Method registry

Score layer = layer whose attention is read; mask from = first masked layer (0 = before the LLM).

| Method | Keep rule | Score layer | Mask from |
|---|---|---|---|
| `full` | all; logs scores at layers 1, 7, 15, RoPE-free scores, ViT-output and layer-1 norms | — | — |
| `text_only` | no image | — | — |
| `random_l2` / `random_l0` | uniform random k | — | 2 / 0 |
| `uniform_l2` | k evenly spaced (raster) | — | 2 |
| `fastv` | top-k score | 1 | 2 |
| `fastv_prellm` | top-k from a first prefill pass (MedSTP-style) | 1 | 0 |
| `ropefree` | top-k of pre-RoPE q·k (Qwen2.5: `q_proj`/`k_proj`; Qwen3: `q_norm`/`k_norm` ✓) | 1 | 2 |
| `gate_fastv` / `gate_ropefree` / `gate_uniform` | as above among gate-passing tokens (τ = 0.5) | 1 / 1 / — | 2 |
| `fastv_k8` / `fastv_k16` | top-k score | 7 / 15 | 8 / 16 |
| `static_map` | top-k of cross-dataset mean rank map (SLAKE ↔ RadImageNet) | — | 2 |
| `inverse_gate` | k evenly spaced among background | — | 2 |
| `oracle_gate` | top-k among SLAKE organ-mask union (SLAKE only) | 1 | 2 |
| `artifact_excl` | FastV excluding ViT-norm outliers (z > 5, median/MAD) | 1 | 2 |
| `gate_quota3` | `gate_fastv` + 3%·N evenly spaced background, total k | 1 | 2 |
| `gate_tau25` / `gate_tau75` | `gate_fastv`, τ = 0.25 / 0.75 | 1 | 2 |
| `divprune` / `gate_divprune` | greedy max–min cosine diversity on LLM-input visual embeddings (DivPrune Alg. 1) | — | 0 |
| `pdrop` / `gate_pdrop` | boundaries {2, 9, 16, 23}, keep {a, a/2, a/4, a/8}·N, a = 26ρ/12.875 (ρ = 10% → 20.2%), token-matched to FastV | 1, 8, 15, 22 | 2, 9, 16, 23 |
| `gate_input_fastv` | physically drop background before the LLM, then FastV (latency, Lingshu only) | 1 | 2 |

## C4. Planning rates (s/question, T4×2, batch 1)

| Model | SLAKE | OmniMedVQA | RadImageNet | Natural | Text-only |
|---|---|---|---|---|---|
| Lingshu-7B | 1.0 | 1.1 | 0.6 | 0.8 | 0.5 |
| Qwen3-VL-8B | 1.1 | 1.2 | 0.66 | 0.88 | 0.55 |
| HuatuoGPT-Vision-7B | 1.1 | — | 0.7 | — | 0.5 |

Masking keeps every token in the forward pass, so pruned runs cost as much as `full`; speedups are measured in E9.
Derivation (Lingshu, 512² SLAKE → 324 tokens ✓): prefill ≈ 2 × 6.5 B × ~420 tokens + ViT ≈ 7.2 TFLOP at ~18 TFLOP/s
≈ 0.4 s; decode ≈ 15.2 GB / ~230 GB/s + overhead ≈ 80 ms × 3–5 tokens; +0.1 s preprocessing. RadImageNet image size is
not on its card: at 224² Lingshu sees 64 tokens and keeps **6** at 10% — state this regime in the paper.
`fastv_prellm` ×1.3. Calibrated in Stage 3.

## C5. Storage

| Item | Size | Where | Kept |
|---|---|---|---|
| One model per GPU session | 16.6–17.6 GB | HF cache on session disk, never `/kaggle/working` | No |
| `medprune-data` bundle | ~0.7 GB | private Kaggle dataset / `$MEDPRUNE_DATA` | Yes |
| OmniMedVQA zip (build only) | 10.70 GB | scratch | No |
| RadImageNet benchmark | 0.36 GB | per session, own token | No |
| Results JSONL + diagnostics NPZ | ~150 + ~200 MB | `results/` | Yes (not in git) |

## C6. Outputs

- **Run ID:** `{exp}__{model}__{dataset}__{method}__r{budget%:03d}__s{seed}__{profile}`, e.g.
  `E3__lingshu7b__slake_en__gate_fastv__r010__s0__t4x2`. The profile suffix prevents mixing hardware in one table.
- **`results/runs/<run_id>.jsonl`** (one line per question, flushed): `qid, image_id, dataset, modality, lang,
  answer_type, question, answer, response, correct, open_recall, n_vis, k_keep, grid_hw, keep_idx, bg_frac,
  elapsed_s, n_new_tokens, truncated`.
- **`results/runs/<run_id>.meta.json`:** model id + revision, versions, GPU, profile, dtype, max_pixels, prompt version,
  git hash, timings.
- **`results/diag/<run_id>/part-XXXX.npz`** (`full` and score-based runs, every 200 questions): `scores_l1`,
  `scores_l7`, `scores_l15`, `scores_ropefree_l1`, `vit_norm`, `llm_norm_l1` (fp16), `qid`, offsets.
- **Tables/figures:** only from `02_analyze_results.ipynb` → `results/tables/`, `results/figures/`.

## C7. Statistics

| Quantity | Definition | Uncertainty |
|---|---|---|
| Accuracy | mean `correct` | cluster bootstrap over images, B = 2,000 |
| Paired difference | per-question A − B on identical questions | cluster bootstrap; McNemar exact (secondary) |
| Random | mean over seeds 0–2 | seeds averaged inside each bootstrap resample |
| Image-dependent subset | {q : `full` correct ∧ `text_only` wrong}, per model | same, restricted |
| BRR | (\|kept ∩ bg\|/k)/(\|bg\|/N), images with b ∈ [0.05, 0.95] | bootstrap images |
| Keep regression | logistic: kept ~ spline(row) + spline(col) + bg + z(vit_norm) + z(llm_norm) + dataset FE (S1, `fastv`) | image-clustered SEs |
| Keep-maps | keep probability on 16×16 normalized grid; quadrant concentration = P(kept in quadrant)/area share | bootstrap |
| Static map | rank-normalized `scores_l1` → 32×32 (bilinear) → mean over the other anchor dataset → resampled per image | — |
| High-norm | per-image z = (norm − median)/MAD > 5; rate = P(kept \| outlier)/ρ | bootstrap |
| Dose-response | per-image Δ = mean(correct_Random − correct_FastV); Spearman vs b; deciles; modality-level over ≥ 13 groups | cluster bootstrap |
| Evidence size | SLAKE question → `mask.txt` labels (synonyms: lung → both lungs; tumor → brain tumor/edema; cancer → liver/kidney/lung cancer); area tertiles; report coverage | bootstrap within tertile |

Primary tests = P1–P14; everything else is exploratory.

---

# Part D — Code design

## D1. Principles

One job per file; dependencies flow one way (`data`, `masks` → `selection` → `attention_hook` → `models` →
`run_eval`). Every method = "choose a keep-set, mask dropped visual keys from layer L" — masking with original
positions equals deleting the tokens (proved in tests), so one ~120-line hook covers all 26 methods. The only global
state is `attention_hook.ACTIVE`, set by a context manager. Notebooks and scripts only orchestrate; nothing in
`medprune/` assumes Kaggle (paths come from `env.py`). Python ≥ 3.10, type hints, docstrings with shapes, `logging`.

## D2. Layout

```
cvpr/
├── README.md                    framing + quick start
├── plan.md                      this file
├── mentor_brief.tex             two-page mentor brief (Overleaf)
├── requirements.txt             pinned (transformers==4.57.1); torch comes from the platform
├── .gitignore                   results, data, third_party, caches
├── scripts/
│   └── setup_env.sh             same setup on Kaggle and SSH servers; prints the detected profile
├── configs/
│   ├── runs.yaml                every run: exp, tier, account, model, dataset, method, budget, seed
│   └── bundle_manifest.json     data counts + hashes (committed provenance)
├── medprune/
│   ├── env.py                   compute profile (t4x2 | a100) and paths
│   ├── data.py                  manifests → Sample; MedEvalKit prompts; `python -m medprune.data build`
│   ├── masks.py                 model view, body mask, per-token foreground fraction
│   ├── selection.py             MethodSpec registry + pure NumPy selectors
│   ├── attention_hook.py        PruneState, `medprune_sdpa`, scoring helpers
│   ├── models.py                Qwen-family adapter (Lingshu-7B, Qwen3-VL-8B)
│   ├── huatuo_llava.py          HuatuoGPT-Vision-7B port to transformers 4.57.1
│   ├── scoring.py               MedEvalKit-identical correctness
│   ├── stats.py                 bootstrap, McNemar, BRR, keep-maps, static map, dose-response, regression
│   └── run_eval.py              run_one / run_many / CLI (`python -m medprune.run_eval`)
├── notebooks/
│   ├── 00_build_data_bundle.ipynb   Kaggle CPU wrapper around `medprune.data build`
│   ├── 01_run_gpu.ipynb             Kaggle GPU wrapper around `run_many`
│   └── 02_analyze_results.ipynb     laptop: all tables and figures
├── tests/
│   ├── test_masks.py
│   ├── test_selection.py
│   ├── test_attention_hook.py
│   └── test_scoring.py
├── results/                     runtime output, git-ignored
└── third_party/                 FastV, MedEvalKit, HuatuoGPT-Vision — read-only, git-ignored
```

## D3. File specifications

**`env.py`**
```python
@dataclass(frozen=True)
class Profile:
    name: str                 # "t4x2" | "a100"
    dtype: torch.dtype        # float16 | bfloat16
    device_map: str | dict    # "auto" | {"": 0}
    max_memory: dict | None   # {0: "14GiB", 1: "14GiB"} | None
    data_root: Path; results_root: Path; hf_home: Path
def detect_profile() -> Profile   # MEDPRUNE_PROFILE overrides; else capability ≥ 8.0 and ≥ 40 GB → a100; 2×T4 → t4x2
```
Paths: `MEDPRUNE_DATA`, `MEDPRUNE_RESULTS`, `HF_HOME`; on Kaggle (`KAGGLE_KERNEL_RUN_TYPE` set) defaults are
`/kaggle/input/medprune-data`, `/kaggle/working/results`.

**`data.py`**
```python
@dataclass(frozen=True)
class Sample:
    qid: str; dataset: str; image_id: str; image_path: str; question: str; answer: str
    answer_type: str          # "open" | "closed" | "yesno" | "mc"
    choices: tuple[str, ...] | None; lang: str; modality: str; meta: dict
def load_manifest(bundle_root: Path, dataset: str) -> list[Sample]
def make_prompt(sample: Sample) -> str                         # MedEvalKit formats incl. Chinese
def build_slake(raw_dir, out_dir) -> dict[str, int]
def build_omnimedvqa(zip_path, out_dir, panel_per_modality=150, seed=0) -> dict[str, int]
def build_radimagenet(parquet_path, out_dir) -> dict[str, int]   # in-session only (license)
def build_natural(gqa_dir, pope_path, out_dir, n_each=500, seed=0) -> dict[str, int]
def main() -> None                                             # `python -m medprune.data build --out DIR`
```

**`masks.py`**
```python
def model_view(image, model) -> tuple[Image.Image, tuple | None]   # transformers smart_resize (bicubic) | expand2square→336
def body_mask(img: np.ndarray, pad_box=None) -> np.ndarray         # [H, W] bool, §A5 steps 1–7
def token_foreground_fraction(mask, cell) -> np.ndarray            # [Ht, Wt] in [0, 1]
def background_fraction(fg_frac, tau=0.5) -> float
def slake_oracle_mask(mask_png, label_values) -> np.ndarray
```
Otsu `skimage.filters.threshold_otsu`; components `skimage.measure.label(connectivity=2)`; holes
`scipy.ndimage.binary_fill_holes`; dilation `binary_dilation` with a cell-sized square.

**`selection.py`**
```python
@dataclass(frozen=True)
class MethodSpec:
    name: str; rule: str; score: str | None; score_layer: int | None; mask_from: int | None
    gate: str | None; tau: float = 0.5; bg_quota: float = 0.0; two_pass: bool = False; image: bool = True
METHODS: dict[str, MethodSpec]                 # = §C3
def keep_count(n, ratio) -> int
def rng_for(seed, qid) -> np.random.Generator  # sha256-based; never Python's salted hash()
def select_random(n, k, rng); select_uniform(n, k, allowed=None); select_topk(scores, k, allowed=None)
def select_inverse(allowed, k); select_divprune(feats, k, allowed=None)
def select_static(static_map, rc, grid_hw, k, allowed=None)
def pdrop_schedule(rho, n_layers, boundaries=(2, 9, 16, 23)) -> list[tuple[int, float]]
```
Selectors return sorted unique indices of length exactly k; fill rule when fewer than k are allowed.

**`attention_hook.py`**
```python
@dataclass
class PruneState:
    vis_pos: torch.LongTensor; k: int; spec: MethodSpec
    select_fn: Callable[[np.ndarray], np.ndarray] | None
    keep: np.ndarray | None = None             # preset when mask_from == 0
    stages: list[tuple[int, float]] | None = None
    log_layers: tuple[int, ...] = ()
    logs: dict[str, np.ndarray] = field(default_factory=dict)
ACTIVE: PruneState | None = None
@contextmanager
def pruning(state): ...
def medprune_sdpa(module, query, key, value, attention_mask, **kwargs)   # AttentionInterface.register(...)
def last_token_attention(query, key, scaling, n_rep) -> torch.Tensor     # [Tk]
def capture_pre_rope(modules) -> handles
```
Control flow of `medprune_sdpa`:
```
L = getattr(module, "layer_idx", None); st = ACTIVE
if L is None or st is None: return sdpa_attention_forward(...)        # vision tower / no pruning
prefill = query.shape[2] > 1
if prefill and L in st.log_layers: log last_token_attention(...)[vis_pos]
if prefill and L == st.spec.score_layer and st.keep is None:
    s = ropefree or attention scores over vis_pos; st.keep = st.select_fn(s); st._dropped_pos = vis_pos[not kept]
if st.keep is not None and L >= st.spec.mask_from:
    attention_mask = allow_mask(attention_mask, query, key, st._dropped_pos)
return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)
```
`allow_mask`: boolean `[1, 1, Tq, Tk]`, causal on prefill, all-true on decode, AND incoming mask, dropped columns
False. Passing a mask disables SDPA's `is_causal` ✓ (`sdpa_attention.py` L78–81), hence explicit causality; SDPA then
repeats KV heads itself ✓ (L66–68). PDrop re-ranks survivors at each stage boundary.

**`models.py`**
```python
@dataclass
class PreparedInputs:
    model_kwargs: dict; vis_pos: torch.LongTensor; grid_hw: tuple[int, int]
    rc: np.ndarray            # (row, col) per token from rope position ids (h − β, w − β)
    view_size: tuple[int, int]
class VLMAdapter(Protocol):
    name: str; num_layers: int; cell_px: int; n_heads: int; n_kv_heads: int
    def prepare(self, sample, image) -> PreparedInputs
    def generate(self, inputs, max_new_tokens) -> tuple[str, int]
    def visual_embeddings(self, inputs) -> torch.Tensor
    def pre_rope_modules(self, layer) -> tuple[nn.Module, nn.Module]
def load_adapter(model: str, profile: Profile, vision_fp32: bool = False) -> VLMAdapter
```

| | Lingshu-7B | Qwen3-VL-8B |
|---|---|---|
| Class | `Qwen2_5_VLForConditionalGeneration` ✓ | `Qwen3VLForConditionalGeneration` ✓ |
| LLM | 28 layers, 28 heads, 4 KV, hidden 3,584 ✓ | 36 layers, 32 heads, 8 KV, head_dim 128, hidden 4,096 ✓ |
| Rotary | M-RoPE [16, 24, 24] ✓ | interleaved M-RoPE [24, 20, 20], θ 5e6 ✓ |
| Vision | 32-layer windowed ViT, patch 14, merge 2 → cell 28 ✓ | 27-layer ViT, patch 16, merge 2 → cell 32; DeepStack (ViT 8/16/24 → LLM layers 0–2, added after each layer) ✓ |
| Pre-RoPE q/k | `q_proj`, `k_proj` | `q_norm`, `k_norm` ✓ |
| `use_cache` in config | false ✓ → force True | true ✓ |

Load with `attn_implementation="sdpa"`, then set only `model.config.text_config._attn_implementation =
"medprune_sdpa"`; assert vision stays `"sdpa"`. Assert `grid_hw == image_grid_thw[1:] // merge ==` precomputed mask grid.

**`huatuo_llava.py`.** `Qwen2ForCausalLM` from the checkpoint's Qwen2 fields (28 layers, 4 KV, θ 1e6 ✓) with
`medprune_sdpa`; `CLIPVisionModel` from `vit/clip_vit_large_patch14_336`, layer −2, drop CLS → 576 × 1,024 ✓;
projector Linear(1024, 3584) → GELU → Linear(3584, 3584) ✓; prompt `"<|user|>\n<image>\n{prompt}\n<|assistant|>\n"`,
576 embeddings spliced at `<image>`; generate from `inputs_embeds`; `rc` raster 24 × 24.

**`scoring.py`.** `parse_response`, `judge_yesno`, `judge_closed`, `judge_open` (em, recall), `judge_mc`, `is_correct`
— line-by-line port of `third_party/MedEvalKit/utils/utils.py` L95–226.

**`stats.py`.** `cluster_bootstrap_ci`, `paired_delta_ci`, `mcnemar_exact`, `background_retention_ratio`, `keep_map`,
`static_map_from_scores`, `dose_response`, `keep_regression` (statsmodels, clustered SEs).

**`run_eval.py`.**
```python
@dataclass
class RunSpec:
    exp: str; model: str; dataset: str; method: str; budget: float = 0.10; seed: int = 0
    def run_id(self, profile: str) -> str
def run_one(adapter, spec, bundle, out, profile, limit=None, max_new_tokens=64) -> dict
def run_many(specs, bundle, out, max_hours=6.0) -> list[dict]   # one model load per model; resumes; time budget
def main() -> None   # python -m medprune.run_eval --exp E3 --model qwen3vl8b [--account B] [--max-hours 5]
```

## D4. Notebooks and scripts

| File | Runs on | Does |
|---|---|---|
| `scripts/setup_env.sh` | Kaggle and SSH | pip install requirements; print torch/CUDA/GPU/profile |
| `00_build_data_bundle.ipynb` | Kaggle CPU | `python -m medprune.data build --out /kaggle/working/medprune-data` → private dataset |
| `01_run_gpu.ipynb` | Kaggle GPU (commit mode) | env check → `setup_env.sh` → clone repo → RadImageNet manifests (in-session) → 5-question smoke test → `run_many` |
| `02_analyze_results.ipynb` | Laptop | all tables, figures, statistics |

## D5. Tests (CPU, < 1 minute)

| File | Guarantees |
|---|---|
| `test_masks.py` | ellipse area within 2%; lungs kept by hole filling; table removed; white-background polarity; padding forced to background; grids for cells 28/32/14 |
| `test_selection.py` | `keep_count`; deterministic Random across processes; exactly k sorted unique indices; gate respected with fill rule; DivPrune distinct clusters; PDrop a = 0.2019 at ρ = 0.10 |
| `test_attention_hook.py` | tiny `Qwen2ForCausalLM` and `Qwen3ForCausalLM`, fp32: no state ≡ vanilla; keep-all ≡ vanilla; mask from 0 ≡ deleting tokens with original positions; mask from K ≡ FastV-style index-select reference; greedy decode identical (cache path); `last_token_attention` ≡ eager probabilities; RoPE-free ≡ q·kᵀ from captured projections |
| `test_scoring.py` | fixed yes/no, closed, open, MC cases match MedEvalKit verdicts |

## D6. Pitfalls and guards

| Pitfall | Guard |
|---|---|
| Hook applied to the vision tower | Only `text_config` set; pass-through without `layer_idx`; assert |
| Mask without causality | `allow_mask` builds causal; physical-reference test |
| FastV off-by-one | score K−1, mask K |
| GQA ignored in scoring | `repeat_kv`, n_rep 7 (Lingshu, HuatuoGPT) / 4 (Qwen3-VL); test |
| Mask missing on decode steps | applied whenever `keep` set and L ≥ `mask_from`; decode test |
| `use_cache: false` in configs | `generate(use_cache=True)` |
| Mask grid ≠ model grid | assert per question |
| Salted `hash()` seeds | sha256 `rng_for` |
| fp16 overflow | `isfinite` check in smoke test; vision fp32 fallback |
| Batch > 1 | assert batch size 1 |
| Gate-input on Qwen3-VL (DeepStack misalignment) | Gate-input only on Lingshu |
| Mixing T4 and A100 runs in one table | profile in run ID; analysis asserts one profile per table |

Reading order: `selection.py` → `attention_hook.py` → `models.py` → `run_eval.py` → `masks.py`, `scoring.py`,
`stats.py` → `huatuo_llava.py` → `tests/test_attention_hook.py`.

---

# Part E — Compute backends: Kaggle T4×2 now, SSH A100 later

The package never assumes Kaggle; `env.py` picks a profile and paths, and the same CLI runs everywhere.

| | `t4x2` (Kaggle, default now) | `a100` (SSH, if the mentor provides one) |
|---|---|---|
| GPUs | 2 × T4 16 GB, sm_75, no bf16 | 1 × A100 40/80 GB, sm_80 |
| dtype | fp16 (vision fp32 if Stage 3 requires) | bf16 (checkpoints' native dtype) |
| Placement | `device_map="auto"`, 14 GiB per GPU | `device_map={"": 0}` |
| Attention | `medprune_sdpa` (text) / `sdpa` (vision) | same |
| Speed | planning rates (§C4) | ≈ 4–5× faster (estimate; calibrate with the Stage 3 timing run) → 66 T4-h ≈ 14–17 h |
| Paths | `/kaggle/input/medprune-data`, `/kaggle/working/results` | `$MEDPRUNE_DATA` (default `~/medprune-data`), `$MEDPRUNE_RESULTS` (default `./results`) |
| Launch | `01_run_gpu.ipynb`, commit mode, ≤ 12 h sessions | `tmux` + CLI |
| Results back | Kaggle output download | `rsync` |

**SSH quick start (A100):**
```bash
git clone git@github.com:pushkar1887/cvpr.git && cd cvpr
python -m venv .venv && source .venv/bin/activate      # install a CUDA-matched torch first if the server lacks one
bash scripts/setup_env.sh
export MEDPRUNE_PROFILE=a100 MEDPRUNE_DATA=~/medprune-data HF_TOKEN=<token with RadImageNet access>
kaggle datasets download -d <kaggle-user>/medprune-data -p ~/medprune-data --unzip   # or: python -m medprune.data build --out ~/medprune-data
tmux new -s e3q "python -m medprune.run_eval --exp E3 --model qwen3vl8b --max-hours 5"
```
On the laptop: `rsync -avz <server>:cvpr/results/ results/`.

**Parity rules.**
1. One profile per model: all runs of a model that enter the same table use the same profile (bf16 vs fp16 shifts
   Full accuracy by ~0.5–1 point).
2. If the A100 arrives mid-project, move a *whole* model to it and rerun its grid there (Qwen3-VL P0 ≈ 3.5–4 A100-h).
3. Never mix profiles in one table; the profile suffix in run IDs lets `02_analyze_results.ipynb` assert this.

**Kaggle operations.** Accelerator GPU T4×2; internet on (phone-verified); secrets `HF_TOKEN`, `GITHUB_TOKEN`;
attach `medprune-data`. Long runs via "Save Version → Save & Run All (Commit)" (12-hour limit ✓); `run_many` stops
scheduling when elapsed + next estimate > 6 h; runs resume from their JSONL. Weekly GPU quota is not published
(reports 15–30 h) — read each account's quota page. Model download costs 3–5 min of quota per session; optionally
mirror a model into a private dataset from a CPU session. `/kaggle/working` (20 GB ✓) is for results only.

---

# Appendix — Sources

**Reference code (clone into `third_party/`, git-ignored):**
```bash
git clone --depth 1 https://github.com/pkunlp-icler/FastV third_party/FastV
git clone --depth 1 https://github.com/alibaba-damo-academy/MedEvalKit third_party/MedEvalKit
git clone --depth 1 https://github.com/FreedomIntelligence/HuatuoGPT-Vision third_party/HuatuoGPT-Vision
```

**Models:** [Lingshu-7B](https://huggingface.co/lingshu-medical-mllm/Lingshu-7B) ·
[Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) ·
[HuatuoGPT-Vision-7B](https://huggingface.co/FreedomIntelligence/HuatuoGPT-Vision-7B)

**Data:** [SLAKE](https://huggingface.co/datasets/BoKelvin/SLAKE) ·
[OmniMedVQA](https://huggingface.co/datasets/foreverbeliever/OmniMedVQA) ·
[RadImageNet-VQA](https://huggingface.co/datasets/raidium/RadImageNet-VQA) (arXiv 2512.17396) ·
[GQA](https://huggingface.co/datasets/lmms-lab/GQA) · [POPE](https://huggingface.co/datasets/lmms-lab/POPE)

**Papers:** FastV arXiv 2403.06764 · FEATHER arXiv 2412.13180 · PoRe arXiv 2508.17807 · Wang et al. arXiv 2512.07580 ·
Registers arXiv 2309.16588 · test-time registers arXiv 2506.08010 · VTC-Bench arXiv 2510.07143 · MedSTP arXiv
2610.22751 · ViToS arXiv 2606.31599 · Photon arXiv 2603.25155 · MedARC arXiv 2607.26554 · EasyLens arXiv 2606.06379 ·
[CVPR 2027 CFP](https://cvpr.thecvf.com/Conferences/2027/CallForPapers)
