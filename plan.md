# Plan — Pruning the Air

Single source of truth for the project: research (Part A), execution by stages (Part B), protocol (Part C), code
design (Part D), compute backends incl. SSH A100 (Part E), agent handbook and task queue (Part F), work log (Part G,
the last section). Version 2 (merged), 2026-11-02.

**AI agents (Codex, Gemini, Claude): read Part F first, then the newest entries of the Work log (Part G).**

Conventions: ✓ = verified on 2026-11-02 against the named source. Stages run in order; a stage starts only when the
previous stage's **exit gate** is met (parallel work is marked). Tick boxes on GitHub as work completes. Predictions in
§A6 are frozen as of 2026-11-02 — record outcomes, never edit predictions.

## Stage tracker

| Stage | Work | Where | GPU-h | Dates (IST) | Exit gate |
|---|---|---|---|---|---|
| 0 | Accounts, access, repo | Laptop, Hugging Face, Kaggle, GitHub | 0 | Nov 2 | Kaggle accounts ready; HF secret `cvpr`; RadImageNet access — **passed Nov 2** |
| 1 | Code and tests | Laptop CPU | 0 | Nov 2–3 | **84 tests passed Nov 3**, incl. identity/resume and actual RIN template regressions |
| 2 | Data bundle | Kaggle CPU (or SSH) | 0 | Nov 2 | **Built and rebuilt with revised masks**; counts in committed manifest and Work log |
| 3 | Validation and calibration (E0) | Kaggle GPU, A (+0.5 h on B) | 3.0 | Nov 2–3 | 50-item component checks passed on both models; **full SLAKE baseline target unresolved** (77.19%, different published judge) |
| 4 | Kill test (E1) + diagnostics (E2) | GPU A + CPU | 3.3 + repairs | Nov 3 | **Primary decisions complete: K1 GO; K2 H1 alive**; registered truncation retry verified; causal controls remain |
| 5 | Main grid (E3) + modality panel (E4) | GPU A, B | 33.1 | Nov 3–5 | **First bounded Qwen E3 block RUNNING (B v2, Nov 3)**; broader grid and scoring-comparability gate remain open |
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
2. Polarity: the border ring's median is on the dark or the bright half of the [1st, 99th]-percentile range.
3. Background B = pixels within 20% of the range from that end **and** absolutely dark (Y ≤ 80) or bright (Y ≥ 225),
   restricted to the 4-connected components that touch the image edge. Foreground F = not B.
4. Keep components (8-connectivity) ≥ 10% of the largest → removes table, text, markers.
5. Fill holes → keeps lungs, airway, bowel gas (they never touch the edge anyway).
6. Dilate by one cell.
7. Force known padding (HuatuoGPT) to background.
8. m_i = mean of F over cell i; gate g_i = [m_i ≥ τ], τ = 0.5.

Guard (no-op where there is no acquisition background): the mask is all-foreground when B covers less than 60% of the
5% border ring — dermoscopy, pathology crops, X-rays whose body fills the frame.

*Revision 2026-11-02 (before any GPU run).* The first version (Otsu + centre-vs-border contrast + background-flatness
guard) left 48% of SLAKE CT and 65% of OmniMedVQA CT ungated: a small head in a large field has an all-air centre crop
(contrast 0), and Otsu puts bowel gas and lung on the background side (not flat). The border-connected definition gates
100% of SLAKE CT/MRI and 92% of OmniMedVQA CT. The absolute limits stop light skin in low-contrast dermoscopy (Y ≈
185–210) from counting as bright background (dermoscopy 9% → 0% gated). Regression tests: `tests/test_masks.py`.

Selection: score-based → top-k among g_i = 1, fill from best background if short; score-free → k evenly spaced (Gate+
Uniform) or max–min diversity (Gate+DivPrune) among g_i = 1. **Budget fairness:** k = round(ρN), N counts all visual
tokens. **Variants:** Gate-score (selection only; causal test of H1); Gate-input (drop background before the LLM; free
compression; latency only). Cost < 5 ms per 512² image, no parameters. Ablations: τ ∈ {0.25, 0.5, 0.75}; background
quota 0/3%; oracle gate (SLAKE masks) as upper bound; inverse gate (background only) as lower bound.

## A6. Pre-registered predictions (frozen 2026-11-02) and decision rules

| # | Experiment | Prediction | Confidence | Outcome |
|---|---|---|---|---|
| P1 | E1, RadImageNet anatomy MC, Lingshu, 10% | Standard FastV < Random@L2 by ≥ 8; pre-LLM FastV < Random@L0 by ≥ 10 (MedSTP: 37.5 vs 55.6) | 65% / 80% | Supported on Lingshu seed 0: 38.1 vs 79.3, Δ −41.2 [−45.0, −37.6]; pre-LLM 37.4 vs 79.2, Δ −41.8 [−45.5, −38.2]. Corrected per-image Random; Nov 3. |
| P2 | E1, SLAKE-en | Full 81–85; FastV − Random ≤ −4 on CT+MRI (66 images), ≥ 0 on X-ray (30) | 55% | Mixed: Full 77.19 misses target; CT/MRI Δ −18.14 [−22.41, −14.00] supports forecast; X-ray Δ −1.94 [−4.93, +0.87] has the wrong point-estimate sign but no significant difference. |
| P3 | E2 | BRR(FastV) ≥ 1.3 on CT/MRI | 65% | Supported for heuristic-mask background: RIN 2.32 [2.25, 2.39], coverage 874/1,000; SLAKE 1.67 [1.57, 1.76], 700/700 CT/MRI questions. Association, not causal proof. |
| P4 | E2 | Keep-mass ≥ 1.5× uniform in bottom-right (Lingshu/Qwen3-VL), bottom rows (HuatuoGPT) | 60% | Contradicted on Lingshu: bottom-right concentration 0.018× uniform (RIN), 0.427× (SLAKE CT/MRI); strongest RIN concentration is top-left (2.52×). Other models untested. |
| P5 | E2/E6 | High-norm artifacts over-selected ≥ 2× | 35% | Norm-threshold proxy supports over-selection: 2.62× RIN and 5.07× SLAKE. These tokens are not independently annotated artifacts; E6 ablation and causal interpretation remain open. |
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
- **Table 2** BRR, quadrant concentration, regression coefficients, high-norm rate, cross-question keep-set Jaccard, static-map accuracy/Jaccard, inverse/oracle gate.
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

This Part is the runbook. It is written so that a person or an agent can open it at any stage and know what to do,
why, with which command, what files appear, what result to expect, and what decision the result feeds.

## B0. How to read this Part

**Every stage** has the same layout:

1. **In one sentence** — what the stage achieves.
2. **Status** — DONE, NEXT, or LATER.
3. **Before you start** — what must already be true.
4. **Steps** — numbered actions, each with the exact command and what you should see.
5. **Experiment cards** — one per experiment (E0–E9), always with the same fields: *Question · Why it matters · Setup ·
   Methods · Command · Time · Outputs · Expected result · How to read it · Decision · Checks*.
6. **Done when** — the exit gate. Do not start the next stage before it is met.
7. **If something goes wrong** — symptom → likely cause → action.

Agents: use commands exactly as written; settings fixed in Part C are never improvised.

### B0.1 Glossary

| Term | Meaning in this project |
|---|---|
| Visual tokens | The image is cut into a grid of patches; each patch becomes one token the language model reads. Lingshu turns a 512×512 image into 18×18 = 324 tokens; Qwen3-VL into 16×16 = 256; HuatuoGPT always 24×24 = 576. |
| Pruning, keep-set | Keeping only some visual tokens to save compute. The keep-set is the list of kept tokens. |
| Budget ρ | Share of visual tokens kept. ρ = 10% keeps 32 of Lingshu's 324 tokens. Main setting: 10%. |
| FastV | The method under study. At an early layer it measures how much the last prompt token attends to each visual token and keeps the most-attended ones. |
| Score layer, mask from | FastV reads attention at layer K−1 = 1 ("score layer") and hides dropped tokens from layer K = 2 on ("mask from 2"). "Mask from 0" means the tokens are removed before the language model sees them — how MedSTP pruned. |
| Random@L2, Random@L0 | A random keep-set applied at FastV's point (layer 2) or before the LLM (layer 0). **The fair comparison for FastV is Random@L2.** |
| Positional bias | Rotary position encoding (RoPE) makes attention favour some grid positions regardless of content. Predicted: bottom-right corner for M-RoPE (Lingshu, Qwen3-VL), bottom rows for 1D RoPE (HuatuoGPT). On axial scans those positions are air. |
| RoPE-free score | FastV's score computed without position encoding (FEATHER's criterion). Removes the positional bias only. |
| Foreground gate | Our fix: a body mask marks tokens outside the body as background; the pruner may then choose only body tokens, within the same budget. |
| Background fraction b | Share of an image's tokens that are background. CT/MRI slices ≈ 0.3–0.6; dermoscopy ≈ 0. |
| BRR | Background retention ratio = (share of kept tokens that are background) ÷ (share of all tokens that are background). Random ≈ 1. Above 1 means the method prefers background. |
| Full, text-only | No pruning (the ceiling) / no image at all (what the model guesses from the question alone). |
| Image-dependent subset | Questions the model answers correctly with the image and wrongly without it. Only these really test vision. |
| Cluster bootstrap CI | 95% interval that resamples images, not questions, because questions about one image are not independent. |
| Run, run ID | One model × one dataset × one method × one budget × one seed. ID: `{model}__{dataset}__{method}__r{budget%}__s{seed}__{profile}`. |
| Group | A named set of datasets run together: K, S1, S2, S3, P, N (B0.2). |
| Profile | Hardware setting in the run ID: `t4x2` (Kaggle) or `a100`. |
| Role A/B/C | Which Kaggle account does the work (Stage 0 table): A = Lingshu, B = Qwen3-VL, C = supplementary + overflow. |
| P#, K# | A pre-registered prediction / a decision rule (§A6). |

### B0.2 Datasets and groups

| Dataset ID | What it is | Questions | Images | Lives in |
|---|---|---|---|---|
| `slake_en` | SLAKE test, English (CT, MRI, X-ray) | 1,061 | 96 (CT 40, MRI 26, X-ray 30) | shared bundle |
| `slake_zh` | SLAKE test, Chinese (same images) | 1,033 | 96 | shared bundle |
| `omni_anat` | OmniMedVQA open-access, "Anatomy Identification", multiple choice | ~3,329 (verify) | verify | shared bundle |
| `omni_panel` | OmniMedVQA, 8 modalities × 150 (CT, MRI, X-ray, ultrasound, OCT, fundus, dermoscopy, microscopy) | 1,200 | ≥ 400 | shared bundle |
| `rin_anat_mc` | RadImageNet-VQA anatomy multiple choice, CT/MRI, one per image | ~1,000 | 1,000 | per session, `/tmp/medprune-rin` |
| `rin_abn_closed` | RadImageNet-VQA abnormality yes/no, same images | ~1,000 | same 1,000 | per session |
| `natural` | 500 GQA + 500 POPE photographs | 1,000 | ~900 | shared bundle |

| Group | Datasets | Questions | Hours per run: Lingshu / Qwen3-VL |
|---|---|---|---|
| K | `slake_en` + `rin_anat_mc` | 2,061 | 0.46 / 0.51 |
| S1 | K + `rin_abn_closed` | 3,061 | 0.63 / 0.69 |
| S2 | `slake_zh` | 1,033 | 0.29 / 0.32 |
| S3 | `omni_anat` | 3,329 | 1.02 / 1.12 |
| P | `omni_panel` | 1,200 | 0.37 / 0.40 |
| N | `natural` | 1,000 | 0.22 / 0.24 |

Hours use the planning rates of §C4 (pruned runs cost as much as `full`, because masking keeps every token in the
forward pass). Stage 3 replaces them with measured rates.

### B0.3 Paths, commands and Kaggle sessions

| Shorthand | Kaggle | Laptop / SSH |
|---|---|---|
| `$DATA` (shared bundle) | `/kaggle/input/medprune-data` — confirm with `ls /kaggle/input` | `~/medprune-data` |
| `$RIN` (RadImageNet, per session) | `/tmp/medprune-rin` | `/tmp/medprune-rin` |
| `$RES` (results) | `/kaggle/working/results` | `D:\cvpr\results` (laptop) |

```bash
# See what a block contains, how much is done and how long the rest takes:
python -m medprune.run_eval --list --exp E1 --model lingshu7b --bundle $DATA --bundle $RIN --out $RES
# Smoke test (5 questions per run):
python -m medprune.run_eval --exp E1 --model lingshu7b --bundle $DATA --bundle $RIN --out $RES --limit 5
# Real run; stops cleanly before exceeding the session budget:
python -m medprune.run_eval --exp E1 --model lingshu7b --bundle $DATA --bundle $RIN --out $RES --max-hours 6
```

**Gates travel with the results:** the first run of each model × dataset copies its mask file to
`$RES/gates/<dataset>__<model>.npz`, so the laptop analysis has RadImageNet's per-token background too (derived
foreground fractions, not images). **Safety stop:** a real run aborts on its first question if the pruning attention
was never called.

**Every run writes** `$RES/runs/<run_id>.jsonl` (one line per question: answer, correctness, kept tokens, background
fraction, timing), `$RES/runs/<run_id>.meta.json` (versions, settings, accuracy) and, for `full` and score-based
methods, `$RES/diag/<run_id>/part-XXXX.npz` (attention scores and token norms). A crash loses at most one question;
rerunning the same command resumes.

**Kaggle keeps nothing between sessions.** Each session starts with an empty `/kaggle/working`. To continue a block in a
new session: (1) add the previous session's notebook output as an input (*Add data → Your Work → the notebook →
Output*); (2) notebook `01_run_gpu` copies it into `$RES` first (parameter `RESTORE_FROM`); (3) run the same command —
finished questions are skipped. After every session, download the output and merge it into `D:\cvpr\results\` on the
laptop; run IDs are unique, so folders merge without conflicts.

### B0.4 Experiment map

| Exp | Question in plain words | Stage | Models | Data | Run IDs (new) | GPU-h | Role | Feeds |
|---|---|---|---|---|---|---|---|---|
| E0 | Is the pipeline correct, and how fast is it? | 3 | Lingshu, Qwen3-VL | `slake_en` | 2 | 3.0 | A (+B) | Stage 3 gate |
| E1 | Is standard FastV worse than random in our pipeline, and does it prefer background? | 4 | Lingshu | K | 14 (13) | 3.3 | A | K1, K2 |
| E2 | Why: position, background or artifact tokens? (analysis only) | 4–6 | — | logs | 0 | 0 | laptop | Table 2, Fig. 3 |
| E3 | Does it hold on both models and all three benchmarks, and does the gate fix it? | 5 | both | S1, S2, S3 | 45 + 45 | 28.9 | A, B | Table 1 |
| E4 | Does the failure grow with the amount of background? | 5 | both | P | 6 + 6 | 4.2 | A, B | Fig. 2 |
| E5 | Does it hold at other budgets? | 6 | both | K | 20 + 12 | 7.7 | A, B | Fig. 4 |
| E6 | Mechanism controls and gate ablations | 6–7 | both | S1, K, `slake_en` | 37 + 10 | 10.4 | A, B, C | Tables 2–3 |
| E7 | Does FastV fail on ordinary photographs too? | 6 | both | N | 10 | 2.5 | B | specificity |
| E8 | Does a 1D-RoPE model show the predicted bottom-row bias? | 7 | HuatuoGPT | S1 | 21 | 5.7 | C | Fig. 3, H2 |
| E9 | How much compute does the gate really save? | 7 | Lingshu | 200 × `slake_en` | — | 0.5 | C | Table 4 |

Total 66.2 GPU-h (≈ 79 h with 20% contingency). After each stage: fill the Outcome column of §A6, tick the boxes, add
a Work log entry (Part G).

---

## Stage 0 — Accounts, access, repository

**In one sentence:** every account can download data and models and run notebooks. **Status: DONE (Nov 2).**

What exists: public repo `pushkar1887/cvpr` (the user is the only writer); laptop venv `D:\cvpr\.venv`; three Kaggle
accounts mapped to roles; Hugging Face token in the Kaggle secret `cvpr`; RadImageNet-VQA access granted.

**Team facts.** Pushkar runs every experiment and is the only GitHub writer. OpenReview and collaborators are handled by
the user and not tracked here. More Kaggle accounts are available if quota runs short.

| Role | Work | Planned GPU-h | Account | Quota on Nov 2 |
|---|---|---|---|---|
| A | Lingshu-7B: E0, E1, E3, E4, E5, E6-P1 | 29.6 | Kaggle-2 | 30 h |
| B | Qwen3-VL-8B: E3, E4, E5, E6, E7 | 26.6 | Kaggle-3 | 30 h |
| C | E6-P2, E8, E9, overflow from A/B | 10.0 | Kaggle-1 | 21 h 50 min left |

A is tight (29.6 of 30 h): anything beyond moves to C (~11.8 h spare). Quotas reset weekly, so Stages 6–7 get a fresh
allowance.

**Secrets.** In notebooks: `from kaggle_secrets import UserSecretsClient` then
`os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("cvpr")`. The repo is public: clone without a token.

**Checks**
- [x] Repo on GitHub (public by decision); the user is the only writer.
- [x] Laptop environment: torch 2.14.1 (CPU), transformers 4.57.1; references in `third_party/`.
- [x] Quotas recorded: Kaggle-1 21.8 h, Kaggle-2 30 h, Kaggle-3 30 h.
- [x] Hugging Face token stored as Kaggle secret `cvpr`.
- [x] RadImageNet-VQA terms accepted on that Hugging Face account (user, Nov 2).
- [ ] Internet switched on in each Kaggle notebook (Settings → Internet) — check when creating the notebooks.

---

## Stage 1 — Code and tests

**In one sentence:** the `medprune` package exists and its tests prove the pruning is implemented exactly.
**Status: DONE (Nov 2).**

What exists (details in Part D):

| File | Job |
|---|---|
| `medprune/attention_hook.py` | One attention function that runs every method: scores at layer K−1, hides dropped tokens from layer K |
| `medprune/selection.py` | The 25 methods (§C3) and the rules that pick tokens |
| `medprune/masks.py` | Body masks, per-token foreground fraction, the guard for images without background |
| `medprune/scoring.py` | Correctness exactly as MedEvalKit |
| `medprune/stats.py` | Image-clustered confidence intervals, BRR, keep-maps, static maps, dose-response |
| `medprune/data.py` | Dataset manifests and builders (`python -m medprune.data` with `build`, `masks`, `summary`) |
| `medprune/models.py` | Lingshu-7B and Qwen3-VL-8B adapter |
| `medprune/run_eval.py` | Runs, resume, `--list` progress, `--dry-run`; copies gates into `results/gates/`; stops if the hook never fires |
| `medprune/checks.py` | Stage 3 checks (`python -m medprune.checks`) |
| `medprune/analysis.py` | Stage 4 tables, keep-maps and the K1/K2 decisions (laptop) |
| `notebooks/00_build_data_bundle.ipynb`, `01_run_gpu.ipynb`, `02_analyze_results.ipynb` | Stage 2; Stages 3–7; analysis |
| `configs/runs.yaml` | Every experiment's runs |

Run the tests on the laptop: `D:\cvpr\.venv\Scripts\python -m pytest -q` → **69 passed** (about 10 s).

**Checks**
- [x] All tests pass, including masking ≡ deleting tokens, masking ≡ FastV's physical removal, cached decoding
      identical, logged scores = eager attention, RoPE-free head mapping.
- [x] Scoring and prompts equal MedEvalKit's on fixed cases.
- [x] `--list` expands E1 to 14 runs; resume verified with `--dry-run`.

---

## Stage 2 — Data bundle

**In one sentence:** freeze the public datasets as question lists, images and background masks in private notebook
outputs built on each role account. **Status: DONE (Nov 2, revised masks).** The original shared-dataset procedure
was superseded by same-owner `kernel_sources` inputs (Work log, Nov 2); use those outputs with notebook 01's bundle
auto-discovery. The committed `configs/bundle_manifest.json` records the actual counts.

**Before you start**
- [x] Notebook `00_build_data_bundle.ipynb` exists (task T1, Part F).
- [x] Kaggle notebook settings: Accelerator *None* (CPU — costs no GPU quota), Internet *on*.

**Steps**

| # | Action | Command | You should see |
|---|---|---|---|
| 1 | Set the token | `os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("cvpr")` | no error |
| 2 | Get the code | `!git clone https://github.com/pushkar1887/cvpr && cd cvpr && bash scripts/setup_env.sh` | `transformers 4.57.1`, `profile: unsupported` (CPU is fine here) |
| 3 | Check scratch space | `!df -h /tmp /kaggle/working` | ≥ 25 GB free on `/tmp` (the OmniMedVQA zip is 10.7 GB and must not go to `/kaggle/working`) |
| 4 | Build questions and images | `!python -m medprune.data build --out /kaggle/working/medprune-data --work /tmp/medprune-work --skip radimagenet` | printed counts: `slake_en 1061`, `slake_zh 1033`, `omni_anat ~3329`, `omni_panel 1200`, `natural 1000`; OmniMedVQA's modality list |
| 5 | Build masks | `!python -m medprune.data masks --bundle /kaggle/working/medprune-data` | one line per dataset × model |
| 6 | Summarize | `!python -m medprune.data summary --bundle /kaggle/working/medprune-data` | `bundle_manifest.json` with counts and median background fraction per modality |
| 7 | Look at masks | display 3 images per modality from `masks/overlays/` | background shaded red outside the body only |
| 8 | Clean up | `!rm -rf /tmp/medprune-work` | — |
| 9 | Save and attach | Save Version on each role; attach that role's completed builder notebook via `kernel_sources` | bundle discoverable inside notebook 01 |
| 10 | Record | copy `bundle_manifest.json` to `configs/` in the repo; commit; Work log entry | — |

Time: about 30–45 minutes of CPU, no GPU quota. Size: about 0.7 GB.

**Outputs (inside the bundle)**
- `manifests/{slake_en,slake_zh,omni_anat,omni_panel,natural}.jsonl` — one question per line (§D3 `Sample`).
- `images/…` — only the images the questions use.
- `masks/<dataset>__<model>.npz` — each image's token grid and foreground fraction per token, for each of the three
  models; `masks/slake_*__<model>__oracle.npz` from SLAKE's organ masks.
- `masks/overlays/<modality>/*.png`, `bundle_manifest.json`.

**Checks**
- [x] Counts: `slake_en` 1,061 with 96 images (CT 40, MRI 26, X-ray 30); `slake_zh` 1,033; `omni_panel` 1,200;
      `natural` 1,000; `omni_anat` 3,329 sampled from 16,448 qualifying open-access questions.
- [x] Median background fractions recorded in `configs/bundle_manifest.json`; actual zero-background X-rays and
      black-surround dermoscopy differ from early guesses. Do not replace observed values with expected ranges.
- [x] Qualitative overlay review and gate revision reported by Claude on Nov 2; known grey-surround misses retained.
      These heuristic masks are not validated clinical segmentations.
- [x] Role-specific bundles built and attached as same-owner notebook outputs; no shared dataset required.

**Done when** all checks are ticked. The bundle is then frozen: any change means a new version and re-running the runs
that used it.

**If something goes wrong**

| Symptom | Likely cause | Action |
|---|---|---|
| `omni_anat` ≠ 3,329 | MedSTP used another filter | keep our rule; record both numbers; mention in the paper |
| Lungs shaded as background | body touches the image border, so the lungs are not enclosed | inspect; if common, close the mask with a 1-cell closing before hole filling (code change + test + log) |
| Disk full | zip downloaded to `/kaggle/working` | use `--work /tmp/medprune-work` |
| Ultrasound fan mostly background | low-contrast speckle | expected; report ultrasound separately |

---

## Stage 3 — Validation and calibration (E0)

**In one sentence:** before any experiment, prove the GPU pipeline gives sound numbers and measure how fast it is.
**Status: COMPONENT CHECKS PASSED; FULL-SET BASELINE TARGET UNRESOLVED.** Claude's 50-question E0 reports passed on
both models (Work log, Nov 2). The subsequent 1,061-question Lingshu Full run scores 77.19% with normalized exact
match, below the registered 81–85% forecast. Published Lingshu/MedSTP results use an LLM judge. The 50-item result
must not be treated as proof that full-set published-baseline comparability has passed; resolve this scoring
contract before claiming a reproduction or committing the large Stage 5 grid. E1 has already run and its within-
pipeline paired comparisons can be reported with this limitation.

**Before you start**
- [ ] Stage 2 done; `medprune-data` attached to the notebook.
- [x] Notebook `01_run_gpu.ipynb` and `medprune/checks.py` exist (task T2).
- [ ] Kaggle settings: Accelerator *GPU T4 ×2*, Internet *on*.

### E0 — Is the pipeline correct, and how fast is it?

| Field | Value |
|---|---|
| Question | Does the pipeline reproduce published accuracy, stay numerically stable in fp16 on a T4, and leave outputs unchanged when it only logs? How many seconds per question? |
| Why it matters | Three silent failures would invalidate every later number: fp16 overflow (the checkpoints are bf16, the T4 has no bf16), logging or the hook changing outputs, and prompts/scoring that miss the published accuracy. The whole schedule depends on the speed. |
| Setup | Lingshu-7B on role A (`slake_en`, 1,061 questions); Qwen3-VL-8B on role B (50 questions). |
| Methods | `full` only (plus fastv at 50% and 10% inside the checks). |
| Time | A: ~2.5 h (including model download and debugging); B: ~0.5 h. |

**Commands (role A; role B is the same with `--model qwen3vl8b` and only steps 1–2)**
```bash
python -m medprune.run_eval --exp E0 --model lingshu7b --bundle $DATA --out $RES --limit 50   # 1. 50-question run
python -m medprune.checks --model lingshu7b --bundle $DATA --out $RES --n 50                 # 2. E0b-E0d checks (T2)
python -m medprune.run_eval --exp E0 --model lingshu7b --bundle $DATA --out $RES             # 3. all 1,061 (resumes)
```

**What each check means**

| ID | Check | Pass when |
|---|---|---|
| E0a | Environment: `nvidia-smi`, `df -h`, `setup_env.sh` | 2 × T4, ≥ 40 GB temp disk, `profile: t4x2` |
| E0b | fp16 stability: `full` in fp16 vs fp16 with the vision tower in fp32 | ≥ 48/50 identical answers, all logits finite; otherwise use `--vision-fp32` for every run of that model |
| E0c | Logging is inert: answers with the logging state equal answers with no state | ≥ 49/50 identical |
| E0d | FastV sanity: keep-sets recomputed offline from the logged scores equal the keep-sets the hook chose; FastV accuracy at 50% and 10% reported next to Full (information only — 50 questions are too few to gate on) | ≥ 49/50 identical keep-sets |
| E0e | Published accuracy: Lingshu `full` on all of `slake_en` | 81–85% (MedSTP reports 83.9 for Lingshu on SLAKE) |
| E0f | Speed: mean, median and 95th percentile seconds per question | written into `configs/runs.yaml` rates |

**Outputs:** `$RES/runs/lingshu7b__slake_en__full__r100__s0__t4x2.jsonl` (reused by E1, E3, E5, E6),
`$RES/e0_report__<model>__t4x2.json`, updated rates in `configs/runs.yaml` (commit).

**How to read it:** `e0_report__<model>__t4x2.json` lists each check with pass/fail and an overall `pass`; Full accuracy is `accuracy` in the run's
`meta.json`. Speed factor = measured seconds ÷ planned seconds (1.0 for Lingshu on SLAKE, 1.1 for Qwen3-VL).

**Decision:** all checks pass → Stage 4. If the speed factor is above 1.5: Random with one seed outside S1, drop
`uniform_l2` from E3, move E5's 50% points to Stage 7.

**Checks**
- [ ] E0b fp16 decision recorded (vision fp32: yes / no).
- [ ] E0c ≥ 49/50. [ ] E0d 50/50. [ ] E0e Full = ____%.
- [ ] Rates: Lingshu ____ s/q, Qwen3-VL ____ s/q; factor ____.

**If something goes wrong**

| Symptom | Likely cause | Action |
|---|---|---|
| Empty, repeated or "!!!!" answers | fp16 overflow | `--vision-fp32`; if still broken, 8-bit LLM (2–3× slower, P0 only) |
| Full accuracy < 79% | chat template, prompt text or answer parsing | print 20 raw responses next to answers; diff prompts with MedEvalKit |
| Very slow decoding | `use_cache` off in the config | the adapter forces `use_cache=True`; check it is passed |
| `CALLS["decoder"] == 0` | hook not installed on the text decoder | check `language_model.config._attn_implementation == "medprune_sdpa"` |

---

## Stage 4 — Kill test (E1) and first diagnostics (E2)

**In one sentence:** in one day, find out whether standard FastV really is worse than random in our pipeline (K1) and
whether it prefers background tokens (K2) — before spending ~30 GPU-hours. **Status: PRIMARY DECISIONS COMPLETE
(Nov 3); registered text-only truncation retry verified.** K1 is GO and K2 leaves H1 alive in this Lingshu setup.
This does not prove the cause, validate the gate as a competitive fix, or close the Stage 3 scoring issue.

**Before you start**
- [x] Stage 3 component checks passed on 50 items per model; full-set baseline comparability remains open above.
- [x] Analysis module and notebook section ready (task T3).
- [x] RadImageNet anatomy built and its 1,000-image identity contract verified in the repair session:
```bash
python -m medprune.data build --out /tmp/medprune-rin --work /tmp/medprune-work --only radimagenet
python -m medprune.data masks --bundle /tmp/medprune-rin --datasets rin_anat_mc,rin_abn_closed --models lingshu7b
```
  The build prints the RadImageNet `question_type` and `content_type` values it filtered on, and the counts
  (expected ~1,000 each). Record them and the image size in the Work log.

### E1 — Kill test

| Field | Value |
|---|---|
| Question | At 10% of tokens, is standard FastV significantly worse than random selection on CT/MRI? Does MedSTP's pre-LLM version behave differently? Does FastV keep more background than random? |
| Why it matters | MedSTP's FastV numbers come from an undocumented pre-LLM adaptation. If standard FastV is not worse than random here, the paper's premise changes. |
| Setup | Lingshu-7B, role A; group K (`slake_en` 1,061 + `rin_anat_mc` ~1,000 questions); budget 10%; seed 0. |
| Runs | 14 run IDs (7 methods × 2 datasets); `full` on `slake_en` already exists from E0, so 13 are new; 13,366 new answers. |
| Time | 3.3 GPU-h (one session). |

**Methods**

| Method | What it does differently from FastV | What its result tells us | Prediction |
|---|---|---|---|
| `full` | no pruning | the ceiling; also logs attention scores, so FastV's choice at any budget can be recomputed offline | — |
| `text_only` | no image | the language-prior floor; defines the image-dependent subset | — |
| `random_l2` | random tokens, hidden from layer 2 | chance at FastV's own insertion point — **the K1 comparison** | P1 |
| `random_l0` | random tokens, removed before the LLM | chance at MedSTP's insertion point | P1 |
| `fastv` | — (the method under test) | is attention-based selection below chance? | P1, P2 |
| `fastv_prellm` | same scores, but tokens removed before the LLM (two passes) | does MedSTP's 18-point gap come from FastV itself or from its insertion point? | P1 |
| `gate_fastv` | background removed before choosing | first test of the fix; a large gain on CT/MRI supports H1 | P6 (early) |

**Command**
```bash
python -m medprune.run_eval --exp E1 --model lingshu7b --bundle $DATA --bundle $RIN --out $RES --max-hours 6
```

**Outputs:** 13 new run files and diagnostics under `$RES`; after the laptop analysis (E2 below):
`results/tables/stage4_*.csv`, `results/figures/stage4_keepmaps.pdf`.

**Expected result (pre-registered, §A6):** P1 — FastV below Random@L2 by ≥ 8 points on `rin_anat_mc`; pre-LLM FastV
below Random@L0 by ≥ 10. P2 — on `slake_en`, Full 81–85%, FastV − Random ≤ −4 on CT+MRI, ≥ 0 on X-ray.

**How to read it:** in `stage4_paired.csv`, the row `rin_anat_mc · fastv − random_l2` gives Δ (points) and its 95%
cluster CI. Negative Δ with the whole CI below 0 = FastV is significantly worse than chance. Repeat for
`fastv_prellm − random_l0`, and for `slake_en` split by modality.

**Decision (K1):** GO if the `fastv − random_l2` CI on `rin_anat_mc` lies entirely below 0. If only the pre-LLM pair
does → reframe to "pre-LLM attention selection fails" and discuss with the mentor. If neither → run the same pair at
5% on `rin_anat_mc` (0.17 h; with 224² images there are only 64 tokens, so 10% may be lenient) before stopping.

**Checks**
- [x] Every pruned answer has `k_keep == round(0.1 × n_vis)`.
- [x] `n_vis` recorded: SLAKE 324 tokens, k = 32; RadImageNet 64 tokens (8×8), k = 6.
- [x] Truncated answers < 1% after the registered retry: 43 original SLAKE text-only items retried at 512, all prefixes
  verified; two remain truncated (0.19%). Text-only accuracy and K1/K2 unchanged; retain the two-item limitation.
- [x] RadImageNet fields/counts and raw dimensions verified in the CPU rebuild: 1,000 images, all 224×224,
  426 CT + 574 MRI; processed Lingshu grid 8×8. Anatomy and abnormality use identical image-ID sets.

### E2 — Why does it fail? (analysis only, laptop)

| Field | Value |
|---|---|
| Question | Is FastV's keep-set driven by position, by background, or by high-norm artifact tokens? |
| Why it matters | This is the mechanism claim (C3). The first look uses E1's logs; E3, E6 and E8 add to it later. |
| Setup | Download the session output, merge into `D:\cvpr\results\`, run `02_analyze_results.ipynb`, section "Stage 4" (task T3). No GPU. |

| Measurement | What it shows | Prediction |
|---|---|---|
| BRR of FastV vs Random on CT/MRI and X-ray | does FastV prefer background beyond its share? Random must be ≈ 1.00 | P3 (BRR ≥ 1.3 → K2) |
| Keep-maps (16 × 16 average of kept positions) | where FastV's kept tokens sit; predicted bottom-right for Lingshu | P4 |
| Keep regression (kept ~ position + background + norms) | separates position from background statistically | H1 vs H2 |
| Query dependence (Jaccard of keep-sets across questions on one image) | high overlap = FastV ignores the question | H1/H2 |
| High-norm token selection rate | are artifact tokens over-selected? | P5 |
| Gate effect by modality | how much the gate recovers where background exists | P6 |

**Decision (K2):** if FastV's BRR CI on CT/MRI includes 1, background capture (H1) is out as the lead explanation and
the story becomes positional (H2).

**Done when (Stage 4):** K1 and K2 decided and logged; P1–P5 outcomes filled in §A6; mentor informed.

**If something goes wrong**

| Symptom | Likely cause | Action |
|---|---|---|
| FastV ≈ Random everywhere | 10% too lenient for small grids | 5% check; look at keep-maps |
| Full on RadImageNet far below MedSTP's 89.4 | option order or answer letters wrong | print 20 prompts and answers |
| BRR missing for many images | background fraction outside 0.05–0.95 (common for X-ray) | expected; report coverage |

---

## Stage 5 — Main grid (E3) and modality panel (E4)

**In one sentence:** produce the paper's two core results — Table 1 (both models, three benchmarks, all main methods)
and Fig. 2 (failure grows with background) — in time for the Nov 10 enrollment. **Status: FIRST BOUNDED QWEN BLOCK
RUNNING (B v2, Nov 3, user requested).** This uses ten existing E3 conditions on English SLAKE and RIN anatomy: Full,
text-only, Random@L2 seed 0, FastV, Gate+FastV. Retention 10%; RIN first; five-question smoke for every condition
before bulk inference. Approximately 2.35 planned inference hours, 3.5-hour scheduling budget, role B/Father T4×2.
The full Lingshu scoring-comparability issue remains open; this within-Qwen comparison does not require claiming
equivalence to published LLM-judged scores. The full 45-condition grid, other datasets/seeds and E4 remain pending.

**Before you start**
- [x] K1 = GO (corrected Lingshu E1, Nov 3).
- [ ] RadImageNet built in each session (Stage 4 commands; use `--models qwen3vl8b` on role B).

### E3 — Main grid

| Field | Value |
|---|---|
| Question | Does FastV fall below random on both models and on all of MedSTP's benchmarks? Does the gate fix it? Is the cause position, background, or both? |
| Why it matters | Table 1 carries claims C1, C4 and C5; reviewers need two models (medical-tuned Lingshu, general Qwen3-VL), variance across Random seeds and the 2×2 factorial. |
| Setup | Lingshu-7B on role A, Qwen3-VL-8B on role B, in parallel. Groups S1 (`slake_en`, `rin_anat_mc`, `rin_abn_closed`), S2 (`slake_zh`), S3 (`omni_anat`). Budget 10%. |
| Runs | 45 run IDs per model (S1: 11 methods × 3 datasets; S2: 5; S3: 7). Lingshu reuses 10 from E1. |
| Time | Lingshu 12.6 GPU-h (~3 sessions); Qwen3-VL 16.3 GPU-h (~3–4 sessions). |

**Methods**

| Method | What it changes vs FastV | What it tells us | Groups | Prediction |
|---|---|---|---|---|
| `full`, `text_only` | — | ceiling and language prior | S1, S2, S3 | — |
| `random_l2` seeds 0, 1, 2 | random selection | chance level and its seed variance (FastV's gap must exceed it) | S1 (seeds 0–2), S2, S3 (seed 0) | P1 |
| `uniform_l2` | evenly spaced tokens | coverage without attention (FEATHER's baseline) | S1 | — |
| `fastv` | — | the method under test | S1, S2, S3 | P1, P2 |
| `ropefree` | positional part removed from the score | how much of the failure is position (H2) | S1, S3 | P8 |
| `gate_fastv` | background removed before choosing | background capture (H1) and the fix | S1, S2, S3 | P6 |
| `gate_ropefree` | both removed | 4th cell of the 2×2: are the causes additive? | S1 | P8 |
| `gate_uniform` | background removed, no attention at all | does attention add anything once background is gone? | S1, S3 | P7 |

`slake_zh` (S2) gives "SLAKE (all)" comparable to MedEvalKit, and doubles as a test of FastV's query dependence
(same images, different language).

**Commands (repeat each session until `--list` shows everything complete)**
```bash
# role A
python -m medprune.run_eval --exp E3 --model lingshu7b --bundle $DATA --bundle $RIN --out $RES --max-hours 5.5
# role B
python -m medprune.run_eval --exp E3 --model qwen3vl8b --bundle $DATA --bundle $RIN --out $RES --max-hours 5.5
```
Each new session: restore the previous output first (B0.3). Check progress any time with `--list` (add
`--profile t4x2` on the laptop).

**Outputs:** run files; laptop: `results/tables/table1_draft.csv` (+ `.tex`).

**Expected result:** P6 — Gate+FastV beats FastV by ≥ 8 points on CT/MRI and changes nothing (|Δ| ≤ 1) on
dermoscopy/microscopy (E4); P7 — Gate+Uniform within 3 points of Gate+FastV; P8 — the gate adds ≥ 3 points on top of
RoPE-free; P10 — Qwen3-VL shows the same direction, smaller.

**How to read it:** Table 1 has one row per method and one column per dataset, both models side by side, with 95%
cluster CIs; MedSTP's numbers appear in a separate grey block, never mixed in. Read across the `fastv` and
`random_l2` rows first (the phenomenon), then `gate_fastv` vs `fastv` (the fix), then the 2×2 block
(`fastv`, `ropefree`, `gate_fastv`, `gate_ropefree`) for the cause.

**Checks**
- [ ] Every run's question count equals its manifest (`--list` shows "complete").
- [ ] Random seed spread (std over 3 seeds) ≤ ~1.5 points per dataset.
- [ ] Speed still within 1.5× of the measured rate; if not, re-plan Stage 6 now.

### E4 — Modality panel (dose-response)

| Field | Value |
|---|---|
| Question | Does the Random − FastV gap grow with the share of background in the image, and vanish where there is none? |
| Why it matters | Fig. 2 carries claim C2: background fraction *predicts* the failure. Background-free modalities are the negative control — the gate must do nothing there. |
| Setup | Both models; `omni_panel` (8 modalities × 150); budget 10%. |
| Methods | `full`, `text_only`, `random_l2`, `fastv`, `gate_fastv`, `gate_uniform`. |
| Runs, time | 6 run IDs per model; Lingshu 2.0 h (A), Qwen3-VL 2.2 h (B). |

```bash
python -m medprune.run_eval --exp E4 --model lingshu7b --bundle $DATA --out $RES --max-hours 5.5   # role A
python -m medprune.run_eval --exp E4 --model qwen3vl8b --bundle $DATA --out $RES --max-hours 5.5   # role B
```

**Expected result:** P9 — Spearman ρ ≥ 0.7 between modality background fraction and the Random − FastV gap; per-image
ρ > 0 with CI excluding 0.

**How to read it:** Fig. 2 plots background-fraction deciles (x) against Random − FastV accuracy (y) with CIs, one line
per model, modality means labelled. A rising line = C2 supported. Check that dermoscopy and microscopy sit near 0 and
that their `gate_fastv` equals `fastv`.

**Done when (Stage 5):** Table 1 shows FastV below Random on CT/MRI-heavy sets (CI excluding 0) with a gate gain, and
Fig. 2 rises (or the reframed claim from K1/K2 holds). Abstract draft to the mentor by Nov 9; **paper enrollment on
OpenReview by Nov 10, evening IST** (deadline Nov 10 23:59 AoE = Nov 11 17:29 IST).

**If something goes wrong**

| Symptom | Likely cause | Action |
|---|---|---|
| Qwen3-VL shows no failure anywhere | DeepStack makes its attention content-aware (H5) | report it as a finding; Lingshu's dose-response still carries C2 |
| `slake_en` and `slake_zh` keep-sets differ a lot | FastV depends on the question after all | weakens the pure-position story; report the Jaccard numbers |
| A session dies | Kaggle limit or out of memory | restore and rerun; finished questions are skipped |

---

## Stage 6 — Budget curves (E5), mechanism controls (E6-P1), natural control (E7)

**In one sentence:** show the effect is not a 10%-only artifact, pin down the mechanism with controls, and test
whether it is specific to medical images. **Status: LATER.** Decisions K3 and K4 are taken here.

**Before you start**
- [ ] Stage 5 done.
- [ ] Static maps built on the laptop (task T4): `results/static_maps/<model>__for_<dataset>.npy` from the `full`
      runs' scores on the *other* anchor dataset; upload them with the restored results.

### E5 — Budget curves

| Field | Value |
|---|---|
| Question | Does the FastV gap widen as the budget shrinks, and does the gate help at every budget? |
| Why it matters | Pruning papers are judged on curves, not one budget (Fig. 4). The 50% points link to ViToS's mild-pruning result. |
| Setup | Group K; Lingshu: `random_l2`, `fastv`, `gate_fastv`, `gate_uniform` at 5% and 25%, plus `fastv`, `gate_fastv` at 50%; Qwen3-VL: `random_l2`, `fastv`, `gate_fastv` at 5% and 25%. |
| Runs, time | Lingshu 20 run IDs, 4.6 h (A); Qwen3-VL 12 run IDs, 3.1 h (B). |

```bash
python -m medprune.run_eval --exp E5 --model lingshu7b --bundle $DATA --bundle $RIN --out $RES --max-hours 5.5
python -m medprune.run_eval --exp E5 --model qwen3vl8b --bundle $DATA --bundle $RIN --out $RES --max-hours 5.5
```

**How to read it:** Fig. 4 — accuracy vs budget for each method; the 10% points come from E3. Random must rise with
budget (sanity); the FastV line should sit below Random at small budgets; the gate line should stay close to Full.

### E6-P1 — Mechanism controls

| Field | Value |
|---|---|
| Question | Which part of FastV's choice is position, which is background, and how much room is left for a better mask? |
| Setup | Lingshu on S1 (+ `oracle_gate` on `slake_en`); Qwen3-VL on K for the key controls; budget 10%. |
| Runs, time | Lingshu 19 run IDs, 4.1 h (A); Qwen3-VL 10 run IDs, 2.5 h (B). |

| Method | What it isolates | Reading | Prediction |
|---|---|---|---|
| `fastv_k8`, `fastv_k16` | scoring deeper in the model | positional decay matters less deeper → the gap should shrink with K | P11 |
| `static_map` | a fixed position-only ranking learned on the other dataset | if it matches FastV, FastV chooses by position | H2 |
| `inverse_gate` | keeping only background tokens | if it matches FastV on CT/MRI, FastV behaves like a background selector | P12 |
| `oracle_gate` | SLAKE's organ masks as the gate | upper bound of any spatial gate; the gap to our mask = room for a better mask | — |
| `divprune`, `gate_divprune` | a diversity pruner without attention, ± gate | does the gate help other families, or does diversity already avoid background? | — |

```bash
python -m medprune.run_eval --exp E6 --model lingshu7b --account A --bundle $DATA --bundle $RIN --out $RES --max-hours 5.5
python -m medprune.run_eval --exp E6 --model qwen3vl8b --account B --bundle $DATA --bundle $RIN --out $RES --max-hours 5.5
```

### E7 — Natural-image control

| Field | Value |
|---|---|
| Question | On ordinary photographs, is FastV still worse than random for the same models? |
| Why it matters | Separates "medical images break FastV" from "FastV is generally weak at 10%". The keep-maps show whether the same positional bias exists but lands on content. |
| Setup | Both models on role B; group N (500 GQA + 500 POPE); `full`, `random_l2`, `uniform_l2`, `fastv`, `ropefree`; no gate (photos have no acquisition background). |
| Runs, time | 10 run IDs, 2.5 h. |

```bash
python -m medprune.run_eval --exp E7 --model lingshu7b --bundle $DATA --out $RES --max-hours 5.5
python -m medprune.run_eval --exp E7 --model qwen3vl8b --bundle $DATA --out $RES --max-hours 5.5
```

**Expected result:** P13 — FastV ≥ Random − 2 points on photographs. If FastV is below Random on photos too, the claim
becomes "background-driven, strongest on medical images", carried by Fig. 2.

**Decisions**
- **K3:** if Gate+FastV recovers less than ⅓ of (Full − FastV) on CT/MRI, the paper becomes analysis-first.
- **K4:** if RoPE-free alone recovers ≥ 90% of the gate's gain and the gate adds nothing on top, reframe as "positional
  bias meets acquisition geometry", with the gate as the cheap geometric alternative.

**Checks**
- [ ] Random accuracy increases with budget (5% < 10% < 25%) on every dataset.
- [ ] Static maps built only from the other dataset (source ≠ evaluation dataset).
- [ ] Oracle gate covers all 96 SLAKE test images.

**Done when (Stage 6):** K3 and K4 decided and logged; P8, P11–P13 filled; Fig. 3, Fig. 4 and Table 3 drafted.

---

## Stage 7 — Supplementary runs (E6-P2, E8, E9)

**In one sentence:** strengthen the mechanism and the fix with gate-robustness and artifact controls, a third model
with 1D RoPE, and a real latency measurement. **Status: LATER** (can overlap Stage 6; results go to the paper if ready
by Nov 14, otherwise to the supplementary, Nov 23).

### E6-P2 — Gate robustness, artifacts, PDrop

| Method | What it isolates | Reading |
|---|---|---|
| `artifact_excl` | FastV with high-norm tokens removed from the candidates | if accuracy recovers, artifact tokens cause part of the failure (H3, P5) |
| `gate_quota3` | gate plus 3% of tokens kept from the background | do plane/modality questions need a little background for context? |
| `gate_tau25`, `gate_tau75` | the gate's single threshold at 0.25 and 0.75 | the gate is not tuned to the test set if results barely move |
| `pdrop`, `gate_pdrop` | multi-stage attention pruning, token-matched to FastV, ± gate | composability with another attention method |

Setup: Lingshu on S1, role C, 18 run IDs, 3.8 h.
```bash
python -m medprune.run_eval --exp E6 --model lingshu7b --account C --bundle $DATA --bundle $RIN --out $RES --max-hours 5.5
```

### E8 — HuatuoGPT-Vision-7B (1D RoPE)

| Field | Value |
|---|---|
| Question | With 1D RoPE, do FastV's kept tokens move to the bottom rows (not the corner), and does the gate still help? |
| Why it matters | A different position encoding with a different predicted bias is the cleanest test that the bias comes from rotary decay (H2). |
| Before running | Task T5: write `medprune/huatuo_llava.py`. Read `model.safetensors.index.json` first — the authors' loader uses `init_vision_encoder_from_ckpt=True`, so the fine-tuned vision tower may be inside the main checkpoint rather than the stock CLIP in `vit/`. Validate on 50 SLAKE questions: Full within 3 points of the authors' numbers. |
| Setup | Role C; S1; `full`, `text_only`, `random_l2`, `fastv`, `ropefree`, `gate_fastv`, `gate_ropefree`. |
| Runs, time | 21 run IDs; 4.7 h + 1.0 h validation. |

```bash
python -m medprune.data masks --bundle /tmp/medprune-rin --datasets rin_anat_mc,rin_abn_closed --models huatuo7b
python -m medprune.run_eval --exp E8 --model huatuo7b --bundle $DATA --bundle $RIN --out $RES --max-hours 5.5
```

**How to read it:** its keep-maps join Fig. 3 — mass in the bottom rows for HuatuoGPT vs the bottom-right corner for the
Qwen models supports H2. Report `expand2square` padding separately in its BRR.

### E9 — Latency

| Field | Value |
|---|---|
| Question | How much prefill time does the gate save in practice? |
| Why it matters | Masking (used for accuracy runs) saves no time, so the deployment claim needs a measured number. |
| Setup | Task T6. Lingshu, 200 `slake_en` questions, role C: `full` vs `gate_input_fastv` (background tokens physically removed before the LLM); CUDA events, 10 warm-up questions, median and IQR; theoretical prefill FLOPs for every method. |
| Time | 0.5 h. |

**Outputs (Stage 7):** `table4_efficiency.csv`, HuatuoGPT keep-maps, supplementary tables; P5 and P14 filled.

**Checks**
- [ ] HuatuoGPT validation passed before E8.
- [ ] Latency timed around prefill only, with `torch.cuda.synchronize()`, on identical questions for both methods.

---

## Stage 8 — Analysis and writing

**In one sentence:** every number in the paper is regenerated from the run files, and the paper and supplementary are
submitted on time. **Status: LATER (Nov 9–23).**

**Steps**
1. Merge all session outputs into `D:\cvpr\results\`; run `02_analyze_results.ipynb` end to end (task T7). It writes
   `results/tables/*.csv` and `*.tex` (the paper `\input`s the `.tex` files) and `results/figures/*.pdf`; every CSV
   keeps the run IDs behind its cells.
2. Write following §A7; limitations: 2D only, heuristic mask, closed-form benchmarks, no clinical claim.
3. Double-blind: no link to the named GitHub repo in the paper; use an anonymized mirror if code is referenced.
4. Cite MedSTP as contemporaneous (online after Sept 15, 2026).

**Checks**
- [ ] Nov 9: abstract draft to the mentor. [ ] Nov 10 (IST evening): enrolled on OpenReview.
- [ ] Nov 15: full draft to the mentor. [ ] Nov 16 (IST evening): submitted (deadline Nov 17, 17:29 IST).
- [ ] Nov 23: supplementary submitted.
- [ ] One notebook run regenerates every figure and table from a clean `results/`.

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
| Chat template | Model's own; HuatuoGPT `"<\|user\|>\n<image>\n{prompt}\n<\|assistant\|>\n"` ✓ |
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
lowercasing and stripping newlines/periods. Open: **normalized** exact match (case, ASCII/CJK punctuation, articles,
whitespace; revised 2026-11-02 — MedEvalKit's published numbers use a GPT-4.1 judge, and its strict-EM fallback scores
Lingshu's "CT." as wrong), with strict EM and token recall
(secondary). Multiple choice: letter, choice text, or difflib nearest choice. All after `parse_response`.

## C2. Data

**RadImageNet identity correction (2026-11-03).** `metadata.question_id` identifies a question template, not an
individual example. Each manifest `qid` is now `{dataset}/{image-content-sha1}/{template-id}`. The template by itself
must never seed Random, identify a completed question, or key an analysis/diagnostic record. Duplicate IDs are a
hard error at manifest writing/loading, result loading, diagnostic loading and resume/listing boundaries.

The first E1 output used template-only IDs. Preserve its original files. Non-random answers and numerical diagnostic
arrays can be recovered offline by joining each original row to its saved image ID and preserving verified shard
order; the recovery writes a separate provenance audit. The two RadImageNet Random conditions require fresh
inference because template-only seeding gave every image one fixed spatial keep-set. They are excluded from K1
until rerun. The anatomy-only repair build explicitly selects `--datasets rin_anat_mc`; its fresh Random outputs
have 1,000 distinct IDs and keep-sets each, and K1 was independently verified after merging them.

**RadImageNet abnormality mapping correction (2026-11-03).** The saved source-schema receipt identifies the intended
generic abnormality task as `metadata.question_id == "abnormal_closed"`, `content_type == "pathology"`,
`question_type == "closed"` (1,000 rows; example: "Does the ankle foot look abnormal?"). The disease-specific
`pathology_closed_pos/neg` templates are different tasks and must be excluded. The builder and regression fixtures
now use these observed values. A real CPU rebuild at `7dec114` verified 1,000 unique IDs/images per subset,
identical image-ID sets, unchanged anatomy inputs, and 224×224 raw images. Abnormality labels are 719 yes / 281 no:
an always-yes baseline scores **71.9%** without an image. Record that analytical class-prior baseline and per-class
outcomes alongside abnormality accuracy before interpreting the Stage 5 results.

| ID | Source | Size ✓ | Access | Subset rule | Questions | Images |
|---|---|---|---|---|---|---|
| `slake_en` | `BoKelvin/SLAKE` | `imgs.zip` 212 MB (masks + boxes per image), `test.json` 636 KB, `mask.txt` 39 labels | CC-BY-4.0 | test, `q_lang == "en"` | **1,061** ✓ (open 645, closed 416) | **96** ✓ (CT 40, MRI 26, X-ray 30) |
| `slake_zh` | same | same | same | test, `q_lang == "zh"` | **1,033** ✓ | 96 |
| `omni_anat` | `foreverbeliever/OmniMedVQA` | zip 10.70 GB | public, per-source licenses (bundle private) | open-access Anatomy Identification: seed-0 sample of 16,448 candidates; not established to be MedSTP's exact subset | 3,329 MC (2,040 ultrasound) | 3,326 |
| `omni_panel` | same | same | same | open-access; 8 modalities (CT, MRI, X-Ray, Ultrasound, OCT, Fundus, Dermoscopy, Microscopy) × 150, seed 0, ≤ 3/image, disjoint from `omni_anat` | 1,200 | 1,198 |
| `rin_anat_mc` | `raidium/RadImageNet-VQA` | benchmark parquet 361 MB (9,000 QA / 1,000 images, CT/MRI) | gated, research-use, **no redistribution** | anatomy × multiple choice, one per image; actual template `anatomy_mc` | 1,000 | 1,000 (426 CT, 574 MRI; 224×224) |
| `rin_abn_closed` | same | same | same | template `abnormal_closed`, content `pathology`, type `closed`; yes/no, one per image | 1,000 (719 yes, 281 no) | same 1,000 verified |
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

- **Run ID:** `{model}__{dataset}__{method}__r{budget%:03d}__s{seed}__{profile}`, e.g.
  `lingshu7b__slake_en__gate_fastv__r010__s0__t4x2`. No experiment name: a run shared by several experiments is
  computed once (`meta.json` lists the experiments). The profile suffix prevents mixing hardware in one table.
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
| Query dependence | mean Jaccard of FastV keep-sets across different questions on the same image (and `slake_en` vs `slake_zh`); high = selection ignores the question | bootstrap images |
| Dose-response | per-image Δ = mean(correct_Random − correct_FastV); Spearman vs b; deciles; modality-level over ≥ 13 groups | cluster bootstrap |
| Evidence size | SLAKE question → `mask.txt` labels (synonyms: lung → both lungs; tumor → brain tumor/edema; cancer → liver/kidney/lung cancer); area tertiles; report coverage | bootstrap within tertile |

Primary tests = P1–P14; everything else is exploratory.

---

# Part D — Code design

## D1. Principles

One job per file; dependencies flow one way (`data`, `masks` → `selection` → `attention_hook` → `models` →
`run_eval`). Every method = "choose a keep-set, mask dropped visual keys from layer L" — masking with original
positions equals deleting the tokens (proved in tests), so one ~120-line hook covers all 25 methods. The only global
state is `attention_hook.ACTIVE`, set by a context manager. Notebooks and scripts only orchestrate; nothing in
`medprune/` assumes Kaggle (paths come from `env.py`). Python ≥ 3.10, type hints, docstrings with shapes, `logging`.

## D2. Layout

```
cvpr/
├── README.md                    framing + quick start
├── plan.md                      this file
├── mentor_brief.tex             two-page mentor brief (Overleaf)
├── requirements.txt             pinned (transformers==4.57.1); torch comes from the platform
├── pyproject.toml               package metadata; lets pytest and `pip install -e .` find `medprune`
├── .gitignore                   results, data, third_party, caches
├── scripts/
│   └── setup_env.sh             same setup on Kaggle and SSH servers; prints the detected profile
├── configs/
│   ├── runs.yaml                run matrix (exp → model → group → methods → budgets → seeds), expanded by run_eval
│   └── bundle_manifest.json     data counts + hashes (committed provenance)
├── medprune/
│   ├── env.py                   compute profile (t4x2 | a100) and paths
│   ├── data.py                  manifests → Sample; MedEvalKit prompts; `python -m medprune.data build`
│   ├── masks.py                 model view, body mask, per-token foreground fraction
│   ├── selection.py             MethodSpec registry + pure NumPy selectors
│   ├── attention_hook.py        PruneState, `medprune_sdpa`, scoring helpers
│   ├── models.py                Qwen-family adapter (Lingshu-7B, Qwen3-VL-8B)
│   ├── huatuo_llava.py          HuatuoGPT-Vision-7B port to transformers 4.57.1 (written at Stage 7)
│   ├── scoring.py               MedEvalKit-identical correctness
│   ├── stats.py                 bootstrap, McNemar, BRR, keep-maps, static map, dose-response, regression
│   ├── run_eval.py              run_one / run_many / CLI (`python -m medprune.run_eval`)
│   ├── checks.py                Stage 3 checks: fp16, inert logging, offline keep-sets, timing
│   └── analysis.py              run files → tables, keep-maps, K1/K2 decisions (laptop)
├── notebooks/
│   ├── 00_build_data_bundle.ipynb   Kaggle CPU wrapper around `medprune.data build`
│   ├── 01_run_gpu.ipynb             Kaggle GPU wrapper around `run_many`
│   └── 02_analyze_results.ipynb     laptop: all tables and figures
├── tests/
│   ├── test_masks.py
│   ├── test_selection.py
│   ├── test_attention_hook.py
│   ├── test_scoring.py
│   ├── test_stats.py
│   ├── test_run_eval.py
│   ├── test_checks.py
│   └── test_analysis.py
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
Edge-connected background `skimage.measure.label(connectivity=1)`; components `label(connectivity=2)`; holes
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
576 embeddings spliced at `<image>`; generate from `inputs_embeds`; `rc` raster 24 × 24. Written at Stage 7: the
authors' loader uses `init_vision_encoder_from_ckpt=True`, so the main checkpoint may carry a fine-tuned CLIP — read
`model.safetensors.index.json` first and load the vision tower from wherever its weights actually are.

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
def load_runs(path="configs/runs.yaml", exp=None, model=None, account=None) -> list[RunSpec]  # expands the matrix form
def main() -> None   # python -m medprune.run_eval --exp E3 --model qwen3vl8b [--account B] [--max-hours 5]
                     #   [--list] print runs + hours · [--limit N] smoke test · [--dry-run] fake adapter, CPU
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
| `test_stats.py` | cluster bootstrap reproducible by seed and wider than the naive interval when questions cluster; paired delta sign; BRR = 1 for uniform random keep-sets in expectation; keep-map and static-map shapes |
| `test_run_eval.py` | `runs.yaml` expands as planned (E1 = 14 runs, shared runs merged); run IDs; bundle-root search; dry run writes one line per question for 7 methods and resumes |
| `test_checks.py` | all Stage 3 checks pass on a deterministic adapter; report written; timing summary |
| `test_analysis.py` | run-ID parsing; accuracy and paired deltas on known values (Random seeds averaged); query dependence; diag round trip; full Stage 4 report on dry-run results (gate never keeps background, decisions present, files written) |

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

**Kaggle operations.** Accelerator GPU T4×2; internet on (phone-verified); secret `cvpr` = Hugging Face token (set
`HF_TOKEN` from it); repo cloned over HTTPS (public, no token); attach `medprune-data`. RadImageNet-VQA is built per
session under `/tmp/medprune-rin` and passed as a second `--bundle` — never under `/kaggle/working`, whose contents
become notebook output. Long runs via "Save Version → Save & Run All (Commit)" (12-hour limit ✓); `run_many` stops
scheduling when elapsed + next estimate > 6 h; runs resume from their JSONL. Weekly GPU quota is not published
(reports 15–30 h) — read each account's quota page. Model download costs 3–5 min of quota per session; optionally
mirror a model into a private dataset from a CPU session. `/kaggle/working` (20 GB ✓) is for results only.

---

# Part F — Agent handbook (Codex, Gemini, Claude)

Any agent picking up this project works from this Part. It says where things are, what is allowed, and what to do next.

## F1. Start here

1. Read `README.md`, the Stage tracker at the top of this file, then Part B for the current stage, then the newest
   entries of the Work log (Part G).
2. **Current stage** = the first row of the Stage tracker whose exit gate is not passed. **Current task** = the first
   task in §F4 whose status is `open`, belongs to the current or an earlier stage, and whose prerequisites are done.
3. Before changing code: `.venv\Scripts\python -m pytest -q` on the laptop (Linux: `python -m pytest -q`) must pass.
   It currently runs 84 tests in about 10 s (verified Nov 3).
4. When you finish (or stop), add a Work log entry (template in Part G) and set the task's status in §F4.

## F2. Rules

- **Frozen science.** Never edit the predictions in §A6 (fill only Outcome). Never change method definitions (§C3 =
  `selection.METHODS`), prompts (§C1.1 = `data.make_prompt`), scoring (`scoring.py`), fixed settings (§C1) or the run-ID
  format without (a) updating this plan, (b) updating tests, (c) a log entry saying why. After Stage 4 starts, any such
  change invalidates affected finished runs: exclude them from active analysis and rerun as needed, while retaining
  immutable originals and provenance. Do not delete historical evidence as a routine protocol change.
- **Where logic lives.** Notebooks only orchestrate; every function goes in `medprune/` with a CPU test in `tests/`.
- **Never commit** data, model weights, `results/`, `third_party/`, `research/` or `scripts/retina_microprobe.py`
  (the last two are a separate project of the user's).
- **RadImageNet-VQA** (license: no redistribution): never in the shared bundle, never under `/kaggle/working`, never in
  git, never in a public notebook output. Build it per session under `/tmp/medprune-rin`.
- **Hardware.** Never put runs from different profiles (`t4x2`, `a100`) in one table.
- **Results are data.** Do not change code to move a result; report it in the log with its run IDs and CIs.
- **Who runs what.** The user's Nov 3 handoff and Kaggle operations instructions authorize agents to submit and
  monitor scoped project runs on configured accounts. Use per-child credentials, preflight account-specific metadata,
  report role/slug/version/cost, preserve receipts and never print credentials. Deleting/cancelling running work or
  unrelated operations still needs an exact explicit instruction. Agents may commit and push to `main` after tests
  pass. Keep commits small; imperative subject ≤ 72 characters; body says what and why; end with agent attribution.

## F3. Environment facts

| Item | Value |
|---|---|
| Laptop | Windows 11; repo `D:\cvpr`; venv `D:\cvpr\.venv` (torch 2.14.1 CPU, transformers 4.57.1); Git Bash and PowerShell |
| GitHub | `https://github.com/pushkar1887/cvpr`, public, branch `main` |
| Kaggle | Roles A/B/C → Kaggle-2/3/1 (Stage 0 table); GPU T4×2; HF token in secret `cvpr`; clone over HTTPS |
| Data paths | Per-role notebook-output bundle, auto-discovered under `/kaggle/input`; RadImageNet `/tmp/medprune-rin` (per session); results `/kaggle/working/results` |
| Commands | `python -m medprune.data build`, `… masks`, `… summary`; `python -m medprune.run_eval` (`--list`, `--limit`, `--dry-run`, repeatable `--bundle`) |
| A100 | Not available yet; Part E describes the switch |

## F4. Task queue

Status values: `open`, `in progress (agent, date)`, `done (commit)`.

| ID | Stage | Task | Status | Prerequisites |
|---|---|---|---|---|
| T1 | 2 | `notebooks/00_build_data_bundle.ipynb` | done — executed/rebuilt on Kaggle; manifest committed Nov 2 | — |
| T2 | 3 | `medprune/checks.py` (E0 checks) + `notebooks/01_run_gpu.ipynb` | done — real E0 component checks on both models; full-set scoring issue recorded | — |
| T3 | 4 | `medprune/analysis.py` + `notebooks/02_analyze_results.ipynb`, Stage 4 section | done — notebook dry-run Nov 2; module executed on recovered real E1 Nov 3 | — |
| T4 | 6 | static-map builder | open | T3 |
| T5 | 7 | `medprune/huatuo_llava.py` + validation | open | Stage 5 done |
| T6 | 7 | E9 latency measurement | open | T2 |
| T7 | 8 | table and figure builders (Tables 1–4, Figs 1–5) | open | T3 |
| T8 | 4 | Recover E1 identities, rerun the two RIN Random conditions, verify K1/K2 | done (Codex, Nov 3) — independent CIs and registered 43-item retry verified | original and corrected receipts preserved |
| T9 | 5 preflight | Resolve full SLAKE scoring comparability before broad grid; report abnormality class-prior baseline | next — real abnormality build/image pairing passed on CPU | T8 done; first Qwen block running; remaining grid held |
| T10 | 5 | First ten registered Qwen E3 conditions on K; retrieve/verify outputs and report cross-model results | in progress (Codex, Nov 3) — B v2 submitted, API RUNNING | K1 GO; data-builder access and 84 CPU tests verified |

**T1 — data bundle notebook (Kaggle CPU, internet on).** Cells, in order:
1. `from kaggle_secrets import UserSecretsClient; os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("cvpr")`.
2. `!git clone https://github.com/pushkar1887/cvpr && cd cvpr && bash scripts/setup_env.sh`.
3. `!df -h /tmp /kaggle/working` — scratch for the 10.7 GB OmniMedVQA zip must be ≥ 25 GB free and must not be
   `/kaggle/working` (20 GB output cap).
4. `!python -m medprune.data build --out /kaggle/working/medprune-data --work /tmp/medprune-work --skip radimagenet`.
5. `!python -m medprune.data masks --bundle /kaggle/working/medprune-data`.
6. `!python -m medprune.data summary --bundle /kaggle/working/medprune-data`.
7. Print the counts against §C2 and display 3 overlays per modality.
8. `!rm -rf /tmp/medprune-work`.

Done when: the notebook is committed and executed; counts match §C2 (or differences are logged); the per-role
private builder outputs are available to their GPU notebooks; `configs/bundle_manifest.json` is committed; a log
entry records counts and median background fraction per modality.

**T2 — GPU notebook and E0 checks.** `medprune/checks.py` provides functions, each returning a dict for
`$RES/e0_report.json`:
- `fp16_check(adapter_fp16, adapter_vision_fp32, samples)`: identical-answer rate and finite logits (§Stage 3 E0b).
- `logging_is_inert(adapter, samples)`: answers with a `full` logging state equal answers with no state.
- `offline_keepsets_match(adapter, samples, budget)`: FastV keep-sets recomputed from logged `scores_l1` equal the
  keep-sets chosen in the hook.
- `timing_summary(run_jsonl)`: mean, p50, p95 seconds per question.
- CLI `python -m medprune.checks --model M --bundle DIR --out RES --n 50` runs the three checks and writes
  `e0_report.json` (this is the command Stage 3 uses).

Unit-test what can run on CPU with `FakeAdapter`. Notebook cells: env check (`nvidia-smi`, `df -h`); secret; clone and
`setup_env.sh`; parameters `EXP, MODEL, ACCOUNT, MAX_HOURS, LIMIT, RESTORE_FROM`; **restore** — if `RESTORE_FROM` names
an attached previous notebook output, copy its `results/` into `/kaggle/working/results` (Kaggle starts every session
with an empty `/kaggle/working`, so this is what makes multi-session blocks resume); RadImageNet build
`python -m medprune.data build --out /tmp/medprune-rin --work /tmp/medprune-work --only radimagenet` and
`python -m medprune.data masks --bundle /tmp/medprune-rin --datasets rin_anat_mc,rin_abn_closed --models $MODEL`;
smoke test (5 questions; assert `attention_hook.CALLS["decoder"] > 0`); E0 checks when `EXP == "E0"`;
`python -m medprune.run_eval --exp $EXP --model $MODEL --bundle /kaggle/input/medprune-data --bundle /tmp/medprune-rin
--out /kaggle/working/results --max-hours $MAX_HOURS`; per-run summary table (no zip — Kaggle's output download
already zips, and a copy would count twice against the 20 GB cap). Reports are named `e0_report__<model>__<profile>.json`.
Done when: both files are committed, CPU tests pass, and a log entry lists the exact parameters for E0 on roles A and B.

**T3 — analysis module and Stage 4 notebook section.** `medprune/analysis.py`:
`load_runs(results_dir) -> DataFrame` (one row per question, §C6 columns plus run fields parsed from the run ID),
`load_diag(results_dir, run_id) -> dict[qid, dict[str, array]]`, `image_dependent(df, model)`,
`accuracy_table(df, by=[...])` with cluster CIs, `paired_delta(df, method_a, method_b, dataset)`,
`retention_by_modality(df, masks)` (BRR), `keep_maps(df)`, `query_dependence(df)` (Jaccard across questions of one
image), `high_norm_rates(df, diag)`. Notebook section "Stage 4" writes `$RES/tables/stage4_*.csv` and
`$RES/figures/stage4_keepmaps.pdf` and prints the K1/K2 decision numbers. Done when: tests on synthetic run files pass
and a log entry shows the notebook output on the dry-run results.

**T4 — static maps.** `python -m medprune.analysis static-maps --results results --bundle ...` builds
`results/static_maps/<model>__for_<dataset>.npy` from the `full` runs' `scores_l1` of the *other* anchor dataset
(SLAKE-en → RadImageNet datasets; RadImageNet anatomy → SLAKE) with `stats.static_map_from_scores`. Done when: tested
and logged.

**T5 — HuatuoGPT-Vision port.** §D3 `huatuo_llava.py`. First read `model.safetensors.index.json` to find the vision
tower weights (the authors' loader uses `init_vision_encoder_from_ckpt=True`). Done when: a 50-question SLAKE
validation is within 3 points of the authors' numbers and the E8 runs list correctly.

**T6 — latency (E9).** A function timing prefill with CUDA events (10 warm-up, 200 SLAKE-en questions) for `full` and
`gate_input_fastv` on Lingshu, where Gate-input physically removes background tokens before the LLM; plus theoretical
FLOPs for every method (§Stage 7). Done when: committed with a notebook cell and logged.

**T7 — paper tables and figures.** Builders in `analysis.py` for Tables 1–4 and Figs 1–5 exactly as §A7 specifies,
writing `.csv`, `.tex` and `.pdf` under `results/`. Done when: one notebook run regenerates every paper number.

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

---

# Part G — Work log

Append-only, newest entry last. Every working session (any agent, or the user) adds one entry. Keep numbers exact:
run IDs, counts, accuracies with CIs. Template:

```
### YYYY-MM-DD — <agent or person> — Stage <n> — <one-line title>
- Changed: files and what changed
- Commits: hashes
- Verified: tests run, checks passed (with numbers)
- Results: run IDs with accuracy and 95% CI (if any runs finished)
- Decisions: what was decided and why
- Open issues: what is unverified or broken
- Next: the next task ID from §F4
```

### 2026-10-31 — Claude — Pre-plan — direction chosen
- Changed: nothing in the repo (exploration only).
- Decisions: medical direction = why attention-guided visual token pruning fails on medical images. A literature check
  found MedSTP (arXiv 2610.22751, Oct 15 2026): a new medical pruner whose own table shows FastV below Random in 5 of
  8 cells without explanation — that unexplained result is this paper's starting point. Positioned against FEATHER
  (ICCV 2025), PoRe, Wang et al. (arXiv 2512.07580) and ViToS (ICML 2026).
- Open issues: MedSTP documents no prompts, subset rules or FastV layer, and prunes every method before the LLM;
  its numbers can only be a reference block, not mixed into our tables.

### 2026-11-02 — Claude — Planning — plan, repo, verification
- Changed: `plan.md`, `README.md`, `mentor_brief.tex`, `requirements.txt`, `scripts/setup_env.sh`, `.gitignore`,
  `.gitattributes`.
- Commits: d13007c, 34e69a4 (plan and setup), 79e5f3c (detailed Part B).
- Verified: CVPR 2027 dates from the CFP (enrollment Nov 10 AoE = Nov 11 17:29 IST; paper Nov 16 AoE; supplementary
  Nov 23 AoE; contemporaneous cut-off Sept 15 2026). Model files on Hugging Face (Lingshu-7B 16.58 GB, Qwen3-VL-8B
  17.53 GB, HuatuoGPT-Vision-7B 15.87 + 1.71 GB). SLAKE test = 2,094 questions (en 1,061 on 96 images: CT 40, MRI 26,
  X-ray 30; zh 1,033). transformers 4.57.1: `AttentionInterface`, SDPA's `is_causal` path, mask interface, DeepStack on
  LLM layers 0–2, M-RoPE text offset (max + 1 on all axes) — the basis of the bottom-right prediction.
- Decisions: Lingshu-7B + Qwen3-VL-8B on MedSTP's three benchmarks; HuatuoGPT-Vision-7B as the 1D-RoPE third model;
  repo public (user's decision).

### 2026-11-02 — Claude — Stage 1 — package and tests
- Changed: `medprune/` (attention_hook, selection, masks, scoring, stats, data, env, models, run_eval),
  `tests/` (6 files), `configs/runs.yaml`, `pyproject.toml`.
- Commits: b447543.
- Verified: 61 CPU tests passed in 7 s, including masking ≡ token deletion and ≡ FastV's physical removal (K = 1, 2) on
  tiny Qwen2 and Qwen3 models, cached decoding token-identical, logged scores = eager attention, RoPE-free GQA mapping.
- Decisions: run IDs carry no experiment name (shared runs execute once); 25 methods (the plan had said 26); mask guard
  so textured, background-free images (dermoscopy, pathology crops) are never gated; HuatuoGPT port moved to Stage 7
  because its fine-tuned vision tower may sit inside the main checkpoint.
- Open issues (verify on GPU or real data): RadImageNet field values and image size; Qwen3-VL `get_image_features`
  return format (used by DivPrune and artifact exclusion); fp16 stability on T4; Lingshu Full accuracy under our
  pipeline; `omni_anat` count vs MedSTP's 3,329.

### 2026-11-02 — Claude — Stage 0 — accounts closed; agent handbook
- Changed: Stage 0 rewritten with the user's facts (sole experimenter and GitHub writer; OpenReview and collaborators
  handled by the user; Kaggle quotas 21.8 / 30 / 30 h mapped to roles C / A / B; HF token in Kaggle secret `cvpr`).
  Added Part F (agent handbook, task queue T1–T7) and this log. Code: `run_eval --bundle` is repeatable (RadImageNet
  lives in a session-only root), `data masks --datasets` restricts mask building.
- Commits: 8a2b2b7.
- Verified: 62 CPU tests pass (new: bundle-root search order).
- Open issues: confirm the RadImageNet terms are accepted on the account behind `cvpr` (Stage 0 check).
- Next: T1 (data bundle notebook), then T2 and T3 — all three are needed before the Nov 3 kill test.

### 2026-11-02 — Claude — Part B rewrite — experiment cards; RadImageNet access
- Changed: Part B rewritten for readability and for agents: glossary (B0.1), datasets and groups (B0.2), paths,
  commands and the Kaggle session/restore rule (B0.3), experiment map E0–E9 (B0.4), and one card per experiment with
  question, rationale, setup, methods, exact command, time, outputs, expected result, how to read it, decision and
  checks. `run_eval --list` now shows done/total and pending hours per run (`--profile t4x2` on the laptop). Part F
  T2 gains the `RESTORE_FROM` cell and the `python -m medprune.checks` CLI.
- Decisions: Kaggle keeps nothing between sessions, so every multi-session block restores the previous output before
  rerunning the same command; a whole block is run by repeating one command until `--list` shows it complete.
- Verified: 62 CPU tests pass.
- Resolved: the user has RadImageNet-VQA access (Stage 0 ticked).
- Next: T1, T2, T3.

### 2026-11-02 — Claude — Tasks T1–T3 — notebooks, checks, analysis
- Changed: `notebooks/00_build_data_bundle.ipynb` (Stage 2), `notebooks/01_run_gpu.ipynb` (Stages 3–7: parameters,
  bundle discovery, `RESTORE_FROM`, per-session RadImageNet, `--list`, checks, runs, summary),
  `notebooks/02_analyze_results.ipynb` (Stage 4 section); `medprune/checks.py` (E0b–E0d + timing, CLI, report
  `e0_report__<model>__<profile>.json`); `medprune/analysis.py` (run loading, image-dependent subset, accuracy with
  image-clustered CIs and seed averaging, paired deltas + McNemar, BRR, keep-maps, query dependence, high-norm rates,
  `stage4_report` with the K1/K2 decisions). `run_eval`: copies each model × dataset gate file into `results/gates/`;
  aborts on the first question if the pruning attention never fires; `FakeAdapter` now emulates logging. Adapter:
  `set_vision_fp32`, `last_logits`.
- Verified: 69 CPU tests pass (new: `test_checks.py`, `test_analysis.py`). Notebook 02's code executed end to end on
  dry-run results (all Stage 4 tables, the decisions JSON and the keep-map PDF produced). Notebooks 00 and 01 contain
  shell steps and cannot run off Kaggle; they are thin wrappers around the tested CLIs.
- Decisions: E0d gates only on keep-set identity (≥ 49/50); FastV accuracy at 50% is reported but not gated (50
  questions are too few). Results are not zipped in notebook 01 (Kaggle zips on download; a copy would double the size).
- Open issues: first real Kaggle runs will verify RadImageNet's field values and image size, Qwen3-VL
  `get_image_features`, fp16 on T4, and the bundle mount path.
- Next (user): run notebook 00 on Kaggle (CPU) → create and share `medprune-data` → commit
  `configs/bundle_manifest.json` → Stage 3 with notebook 01 (`EXP = "E0"`, `LIMIT = 50` first).

### 2026-11-02 — Claude — Stage 2 — data bundle on Kaggle, gate revision, Stage 3 launched
- Ran notebook 00 on `pushkarsiuuarora` (v1 crashed: one OmniMedVQA source names the field `modality`; v2 fixed in
  `f8e78d9`), then on the role accounts. Counts: `slake_en` 1,061 (96 images), `slake_zh` 1,033 (96 images; only 12
  shared with `slake_en`), `omni_panel` 1,200 (8 × 150), `natural` 1,000. `omni_anat`: open-access has **16,448**
  Anatomy Identification questions (10,018 ultrasound), not 3,329 → seed-0 sample of 3,329 to match MedSTP's count and
  the S3 budget (`834bda2`); report both numbers in the paper.
- Accounts: A = `pushkararora88` (Lingshu), B = `pushkararora` (Qwen3-VL), C = `pushkarsiuuarora`. Keys live in
  `~/.kaggle-accounts/account-{3,4}` and `~/.kaggle`; `riddhitaneja`'s key is revoked (401).
- Decision: no shared dataset. Each role account builds its own bundle with notebook 00 (CPU, free) and notebook 01
  takes that kernel's output as a same-owner `kernel_sources` input. Reason: `kaggle kernels output` lists only the
  first 500 files, and private datasets cannot be shared across accounts through the API. Notebook 00 now asserts every
  dataset was built and writes `meta.tar.gz` (manifests + masks + manifest JSON) for the laptop.
- Gate revised before any GPU run (plan.md §A5, revision note): the Otsu/contrast/flatness version gated only 52% of
  SLAKE CT and 35% of OmniMedVQA CT, and failed exactly on the most background-heavy slices. Border-connected
  background with absolute limits: SLAKE CT/MRI 100%/100%, X-ray 2%; OmniMedVQA sample (12/modality) CT 92%, MRI 83%,
  ultrasound 92%, OCT 83%, fundus 50%, X-ray 25%, microscopy 8%, dermoscopy 0%. Known misses: grey (not black)
  surrounds — lung-window CTs and blue-tinted fundus. 72 CPU tests pass (3 new regression tests).
- Stage 3: E0 (`LIMIT = 50`) pushed to A (`medprune-e0-lingshu7b`) and B (`medprune-e0-qwen3vl8b`) at 16:39/16:43 on
  the v1 bundles. E0 does not read the gate, so the revision does not invalidate it; bundles are being rebuilt with the
  new masks before E1.
- Next: collect E0 reports; commit `configs/bundle_manifest.json` from the rebuilt bundle; Stage 4 (E1) on A.

### 2026-11-02 — Claude — Stage 3 — E0 results, open-question scoring fix
- E0 (`LIMIT = 50`, `slake_en`, T4×2 fp16, git `834bda2`) passed on both models: E0b fp16 50/50 identical with finite
  logits; E0c logging inert 50/50; E0d offline keep-sets 50/50. Full-model time 1.00 s/q (Lingshu, 18×18 = 324 visual
  tokens) and 0.85 s/q (Qwen3-VL, 16×16 = 256). E0f timing reported "not available" (to fix with T4).
- [FLAG fixed] Lingshu Full scored 38% under the ported strict exact match: it ends answers with a period ("CT."). The
  published Lingshu/MedSTP numbers come from MedEvalKit's GPT-4.1 judge (`eval.sh` sets `USE_LLM_JUDGE="True"`).
  Open questions now use normalized exact match; `analysis.load_runs` re-judges open answers from the saved responses
  (`rescore_open`), so scoring changes never need a GPU rerun. Rescored E0: Lingshu 82% (open 76%, closed 90%; paper
  ≈ 83%), Qwen3-VL 84%. 72 CPU tests pass.
- Rebuilt bundles (git `3188c2f`, new gate) on A and B; `configs/bundle_manifest.json` committed. Full gate rates
  (Lingshu view, share of images gated / median background): SLAKE CT 100%/0.66, MRI 100%/0.57, X-ray 0%;
  omni_panel CT 93%, MRI 85%, ultrasound 95%, OCT 63%, fundus 56%, X-ray 16%, dermoscopy 13% (all checked:
  dermatoscope discs with a black surround — real background), microscopy 7%; omni_anat CT 15% (COVID-CT lung-window
  crops with a grey surround or no surround — known limitation).
- Blocker for E1: RadImageNet-VQA needs the `cvpr` secret, which API-pushed notebooks do not receive.

### 2026-11-03 — Codex — Stage 4 takeover — identity audit and targeted recovery
- Recovered the state at `5324d86`; both long-lived notebooks were idle/complete. Independently downloaded the E1
  output to `results/receipts/e1_lingshu_v2_20261103` (original receipt retained). E1 contains all 14 run files,
  including the completed 1,061-question English SLAKE reference and 1,000 anatomy questions per RIN condition.
- Found the blocking defect: all 1,000 RIN rows had `qid=rin_anat_mc/anatomy_mc`; analysis collapsed them, diagnostic
  dictionaries overwrote them, and Random selected the same six positions for every image. Added unique per-image
  IDs and rejection of duplicate/stale identities across data, analysis and run-resume/list entry paths.
- Verified baseline: 72 tests passed before edits. Seven regression tests failed on the old implementation; four
  further tests reproduced count-only completion shortcuts, anatomy-only building and the missing registered BRR
  `[0.05,0.95]` filter. Corrected all those paths; 83 tests pass. A fresh read-only review identified the completion
  shortcut and independently checked the non-random recovery against the logged attention vectors/top-k selections.
- Recovery: `results/ops/recover_e1_identity.py` materialized `results/stage4_lingshu_20261103` without altering the
  original receipt. It preserves every response, answer and numeric diagnostic array; all 1,000 RIN rows and
  diagnostic records now have unique IDs. Legacy fixed-pattern Random runs are retained separately and excluded.
- Ruling: continue the approved main-checkout handoff; preserve unrelated/untracked work. Treat this as an
  implementation correction to the registered per-question identity contract, not a change to the predictions.
- Ruling: do not infer GO from the legacy Random pattern. Run only corrected `random_l2` and `random_l0` on
  `rin_anat_mc`, seed 0, 10% budget, role A (University), in a fresh output namespace; no Qwen main grid yet.
- Additional source mismatch: the original notebook continued after `rin_abn_closed` failed to build. Anatomy output
  exists and is usable, but the Stage 5 abnormality subset must be resolved from real template metadata first.
- Next: targeted Random repair, independent paired/statistical verification, then the K1/K2 report. Raw data,
  response files and diagnostic arrays remain in git-ignored `results/`.

### 2026-11-03 — Codex — corrected E1 decisions and source-template repair
- Committed/pushed the identity/resume correction as `dbe9ab5` after 83 CPU tests and a focused review. All original
  E1 files remain in `results/receipts/e1_lingshu_v2_20261103`; recovered data live separately in
  `results/stage4_lingshu_20261103` with `identity_recovery_audit.json`.
- Role A (University), long-lived notebook `medprune-gpu-lingshu7b`: v3 stopped before model loading because Kaggle's
  base Transformers was 5.16.1, not E1's 4.57.1. V4 restored the existing project pin and completed the two corrected
  Random conditions. Both have 1,000 unique IDs, images and per-image keep-sets; all rebuilt input fields and gate
  arrays match recovered E1. Torch 2.11.0+cu128, Transformers 4.57.1, Python 3.13.15, two T4s, fp16; Lingshu model
  revision `b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9`. No model changes or new dependency-version choice.
- Fresh Random inference: 1,000×0.356 s (L2) + 1,000×0.352 s (L0), about 11.8 minutes of question evaluation,
  excluding download/model load. Receipt: `results/receipts/random_repair_v4_20261103`.
- **K1 GO:** RIN Full 89.0, Random@L2 79.3, FastV 38.1, Gate+FastV 51.3 (%). FastV−Random = −41.2 points,
  paired 95% CI [−45.0, −37.6]. Pre-LLM: 37.4 vs 79.2, Δ −41.8 [−45.5, −38.2]. Independently reproduced all six
  all-question paired CIs from image-cluster sums (2,000 bootstrap draws, seed 0), outside the analysis helper.
- **K2 H1 alive:** FastV BRR 2.320 [2.251, 2.391] on 874 eligible RIN images and 1.668 [1.573, 1.764] on SLAKE
  CT/MRI; corrected Random BRR 0.974 [0.924, 1.025] and 0.999 [0.983, 1.014]. The registered background-fraction
  [0.05,0.95] eligibility rule is applied. This is over-selection against a heuristic mask, not causal attribution.
- The gate gains 13.2 [10.3,16.2] points on RIN, but recovers only 25.9% of Full−FastV and remains 28.0 points below
  Random. On SLAKE CT/MRI it gains 19.29 [15.41,23.33]. Do not claim a universally competitive fix from this pilot.
- Preserved failed predictions: Full SLAKE 77.19 misses 81–85; X-ray FastV−Random point estimate is −1.94
  [−4.93,+0.87]; Lingshu attention-selection mass is top/top-left rather than predicted bottom-right. P1–P5 Outcome
  cells updated without changing any frozen forecast. Keep regression and positional/norm causal controls remain.
- The V4 source-schema summary resolves the empty abnormality subset: `abnormal_closed` / `pathology` / `closed`,
  not `content_type` containing `abnormal`. Corrected the selector and replaced invented fixture values with the
  actual schema; added a distractor-first regression excluding disease-specific templates. Two tests failed before
  the correction; **84 CPU tests now pass**. Focused independent review found no important defect. Real 1,000-image
  build/pairing remains a Stage 5 prerequisite.
- Role A v5 is running only the 43/1,061 truncated SLAKE text-only items at the already registered 512-token limit.
  It pins the same model revision/runtime and verifies unchanged greedy prefixes. K1/K2 do not depend on these
  text-only answers; image-dependent-subset tables remain provisional until this retry is checked and merged.
- The Sonatype dependency-check endpoint returned HTTP 401; no security-audit success is claimed. The remote
  installation restores the existing approved Transformers pin to match E1, rather than adding a dependency.
- Qwen main grid remains idle pending the full-baseline scoring and real abnormality-build preflight. No `rin_open`
  condition was added. Raw responses, numerical diagnostics, operational notebooks and receipts stay ignored.

### 2026-11-03 — Codex — Stage 4 handoff complete; preflight evidence reconciled
- Role A v5 completed the registered 512-token retry for all and only the 43 originally truncated SLAKE text-only
  answers. All 43 greedy prefixes match the original 64-token outputs. Two remain truncated (2/1,061 = 0.19%, below
  the registered 1% threshold). Accuracy remains 24.51%; every primary paired contrast and K1/K2 is unchanged.
  Retained originals and merged only those 43 rows in the derived tree; receipt `results/receipts/text_retry_v5_20261103`.
- Role A v6 ran the corrected real RIN builder on **CPU, no GPU**. Both subsets have 1,000 unique questions/images
  and identical image-ID sets; rebuilt anatomy inputs match E1. All raw images are 224×224; modalities 426 CT and
  574 MRI. Abnormality uses `abnormal_closed`, labels 719 yes / 281 no; always-yes accuracy is analytically 71.9%.
  Receipt `results/receipts/rin_cpu_preflight_v6_20261103`; summary copied to the derived analysis root. No RIN images
  or source parquet were put in saved outputs.
- Reconciled stale tracker, Stage 2 source-sharing instructions, C2 counts and F4 task statuses with the actual
  completed runs. User-authorized agent Kaggle operations supersede the early manual-only run rule. Scientific
  amendments preserve original evidence and exclude invalid runs rather than deleting their history.
- Reproduced the first-50 Lingshu score: 82% on only six images. Full English SLAKE: 77.19% on 96 images; open
  71.94% (645 questions), closed 85.34% (416). The full-set scoring-comparability gate is unresolved; no scorer was
  changed to raise these results. Image-dependent-subset results are secondary and selected by construction.
- Wrote `results/stage4_lingshu_20261103/STAGE4_REPORT.md`, `stage4_verification.json`, updated tables/figures and
  retained `identity_recovery_audit.json`. Primary K1/K2 and P1–P5 outcomes are reported; E2 keep regression and later
  positional/norm causal controls remain pending. T8 is complete. T9 is the next preflight task before the large grid.
- Last status checks: role A's v6 CPU job COMPLETE; role B's long-lived Qwen notebook COMPLETE/idle. No new Qwen
  main-grid run was submitted, and no `rin_open` condition was added. The original takeover request to recover E1,
  independently analyze it and report GO/REFRAME/STOP is complete; this is not a claim of a validated remedy or paper.

### 2026-11-03 — Codex — user-requested first Qwen E3 block
- User request: "now we run the qwen nb?" Ruling: execute a bounded within-Qwen replication using ten existing E3
  conditions, rather than treating the request as permission to spend all 16.3 planned Qwen grid hours. Lingshu's
  published-baseline scoring comparison remains unresolved. No scorer, prompt, method, budget or prediction changes.
  Cost if the effect does not generalize: approximately 2–3.5 session hours, not the entire Qwen campaign.
- Selected role B/Father, existing private `medprune-gpu-qwen3vl8b`. Verified idle state and its same-owner
  `medprune-00-build-data-bundle` input COMPLETE/accessibility. Downloaded previous output to
  `results/receipts/qwen_before_stage5_20261103` and pulled the source/metadata before replacing the version.
- Prepared and reviewed `results/ops/qwen_stage5_block_submission_v2_reviewed`: `rin_anat_mc` and `slake_en`, each with Full, text-only,
  Random@L2 seed 0, FastV, Gate+FastV. 10,305 primary question evaluations plus 50 smoke questions. Rebuild RIN anatomy
  only under `/tmp`; require exact matching input-field signatures against recovered Lingshu and exact SLAKE manifest
  SHA. No `rin_open`, abnormality grid, extra seeds, OmniMedVQA, or E4 in this first block.
- Pinned code `f53a663`, Qwen model revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` (live official model API),
  existing Transformers 4.57.1 requirement, Torch 2.11.0+cu128, fp16 T4×2; seed 42 for Torch, seed 0 per-question Random.
  Five-question smoke runs for all ten conditions precede the bulk loop. Registered truncation retries at 512 are
  saved separately for verified merge; original responses remain preserved. No E0 inference is silently restored.
- Verification before submission: all code cells compile, selected run IDs match the registered E3 matrix, planned
  inference 2.345 h, metadata GPU/internet/private settings checked; **84 CPU tests freshly passed in 7.63 s**.
  The dependency-check endpoint again returned HTTP 401; no component-security assessment is claimed. Installation
  restores the existing approved experimental pin; no new dependency or version recommendation was introduced.
- Focused review found two important gaps before submission: historical RIN signatures omitted MC choices, and long
  retries lacked a separate budget check. Resolved both. Pin current RIN source revision
  `9add2548cca069f088b9d0e44bfac849a53a5c31`, record full prompts/ordered options, manifest and parquet digests, and
  explicitly mark historical Lingshu full-MC-prompt equivalence unverified because that digest was never saved.
  The recorded historical fields match; current within-Qwen inputs are frozen across every condition. Rebuild SLAKE
  masks from pinned code as well, using immutable input images/manifests via a temporary symlinked bundle.
- Retries are queued after bulk inference, estimated conservatively from measured truncated-item time, and only
  scheduled if they fit. Pending IDs are saved. The budget is checked even before the first bulk run. Completion
  markers distinguish bulk completion from verification readiness and final analysis; 3.5 h is a scheduling
  allowance, not a hard kill timer. Four synthetic control-flow checks passed (normal, defer retry, stop before
  first bulk run, no truncation); these are orchestration evidence, not model results. Focused re-review found no
  important remaining defect.
- Submitted **role B/Father, `medprune-gpu-qwen3vl8b`, version 2**, private, GPU requested (T4×2), internet enabled,
  same-owner builder input attached. API accepted the submission and returned **RUNNING**. Receipt:
  `results/ops/qwen_stage5_block_submission_v2_reviewed/account_operation_receipt.json`. This confirms the notebook
  run started; runtime/input assertions, smoke checks, inference and final accuracy still require its outputs.
