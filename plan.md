# Plan — Pruning the Air

Single source of truth for the project: research (Part A), execution by stages (Part B), protocol (Part C), code
design (Part D), compute backends incl. SSH A100 (Part E), agent handbook and task queue (Part F), work log (Part G,
the last section). Version 2 (merged), 2026-11-02.

**AI agents (Codex, Gemini, Claude): read Part F first, then the newest entries of the Work log (Part G).**

Conventions: ✓ = verified on 2026-11-02 against the named source. Stages run in order; a stage starts only when the
previous stage's **exit gate** is met (parallel work is marked). Tick boxes on GitHub as work completes. Predictions in
§A6 are frozen as of 2026-11-02 — record outcomes, never edit predictions.

## Stage tracker

Updated 2026-11-07 03:20 IST. **Update after every Kaggle session:** status, the Spent column, the ledger below, and a
Work log entry. GPU-h = Kaggle T4×2 session hours (what the quota counts), measured from the kernel logs.

| Stage | Work | Status | GPU-h spent | GPU-h left | Dates (IST) | Exit gate / result |
|---|---|---|---|---|---|---|
| 0 | Accounts, access, repo | ✅ DONE | 0 | 0 | Nov 2 | Accounts, HF secret `cvpr`, RadImageNet access |
| 1 | Code and tests | ✅ DONE | 0 | 0 | Nov 2–3 | 116 CPU tests pass (Nov 4: + judge rubric guard) |
| 2 | Data bundle | ✅ DONE | 0 (CPU) | 0 | Nov 2–3 | A, B and C rebuilt with the revised gate; C (v3, git `1c4fef2`) has the same manifest hashes and mask summaries as A |
| 3 | Validation (E0) | ✅ DONE | 0.39 | 0 | Nov 2 | E0b–E0d 50/50 on both models; SLAKE scoring contract moved to the Stage 5 preflight |
| 4 | Kill test (E1) + diagnostics (E2) | ✅ DONE | 3.35 | 0 | Nov 2–3 | **K1 GO** (FastV − Random −41.2 [−44.9, −37.6], RIN, Lingshu); **K2: H1 alive** (BRR 2.32) |
| 4b | Gate + coverage selectors (E1g) | ✅ DONE | 2.56 | 0 | Nov 3 | Rule met on Lingshu only (`gate_random` − Random +3.2 [+1.2, +5.2]); Qwen +0.5 [−1.3, +2.3] → **diagnosis-first paper**, gate = partial remedy |
| 5 | Main grid (E3) + mechanism controls on K (E6, pulled forward) | ✅ DONE (A v8 9.46 h, B v4 9.76 h; 85 runs, all complete; merged Nov 4 17:30) | 20.95 | 0 (rest of E3 in session 2) | Nov 3–4 | **H3 dead** (`artifact_excl` = FastV); **P12 contradicted** (background-only beats FastV); **K16 beats Random** (P11); static map ≈ FastV on Lingshu SLAKE; OmniMedVQA anatomy −29.4 / −4.3 |
| 5b | Confound controls: natural (E7), resolution (E10, E10r), flip + bottom-k (E11, E11r) | ✅ DONE. C half Nov 3 (38 runs); RadImageNet arms (E10r, E11r) ran in A/B session 2 (stage 6) | 6.14 | 0 | Nov 3–4 | **P13 contradicted**: FastV < Random on photographs (−8.7 / −4.9). **G1**: token count is a modest amplifier; RadImageNet's collapse is dataset-specific. **G3**: Lingshu's keep-set is positional (flip). **G4**: Lingshu's least-attended tokens beat its most-attended (+6.2) |
| 6 | **E12 layer profile + text readout (D1, D2)**, modality panel (E4), budget curves (E5), RIN 448² / flip (E10r, E11r) | ✅ DONE. B v5 6.30 h (79 runs; merged Nov 5 00:30); A v9 9.43 h (87 runs, 86 complete; merged Nov 5 03:40; judge pass 7 merged). Missing: Lingshu SLAKE-zh `gate_fastv` (0.32 h, did not fit) | 15.73 | 0.32 (A) | Nov 4–5 | **E10r**: more tokens do not help (Lingshu −48.9 at 256 tokens vs −41.7; Qwen3-VL −29.2 vs −33.7). **`fastv_text` = FastV** (Lingshu +0.5 / +0.2). **E5**: Lingshu RIN loss flat 5→25% (−39.8 / −41.7 / −43.9). **P9 ✗**: no background dose-response across modalities. **E12**: question dependence starts at layer 14 (Lingshu) / 17 (Qwen3-VL) |
| 6b | **E13 published pruners**: SparseVLM, VisPruner, PruneSID, BTP (official selection code, medprune prompts; `medprune/baselines/`) on SLAKE, RIN anatomy, RIN abnormality, OmniMedVQA anatomy | ✅ DONE (A 4.74 h, B 4.34 h; 32 runs, all complete; merged Nov 5 00:10). SLAKE judge-scored (pass 5 merged) | 9.23 | 0 | Nov 4–5 | **Only LLM-attention scoring collapses**: SparseVLM fails like FastV on Lingshu (−44.8 RIN anatomy), not on Qwen3-VL (+0.1); VisPruner / PruneSID ≈ Random. Gate+Uniform ≥ all published on Lingshu RIN; on SLAKE (judge) no non-attention method separates from another. P15 ✗, P16 ✗, P17 ✗, P18 ✓ |
| 7 | Supplementary (E6-P2, E8 HuatuoGPT, E9 latency) + open-answer judge | ✅ DONE. E8r, E6-P2 (both halves), judge passes 4–8 merged. **C v4 ✅** (2.71 h, 10 runs, merged Nov 5 07:30): the 8 registered Lingshu runs A did not reach, Lingshu DivPrune on OmniMedVQA anatomy, **E9 latency** and Lingshu prompt tokens. **B v6 ✅** (1.38 h, 4 runs): Qwen3-VL DivPrune on RIN abnormality + OmniMedVQA anatomy. Registry: 324 of 324 registered runs complete. Judge: **κ 0.81 (as pre-registered) → 0.90 with the rubric guard** vs Codex's labels (AI labels, not human — §G Nov 4) | 14.45 | 0 | Nov 3–5 | G6 answered; on RadImageNet the gate repairs a third of FastV's deficit at most. **E9**: FastV at 10% → prefill 2.95× (paired 3.07×), first-token path 1.69× (vision encoder 183 ms is untouched); the gate alone saves nothing. **Oracle organ gate = heuristic gate** on SLAKE (−0.9, n.s.) |
| 7b | **E14** six more published pruners (VisionZip ± gate, HiPrune, CDPruner ± gate, FEATHER, VScan, TransPrune; G19) on the four baseline cells; **E15** RadImageNet open questions (`rin_open`, D5); **E16** SparseVLM raters ablation (D-15) | 🔄 RUNNING. Nov 7 03:14: C v5 (E15, Lingshu, judged in-session), A v10 (E16, Qwen3-VL then Lingshu); D v1 × 2 E14 smoke ✅ (03:22, 32 runs × 3 questions); 03:24: D v2 × 2 (E14 SLAKE + OmniMedVQA, SLAKE judged in-session), B `medprune-baselines-qwen3vl8b` v3 + `medprune-gpu-qwen3vl8b` v7 (E14 RadImageNet cells) | 0 | ≈ 21 (D 14, B 3.3, C 2.4, A 1.2) | Nov 7 | G19 closes with vision-encoder methods in the table; CDPruner tests question-awareness without LLM attention; FEATHER tests the RoPE fix on M-RoPE models |
| 8 | Analysis and writing | ⏳ LATER | 0 | 0 | Nov 6–23 | **Paper Nov 16 AoE**, supplementary Nov 23 AoE |

**Nov 7, 03:20:** E14–E16 launched (Stage 7b): ≈ 21 GPU-h planned, within every account's free hours (C 2.8 h left
before v5 by the user's count, A 1.4, B 5.0, D 30). **Totals (Nov 5, 07:40):** spent 72.8 GPU-h (A 28.58, B 25.02, C 19.21) · running 0 · free this week ≈ 9.0
(A 1.4, B 5.0, C 2.6) + D 30 (bundle built; GPU unused). Every registered run is complete. Earlier estimate (measured rates, §C4): **≈ 61
GPU-h for the whole project** (48 before Nov 3 evening; the added hours are the diagnostic controls of §A9 — E6 on both
models before the grid, E10, E11, Random seeds — and E12, the layer profile and text readout of §A10, 1.0 + 0.9). The
judge moved from B to C. On an A100 (Part E): ≈ 14–16 h.

**Compute ledger (Kaggle T4×2 hours)**

| Role | Account | Weekly quota | Spent | Running | Left in plan | Slack |
|---|---|---|---|---|---|---|
| A — Lingshu | `pushkararora88` | 30 | 29.70 (E0 0.19, E1 2.92, repairs 0.44, E1g 1.32, session 1 9.46, E13 4.74 + smoke ≈ 0.08, session 2 9.43, v10 E16 1.12) | 0 | 0 | ≈ 0.3 |
| B — Qwen3-VL | `pushkararora` | 30 | 26.52 (E0 0.20, Qwen K-block 1.73, E1g 1.24, session 1 9.76, E13 4.34 + smoke ≈ 0.07, session 2 6.30, v6 DivPrune 1.38, E14 Qwen3-VL RIN 1.50) | `medprune-gpu-qwen3vl8b` v7 (E14 Lingshu RIN) | 0 | weekly reset ≈ 05:30 IST Nov 7 |
| C — supplementary | `pushkarsiuuarora` | 21.8 | 19.21 (E7/E10/E11 6.14; HuatuoGPT E0 0.23, E8 SLAKE 3.34, E8r 2.70; E6-P2 1.93 + 1.29; judge passes 1–8 0.10 + 0.21 + 0.11 + 0.11 + 0.06 + 0.08 + 0.10 + 0.10; Lingshu v4 2.71: 8 missing runs, DivPrune OmniMedVQA, E9) | v5 E15 (Nov 7 03:14, ≈ 2.4 h; the user counted 2 h 47 min left) | 0 | ≈ 0.4 |
| D — 4th account (added Nov 5) | token via `KAGGLE_API_TOKEN` for one command; never stored | 30 | 0 (bundle built on CPU, 0.35 h, Nov 5 04:51; manifest sha256 = C's) | v1 × 2 E14 smoke (Nov 7 03:14, ≈ 0.6 h) | E14 SLAKE + OmniMedVQA, both models (≈ 13.5 h) | ≈ 16; the `cvpr` secret is not attached to D's GPU notebooks (created Nov 7 by API), so D runs no RadImageNet cell |

Slack ≈ 20 GPU-h. E12 (D1 layer profile, D2 `fastv_text`) is now scheduled in A/B session 2. Next uses of slack:
RadImageNet open questions (`rin_open`, MedSTP's strongest column, scored by the judge). HuatuoGPT P2 blocks (incl.
its layer profile, the 1D-RoPE case) are done. Restore datasets for the A/B sessions:
`medprune-restore-<model>` (private, one per account; run files + static maps, `RESTORE_FROM = "auto"`).

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

*Revision 2026-11-03 (evidence from E1, E1g and the offline diagnostics, §A9; the frozen predictions in §A6 stand).*
The rotary component is not supported: FastV's keep-set sits in the **top** rows (P4 contradicted), and RoPE-free
layer-1 scores pick almost the same tokens (Jaccard 0.94 Lingshu, 0.77 Qwen3-VL on RadImageNet). Layer-1 attention
from the last prompt token behaves as a **question-blind, largely image-blind prior**: keep-sets of different questions
about one image overlap 98% (Lingshu) / 82% (Qwen3-VL); at 64 tokens the keep-set is nearly the same for every image
(cross-image Jaccard 0.77 / 0.47); and within Random's own draws, high-attention tokens predict *lower* accuracy for
Lingshu. Background capture is real but is only one of two regimes: on SLAKE, FastV loses 16–20 points on CT/MRI and
1.9 on X-ray, and the gate recovers all of it; on RadImageNet, FastV loses 45 points even on images with no background.
**Working thesis:** early-layer attention pruning applies a fixed prior instead of reading the image or the question;
it fails where the prior lands on acquisition background (CT/MRI) and where the budget is too small for a fixed prior to
cover the content (64-token inputs). E10 (resolution), E11 (flip, bottom-k) and E7 (photographs) test which half is
domain-specific.

## A2. Claims and required evidence

| Claim | Evidence (reviewer bar) | Experiments | Paper element |
|---|---|---|---|
| **C1 Phenomenon**: early attention-guided pruning selects worse than random on medical images | 2 models (medical + general) × 3 benchmarks; paired CIs excluding 0; several budgets; image-dependent subset; natural-image contrast | E1, E3, E5, E7 | Table 1, Fig. 4 |
| **C2 Predictability**: background fraction predicts the size of the failure | Monotone dose-response across 8 modalities and across images; none in background-free modalities | E3, E4 | Fig. 2 |
| **C3 Mechanism**: the keep-set follows position and background, not content | Keep-maps match the rotary prediction (1D vs M-RoPE); position-only static map ≈ FastV; background-only keep-set ≈ FastV; regression separates position from background; K-sweep shrinks the gap | E2, E6, E8 | Fig. 3, Tables 2–3 |
| **C4 Remedy**: a zero-parameter gate recovers the loss and composes with pruners | Gate+FastV ≫ FastV where background exists, ≈ where not; 2×2 factorial vs RoPE-free; DivPrune/PDrop ± gate; budget curves; latency | E3, E5, E6, E9 | Tables 3–4, Fig. 4 |
| **C5 Protocol**: report image-dependent accuracy, image-clustered CIs, background fraction | At least one conclusion that shrinks or flips under the stricter protocol | E3 analysis | §3 + supp |

If C3 fails but C1, C2, C4 hold, the paper stands as empirical + fix. If C1 fails in our pipeline, stop (K1).

**Status (Nov 3, evening).** C1 holds on RadImageNet for both models and on SLAKE for Lingshu; Qwen3-VL SLAKE waits for
the open-answer judge; generality waits for OmniMedVQA (E3) and photographs (E7). C2 holds on SLAKE (per-image Spearman
−0.35 Lingshu, −0.27 Qwen3-VL) but **not** on RadImageNet (FastV − Random is −45 on its 111 background-free images), so
C2 becomes "two regimes": background capture, measured by the dose-response, plus a background-independent collapse at
small token counts (E10 decides whether the second is about tokens or domain). C3 is now carried by question
independence, image independence, the RoPE-free control and the attention-informativeness test, not by rotary
geometry. C4 is partial: the gate closes FastV's whole SLAKE gap but recovers only 13–15 of 34–41 RadImageNet points,
and gate + coverage selection beats Random by 2–3 points on Lingshu only. The paper is diagnosis-first (§A9).

*Nov 3, night (offline, exploratory; §A9 G16–G18).* Two measurements sharpen C3. (i) **Variance decomposition** of the
layer-1 token ranks: on SLAKE the question explains 0.3% (Lingshu) / 3.1% (Qwen3-VL) of the variance, the image beyond
position 94% / 85%, position 5% / 12%; on RadImageNet's 8×8 grid position alone explains 69% / 51%, and RoPE-free
scores give the same split (69% / 47%) — the positional prior is carried by the visual tokens' keys, not by rotary
decay. Question dependence appears only by layer 15 and only in Lingshu (23%). (ii) **Evidence grounding** with
SLAKE's organ masks (`analysis evidence`; 358 / 364 questions that name an annotated structure): FastV keeps the
asked-about structure at 0.84× [0.57, 1.11] (Lingshu) / 0.61× [0.43, 0.80] (Qwen3-VL) of chance — **0.20× / 0.23×
for large organs** (top area tertile, median 5.2% of the image), dropped entirely in 58% / 71% of questions (Random
13% / 11%), but above chance on small structures for Lingshu (1.46×). The large-organ tertile carries Lingshu's SLAKE
loss: −16.8 [−25.2, −7.1] vs −5.0 (medium) and −2.5 (small); Gate+FastV − Random there is +1.7 [−5.5, +8.5]. Qwen3-VL's
large-organ loss is −5.0 [−10.6, +1.7]. H4 (small evidence) is therefore reversed: FastV fails on the big organs.

*Nov 3, 21:45 (C's controls, E7/E10/E11).* C1 must drop "medical-specific": FastV is below Random on photographs too
(−8.7 Lingshu, −4.9 Qwen3-VL; P13 contradicted). What is medical is the size of the loss: SLAKE CT/MRI (background,
gate-recoverable) and above all RadImageNet (−41 / −34), which 64-token inputs alone do not reproduce (SLAKE at 224²:
−16.3 / −0.5; photographs at 224²: −10.7 / −2.7). C3 gains two interventions: flipping SLAKE images leaves Lingshu's
keep-set in place (positional), and Lingshu's least-attended 10% beats its most-attended 10% by 6.2 points.

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
| P4 | E2 | Keep-mass ≥ 1.5× uniform in bottom-right (Lingshu/Qwen3-VL), bottom rows (HuatuoGPT) | 60% | Contradicted on Lingshu: bottom-right concentration 0.018× uniform (RIN), 0.427× (SLAKE CT/MRI); strongest RIN concentration is top-left (2.52×). **Supported for HuatuoGPT** (Nov 4, E8 SLAKE): 77% of FastV's keep mass in the bottom quarter of rows = 3.1× uniform. |
| P5 | E2/E6 | High-norm artifacts over-selected ≥ 2× | 35% | Norm-threshold proxy supports over-selection: 2.62× RIN and 5.07× SLAKE. **Not causal** (E6, Nov 4): excluding them (`artifact_excl`) leaves FastV unchanged — Lingshu −42.0 vs −41.7 (RIN), −12.3 vs −12.4 (SLAKE); Qwen3-VL identical. H3 dead as a cause. |
| P6 | E3 | Gate+FastV − FastV ≥ +8 on CT/MRI; \|Δ\| ≤ 1 on dermoscopy/microscopy | 55% | CT/MRI half (Nov 4): **supported on Lingshu** (SLAKE CT/MRI +19.3 [+15.4, +23.3]; RIN +13.2) and on Qwen3-VL RIN (+15.0); Qwen3-VL SLAKE CT/MRI +2.4 (n.s.). Dermoscopy/microscopy half (Nov 5, E4 panel, n = 150 each): Gate+FastV − FastV microscopy +1.3 / +1.3, dermoscopy +4.7 / −0.7 (Lingshu / Qwen3-VL) → met within 1 point in 1 of 4 cells, within 1.3 in 3; the gate leaves almost every token in these 5%-background images, so the +4.7 is noise or ranking, not removed background. |
| P7 | E3 | Gate+Uniform within 3 of Gate+FastV on CT/MRI | 50% | SLAKE CT/MRI: supported (+2.4 both models). RIN: contradicted — Gate+Uniform beats Gate+FastV by 30.3 (Lingshu) / 20.0 (Qwen3-VL); the gate leaves FastV's RIN collapse mostly in place. |
| P8 | E3/E6 | Gate adds ≥ 3 on top of RoPE-free on CT/MRI | 50% | Lingshu supported (SLAKE CT/MRI +16.7, RIN +15.0); Qwen3-VL RIN +15.5, SLAKE CT/MRI +0.4 (n.s.). |
| P9 | E4 | Modality-level Spearman ρ ≥ 0.7; per-image ρ > 0 (CI excl. 0) | 55% | **Contradicted** (Nov 5, OmniMedVQA panel, 8 modalities × 150, judge-free): modality-level ρ(background fraction, FastV loss) −0.26 (Lingshu), −0.82 (Qwen3-VL, wrong sign); per-image ρ +0.075 [+0.022, +0.130] / −0.073 [−0.132, −0.015]. Lingshu loses 12–13 on microscopy and dermoscopy (5% background) and 56 on ultrasound; Qwen3-VL loses only on microscopy (−6.0). C2 as worded is not supported. |
| P10 | E3 | Qwen3-VL same sign as Lingshu on CT/MRI, smaller magnitude | 60% | Supported: FastV − Random on SLAKE CT/MRI −4.7 vs −18.0, RIN anatomy −33.7 vs −41.7, RIN abnormality −2.2 vs −14.2, OmniMedVQA anatomy −4.3 vs −29.4. |
| P11 | E6 | Gap shrinks monotonically with K; K=16 ≈ Random | 65% | Contradicted in form: not monotonic — K=8 is as bad as K=2 (Lingshu RIN −43.9 vs −41.7; SLAKE −12.4 both), then K=16 **beats** Random (Lingshu +8.5 RIN, +6.9 SLAKE; Qwen3-VL +5.6 RIN, +0.9 SLAKE). The failure belongs to the early layers (≤ 8); E12's layer profile locates the switch. **RIN abnormality (Lingshu, Nov 5):** K=8 −17.4, K=2 −14.2, K=16 −6.6 [−9.3, −3.9] → same non-monotone shape, but K=16 stays **below** Random on the abnormality task. |
| P12 | E6 | Inverse gate within 5 of FastV on CT/MRI | 50% | **Contradicted**: background-only tokens beat FastV — SLAKE CT/MRI −7.5 vs −18.0 (Lingshu), +2.0 vs −4.7 (Qwen3-VL); RIN −23.9 vs −41.7 / −7.6 vs −33.7. FastV is worse than background. RIN abnormality (Lingshu, Nov 5) is the one cell that meets P12: inverse gate −14.5 vs FastV −14.2. |
| P13 | E7 | Natural images: FastV ≥ Random − 2 | 50% | **Contradicted** (Nov 3): FastV − Random on 1,000 photographs (500 GQA + 500 POPE) = −8.7 [−11.7, −5.8] Lingshu, −4.9 [−7.8, −2.1] Qwen3-VL; RoPE-free the same (−8.5 / −4.7). The failure is not medical-only; medical scans amplify it (§A2). |
| P14 | E6 | Residual largest in smallest evidence tertile | 55% | Preliminary (E1/E1g runs, `analysis evidence`, Nov 3): contradicted. Gate+FastV − Random is flat across SLAKE evidence-area tertiles (Lingshu +3.3 small, −3.4 medium, +1.7 large; all CIs include 0); FastV's own loss sits in the **largest** tertile (−16.8 [−25.2, −7.1] vs −2.5 small), where it keeps the named organ at 0.20× chance. **Final (Nov 5, judge-scored, `oracle_gate`): contradicted.** Gate+FastV − Random by tertile +1.7 small, −3.4 medium, +4.2 large (all CIs include 0); the oracle organ gate's gain sits in the **largest** tertile (+10.1 [+1.7, +17.1]; small 0.0). |
| P15 | E13 | SparseVLM (text raters, from layer 2) − Random ≤ −10 on Lingshu RIN anatomy; ≤ 0 on Qwen3-VL RIN anatomy. *Added Nov 4, 20:10, before any E13 result.* | 60% | **Half contradicted** (Nov 5): Lingshu −44.8 [−48.0, −41.5] ✓; Qwen3-VL +0.1 [−1.8, +2.1] ✗ — where FastV loses 33.7, SparseVLM ties Random. Its first keep-set overlaps FastV's near the maximum on Lingshu (Jaccard 0.66–0.68 of ≤ 0.70), not on Qwen3-VL (0.33). |
| P16 | E13 | VisPruner within 3 of Random on SLAKE; its background retention ratio > 1 on CT/MRI (vision-tower attention favours flat background tokens) | 50% | **Contradicted on Lingshu, supported on Qwen3-VL**: SLAKE − Random +3.2 [+1.5, +5.1] (better than the band) / +1.1; BRR 0.96 / 1.34. Vision-tower attention does not prefer background on Lingshu. |
| P17 | E13 | PruneSID within 3 of Random on all four datasets, both models (coverage family) | 60% | **Contradicted** (5 of 8 cells): misses are Lingshu RIN abnormality −4.1, Lingshu OmniMedVQA +3.5, Qwen3-VL SLAKE +4.5; no collapse anywhere (range −4.1 to +4.5). |
| P18 | E13 | BTP ≥ FastV + 10 on Lingshu RIN anatomy (diversity in its first two stages) and within 3 of Random on SLAKE | 55% | **Supported**: Lingshu RIN anatomy 70.0 vs 38.1 (+31.9); SLAKE − Random −2.7 / +2.5. Still −9.8 vs Random on Lingshu RIN: the attention half of its stages carries the bias. |

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

## A9. Gap register — reviewer objections → tests (opened 2026-11-03)

The diagnosis carries the paper, so every objection a reviewer can raise against it gets a named test here. Update the
Status column as results land; numbers come from the Work log (Nov 3, "Offline diagnostics").

| ID | Objection a reviewer will raise | Evidence now | Decisive test | Status | Pri |
|---|---|---|---|---|---|
| G1 | "It is the 64-token input, not the medical domain" | RIN (224², 64 tokens) −41.2 vs SLAKE (324 tokens) −12.6; background-free RIN images −45 but SLAKE X-ray −1.9 | E10: SLAKE + photographs at 224²; E10r: RIN at 448²; E7: photographs at native size | **E10 read (Nov 3)**: at 64 tokens FastV − Random is −10.7 photos, −16.3 SLAKE, −41.2 RIN (Lingshu); −2.7, −0.5, −33.7 (Qwen3-VL). Token count is a modest amplifier (Lingshu −8.7 → −10.7, −12.6 → −16.3; none for Qwen3-VL); RadImageNet's collapse is not reproduced at matched tokens → dataset-specific (G22). **E10r read (Nov 5)**: RIN at 448² (256 tokens) FastV − Random −48.9 [−52.3, −45.4] Lingshu (−41.7 at 64), −29.2 Qwen3-VL (−33.7) → token count does not explain RadImageNet | Closed | P0 |
| G2 | "Your FastV is not FastV" | Masking ≡ physical removal (`fastv_prellm` 37.4 vs `fastv` 38.1); our layer-1 scores reproduce the FastV run's keep-set (Jaccard 1.000 / 0.999); last-token scoring follows the official code (§A4) | Variant scoring by mean attention of all text tokens after the image (the FastV paper's wording) | **Closed (Nov 5)**: `fastv_text` − FastV +0.5 [−0.4, +1.4] RIN, +0.2 SLAKE (Lingshu); Qwen3-VL RIN −27.8 vs −33.7 vs Random. The readout is not the cause | P1 |
| G3 | "The top rows are just uninformative content" | Same positions across images (cross-image Jaccard 0.77 / 0.47 on RIN) | E11/E11r: same images upside down — a keep-set that stays in place is positional; `static_map` (E6) | **E11 read**: on flipped SLAKE, FastV keeps the same grid positions (Jaccard 0.64 Lingshu, 0.36 Qwen3-VL) rather than the moved content (0.13 / 0.20; random 0.05) → Lingshu positional, Qwen3-VL mixed. **E11r (Nov 5)**: on flipped RIN the collapse stays (FastV − Random −33.6 Lingshu, −28.3 Qwen3-VL) | P0 |
| G4 | "Attention is noisy but informative on average" | Within Random's draws, +1 SD of kept-token attention rank → −0.14 to −0.35 log-odds for Lingshu (p < 0.05 in 4/6 cells); ≈ 0 for Qwen3-VL | `fastv_bottom` (keep the least-attended tokens; E11/E11r); re-fit with Random seeds 1–2 (E6) | **E11 read**: on SLAKE, bottom-10% − top-10% = **+6.2 [+2.5, +10.1] Lingshu** (attention ranks useful tokens last) and −4.2 [−7.2, −1.4] Qwen3-VL (weakly informative); both below Random (−6.4 / −6.6). **E11r (RIN, Nov 5)**: least- minus most-attended **+40.0** [+36.2, +43.7] Lingshu, +29.9 Qwen3-VL; least-attended − Random −1.7 (n.s.) / −3.8 → on RIN early attention ranks the useful tokens last | P0 |
| G5 | "FastV is text-guided; you test the wrong questions" | Keep-sets of different questions about one SLAKE image overlap 0.978 [0.974, 0.982] (Lingshu) / 0.820 [0.805, 0.836] (Qwen3-VL), Random 0.054; the question explains 0.3% / 3.1% of the variance of layer-1 token ranks (exact grid positions: position 40% / 27%, image 59% / 70%; layer 15: question 23% / 3.5%) — and ≤ 1.2% on photographs too, so question-blindness is general, not medical; the scoring token is a fixed template token | Report as a finding; the layer profile (D1) shows where question dependence starts, per readout | Analysis done | P0 |
| G6 | "Two Qwen-family models with M-RoPE" | — | HuatuoGPT-Vision-7B (LLaVA, 1D RoPE, fixed 576 tokens; also a token-count control for RIN) on K | **E8 SLAKE read**: FastV − Random −0.6 [−2.7, +1.4] (closed +2.6, n.s.) — the failure is **not universal**; RoPE-free − Random +3.1 [+1.1, +5.0] (closed +7.5), Gate+FastV +2.5; least- minus most-attended −10.7 (attention informative). FastV keeps 77% of its tokens in the bottom quarter of rows (P4 for HuatuoGPT: 3.1× uniform) and RoPE removal halves the positional share (52% → 23%): in 1D-RoPE LLaVA the prior *is* rotary (FEATHER), in M-RoPE Qwen models it is not (G18). Open answers wait for judge pass 3. E8r (RIN) needs C's `cvpr` secret | P1 |
| G7 | "10% is an extreme budget" | — | E5 (5/25/50%) | **Read (Nov 5)**: Lingshu RIN FastV − Random −39.8 / −41.7 / −43.9 at 5 / 10 / 25%, FastV@50% − Full −5.0; SLAKE −12.4 / −13.4 / −6.2, @50% − Full −1.4. Qwen3-VL RIN −32.2 / −33.7 / −15.9. The failure holds to 25% | P0 |
| G8 | "Score at a deeper layer and it works" | Offline: top-10% by layer-7/15 attention is still image-independent on RIN (cross-image Jaccard 0.62 / 0.38) and background-heavy (BRR 2.6 / 2.0, Lingshu) | `fastv_k8`, `fastv_k16` (E6) | Running | P0 |
| G9 | "Open-answer scoring is unfair to Qwen3-VL" | Synonyms ("axial" vs "Transverse Plane"); Qwen3-VL SLAKE Full 58.2 | Local LLM judge on saved responses (Qwen2.5-7B-Instruct, one-token Yes/No verdict), validated against 200 blind hand labels (population-weighted κ) | Pass 1 (v2 prompt, 2,141 answers, 0.1 GPU-h) exposed two failure modes — it overruled the reference with its own knowledge (16 of 364 exact matches rejected, e.g. "largest organ? Liver" → No) and accepted partial/refusing answers (text-only open accuracy 11% → 25%). Fixed in v3 (`d9d678c`): explicit rubric + scorer = exact match OR judge; v3 pass over all 2,855 answers queued on C; labelling sheet after it. Preliminary (v2): Qwen3-VL SLAKE-open FastV − Random flips from +0.8 (rule) to −3.4 [−6.4, −0.8]; Lingshu unchanged (−11.5 / −11.6). Rule fixed now: rule OR judge becomes primary iff its κ with the hand labels ≥ 0.8 and > κ(rule, human) | P0 |
| G10 | "Yes/no accuracy on an imbalanced set" | RIN abnormality is 71.9% yes | Report the majority baseline and balanced accuracy for every yes/no cell | Analysis task | P0 |
| G11 | "Coverage explains it; nothing new" | Selector-level coverage correlates with accuracy (r −0.91 to −0.99) but per question it explains 0–46% of FastV's penalty (logistic mediation) | Report both; DivPrune (max coverage) in E6 | Analysis done; E6 running | P1 |
| G12 | "The gate removes informative tokens" / "background is not empty" | Gate = 100% of SLAKE CT/MRI; 889/1000 RIN questions gated | `inverse_gate` (background only) should sit near text-only; `oracle_gate` (SLAKE organ masks) | E6 running | P1 |
| G13 | "Post-hoc analyses" | Predictions frozen Nov 2 (§A6); E10/E11/G-tests added Nov 3 after seeing E1/E1g | Label E10, E11 and the offline diagnostics *exploratory* in the paper; list every deviation (gate revision, scoring, RIN builder) | Writing | P0 |
| G14 | "How blind is a pruned model?" | FastV's answer equals the text-only answer for 33% of RIN questions (Random 22%); FastV keeps 26% of the image's contribution, Random 86% | Report retained visual gain (acc − text-only) / (Full − text-only) for every cell | Analysis done | P0 |
| G15 | "Contamination" | Lingshu was trained on SLAKE-style data | Paired comparisons only; limitation paragraph | Writing | P2 |
| G16 | "Show me FastV misses the evidence, not just 'background'" | SLAKE organ masks, 358 / 364 questions naming an annotated structure: FastV keeps it at 0.84× [0.57, 1.11] / 0.61× [0.43, 0.80] chance (Random 1.00 / 1.05; Gate+FastV 2.81 / 1.68); large organs 0.20× / 0.23×, dropped entirely in 58% / 71% of questions (Random 13% / 11%); small structures 1.46× / 0.88×; FastV − Random −16.8 [−25.2, −7.1] on large organs vs −2.5 on small (Lingshu) | `oracle_gate` (E6-P1); within-question test of "evidence kept → correct" with Random seeds 0–2 (conditional logit, D4) | **`oracle_gate` read (Nov 5, judge)**: it keeps the named structure at 4.89× chance (hit 100%). On the 358 questions naming a structure, Oracle − Random +5.3 [+2.0, +8.9] (p 0.003); on the other 703, −3.6 [−6.9, −0.3] (p 0.033); net −0.6 (n.s.). **[FLAG] FastV loses more where no structure is named** (−16.4 [−20.2, −12.5] vs −7.5 [−11.4, −3.8]): missed evidence is a real but minor part of the loss, so G16 cannot be the headline mechanism. D4 still open | P0 |
| G17 | "This is the visual attention sink (Kang et al., ICLR 2025), already known" | Spearman(layer-1 score, layer-1 input norm) 0.32 / 0.44 Lingshu (RIN / SLAKE), ≈ 0 / 0.16 Qwen3-VL; FastV's top-10% overlaps the top-10% by norm at 2.3–3.9× chance; ViT-norm outliers selected 2.6–5.1× (P5) → a partial mechanism for Lingshu, not for Qwen3-VL | `artifact_excl` (E6, running); sink-dimension count per token in the layer profile (D1); cite and contrast | Partial | P1 |
| G18 | "FEATHER already explained early-layer bias by RoPE" | RoPE-free scores give the same positional variance share as RoPE scores (RIN: 69% vs 69% Lingshu, 47% vs 51% Qwen3-VL) and keep-set overlap 0.94 / 0.77 → in M-RoPE models the positional prior sits in the visual keys | HuatuoGPT (1D RoPE, E8) as the FEATHER-like case; D1 per layer | Analysis done | P0 |
| G19 | "Only FastV; the title says attention-guided pruning" | — | PDrop ± gate (E6-P2), `fastv_k8/k16`, `fastv_text` (D2); scope the title to early-layer LLM attention unless a ViT-side method (VisionZip-style) is added | **E14 running (Nov 7)**: VisionZip and HiPrune (vision-encoder attention), CDPruner (instruction-conditioned DPP, no LLM attention), FEATHER, VScan, TransPrune | P1 |
| G20 | "Prior work already shows Random ≥ attention pruning" (Wen et al. 2025, ACL Findings; DART; VTC-Bench) | Not yet positioned | Related-work check (D6); our delta = medical regime + quantified prior + anatomy grounding + E7/E10 controls | Writing | P0 |
| G21 | "Many exploratory analyses: forking paths" | §A6 frozen Nov 2; G-tests added Nov 3 | Confirmatory set = P1–P14 with Holm correction; everything else labelled exploratory with its date (extends G13) | Writing | P0 |
| G22 | "Why does RadImageNet collapse 2.5× more than SLAKE at the same 64 tokens?" | FastV − Random −41.2 / −33.7 on RIN vs −16.3 / −0.5 on SLAKE at 224² and −10.7 / −2.7 on photographs at 224²; Random alone loses only 9.7 / 5.1 points on RIN (anatomy is readable from any 6 tokens); position explains 69% / 51% of RIN's layer-1 ranking vs 64% / 40% on SLAKE at 224² — similar priors, very different damage | E11r (flip RIN), E6 `static_map` on RIN (is a fixed map as bad as FastV?), E10r (RIN at 448²); in-session token statistics of RIN images at FastV-kept vs random cells (aggregates only, license) | Partly answered (Nov 5): a position-only map built on SLAKE reproduces FastV on both RIN tasks (Lingshu static map − Random −50.9 anatomy, −15.6 abnormality vs FastV −41.7 / −14.2; static − FastV on abnormality −1.4, p 0.09) → RIN's collapse is what a fixed position prior does to these images. Open: why the same prior hurts RIN more than SLAKE | P0 |
| G23 | "Evidence retention does not explain accuracy" | Lingshu's bottom-10% keeps less of the named organ than FastV (0.59× vs 0.83×) yet answers better (+6.2); per-question mediation 7% | Report evidence retention as a description of *where* FastV looks, not as the causal path; causal claims rest on interventions (bottom-k, flip, gate, oracle) | Writing | P0 |

**Paper consequence.** Section 5 (Diagnosis) leads with G5 + G3 (question- and image-blind prior), G4 (attention is
not informative), G14 (the pruned model falls back on the language prior), then splits the loss into the two regimes
(background capture on CT/MRI, prior collapse at small token counts — G1). The gate (C4) is a remedy for the first
regime; coverage-preserving selection (Random, Uniform, DivPrune) is the baseline any attention pruner must beat.
*Nov 3 night:* G16 (FastV drops the large organs the questions ask about) is the most concrete, clinically legible
result so far and should anchor Fig. 1; G18 replaces "rotary decay" with "a prior carried by the visual keys".
*Nov 3, 21:45 (E7/E10/E11 read):* the failure is **not medical-specific** — FastV loses 8.7 / 4.9 points to Random on
photographs (P13 contradicted), and the question is irrelevant to its ranking on photographs too. The defensible
thesis is general mechanism + medical amplification: early-layer attention pruning applies a question-blind, partly
positional prior that loses to random selection everywhere at 10%; medical scans amplify it — background capture and
large-organ avoidance on CT/MRI (gate fixes SLAKE), and a dataset-specific collapse on RadImageNet (G22). Title and
abstract need this framing before enrollment (Nov 10); decision with Pushkar and the mentor.
*Decision (Pushkar, Nov 4): the paper is about medical MLLMs only; no general-domain claim or section.* Consequences
for the writing: (i) never claim the failure is medical-*specific* — the photograph control (E7) contradicts that, so
the claims are "FastV fails on medical VQA, and here is why", with the medical-only mechanisms (background capture on
CT/MRI, large-organ avoidance, RadImageNet's collapse) as the contribution; (ii) E7 stays in the supplementary as a
one-paragraph scope note (a reviewer will ask; prior work — Wang et al. 2512.07580, VTC-Bench — already reports weak
attention pruning on natural images at small budgets, so the note costs nothing); (iii) the main text uses only
medical datasets.
*Nov 4 (E6 read, session 1):* the gate is not "remove background and FastV works". FastV loses more than a keep-set
of evenly spaced **background** tokens (`inverse_gate`; Lingshu SLAKE CT/MRI −18.0 vs −7.5; Qwen3-VL −4.7 vs +2.0),
so FastV's failure is not that it picks background; it is that it concentrates the budget on a fixed region. Background
tokens are not empty for the LLM (consistent with register-like global tokens, Darcet et al.). Coverage (Random,
Uniform, DivPrune) is the property that matters; the gate helps where it restores coverage of the body.

## A10. Depth plan — what the diagnosis still needs (Nov 3 night)

The paper is a chain of claims; every link needs an intervention, not only a correlation.

| Link | Claim | Have | Missing | Experiments |
|---|---|---|---|---|
| L1 | FastV < Random on medical VQA, robustly | RIN −41 / −34; SLAKE (Lingshu) −12.6; image-clustered CIs | budgets; third model; third benchmark; open-answer metric | E5, E8, E3 (OmniMedVQA), judge |
| L2 | The layer-1 ranking is a fixed, question-blind prior | Jaccard 0.98 / 0.82; question share 0.3% / 3.1%; RoPE-free identical | where along depth it becomes question-aware, and for which readout | D1, D2, E6 `fastv_k8/k16` |
| L3 | Positional at 64 tokens, content (background) at 300+ | position share 69% / 51% (RIN) vs 5% / 12% (SLAKE); BRR; dose-response | position vs content by intervention; token count vs domain | E11 flip, E6 `static_map` / `inverse_gate`, E10 / E10r |
| L4 | It drops the anatomy the question asks about | evidence retention 0.23× on large organs; **oracle (Nov 5)**: keeping the named organ helps those questions (+5.3) but costs the rest (−3.6), and FastV loses more on the rest (−16.4 vs −7.5) | within-question evidence effect | D4 |
| L5 | Attention ranks useful tokens last (Lingshu) | within-Random −0.12 to −0.35 log-odds per SD | keep the least-attended tokens | E11 `fastv_bottom` |
| L6 | The pruned model falls back on its language prior | text-only agreement 33% vs 22%; retained gain 0.26 vs 0.86 | — (descriptive suffices) | — |
| L7 | Medical-specific or regime-specific? | — | photographs at native size and at 64 tokens | E7, E10 |
| L8 | Remedy | gate closes SLAKE's gap, 13–15 of 34–41 RIN points | composition with other pruners; budget curves; latency | E6 DivPrune ± gate, PDrop ± gate, E5, E9 |

New work, in order of decision value per GPU-hour (none is in `configs/runs.yaml` yet):

| ID | What | Answers | Cost | Output |
|---|---|---|---|---|
| D1 | **Layer profile**: `full` re-run on `slake_en`, `rin_anat_mc`, `natural` that logs, at every LLM layer, last-token and text-token (every token after the image) attention over the visual tokens, plus per-token sink-dimension activations | G2, G5, G8, G17, G18 in one figure: per model and layer, the position / image / question variance split, evidence retention and background retention of the top 10% | code ≈ 2 h; ≈ 0.6 GPU-h per model | Fig. 3; a "where to score" rule |
| D2 | `fastv_text`: prune with the text-token readout at layer 1 | G2 ("your FastV is not FastV") | ≈ 0.4 GPU-h per model on K | L2 |
| D3 | Evidence grounding as a tested `analysis.py` function, on every SLAKE cell (E6, E10 224², E11 flipped) | G16 | laptop | Fig. 1, Fig. 5 |
| D4 | Within-question evidence test: Random seeds 0–2, conditional logit of correct on evidence kept | is evidence retention causal for accuracy? | laptop, after E6 seeds | L4 |
| D5 | RadImageNet open questions (`rin_open`), judge-scored | MedSTP's strongest column | ≈ 1.0 GPU-h per model | L1 — **E15 running (Nov 7, C, Lingshu only, 1,000 questions)** |
| D6 | Related-work check: question independence of FastV, Random ≥ attention, visual attention sinks | G20 | 1 h | §2 |
| D7 | Holm correction over P1–P14; every other analysis labelled exploratory with its date | G21 | laptop | §3 |

Status (Nov 3, 22:00): **D1, D2 code done** — methods `full_profile` (logs `lp_last_l{L}`, `lp_text_l{L}` = log
attention of the last token / of every token after the image over the visual tokens, and `sink_l{L}` = Kang et al.'s
sink score at the input of layer L, for every layer) and `fastv_text`; scheduled as **E12** in A/B session 2 (K and
photographs for `full_profile`, K for `fastv_text`), HuatuoGPT's profile in E8 P2; read with `python -m
medprune.analysis profile --results R --slake-images DIR --labels mask.txt` (table `layer_profile.csv`, figure
`layer_profile.pdf`). **D3 done** — `python -m medprune.analysis evidence` (tables `evidence_*.csv`; numbers in §A2,
G16). Tests: 114.

**Depth target.** A CVPR analysis paper is accepted on insight per page, not on method count: four insight figures
and two tables. Fig. 1 teaser (keep-sets on a CT and an X-ray; FastV skips the organ the question names). Fig. 2
phenomenon (models × datasets × budgets, with the photograph controls). Fig. 3 the prior (D1: question / position /
image shares and evidence retention by layer). Fig. 4 the consequence (evidence retention by organ size; text-only
fallback). Table 1 the main grid; Table 2 the interventions (flip, bottom-k, static map, inverse gate, RoPE-free,
sink exclusion, K-sweep, oracle gate). §A7's figure list is revised to this at Stage 8. **Experiment freeze: Nov 12**;
after it, GPU time only re-runs cells that fail their checks.

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

| Group | Datasets | Questions | Hours per run: Lingshu / Qwen3-VL | Basis |
|---|---|---|---|---|
| K | `slake_en` + `rin_anat_mc` | 2,061 | 0.41 / 0.37 | measured |
| S1 | K + `rin_abn_closed` | 3,061 | 0.50 / 0.44 | K measured, `rin_abn_closed` predicted |
| S2 | `slake_zh` | 1,033 | 0.32 / 0.29 | predicted |
| S3 | `omni_anat` | 3,329 | 0.46 / 0.40 | predicted (174 / 164 visual tokens on average) |
| P | `omni_panel` | 1,200 | 0.27 / 0.23 | predicted (361 / 320 tokens) |
| N | `natural` | 1,000 | 0.20 / 0.16 | predicted (351 / 268 tokens) |

Hours use the measured rates and cost model of §C4 (pruned runs cost as much as `full`, because masking keeps every
token in the forward pass; `fastv_prellm` ×1.22; text-only is cheaper). Add ≈ 0.1 h per Kaggle session for setup.

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
| E0 | Is the pipeline correct, and how fast is it? | 3 | Lingshu, Qwen3-VL | `slake_en` | 2 | 0.39 ✅ | A (+B) | Stage 3 gate |
| E1 | Is standard FastV worse than random in our pipeline, and does it prefer background? | 4 | Lingshu | K | 14 (13) | 3.35 ✅ | A | K1, K2 |
| E1g | Can a coverage-preserving selector inside the gate beat Random? | 4b | both | K | 6 + 6 | 2.56 ✅ | A, B | method claim |
| E2 | Why: position, background or artifact tokens? (analysis only) | 4–6 | — | logs | 0 | 0 | laptop | Table 2, Fig. 3 |
| E3 | Does it hold on both models and all three benchmarks, and does the gate fix it? | 5 | both | S1, S2, S3 | 45 + 45 | 15.2 (1.7 ✅) | A, B | Table 1 |
| E4 | Does the failure grow with the amount of background? | 5 | both | P | 6 + 6 | 2.8 | A, B | Fig. 2 |
| E5 | Does it hold at other budgets? | 6 | both | K | 20 + 12 | 6.5 | A, B | Fig. 4 |
| E6 | Mechanism controls and gate ablations | 6–7 | both | S1, K, `slake_en` | 37 + 10 | 8.4 | A, B, C | Tables 2–3 |
| E7 | Does FastV fail on ordinary photographs too? | 6 | both | N | 10 | 2.0 | B | specificity |
| E8 | Does a 1D-RoPE model show the predicted bottom-row bias? | 7 | HuatuoGPT | S1 | 21 | 6.3 | C | Fig. 3, H2 |
| E9 | How much compute does the gate really save? | 7 | Lingshu | 200 × `slake_en` | — | 0.5 | C | Table 4 |

Total ≈ 48 GPU-h with measured rates (first plan: 66.2); ✅ = spent, 🔄 = running. After each stage: fill the Outcome column of §A6, tick the boxes, add
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
| A | Lingshu-7B: E0, E1, E1g, E3, E4, E5, E6-P1 | 29.6 → **21.2** (measured) | `pushkararora88` | 30 h |
| B | Qwen3-VL-8B: E1g, E3, E4, E5, E6, E7 | 26.6 → **17.0** (measured) | `pushkararora` | 30 h |
| C | E6-P2, E8, E9, overflow from A/B | 10.0 → **9.9** | `pushkarsiuuarora` | 21 h 50 min left |

Measured rates (§C4) leave slack on every account (A 8.8 h, B 13.1 h, C 11.9 h); see the ledger under the Stage
tracker. Quotas reset weekly. The `riddhitaneja` key is revoked (401).

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
**Status: ✅ DONE (Nov 2); one item moved to the Stage 5 preflight.** E0b–E0d passed 50/50 on both models (Work log,
Nov 2); sessions took 11.6 / 11.8 min (A / B), 0.39 GPU-h in total. The subsequent 1,061-question Lingshu Full run scores 77.19% with normalized exact
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
| Time | Planned A ~2.5 h, B ~0.5 h. **Measured: A 0.19 h, B 0.20 h** (11.6 / 11.8-min sessions incl. setup and the 17 GB model download). |

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
whether it prefers background tokens (K2) — before spending ~30 GPU-hours. **Status: ✅ DONE (Nov 3), 3.35 GPU-h.
Follow-up E1g ✅ DONE (A v7 1.32 GPU-h, B v3 1.24 GPU-h; result in the Work log, Nov 3).** Registered text-only truncation retry verified. K1 is GO and K2 leaves H1 alive in this Lingshu setup.
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
| Time | Planned 3.3 GPU-h. **Measured: 2.92 h** (one 175-min session; 170 min of questions) **+ 0.44 h repairs** (v4 Random rerun 16 min, v5 truncation retry 10 min). |

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
and Fig. 2 (failure grows with background) — in time for the Nov 10 enrollment. **Status: 🔄 RUNNING (Nov 3, 17:50 IST) — A
`medprune-gpu-lingshu7b` v8 and B `medprune-gpu-qwen3vl8b` v4 run `EXP = "E6,E3"`: the E6 mechanism controls on K
first (pulled forward, §A9), then the rest of E3 (RIN abnormality, OmniMedVQA anatomy, SLAKE-zh last). Finished runs
are restored from the private datasets `medprune-restore-<model>` (`RESTORE_FROM = "auto"`). Estimated 11.2 / 9.0
GPU-h; `MAX_HOURS = 9.5` leaves ≈ 1.7 h of Lingshu SLAKE-zh for session 2. Scoring the open SLAKE answers is offline
(saved responses), so it no longer blocks inference.** Qwen K-block ✅ DONE (B v2, 1.73 GPU-h: RIN FastV − Random −33.7
[−37.3, −30.1]; SLAKE −2.4 [−4.9, +0.2]); E1g ✅ done (diagnosis-first). Record of that first block: This uses ten existing E3 conditions on English SLAKE and RIN anatomy: Full,
text-only, Random@L2 seed 0, FastV, Gate+FastV. Retention 10%; RIN first; five-question smoke for every condition
before bulk inference. Approximately 2.35 planned inference hours, 3.5-hour scheduling budget, role B/Father T4×2.
The full Lingshu scoring-comparability issue remains open; this within-Qwen comparison does not require claiming
equivalence to published LLM-judged scores. The full 45-condition grid, other datasets/seeds and E4 remain pending.

**Before you start**
- [x] K1 = GO (corrected Lingshu E1, Nov 3).
- [x] RadImageNet built in each session (the notebook does it when `NEEDS_RIN`; masks for `MODEL` only).
- [x] E1g read (gate + coverage selectors vs Random) — decides whether `gate_uniform`/`gate_random` are the headline
  method rows of Table 1. **Outcome (Nov 3): not headline.** The gate beats Random on Lingshu (+2–3 points, CI > 0 on
  RIN and SLAKE) but not on Qwen3-VL. `gate_uniform` stays in E3 as the remedy row (deterministic, so no seed
  variance; `gate_random` is indistinguishable from it on all four cells). Table 1 is organised around the
  air / ranking decomposition (Work log, Nov 3).
- [ ] **Scoring preflight (open SLAKE answers).** Normalized exact match under-credits synonyms: Qwen3-VL answers
  "axial" for "Transverse Plane", "brain" for "Head" (Full 58.2% vs Lingshu 77.2%, which copies SLAKE's wording).
  Decide before the S1/S2 SLAKE runs are read: (a) a local LLM judge re-scoring the saved responses offline (no new
  inference; ~0.3 s per open answer on T4, ≈ 3 GPU-h for all SLAKE runs of both models), reported next to (b) closed
  questions only and (c) normalized EM. RIN and OmniMedVQA are multiple choice and need none of this.

### E3 — Main grid

| Field | Value |
|---|---|
| Question | Does FastV fall below random on both models and on all of MedSTP's benchmarks? Does the gate fix it? Is the cause position, background, or both? |
| Why it matters | Table 1 carries claims C1, C4 and C5; reviewers need two models (medical-tuned Lingshu, general Qwen3-VL), variance across Random seeds and the 2×2 factorial. |
| Setup | Lingshu-7B on role A, Qwen3-VL-8B on role B, in parallel. Groups S1 (`slake_en`, `rin_anat_mc`, `rin_abn_closed`), S2 (`slake_zh`), S3 (`omni_anat`). Budget 10%. |
| Runs | 45 run IDs per model (S1: 11 methods × 3 datasets; S2: 5; S3: 7). Lingshu reuses 10 from E1. |
| Time | Planned Lingshu 12.6 / Qwen3-VL 16.3 GPU-h. **Measured-rate estimate: Lingshu 7.2 GPU-h left (2 sessions); Qwen3-VL 6.3 left (2 sessions) + 1.73 spent (K-block).** |

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
| Runs, time | 6 run IDs per model; Lingshu 1.5 h (A), Qwen3-VL 1.3 h (B) at measured rates (planned 2.0 / 2.2). |

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
whether it is specific to medical images. **Status: ⏳ NEXT — partly pulled forward on Nov 3: E6-P1's controls on K run
in the Stage 5 sessions (both models), E7 moved to C with the new E10/E11 controls (Stage 5b); E5 and E4 run in A/B
session 2.** Decisions K3 and K4 are taken here.

**Before you start**
- [ ] Stage 5 done.
- [x] Static maps built on the laptop (task T4): `results/static_maps/<model>__for_<dataset>.npy` from the `full`
      runs' scores on the *other* anchor dataset; uploaded in `medprune-restore-<model>` (Nov 3).

### E5 — Budget curves

| Field | Value |
|---|---|
| Question | Does the FastV gap widen as the budget shrinks, and does the gate help at every budget? |
| Why it matters | Pruning papers are judged on curves, not one budget (Fig. 4). The 50% points link to ViToS's mild-pruning result. |
| Setup | Group K; Lingshu: `random_l2`, `fastv`, `gate_fastv`, `gate_uniform` at 5% and 25%, plus `fastv`, `gate_fastv` at 50%; Qwen3-VL: `random_l2`, `fastv`, `gate_fastv` at 5% and 25%. |
| Runs, time | Lingshu 20 run IDs, 4.2 h (A); Qwen3-VL 12 run IDs, 2.3 h (B) at measured rates (planned 4.6 / 3.1). |

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
| Runs, time | Lingshu 19 run IDs, 3.4 h (A); Qwen3-VL 10 run IDs, 1.9 h (B) at measured rates (planned 4.1 / 2.5). |

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
| Setup | Both models on role C (moved from B on Nov 3, now P0); group N (500 GQA + 500 POPE); `full`, `random_l2`, `uniform_l2`, `fastv`, `ropefree`; no gate (photos have no acquisition background). |
| Runs, time | 10 run IDs, 2.0 h at measured rates (planned 2.5). |

```bash
python -m medprune.run_eval --exp E7 --model lingshu7b --bundle $DATA --out $RES --max-hours 5.5
python -m medprune.run_eval --exp E7 --model qwen3vl8b --bundle $DATA --out $RES --max-hours 5.5
```

**Expected result:** P13 — FastV ≥ Random − 2 points on photographs. If FastV is below Random on photos too, the claim
becomes "background-driven, strongest on medical images", carried by Fig. 2.

### E10 / E10r — Resolution control (added Nov 3, exploratory)

| Field | Value |
|---|---|
| Question | Is RadImageNet's large loss about 64-token inputs or about the domain? (G1) |
| Setup | Same questions, other resolution (`python -m medprune.data resize`, IDs unchanged → paired with the native runs): `slake_en_s224` and `natural_s224` (64 tokens) on C; `rin_anat_mc_s448` (256 tokens Lingshu, 196 Qwen3-VL) on A/B. Methods `full`, `random_l2`, `uniform_l2`, `fastv` (+ `gate_fastv`, `gate_uniform` on the medical sets). |
| Runs, time | C: 10 runs per model, ≈ 1.5 / 1.4 GPU-h; A/B: 6 runs per model, 1.1 / 0.8 GPU-h. |
| Reading | FastV − Random grows on SLAKE at 224² and shrinks on RIN at 448² → token count drives the collapse; unchanged → domain. Photographs at 224² decide whether the collapse is medical at all. |

### E11 / E11r — Position or content, and is attention informative? (added Nov 3, exploratory)

| Field | Value |
|---|---|
| Question | Does FastV's keep-set follow the content or the grid position (G3), and are the least-attended tokens any worse than the most-attended ones (G4)? |
| Setup | `slake_en_vflip` / `rin_anat_mc_vflip`: the same images upside down (`python -m medprune.data vflip`), `full`, `random_l2`, `fastv`; `fastv_bottom` (lowest layer-1 attention) on `slake_en` / `rin_anat_mc`. |
| Runs, time | C: 4 runs per model, ≈ 1.3 / 1.2 GPU-h; A/B: 4 runs per model, 0.4 / 0.3 GPU-h. |
| Reading | Keep-set positions on flipped images ≈ original positions → positional prior; ≈ flipped positions → content-driven. `fastv_bottom` ≥ `fastv` → attention ranks informative tokens last. Up/down answers change under the flip: compare keep-sets, and accuracy only on questions without a vertical direction. |

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
with 1D RoPE, and a real latency measurement. **Status: ⏳ LATER on C, can start now in parallel (9.9 GPU-h)** (can overlap Stage 6; results go to the paper if ready
by Nov 14, otherwise to the supplementary, Nov 23).

### E6-P2 — Gate robustness, artifacts, PDrop

| Method | What it isolates | Reading |
|---|---|---|
| `artifact_excl` | FastV with high-norm tokens removed from the candidates | if accuracy recovers, artifact tokens cause part of the failure (H3, P5) |
| `gate_quota3` | gate plus 3% of tokens kept from the background | do plane/modality questions need a little background for context? |
| `gate_tau25`, `gate_tau75` | the gate's single threshold at 0.25 and 0.75 | the gate is not tuned to the test set if results barely move |
| `pdrop`, `gate_pdrop` | multi-stage attention pruning, token-matched to FastV, ± gate | composability with another attention method |

Setup: Lingshu on S1, role C, 18 run IDs, 3.1 h at measured rates (planned 3.8). **Result (Nov 4, 3.22 h):** SLAKE —
the gate repairs FastV and PDrop and is flat in τ; RadImageNet (64 tokens) — it repairs a third at most and τ matters
(work log, Nov 4 22:40).
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
| Runs, time | 21 run IDs; 6.3 h predicted (576 tokens per image at 1.6 ms/token) + 1.0 h validation (planned 4.7 + 1.0). |

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

**Result (C v4, Nov 5; `results/latency/lingshu7b__slake_en__t4x2.json`, `tables/flops.csv`).** 200 SLAKE questions
(81 / 324 / 1024 visual tokens per image, so the per-question distribution has three modes and the paired ratio is
the statistic to quote, not the IQR). Prefill median: Full 296 ms; `gate_input` 294 ms (88.5% of visual tokens kept,
1.01×); `prellm_10` 90.6 ms (3.27×, paired 3.36×); `fastv_10` 100.3 ms (2.95×, paired 3.07×); `k16_10` 202 ms
(1.47×). The vision encoder plus embedding (183 ms) is untouched by any pruner, so the first-token path speeds up
1.75× (pre-LLM), 1.69× (FastV) and 1.24× (K=16). Theoretical prefill FLOPs (FastV formula, every run's own keep
schedule; Lingshu SLAKE): FastV 10% 0.220 of Full, pre-LLM 10% 0.160, K=16 0.640, text-only 0.070 → measured
speedup reaches ≈ 65% of the FLOPs bound at 10%. Gate+FastV costs exactly FastV's FLOPs (the gate restricts the
candidates, not k); the gate's mask is computed on the CPU from the image and was not timed. Decode not timed.
Fix found on the way: `flops_table` took the Full reference from each row's own `n_vis`, so text-only came out at
1.0; the reference is now the image's token count from any run that saw it (test added).

**Checks**
- [ ] HuatuoGPT validation passed before E8.
- [x] Latency timed around prefill only, with device-wide syncs (both T4s), on identical questions for every
  schedule, in shuffled order; the layer loop equals the model's forward on the first question (max relative diff
  0.0) and the text-token count matches the processor on every question.

### E14 / E15 / E16 — more published pruners, RadImageNet open questions, SparseVLM raters (added Nov 7)

**Why.** G19: a reviewer will say the paper tested one family. E14 adds the methods that avoid early-layer LLM
attention or claim to fix it, each ported from its official code at a pinned commit (`medprune/baselines/`; equivalence
tests in `tests/test_new_baselines.py` run the official functions and inline selection blocks from `third_party/`).
E15 adds the column MedSTP is strongest on (open answers, RadImageNet). E16 asks why SparseVLM survives on Qwen3-VL
where FastV collapses (D-15): rater subset, or its three-stage schedule?

| Method | What it reads | Acts | Schedule at our 10% budget | Port notes |
|---|---|---|---|---|
| VisionZip (CVPR 2025) ± gate | attention received at the last vision block; keys for merging | pre-LLM; contextual tokens merge the rest | dominant = budget − contextual, contextual = max(int(0.05 N), 1) (official Qwen2.5-VL code) | merged embeddings written into the tower output (Qwen3-VL: also its deepstack features) |
| HiPrune (AAAI 2026 SA) | attention at the object block (16 of 32; Qwen3-VL 14 of 27) and the last block | pre-LLM | anchors round(T × 0.1 / 5) + 4-neighbour buffers + registers | official block, verbatim |
| CDPruner (NeurIPS 2025) ± gate | projected visual tokens; mean LLM embedding of the instruction | pre-LLM | DPP MAP, T tokens | paper Eq. 5 relevance for models without a CLIP text tower (no negation) |
| FEATHER (ICCV 2025) | RoPE-free last-token attention; stride-3 grid | LLM layers 8 and 16 | (1 − R) = 0.1^(1/3): 46.4% at 8 (grid + attention), 10% at 16 | no public code; not token-matched (8 unpruned layers) |
| VScan (TMLR 2026) | local (block 7, 4 × 4 windows) + global (last block) vision attention; RoPE-free attention at L/2 | pre-LLM merge + LLM L/2 | stage 1 24.6% (Lingshu) / 22.5% (Qwen3-VL), then 1/3 of it; token-matched to FastV | official Qwen2.5-VL functions; merge = mean of kept + merged |
| TransPrune (CVPR 2026) | token transition (attention + FFN, layers 6–11) + instruction attention | LLM layers 7, 9, 12 | 87.5%, 62.5%, 10% | not token-matched (7 unpruned layers); softmax over the live sequence |

E15: `rin_open` = 500 `anatomy_open` + 500 `pathology_open` questions (one per image and template, seed 0), MedEvalKit's
open prompt, judged in the same session (`JUDGE = True`: the judge items with RadImageNet text stay in `/tmp`; only
verdicts reach the output). Methods: Full, text-only, Random (seeds 0–2), Uniform, FastV, Gate+FastV, Gate+Uniform,
PruneSID. E16: `fastv_raters` (SparseVLM's raters, layer 1 → 2, 10%: the FastV schedule) and `sparsevlm_single`
(raters at SparseVLM's first stage, layer 2 → 3, one stage at 10%), on RIN anatomy + SLAKE (Qwen3-VL) and RIN anatomy
(Lingshu).

**Predictions (written Nov 7 03:20, before any E14–E16 result; exploratory, outside the Holm family).**
- P19 (E14): the three pre-LLM methods that never read LLM attention (VisionZip, HiPrune, CDPruner) are within 3 points
  of Random on Lingshu RIN anatomy. 55%.
- P20 (E14): VScan and TransPrune (first LLM-attention stage at layer ≥ 7) stay above Random − 5 on Lingshu RIN
  anatomy. 50%.
- P21 (E14): FEATHER stays above Random − 5 on Lingshu RIN anatomy — its K = 16 stage rescues it, not the RoPE removal
  (RoPE-free = FastV on these models, G18). 50%.
- P22 (E16): on Qwen3-VL RIN anatomy, `fastv_raters` − Random ≤ −15 (the raters alone do not rescue SparseVLM; its
  schedule does). 55%. **Supported (Nov 7): −25.9 [−28.9, −22.7]** — but the larger share of the rescue is the scoring
  layer (1 → 2), not the schedule (E16 result below).
- P23 (E15): on `rin_open` (Lingshu, judge), FastV − Random ≤ −8 and Gate+Uniform ≥ Random − 3. 55%.

**E16 result (A v10, Nov 7 04:20; 5 of 6 runs; Lingshu `sparsevlm_single` did not fit the 1.0 h cap, queued on B).**
Where SparseVLM's 33.8-point lead over FastV on Qwen3-VL RIN anatomy comes from, each step paired on the same 1,000
questions (X − Random; Random = mean of seeds 0–2):

| Qwen3-VL, RIN anatomy | readout | scored at → masked from | stages | − Random | step |
|---|---|---|---|---|---|
| `fastv` | last token | 1 → 2 | 1 | −33.7 [−36.8, −30.5] | |
| `fastv_text` | every token after the image | 1 → 2 | 1 | −27.8 | +5.9 |
| `fastv_raters` | SparseVLM's raters | 1 → 2 | 1 | −25.9 [−28.9, −22.7] | +1.9 [+0.6, +3.3] vs `fastv_text` (p 0.010) |
| `sparsevlm_single` | raters | 2 → 3 | 1 | −8.7 [−11.1, −6.0] | **+17.2 [+14.2, +20.3]** (one layer later) |
| `sparsevlm` | raters | 2/6/15 → 3/7/16 | 3, token-matched | +0.1 [−1.8, +2.1] | +8.8 [+6.6, +11.0] |

Qwen3-VL SLAKE (judge; 4 new open answers rule-scored): FastV −5.0, `fastv_raters` −0.8 (+4.2 vs FastV), `sparsevlm_single`
−1.0, SparseVLM +0.8 — the readout matters there, the layer does not. Lingshu RIN anatomy: `fastv_raters` −41.3 = FastV
(+0.4, n.s.), as `fastv_text` was: on Lingshu no readout change helps. **Reading:** on Qwen3-VL the layer-1 ranking is
the broken object; one more layer removes half of FastV's RadImageNet loss. The E12 profile (`tables/layer_profile.csv`,
last-token readout, Qwen3-VL RIN anatomy) agrees: position explains 32% / **51%** / **36%** / 46% / 52% of the ranking
at layers 0–4 and the visual attention mass is 0.027 / **0.009** / **0.040** / 0.030 / 0.020 — layer 2 is a local
minimum of the positional prior and a 4× maximum of attention to the image, layer 1 the opposite (SLAKE: position 12% /
10%, mass 0.043 / 0.085). [FLAG] A candidate cause is Qwen3-VL's DeepStack, which adds vision-encoder features to the
visual hidden states after decoder layers 0–2 (layer 2 is the first to see two injections), but layer 3 sees three
and is positional again: a hypothesis, not tested.

**E14 result, Qwen3-VL RadImageNet (B `medprune-baselines-qwen3vl8b` v3, Nov 7 04:54, 1.5 h, 16 runs × 1,000).**
X − Random (seeds 0-2), with each method's background retention ratio (BRR, gate masks) and the number of distinct
keep-sets over the 1,000 anatomy images:

| Qwen3-VL | RIN anatomy | BRR | distinct keep-sets | RIN abnormality |
|---|---|---|---|---|
| FastV (ref.) | −33.7 | 2.96 | 508 | −2.2 |
| VScan | **+5.7** [+4.5, +7.0] (above Full, 89.7) | 2.14 | 766 | +0.4 |
| TransPrune | +4.5 [+2.9, +6.2] | — | — | +0.2 |
| FEATHER | +3.9 [+2.4, +5.5] (− `fastv_k16` −1.7; − `fastv_k8` +31.0) | 2.79 | 455 | +0.6 |
| Gate+VisionZip | +3.0 [+1.6, +4.4] | — | — | −0.2 |
| VisionZip | +1.8 [+0.3, +3.3] | 1.80 | **59** (one set on 495 images) | +0.2 |
| HiPrune | **−12.3** [−14.9, −9.5] | 2.31 | 322 | −0.9 |
| CDPruner | **−23.7** [−26.6, −20.6] | 2.48 | 943 | −1.1 |
| Gate+CDPruner | −9.3 (+14.4 vs CDPruner) | 0.00 | 1000 | −0.7 |

RIN abnormality stays the text-answerable cell (every method within 1.1 of Random). **Reading:** a biased selection
only hurts when tokens are dropped early. VisionZip keeps the same positions on half the images and VScan / FEATHER
are as background-heavy as CDPruner, yet all three beat Random: merging (VisionZip, VScan) routes the dropped tokens'
content into the kept ones, and FEATHER's hard cut comes at layer 16. HiPrune (vision-encoder attention, no merge) and
CDPruner (early hard drop) collapse. [FLAG] CDPruner's relevance sign: the paper's Eq. 5 (used) has none, the released
LLaVA code negates it; with Eq. 5 the relevance points at background (BRR 2.48). `cdpruner_neg` (released sign) is
added to E17; the paper reports CDPruner only with both signs.

**E17 / E15b (added Nov 7 04:45, after E16, before any E17 result).** E17 sweeps FastV's scoring layer over 2-5
(`fastv_k3`-`fastv_k6`) on RIN anatomy, both models (+ Lingshu `sparsevlm_single`); E15b runs `rin_open` on Qwen3-VL
(account B). Both start after the weekly quota reset (≈ 05:30 IST). The E12 profile's position share of the layer-(K-1)
last-token ranking is the prediction: Qwen3-VL RIN 51 / 36 / 46 / 52 / 33% at layers 1-5 (K = 2-6), 50% at 7, Lingshu
69 / 69 / 73 / 65 / 69%, 67% at 7.
- P24 (E17): on Qwen3-VL RIN anatomy, `fastv_k3` and `fastv_k6` each beat FastV (K = 2) by ≥ 10 points, and `fastv_k4`,
  `fastv_k5` are each ≥ 5 points below `fastv_k3`; across K = 2, 3, 4, 5, 6, 8, Spearman(position share, accuracy)
  ≤ −0.6. 45%.
- P25 (E17): on Lingshu RIN anatomy every K in 3-6 stays ≥ 30 points below Random (its prior is flat at 65-73%). 75%.

**E15 result (C v5, Nov 7 05:04, ≈ 1.8 h, 10 runs × 1,000, judge-scored in-session: 1,965 verdicts merged).**
`rin_open`, Lingshu, X − Random (seeds 0-2; Random 24.1, Full 33.2, text-only 4.0): FastV **−12.4** [−14.7, −10.0],
Gate+FastV −9.0 (+3.4 vs FastV), Uniform +0.5, Gate+Uniform −1.0 [−2.6, +0.7], PruneSID −0.3. By template: anatomy
(Random 46.1, Full 56.6) FastV −25.1, Gate+FastV −20.5, Uniform +0.7, Gate+Uniform −2.5; pathology is at the floor (Full
9.8, Random 2.5) and separates nothing. Rule scoring gives the same order (FastV −5.6). FastV keeps 27% of the image's
contribution ((acc − text-only) / (Full − text-only)), Random 69%. **P23 supported.** The open-answer column shows the
collapse that the multiple-choice column shows (MedSTP's strongest column is no exception).

**E18 / E14o (added Nov 7 05:15, after the Qwen3-VL E14 read, before any E18 / E14o result).**
- P26 (E18): on Qwen3-VL RIN anatomy, dropping the merge costs VisionZip ≥ 8 points and VScan ≥ 8 points
  (`*_nomerge` − merged), and FastV's own pre-LLM keep-set gains ≥ 15 points from VScan's merge (`fastv_merge` −
  `fastv_prellm`). 50%.
- P27 (E14o): on `rin_open` anatomy questions (Lingshu, judge), VScan and FEATHER are within 3 points of Random, and
  CDPruner (either sign) and HiPrune lose ≥ 8. 45%.

**Accounts.** D's two GPU notebooks were created by API on Nov 7 and have no `cvpr` secret, so D runs SLAKE and
OmniMedVQA (both models, 8 methods each); B's notebooks have the secret and run the RadImageNet cells (both models).
`KEEP_GOING = True`: a failing method is reported and the next run starts.

**Checks**
- [x] Ports equal the official code on random inputs (VisionZip and HiPrune inline blocks, CDPruner `encode_images`,
  VScan `token_merging` / `window_selection`, TransPrune `compute_transf_weight`); vision cues equal the pre-Nov-7
  VisPruner cues on tiny Qwen2.5-VL / Qwen3-VL towers; TTV under masking equals the official weights on the shortened
  sequence; RoPE-free scores exclude dropped keys (162 tests).
- [x] Smoke (D v1, 3 questions × 16 runs per model, Nov 7 03:22): every run complete, k = 10% of N, stage layers
  [8, 16] / [7, 9, 12] / [0, L/2], sane answers on both models. Cost per question on OmniMedVQA ≈ 0.9 s (Lingshu,
  vision-tower cue pass) vs 0.5 s in the rate table, VScan ≈ 1.5 s (merge on the CPU) → Lingshu cap raised to 10.5 h.

---

## Stage 8 — Analysis and writing

**In one sentence:** every number in the paper is regenerated from the run files, and the paper and supplementary are
submitted on time. **Status: ⏳ LATER (writing can start Nov 6, once E3 is in).**

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

## C4. Measured rates and cost model (s/question, T4×2 fp16, batch 1)

Measured 2026-11-03 from 18,549 question records (Stage 4 Lingshu, Stage 5 Qwen K-block; `elapsed_s` per question).
Bold = measured mean; the rest are predicted with the fitted model below.

| Model | `slake_en` | `slake_zh` | `rin_anat_mc` | `rin_abn_closed` | `omni_anat` | `omni_panel` | `natural` | text-only (SLAKE / RIN) |
|---|---|---|---|---|---|---|---|---|
| Lingshu-7B | **1.07** | 1.10 | **0.35** | 0.30 | 0.50 | 0.80 | 0.72 | **0.66 / 0.23** |
| Qwen3-VL-8B | **0.98** | 1.00 | **0.28** | 0.26 | 0.43 | 0.68 | 0.58 | **0.35 / 0.18** |
| HuatuoGPT-Vision-7B | 1.15 | 1.15 | 1.15 | 1.15 | — | — | — | 0.30 |

**Cost model** (least squares over every non-text-only question; R² 0.98 Lingshu, 0.99 Qwen3-VL):
s/question ≈ **1.6 ms × visual tokens + 76–86 ms × generated tokens**, with a negligible intercept. Both models share
the constants because both run a 7–8 B Qwen-family LLM. Visual tokens per question (mean, from the mask grids;
Lingshu / Qwen3-VL): SLAKE 510 / 476 (median 324 / 256; X-rays reach 1,024), RIN 64 / 64, `omni_anat` 174 / 164,
`omni_panel` 361 / 320, `natural` 351 / 268; HuatuoGPT is always 576. Generated tokens: multiple choice ≈ 2.4–3.0,
SLAKE closed ≈ 2.1–2.2, SLAKE open ≈ 3.7–4.7.

Other measured factors: `fastv_prellm` (two forward passes) ×1.22 (SLAKE 1.30, RIN 1.16); pruned runs cost the same as
`full` (masking keeps every token; FastV 0.93–1.06× Full); session overhead ≈ 6 min (setup, 17 GB model download,
RadImageNet build). The 224² RadImageNet images give **64 tokens, so 10% keeps 6** — state this regime in the paper.
These rates are in `configs/runs.yaml`, which `run_eval --list` uses for its hour estimates.

## C5. Storage

| Item | Size | Where | Kept |
|---|---|---|---|
| One model per GPU session | 16.6–17.6 GB | HF cache on session disk, never `/kaggle/working` | No |
| `medprune-data` bundle | printed by notebook 00 (`du -sh`) | each role account's notebook-00 output; `meta.tar.gz` (manifests + masks) for the laptop | Yes |
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
| Paired difference | per-question A − B on identical questions | cluster bootstrap CI; image-level sign-flip permutation p (Nov 5: replaced McNemar, see work log) |
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

- **Keep the Stage tracker current.** After every Kaggle session (or any finished stage): update the Stage tracker's
  Status, *GPU-h spent* and *GPU-h left* columns and the Compute ledger at the top of this file (session hours from the
  kernel log: the last entry's `time`), set the stage's **Status:** line in Part B to ✅ DONE / 🔄 RUNNING / ⏳, fill
  the card's measured *Time*, and add a Work log entry. Re-estimate hours with `run_eval --list` (it reads the measured
  rates in `configs/runs.yaml`, §C4).
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
| T4 | 6 | static-map builder | done (Nov 3, `analysis.build_static_maps`) | T3 |
| T5 | 7 | `medprune/huatuo_llava.py` + validation | done (Nov 3–4; E8, E8r) | Stage 5 done |
| T6 | 7 | E9 latency measurement | done (`59c141c`, `medprune/latency.py` + `tests/test_latency.py`); runs in C v4 | T2 |
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
and logged. ✅ Done Nov 3 (`analysis.build_static_maps`, test in `tests/test_analysis.py`; maps peak in the top rows for
both models: mean row of the 20 strongest cells 0.02–0.21).

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

### 2026-11-03 — Claude — Stage 4 verification, Qwen block check, E1g launched
- Re-derived Codex's Stage 4 numbers from the raw run files (`results/stage4_lingshu_20261103/runs`): all match.
  Lingshu RIN anatomy: Full 89.0, Random@L2 79.3, FastV 38.1, Gate+FastV 51.3; FastV − Random −41.2 [−44.9, −37.6]
  (image bootstrap). K1 = GO stands.
- Verified Codex's Qwen E3 block (B, `medprune-gpu-qwen3vl8b` v2, git `f53a663`). RIN anatomy: Full 89.7, Random@L2
  84.6, FastV 50.9, Gate+FastV 65.9, text-only 30.1; FastV − Random −33.7 [−37.3, −30.1]; Gate+FastV − Random −18.7
  [−22.0, −15.3]. SLAKE-en: Full 58.2, Random 52.6, FastV 50.2, Gate+FastV 51.8; FastV − Random −2.4 [−4.9, +0.2].
- [FLAG] Qwen's SLAKE Full (58.2) is depressed by vocabulary, not errors: it answers "axial" for gold "Transverse
  Plane", "coronal" for "Coronal Plane", "brain" for "Head". Lingshu was trained on SLAKE-style labels and copies the
  wording. Normalized EM therefore compresses Qwen's SLAKE headroom; RIN (multiple choice) is unaffected. Needs a
  semantic open-answer judge (or closed-only reporting) before SLAKE-Qwen numbers go into the paper.
- New statistic `analysis.image_dependence` (one keep-set per image; distinct sets, share of the most common set,
  mean cross-image Jaccard; Random ≈ 0.05). FastV is close to image-independent, and the loss to Random follows it:
  | model, dataset | FastV distinct sets / images | top share | cross-image Jaccard | FastV − Random |
  |---|---|---|---|---|
  | Lingshu, RIN | 38 / 1000 | 0.447 | 0.769 | −41.2 |
  | Qwen, RIN | 508 / 1000 | 0.030 | 0.465 | −33.7 |
  | Lingshu, SLAKE | 73 / 96 | 0.250 | 0.303 | −12.6 |
  | Qwen, SLAKE | 80 / 96 | 0.115 | 0.174 | −2.4 |
  Reading: at layer 2 FastV mostly applies a fixed positional prior, and fewer tokens (RIN 224² → 64) make it worse.
  This is a sharper paper claim than "attention points at air"; H1 (background over-selection, 2.3× on RIN) may be
  a by-product of the prior's location. Gate+FastV breaks the collapse (Jaccard 0.175 / 0.087) but stays below
  Random, so a gated top-k is not the method.
- Added `gate_random` (random within the gate; ungated draw unchanged — tested) and block **E1g** = {gate_uniform,
  gate_random, uniform_l2} × K for both models. Decision rule (set before results): the gate is a method only if
  gate_uniform or gate_random beats random_l2 on RIN with the paired CI above 0; otherwise the paper is diagnosis-first
  and the gate is reported as a partial remedy. 87 CPU tests pass.
- Launched E1g with the standard `notebooks/01_run_gpu.ipynb` (clones `main`): A `medprune-gpu-lingshu7b` v7,
  B `medprune-gpu-qwen3vl8b` v3, T4×2, each on its own `medprune-00-build-data-bundle` output.

### 2026-11-03 — Claude — Plan maintenance — measured rates, stage tracker, compute ledger
- Stage tracker rewritten with Status (✅/🔄/⏳), GPU-h spent and left per stage; a Compute ledger per account added
  under it; Part F rule F2 now requires updating both after every Kaggle session. Stage 3 and Stage 4 marked DONE.
- Rates measured from 18,549 per-question records (§C4). RadImageNet 0.35 / 0.28 s/q (Lingshu / Qwen3-VL; planned
  0.6 / 0.66), SLAKE-en 1.07 / 0.98 (planned 1.0 / 1.1), `fastv_prellm` ×1.22 (planned ×1.3), session overhead ≈ 6 min.
  Fitted cost model: 1.6 ms × visual tokens + ~80 ms × generated tokens (R² 0.98–0.99); used to predict the datasets
  not yet run from their measured token counts (`omni_anat` 0.50 / 0.43, `omni_panel` 0.80 / 0.68, `natural`
  0.72 / 0.58).
- `configs/runs.yaml` now holds per-dataset rates (family fallback kept); `run_eval.estimate_hours` reads them and the
  measured two-pass factor. Test updated (no hard-coded rate).
- Re-estimated budget: whole project ≈ 48 GPU-h (was 66.2): 5.5 spent, 2.5 running (E1g), 40.0 left (A 16.3, B 13.8,
  C 9.9). Slack ≈ 34 GPU-h across accounts; proposed uses listed under the ledger. Stage 5 checklist gains the E1g
  read and the open-answer scoring preflight (Qwen synonyms).

### 2026-11-03 — Claude — E1g read: the gate is a partial remedy; FastV's loss is coverage, not image dependence
- Sessions: A `medprune-gpu-lingshu7b` v7 1.32 GPU-h, B `medprune-gpu-qwen3vl8b` v3 1.24 GPU-h (predicted 1.3 / 1.2).
  All 12 runs complete (1000 RIN / 1061 SLAKE each), no errors or truncations in the kernel logs; git `bf4a944`.
- Paired deltas vs `random_l2`, 10% retention, image bootstrap (4000 draws):
  | model, dataset | `gate_uniform` | `gate_random` | `uniform_l2` | Full / Random |
  |---|---|---|---|---|
  | Lingshu, RIN | +2.3 [+0.0, +4.6] | **+3.2 [+1.2, +5.2]** | −0.4 [−2.8, +2.0] | 89.0 / 79.3 |
  | Lingshu, SLAKE | **+2.4 [+0.4, +4.3]** | **+1.7 [+0.1, +3.3]** | +1.0 [−1.0, +3.0] | 77.2 / 68.8 |
  | Qwen3-VL, RIN | +1.3 [−0.9, +3.4] | +0.5 [−1.3, +2.3] | −0.7 [−2.8, +1.5] | 89.7 / 84.6 |
  | Qwen3-VL, SLAKE | +0.6 [−1.3, +2.5] | −0.6 [−2.3, +1.1] | **+3.1 [+1.2, +5.1]** | 58.2 / 52.6 |
- Decision rule (registered before results): met on Lingshu (`gate_random`, CI above 0), not on Qwen3-VL. A remedy
  that replicates on one of two models and recovers ≤ 3 of the 5–10 points between Random and Full is not a headline
  method. **Paper is diagnosis-first**; `gate_uniform` stays in E3 as the remedy row. `gate_random` − `gate_uniform`
  is within ±1.1 with CIs spanning 0 on all four cells, so how tokens are drawn inside the gate does not matter.
- Decomposition of FastV's loss (accuracy points): FastV − Random = (FastV − Gate+FastV) "air" +
  (Gate+FastV − Random) "ranking".
  | model, dataset | FastV − Random | air | ranking |
  |---|---|---|---|
  | Lingshu, RIN | −41.2 | −13.2 | −28.0 [−31.7, −24.4] |
  | Qwen3-VL, RIN | −33.7 | −15.0 | −18.7 [−22.0, −15.4] |
  | Lingshu, SLAKE | −12.6 | −12.7 | +0.1 [−2.1, +2.2] |
  | Qwen3-VL, SLAKE | −2.4 | −1.6 | −0.8 [−2.6, +1.1] |
  On SLAKE (CT/MRI with large air regions) background explains all of FastV's loss. On RIN (224² crops, gate active on
  889/1000 questions, mean background 33% of tokens) it explains a third to a half; the rest is attention ranking
  inside the foreground. [FLAG] "Pruning the Air" describes SLAKE, not RIN; title and abstract must carry both terms.
- Correction to the previous entry's reading. `uniform_l2` keeps the *same* 6 tokens on every RIN image (cross-image
  Jaccard 1.000) and matches Random (78.9 vs 79.3). Image independence is therefore harmless by itself; what FastV
  loses is spatial coverage. Mean distance from a grid cell to its nearest kept token (token cells; lower = better
  spread): Random 1.75, Uniform 1.38, Gate+FastV 2.07 / 2.00, FastV 2.60 / 2.25 (Lingshu / Qwen3-VL, RIN); FastV's
  kept tokens sit at mean row 0.18 / 0.36 of the grid (top), Random at 0.50. Across the six selectors on RIN,
  accuracy vs coverage gap r = −0.91 / −0.88 (six points per model: descriptive, not a test). This whole-grid
  statistic penalises gated selectors for leaving background empty; the paper version must measure coverage of the
  foreground only (`analysis.py`, needs the gate masks from the bundle).
- Qwen3-VL SLAKE stays unreadable until the scoring preflight: `uniform_l2` +3.1 over Random but `gate_uniform` −2.5
  below `uniform_l2` [−4.6, −0.5] is the pattern a noisy open-answer metric produces (Full 58.2).
- Ledger: A 4.87 spent, B 3.17 spent; project 8.0 spent, 40.0 left, slack ≈ 34 (A 8.8, B 13.0, C 11.9). Added a
  `uniform_l2` row in E3 to the slack list.

### 2026-11-03 — Claude — Stage 5 launched; diagnostic controls; offline diagnostics; gap register
- **Launched** (17:50 IST): A `medprune-gpu-lingshu7b` v8 and B `medprune-gpu-qwen3vl8b` v4, `EXP = "E6,E3"`,
  `MAX_HOURS = 9.5`, git `1c4fef2`. E6's controls on K run first: `static_map`, `inverse_gate`, `divprune` ± gate,
  `ropefree` ± gate, `fastv_k8`, `fastv_k16`, `artifact_excl`, Random seeds 1–2. Earlier runs are restored from the
  new private datasets `pushkararora88/medprune-restore-lingshu7b` and `pushkararora/medprune-restore-qwen3vl8b`
  (run files + static maps, marker `medprune_restore.json`). C's bundle is rebuilding (v3, CPU); C then runs
  `EXP = "E7,E10,E11"` for both models (3.8 / 3.4 GPU-h). Nothing on C needs the RadImageNet token.
- **Code** (90 tests): `run_eval --exp` takes a comma list; `analysis static-maps` (T4); `data resize` / `data vflip`
  (same-question variants, IDs unchanged); method `fastv_bottom` (`MethodSpec.invert`); notebook restore from an
  attached dataset and E10/E10r/E11/E11r variant building under `/tmp`. Commits `1c4fef2`, `106b85d`, `cd25732`.
- **Offline diagnostics** on the 36 finished runs (laptop, no GPU; scripts in the session scratchpad, to be folded into
  `analysis.py`):
  | test | Lingshu RIN | Qwen3-VL RIN | Lingshu SLAKE | Qwen3-VL SLAKE |
  |---|---|---|---|---|
  | FastV answer = text-only answer (Random) | 33.3% (21.6%) | 46.7% (33.2%) | 37.3% (31.5%) | 32.8% (33.6%) |
  | retained visual gain, FastV / Random / Gate+FastV | 26% / 86% / 45% | 35% / 91% / 60% | 60% / 84% / 84% | 76% / 83% / 81% |
  | keep-set Jaccard, same image, other question (FastV; Random 0.054) | — | — | 0.978 | 0.820 |
  | cross-image Jaccard, FastV top-10% by layer-1 score | 0.773 | 0.460 | 0.299 | 0.170 |
  | same, RoPE-free layer-1 score; overlap with FastV | 0.770; 0.94 | 0.489; 0.77 | 0.305; 0.91 | 0.152; 0.68 |
  | same, layer-15 score; BRR | 0.384; 1.99 | 0.489; 2.89 | 0.126; 1.11 | 0.195; 1.62 |
  | attention-rank effect inside Random (log-odds per SD; p) | −0.12 (0.18) | −0.05 (0.57) | −0.14 (0.03) | +0.06 (0.39) |
  | same inside Gate+Random | −0.35 (<0.001) | −0.18 (0.06) | −0.19 (0.005) | −0.05 (0.42) |
  | FastV penalty explained by foreground coverage (mediation) | 11% | 0% | 46% | n.s. |
  Retained visual gain = (acc − text-only) / (Full − text-only). Text-only: 20.1 / 30.1 / 24.5 / 24.2.
- **Dose-response** (FastV − Random by the image's background fraction): RIN Lingshu −45.0 (no background, n = 111),
  −51.0, −42.5, −28.6, −50.0 → flat; Gate+FastV − Random −45.0 → +5.6 as background grows (ρ = +0.33). SLAKE Lingshu
  −1.9 (X-ray, no background), −17.0, −20.3, −16.3 (ρ = −0.35, p = 0.001); Qwen3-VL ρ = −0.27 (p = 0.008).
- **SLAKE by question type** (Lingshu FastV − Random): Organ −25.3 (n = 253), Size −20.0, Quantity −13.5, Position −13.4,
  Modality −11.1, Plane −8.6, Abnormality −2.7, KG −0.7; Gate+FastV is within ±6 everywhere.
- **High-norm tokens** (robust z > 5 on layer-1 norms): 3.3% of FastV's picks vs 1.0% base rate (Lingshu RIN), 1.9% vs
  0.3% (SLAKE); Qwen3-VL 0% and 0.5% vs 0.1% → over-selected (P5) but too rare to explain the loss.
- **Reading.** FastV's layer-1 ranking ignores the question, is close to a fixed map at 64 tokens, is not caused by
  RoPE, and — at least for Lingshu — prefers tokens that are *less* useful than random ones; the pruned model then falls
  back on the language prior. Background capture explains SLAKE (CT/MRI) fully but not RadImageNet. Coverage correlates
  with accuracy across selectors but does not explain FastV per question, so the claim is about *which* tokens attention
  picks, not only how spread they are. Thesis revised (§A1), claims status (§A2), gap register opened (§A9).
- [FLAG] The 224² / 64-token confound (G1) can still reframe the paper: if photographs at 224² collapse like
  RadImageNet, the paper becomes "attention pruning fails at small token counts; medical benchmarks are small and CT/MRI
  add background capture". E10 decides it tonight.
- 18:10 IST: C's bundle (v3) finished; its manifest hashes and mask summaries equal A's committed
  `configs/bundle_manifest.json` for all five datasets. C's two GPU notebooks (`pushkarsiuuarora/medprune-gpu-lingshu7b`,
  `.../medprune-gpu-qwen3vl8b`, v1, `EXP = "E7,E10,E11"`) started in parallel; C accepts two concurrent T4×2 sessions.

### 2026-11-03 — Claude — Open-answer judge, HuatuoGPT port, deeper offline diagnostics, depth plan
- **Judge (G9)**, `medprune/judge.py` (commit `6e69839`): unique (question, reference, response) items → Qwen2.5-7B-
  Instruct on a GPU, one forward pass each, verdict = P(Yes) / (P(Yes) + P(No)) with MedEvalKit's comparison
  instructions; blind 200-item hand-label sheet stratified by judge/rule agreement; population-weighted Cohen's κ with
  stratified bootstrap. MedEvalKit's own coding ("<judge>0" = correct) is **not** used: a CPU smoke test (Qwen2.5-0.5B)
  read it backwards, scoring "Lung" vs "Lung." as wrong; with Yes/No the same small model accepts 14/14 exact matches.
  `analysis.load_runs(open_scoring="judge")`, `analysis diagnosis --open-scoring judge`. Kaggle wrapper
  `notebooks/03_judge.ipynb`; items dataset `pushkarsiuuarora/medprune-judge-items` (2,141 unique answers, private).
- **HuatuoGPT port (T5, G6)**, `medprune/huatuo_llava.py`: stock `Qwen2ForCausalLM` with the pruning attention (the
  checkpoint's 339 LLM keys match it exactly) + the fine-tuned CLIP read from shard 3 (391/391 keys match a
  `CLIPVisionModel` built from `vit/`'s config; the stock CLIP weights in `vit/` are not used) + the mlp2x_gelu
  projector; prompt, `expand2square` colour (122, 116, 104) and generation follow `cli.py`. Tests: template, 576 raster
  tokens after `"<|user|>\n"`, masking ≡ token removal through `inputs_embeds`, FastV state logs scores, RoPE-free
  scores and norms. `runs.yaml`: HuatuoGPT E0 check (C, `--limit 50`, full/text-only/random/FastV); E8 runs K first.
- **Queued on C** (launcher `launch_c_next.sh`, pushes as C's two slots free, ~22:00): `pushkarsiuuarora/medprune-gpu-
  huatuo7b` (EXP E0, LIMIT 50) and `pushkarsiuuarora/medprune-judge` (pass 1).
- **Deeper offline diagnostics** (36 finished runs; scratchpad `deep_diag.py`, to be folded into `analysis.py` as D3):
  | test | Lingshu RIN | Qwen3-VL RIN | Lingshu SLAKE | Qwen3-VL SLAKE |
  |---|---|---|---|---|
  | variance of layer-1 token ranks: position / image / question | 69% / 31% / — | 51% / 49% / — | 5% / 94% / 0.3% | 12% / 85% / 3.1% |
  | same, RoPE-free layer-1 score | 69% / 31% / — | 47% / 53% / — | 7% / 93% / 0.0% | 8% / 92% / 0.1% |
  | same, layer 15 | 38% / 62% / — | 44% / 56% / — | 6% / 72% / **23%** | 23% / 74% / 3.5% |
  | Spearman(layer-1 score, layer-1 input norm); FastV top-10% ∩ top-10% by norm (chance 0.10) | 0.32; 0.32 | 0.00; 0.28 | 0.44; 0.39 | 0.16; 0.23 |
  RadImageNet has one question per image, so its "image" share includes the question. Position = 8×8 bins of
  normalized coordinates.
- **Evidence grounding** (SLAKE `mask.png` + `mask.txt`; questions whose text or answer names an annotated structure;
  token cells from each model's resize; evidence = cells touching the structure):
  | | Lingshu (356 q, 84 img) | Qwen3-VL (363 q, 85 img) |
  |---|---|---|
  | evidence retention ratio (kept share / chance), FastV · Random · Gate+FastV · Gate+Random | 0.78 · 0.99 · 2.70 · 2.23 | 0.59 · 1.04 · 1.61 · 2.28 |
  | same, large organs (top area tertile, median 5.3% of the image) | 0.23 · 0.99 · 1.91 · 2.02 | 0.22 · 1.00 · 1.40 · 2.13 |
  | structure dropped entirely, large organs (FastV / Random) | 56% / 10% | 70% / 10% |
  | FastV − Random accuracy by tertile small / medium / large | −1.6 / −3.4 / **−17.5 [−25.7, −8.8]** | −0.8 / +3.4 / −4.2 |
  | Gate+FastV − Random by tertile | +1.6 / +1.7 / −1.7 | 0.0 / +5.9 / −3.3 |
  Per question, evidence recall explains only 7% of FastV's penalty (Lingshu, logistic mediation) — same pattern as
  coverage (G11): the selector-level effect is large, the per-question proxy is noisy. Within Random, "any evidence
  token kept" predicts correctness for Lingshu (+0.54 log-odds, p = 0.04, controlling for organ area).
- **Answer collapse**: mild. RIN: FastV's most frequent letter 31% (Random 25%, text-only 36%); SLAKE Lingshu: 93
  distinct answers under FastV vs 116 under Random, entropy 4.85 vs 5.09 bits.
- **Reading.** H4 is reversed: FastV's SLAKE loss is on the *large* organs the questions name, which it keeps at about a
  fifth of chance. The positional prior survives the removal of RoPE, so it lives in the visual keys (G18). Question
  dependence only appears deep (layer 15) and only in Lingshu. P14 filled (preliminary, contradicted); §A2 status,
  §A9 rows G2/G5/G6/G9 and new G16–G21, §A10 depth plan (D1–D7) added.
- [FLAG] D1 (layer profile) is the highest-value missing experiment: one run answers G2, G5, G8, G17 and G18 and gives
  Fig. 3. It needs ≈ 2 h of hook code before A/B session 2 (~03:30) or it moves to C.

### 2026-11-03 — Claude — D1–D3: layer profile, text readout, evidence grounding as tested code
- **Hook** (`attention_hook.py`): `text_attention` (head- and query-mean causal attention of every token after the
  image; equals eager attention in the tests), `sink_scores` (Kang et al., ICLR 2025: sink dimensions = where the first
  token's RMS-normalized state exceeds 20; score = max over them of a visual token's normalized |activation|),
  `install_profile_hooks` (pre-hooks on every decoder layer, active only under a profile state). Profile states log
  `lp_last_l{L}`, `lp_text_l{L}` (log-probabilities: deep-layer visual attention underflows float16 probabilities) and
  `sink_l{L}` for every layer. Adapters (Qwen, HuatuoGPT) pass `text_pos` = every position after the last visual token.
- **Methods**: `fastv_text` (top-k by the text readout at layer 1, mask from 2; G2) and `full_profile` (no pruning,
  answers = Full; D1). 29 methods.
- **Analysis**: `evidence_fractions` / `evidence_retention` / `evidence_report` (D3; SLAKE label masks mapped into each
  model's view by `masks.label_view`, variants `_s224` / `_vflip` transformed like their images; "left lung" names one
  side, "lungs" both), `rank_variance_shares` (position / image / question split, exactly additive), `layer_profile` and
  `plot_layer_profile` (D1), CLI `analysis evidence|profile`. 114 tests.
- **D3 on the finished runs** (canonical; supersedes the scratch numbers in the previous entry — the tested version
  maps sides exactly and matches more structures):
  | | Lingshu (358 q, 84 img) | Qwen3-VL (364 q, 85 img) |
  |---|---|---|
  | evidence retention ratio, FastV · Random · Gate+FastV · Gate+Random | 0.84 [0.57, 1.11] · 1.00 · 2.81 · 2.28 | 0.61 [0.43, 0.80] · 1.05 · 1.68 · 2.33 |
  | FastV by area tertile small · medium · large | 1.46 · 0.84 · **0.20** | 0.88 · 0.72 · **0.23** |
  | large organs dropped entirely, FastV / Random | 58% / 13% | 71% / 11% |
  | FastV − Random accuracy, small · medium · large | −2.5 · −5.0 · **−16.8 [−25.2, −7.1]** | −2.5 · +6.6 · −5.0 [−10.6, +1.7] |
  | Gate+FastV − Random, small · medium · large | +3.3 · −3.4 · +1.7 | +0.8 · +5.0 · −3.3 |
- **Scheduling**: `runs.yaml` reordered so A/B session 2 runs E10r → E11r → E12 → rest of E3 → E4 → E5 → E3 S2
  (SLAKE-zh, lowest priority); E12 = `full_profile` on K + photographs and `fastv_text` on K, both models (1.0 / 0.9
  GPU-h). HuatuoGPT's E8 P2 block gains `full_profile` and `fastv_text`. Notebook 01: E12 needs RadImageNet.
- [FLAG] Lingshu's overall evidence retention CI includes 1; the claim is the size dependence (0.20× on large organs,
  1.46× on small ones), not "FastV ignores the evidence" in general.

### 2026-11-03 — Claude — C's controls read (E7 photographs, E10 resolution, E11 flip + bottom-k)
- **Sessions**: `pushkarsiuuarora/medprune-gpu-qwen3vl8b` v1 COMPLETE 21:14 (3.04 GPU-h), `.../medprune-gpu-lingshu7b`
  v1 COMPLETE 21:18 (3.10 GPU-h), git `c809ed4`; 19 runs each, all complete; merged into `results/all` (74 run files).
  C's freed slots now run the HuatuoGPT check (pushed 21:15) and judge pass 1 (21:18).
- **FastV − Random** (paired, image-clustered 95% CI; uniform − Random in brackets):
  | | photographs | photographs @224² | SLAKE | SLAKE @224² | RadImageNet (224²) |
  |---|---|---|---|---|---|
  | Lingshu | −8.7 [−11.7, −5.8] (+1.3) | −10.7 [−13.8, −7.8] (−0.3) | −12.6 [−15.9, −9.3] (+1.0) | −16.3 [−19.6, −13.2] (+1.2) | −41.2 [−45.0, −37.6] (−0.4) |
  | Qwen3-VL | −4.9 [−7.8, −2.1] (−2.5) | −2.7 [−5.7, +0.3] (+0.8) | −2.4 [−4.9, +0.2] (+3.1) | −0.5 [−3.4, +2.5] (+6.3) | −33.7 [−37.2, −30.3] (−0.7) |
  Photographs: GQA −9.2 / −7.8, POPE −8.2 / −2.0 (Lingshu / Qwen3-VL); RoPE-free = FastV (−8.5 / −4.7). Random − Full:
  photographs −6.0 / −6.0, RIN −9.7 / −5.1. Gate+FastV − Random on SLAKE @224²: −5.3 [−8.3, −2.2] Lingshu (native +0.1),
  −0.6 Qwen3-VL — at 64 tokens the gate no longer closes Lingshu's SLAKE gap.
- **Flip (E11)**: Jaccard of the flipped image's FastV keep-set with the upright keep-set at the same grid positions
  0.637 (Lingshu) / 0.359 (Qwen3-VL), with the mirrored positions (follows the content) 0.130 / 0.200, random 0.052.
  FastV − Random on flipped SLAKE −12.3 / −0.7 (upright −12.6 / −2.4).
- **Bottom-k (E11)**: SLAKE `fastv_bottom` − `fastv` = **+6.2 [+2.5, +10.1] Lingshu**, −4.2 [−7.2, −1.4] Qwen3-VL;
  `fastv_bottom` − Random −6.4 / −6.6.
- **Variance of layer-1 token ranks, exact grid positions** (`rank_variance_shares(position="grid")`, grid shapes with
  ≥ 5 images; the earlier normalized-bin numbers pooled SLAKE's five image sizes and understated position):
  | position / image / question | photographs | photos @224² | SLAKE | SLAKE flipped | SLAKE @224² | RIN |
  |---|---|---|---|---|---|---|
  | Lingshu | 31 / 69 / 0.1% | 39 / 61 / 0.2% | 40 / 59 / 0.3% | 45 / 54 / 0.2% | 64 / 36 / 0.3% | 69 / 31 / — |
  | Qwen3-VL | 19 / 80 / 1.2% | 22 / 77 / 0.8% | 27 / 70 / 3.1% | 30 / 67 / 3.0% | 40 / 59 / 1.5% | 51 / 49 / — |
  At layer 15 the question share reaches 14–32% for Lingshu and stays ≈ 3% for Qwen3-VL on every dataset.
- **Evidence grounding** on the variants: FastV keeps large named organs at 0.20–0.29× chance upright, flipped and at
  224² for both models; large-organ loss Lingshu −16.8 / −11.8 / −10.3 (upright / flipped / 224²). Lingshu's
  `fastv_bottom` keeps the named organ less often than FastV (0.59× vs 0.83×) yet answers 6.2 points better → G23.
- **Decisions.** P13 contradicted (both models). G1: token count amplifies modestly (Lingshu only); RadImageNet's
  collapse is dataset-specific → new G22, tested by E11r, E6 `static_map` on RIN and E10r. G3: positional for Lingshu.
  G4: answered by intervention for Lingshu (anti-informative), weakly informative for Qwen3-VL. G5: question-blindness
  holds on photographs too.
- [FLAG] The thesis changes from "medical images break attention pruning" to "early-layer attention pruning is a
  question-blind prior that loses to random selection on any image at 10%; medical scans amplify it (background,
  large-organ avoidance) and RadImageNet collapses". "Pruning the Air" still names the CT/MRI amplifier but no longer
  the whole paper. Needs a decision before the abstract (Nov 9) and enrollment (Nov 10).

### 2026-11-03 — Claude — Judge pass 1 read; v3 prompt
- `pushkarsiuuarora/medprune-judge` v1 COMPLETE 21:28 (0.10 GPU-h): 2,141 unique open answers from the 36 Stage 4/4b
  runs; verdict mass on Yes/No ≈ 1.000 for every item, 2.2% undecided (P between 0.2 and 0.8).
- Against the rule (normalized exact match), row level: agreement 83.8%; the judge accepts 89.2% of rule-correct rows
  and 21.6% of rule-rejected rows. Two failure modes: (1) it overrules the reference with its own knowledge — 16 of 364
  unique exact matches rejected, all of the type "What is the largest organ in the picture? — Liver / Liver." (p =
  0.00) or "Which organs appear in pairs? — Temporal Lobe"; (2) leniency — "Not applicable — no picture provided." for
  "None", "Left" for "Upper Left", "Pneumonia" for "Pulmonary Infiltration"; text-only open accuracy 10.9% → 25.1%
  (Lingshu). Legitimate rescues also occur: "There is one heart shown in the image." for "1", "magnetic resonance
  imaging" for "MRI".
- Open-question FastV − Random (question-level CI), rule vs judge-v2: Lingshu −11.5 vs −11.6; **Qwen3-VL +0.8 vs −3.4
  [−6.4, −0.8]**; Qwen3-VL Full open accuracy 45.6 → 55.8 (exact match penalizes its longer answers). Preliminary until
  the v3 pass and the hand labels.
- **v3** (`d9d678c`): rubric prompt (reference = ground truth, the judge cannot see the image; partial, less specific
  or refusing answers are wrong; consistent extra detail is fine), scorer = exact match OR judge, verdicts filtered by
  prompt version, label sheet stratified as exact (10%) / rule-rejected-judge-accepts (45%) / rule-rejected-judge-
  rejects (45%) with the rubric in a README. v3 pass: 2,855 unique answers from all 74 runs (dataset version 2), queued
  on C.

### 2026-11-03 — Claude — HuatuoGPT port validated on C; E8 (SLAKE) launched
- `pushkarsiuuarora/medprune-gpu-huatuo7b` v1 (EXP E0, LIMIT 50, git `55231d1`) COMPLETE 21:34, 0.23 GPU-h incl. the
  17.6 GB download. `e0_report__huatuo7b__t4x2.json` passes: fp16 vs fp32 vision 50/50 identical, logits finite,
  logging inert 50/50, offline keep-sets = FastV's 50/50. Responses are clean (no template text; 3 of 50 Full answers
  hit 64 tokens). Rule-scored accuracy on the 50 questions: Full 42, FastV 56, Random 54, text-only 18 — open answers
  are verbose and correct in meaning ("CT scan", "Thorax", "Lungs"), so exact match fails them (open 21%, closed 71%
  of 21); the judge scores them. Not comparable with the authors' published numbers (different scorer); the
  mechanical checks are the validation. 1.13 s per question (predicted 1.15).
- C's notebooks have no `cvpr` secret (log: "RadImageNet steps will fail"), so `runs.yaml` splits E8 into E8 (SLAKE:
  6 P1 + 5 P2 methods incl. `full_profile`, `fastv_text`) and E8r (RadImageNet anatomy + abnormality). Notebook 01:
  E8r (not E8) needs RadImageNet. E8 v2 launched on C 21:40 (`40914ce`, MAX_HOURS 5).
- Needs Pushkar (UI only): attach the `cvpr` secret to C's `medprune-gpu-huatuo7b` notebook (Add-ons → Secrets) so
  E8r can run; the same is needed for E6-P2 on C (S1 includes RadImageNet).

### 2026-11-04 — Claude — HuatuoGPT E8 (SLAKE) read; judge v3 read; first layer profile
- **C sessions**: `medprune-gpu-huatuo7b` v2 (E8, git `40914ce`) COMPLETE ~00:50, 3.34 GPU-h, 11 runs × 1,061 all
  complete, merged (85 run files). `medprune-judge` v2 (v3 prompt) COMPLETE, 0.21 GPU-h, all 2,855 answers. Judge pass
  3 (1,169 HuatuoGPT answers, dataset version 3) running.
- **HuatuoGPT on SLAKE** (exact-match scoring; open answers wait for the judge — its long answers fail exact match):
  | vs Random (paired, 95% CI) | all 1,061 | closed 416 |
  |---|---|---|
  | FastV | −0.6 [−2.7, +1.4] | +2.6 [−1.3, +6.6] |
  | RoPE-free | **+3.1 [+1.1, +5.0]** | **+7.5 [+3.3, +11.6]** |
  | Gate+FastV | +2.5 [+0.7, +4.3] | +6.7 [+3.0, +10.6] |
  | Uniform | +0.7 | +1.9 |
  | Full (Random − Full) | −1.0 [−2.9, +1.0] | −4.8 [−8.2, −1.5] |
  Least-attended − most-attended −10.7 (closed −16.3); `fastv_text` − FastV −1.4 (n.s.). Text-only 19.7 (closed 44.0).
  FastV's keep mass by row quarter (top → bottom) 0.12 / 0.04 / 0.07 / **0.77**; columns flat. Layer-1 rank variance
  (exact grid): position 52% / image 42% / question 6.2%; RoPE-free position 23%. Same-image question Jaccard 0.70.
- **Reading.** The failure depends on the architecture. In the 1D-RoPE LLaVA model the early-layer prior is rotary and
  lands on the bottom rows (FEATHER's finding), RoPE-free scoring removes it and beats Random, and attention is
  informative. In the M-RoPE Qwen-family models the prior survives RoPE removal (it sits in the visual keys) and, for
  Lingshu, attention is anti-informative. The paper's mechanism section should present this contrast (G18), and C1
  ("FastV < Random") is a statement about the Qwen-family models plus photographs, not about every VLM.
- **First layer profile (D1, HuatuoGPT SLAKE)** — `analysis profile`: position share 63% (layer 0) → 40% (1) → 7–10%
  (6–12) → ≈ 20% (late); question share peaks at layer 21 (27%, last-token readout; the text readout stays ≤ 10%);
  top-10% evidence retention 1.6× at layer 1, 2.7× at layer 21 (HuatuoGPT's attention finds the named organ, unlike
  Lingshu/Qwen3-VL). [FLAG] The sink score flags 64–99% of visual tokens from layer 6 on — not discriminative as
  implemented (likely: once the first token has a massive dimension, every RMS-normalized token is large there). Fix
  the definition (e.g. ratio to the token's median |activation|, or Sun et al.'s absolute threshold) before any G17
  claim.
- **Judge v3** (Lingshu/Qwen3-VL answers): exact matches it would reject 7.8% (v2 10.8%) — irrelevant now that exact
  match wins; rescues 18.5% of rule-rejected rows (v2 21.6%); text-only open accuracy still rises 10.9 → 26.0
  (Lingshu) → leniency persists; the hand labels decide (κ ≥ 0.8 or the rule stays). Open-question FastV − Random,
  rule → rule-OR-judge: Lingshu −11.5 → −13.5; **Qwen3-VL +0.8 → −3.6 [−6.7, −0.5]**; leniency would favour FastV's
  generic answers, so the flip is unlikely to be a leniency artefact.
- **Session 2 prepared** (`push_s6_<model>` in the scratchpad): EXP `E10r,E11r,E12,E3,E4,E5`, RESTORE_FROM auto,
  MAX_HOURS 9.5 — launch after session 1 ends (~03:30) and the restore datasets are re-versioned with its outputs.

### 2026-11-04 — Claude — Judge pass 3; label sheet; SLAKE under rule-OR-judge
- Judge pass 3 (1,169 HuatuoGPT answers, 0.11 GPU-h) done: every open row of all 85 runs (31,605) has a v3 verdict
  (4,024 unique answers).
- **Label sheet** for Pushkar: `results/all/judge/label_sheet.csv` (200 blind items: 20 exact matches, 90 rule-rejected
  answers the judge accepts, 90 it rejects; populations 13,602 / 3,443 / 14,560 rows), rubric in
  `label_sheet_README.txt`, strata and weights in `label_sheet_key.csv`. Then `python -m medprune.judge kappa
  --results results/all`.
- **SLAKE (open + closed) FastV − Random, exact match → rule-OR-judge (provisional until κ)**: Lingshu −12.6 → −13.9
  [−17.1, −10.7]; **Qwen3-VL −2.4 (n.s.) → −5.0 [−7.9, −2.0]**, Gate+FastV −0.1; HuatuoGPT −0.6 → −1.6 [−3.5, +0.4],
  RoPE-free +3.6 [+1.5, +5.7], Gate+FastV +2.7 [+0.7, +4.9]. Full accuracy under the judge: Lingshu 82.4, Qwen3-VL
  64.8, HuatuoGPT 56.6 (exact match 77.2 / 58.2 / 44.7). If κ holds, C1 on SLAKE holds for both Qwen-family models
  and the gate closes both gaps.

### 2026-11-04 — Claude — Session 1 merged; session 2 and C launched; label audit; rubric guard; sink fix
- **A/B session 1** COMPLETE: `medprune-gpu-lingshu7b` v8 9.46 GPU-h, `medprune-gpu-qwen3vl8b` v4 9.76 GPU-h (both
  stopped at the 9.5 h budget). 41 + 44 new run files, all complete (no partial runs); merged into `results/all`
  (170 runs) with their diag parts. Restore datasets re-versioned (126 / 124 files). The A/B watcher had died overnight
  (fork limit), so A and B sat idle from ~03:30 to 17:45.
- **Launched 17:44–17:46**: A v9 and B v5 (EXP `E10r,E11r,E12,E3,E4,E5`, RESTORE_FROM auto, MAX_HOURS 9.5; pending 46
  / 35 runs); C `medprune-gpu-huatuo7b` v3 (E8r, 16 runs, MAX_HOURS 5) and C `medprune-gpu-lingshu7b` v2 (E6-P2, 18
  runs, MAX_HOURS 4). [FLAG] Kaggle shows no live log, so whether C's Lingshu notebook also has the `cvpr` secret is
  known only when it ends; without it the RadImageNet half of E6-P2 fails. Judge pass 4 (1,145 answers: 863 SLAKE-zh,
  282 SLAKE-en; dataset version 4) is pushed by a watcher when a C slot frees.
- **Session 1 read** (FastV-family − Random@L2, 10% kept, paired, image-clustered 95% CI; rule scoring):
  | | Lingshu RIN anat | Lingshu SLAKE | Qwen3-VL RIN anat | Qwen3-VL SLAKE |
  |---|---|---|---|---|
  | FastV (K=2) | −41.7 | −12.4 | −33.7 | −2.4 |
  | FastV K=8 / K=16 | −43.9 / **+8.5** | −12.4 / **+6.9** | −27.1 / **+5.6** | −0.3 / +0.9 |
  | RoPE-free | −42.3 | −12.5 | −31.9 | −1.4 |
  | static map (position only) | −50.9 | −13.0 | −27.8 | +2.0 |
  | inverse gate (background only) | −23.9 | −4.8 | −7.6 | +1.7 |
  | artifact exclusion | −42.0 | −12.3 | −33.7 | −2.4 |
  | DivPrune / Gate+DivPrune | −11.3 / −1.7 | +0.6 / +3.2 | +1.5 / +3.0 | +0.3 / +0.9 |
  | Gate+FastV | −28.5 | +0.3 | −18.7 | −0.8 |
  New datasets: OmniMedVQA anatomy (3,329 Q) FastV −29.4 (Lingshu) / −4.3 (Qwen3-VL); RIN abnormality −14.2 / −2.2.
  Random seeds 0–2 agree within 1.5 points. SLAKE CT/MRI: FastV −18.0 / −4.7, inverse gate −7.5 / +2.0, static map
  −15.5 / +1.0, K=16 +8.0 / +3.9.
- **Reading.** (1) The failure is early-layer: K=8 is as bad as K=2, K=16 beats Random on both models. (2) It is not
  background selection: background-only tokens beat FastV everywhere (P12 contradicted) — background tokens carry
  usable information and FastV's concentrated keep-set is worse than any spread-out one. (3) For Lingshu on SLAKE a
  position-only map from another dataset reproduces FastV's loss (−13.0 vs −12.4) — positional prior, matching the flip
  test; for Qwen3-VL it does not. (4) High-norm tokens are over-selected but irrelevant (P5 not causal, H3 dead).
  (5) RoPE is not the cause in the Qwen family (RoPE-free = FastV), unlike HuatuoGPT. (6) Coverage-preserving
  selectors (Random, Uniform, DivPrune) are the bar; Gate+DivPrune is the best training-free selector on RIN.
- **Framing decision (Pushkar)**: medical only, no general-domain section; consequences recorded in §A9 (no
  "medical-specific" claim; E7 as a supplementary scope note).
- **Label audit.** Codex filled all 200 cells (`label_sheet_provenance.txt`: AI labels, blinded to the judge). Audit:
  rubric applied faithfully on less-specific locations ("Left Lung" for "Lower Left Lung" → 0, 17 items) and partial
  lists; two inconsistencies — organ-system questions ("brain" for "Head" accepted, "respiratory" for "Chest" and
  "gastrointestinal" for "Abdomen" rejected; SLAKE answers organ-system questions with body regions) and extra
  consistent items (accepted for symptoms, rejected for "medical, supportive, surgical treatment"). Adjudicated copy
  `label_sheet_adjudicated.csv` flips those 12 items to 1. κ (population-weighted, stratified bootstrap):
  | scorer | Codex labels | adjudicated |
  |---|---|---|
  | exact match | 0.806 [0.744, 0.859] | 0.763 |
  | rule OR judge (pre-registered) | 0.813 [0.748, 0.872] | 0.816 |
  | **rule OR synonym OR (judge AND NOT less-specific location)** | **0.898 [0.847, 0.939]** | **0.902** |
  The pre-registered gate (κ ≥ 0.8 and > rule) passes by 0.007 — not a meaningful win. Of the answers the judge
  rescues, only 52% are correct (Codex) / 63% (adjudicated): it ignores the rubric's less-specific clause on locations
  and rejects "axial" for "Transverse Plane" (4 of 90 rejected items). `judge.apply_judge` now applies both
  deterministically (`SCORER_VERSION`, default on; `guard=False` gives the pre-registered scorer); `kappa` reports all
  three. [FLAG] The guard was written after reading these labels, and the labels are AI labels: the κ of 0.90 is
  in-sample and AI-to-AI. Before the paper reports it, a held-out sheet (100 new items) needs human labels.
- **SLAKE FastV − Random by scorer** (rule / rule|judge / guarded; runs with full verdict coverage): Lingshu −12.6 /
  −13.9 / −13.5; **Qwen3-VL −2.4 [−4.9, +0.2] / −5.0 / −4.8 [−7.6, −1.9]**; HuatuoGPT −0.6 / −1.6 / −1.6. The Qwen3-VL
  SLAKE loss is significant under both judge scorers and not under exact match — report both.
- **Sink score fix (D1).** HuatuoGPT's per-layer sink scores are unimodal, with a scale that grows with depth (median
  0.8 at layer 0, 22 at layer 6, 43 at layer 27), and no image has a token above median + 5·MAD. The fixed τ = 20 was
  measuring scale. `analysis.layer_profile` now uses the per-image robust rule (`sink_z = 5`; `sink_tau` keeps the
  absolute option). Reading: no visual-sink tokens in HuatuoGPT by Kang et al.'s criterion; G17 rests on
  `artifact_excl` (no effect) instead.
- 116 CPU tests pass.

### 2026-11-04 — Claude — E13 ported and launched; C E6-P2 SLAKE half read
- **E13** (commits `ec9133c`, `d6fd1fa`): SparseVLM, VisPruner, PruneSID, BTP in `medprune/baselines/` — selection
  code from pinned official commits, medprune prompts and scoring; `tests/test_baselines.py` checks the kept indices
  against the official functions from the clones in third_party/ (all four pass; 128 tests). Deviations, each in
  its module docstring: multi-stage methods token-matched to FastV (mean visual tokens per layer), SparseVLM without
  recycling and with original position ids (the authors' v2.0 choices), VisPruner's [CLS] score replaced by the
  attention each patch receives in the last vision block (no [CLS] in Qwen towers) plus a guard where the official
  loop crashes on odd residual counts, PruneSID without its 16-token floor, BTP's stage layers scaled by depth on
  Qwen3-VL. Smoke v1 (3 SLAKE questions per method, both models) passed: keep counts Lingshu 324 tokens → PruneSID /
  VisPruner 32, BTP 56/28/14/0 at layers 3/6/15/22, SparseVLM 46/21/12 at 3/7/16 (mean 53.2 visual tokens per layer
  vs FastV's 52.9); VisPruner's alignment check (merger of the captured block output = the tower's output) passed
  on both towers. Full runs v2 launched 19:07 on the second slot of A and B (`medprune-baselines-<model>`).
  Predictions P15–P18 added to §A6 before any result.
- **C `medprune-gpu-lingshu7b` v2** (E6-P2, no `cvpr` secret then): the 6 SLAKE runs COMPLETE 19:40; the 12
  RadImageNet runs relaunch when both C sessions stop (secret now attached). Lingshu SLAKE, − Random (all / CT/MRI):
  PDrop −13.8 [−16.8, −10.7] / −20.0; **Gate+PDrop −0.6 / −0.0**; Gate τ 0.25 −1.0, τ 0.5 (Gate+FastV) +0.3, τ 0.75
  +0.9; background quota 3% −2.5. Reading: multi-stage attention pruning token-matched to FastV fails exactly like
  FastV, so the failure is not specific to single-shot pruning (D5 revised: PDrop's published 80%-kept robustness
  does not survive 10%); the gate repairs both and is insensitive to τ. Cross-account determinism: C's
  `artifact_excl` rerun equals A's on all 1,061 questions (keep-sets and responses).
- Judge pass 4 (1,145 answers) running on C since 19:40.

### 2026-11-04 — Claude — HuatuoGPT E8r (RadImageNet) read; judge pass 4 merged
- C `medprune-gpu-huatuo7b` v3 (E8r, `cvpr` secret attached) COMPLETE ~20:25: 16 runs × 1,000, all complete, merged
  (191 run files). Judge pass 4: 1,145 v3 verdicts merged (7,310 lines); 33 answers from C's E6-P2 SLAKE runs remain
  for pass 5 (with E13's SLAKE answers). C `medprune-gpu-lingshu7b` v3 (E6-P2, RadImageNet half only) launched 20:28.
- **RadImageNet anatomy, − Random (10% kept):**
  | | HuatuoGPT | Lingshu | Qwen3-VL |
  |---|---|---|---|
  | Full / Random / text-only | 88.6 / 84.5 / 26.9 | 89.0 / 79.8 / 20.1 | 89.7 / 84.6 / 30.1 |
  | FastV | **−0.1 [−1.9, +1.7]** | −41.7 | −33.7 |
  | RoPE-free / Gate+FastV | +2.5 / +2.0 | −42.3 / −28.5 | −31.9 / −18.7 |
  | `fastv_text` | −6.5 | — | — |
  | least-attended (`fastv_bottom`) | **−33.1** | (SLAKE: Lingshu's least-attended beat its most-attended) | — |
  FastV's keep mass by row quarter (top → bottom) 0.16 / 0.06 / 0.21 / 0.57 (1D-RoPE bottom-row prior, as on SLAKE).
- **Reading (G22).** RadImageNet's collapse is **not a property of the dataset**: HuatuoGPT's FastV ties Random there,
  and its attention is strongly informative (most-attended beat least-attended by 33 points). [FLAG] Confound:
  HuatuoGPT always sees 576 tokens (RadImageNet's 224² images padded to 336²), the Qwen-family models 64. The
  session-2 run E10r (RadImageNet at 448², 256 tokens for Lingshu / Qwen3-VL) separates token count from
  architecture; until then the claim is "Qwen-family models at 64 tokens", not "RadImageNet".
- [FLAG] RadImageNet abnormality is not image-dependent for Qwen3-VL: text-only 71.4 vs Full 71.8 (Lingshu 27.7 vs
  46.8; HuatuoGPT 30.5 vs 41.3, with Full below Random 42.2). Pruning deltas on that cell measure nothing for Qwen3-VL;
  report it only with this caveat, or drop it from Table 1 for Qwen3-VL.

### 2026-11-04 — Claude — C E6-P2 RadImageNet half read (22:40)
- C `medprune-gpu-lingshu7b` v3 (E6-P2, RadImageNet only, `cvpr` secret loaded) COMPLETE, 1.29 h: 12 runs × 1,000,
  all complete, merged (202 run files). The `artifact_excl` rin_anat rerun equals A's session-1 file on every
  question once timing and the new `keep_stages: null` field are dropped (torch cu130 vs cu128): determinism holds
  across accounts on RadImageNet too; A's copy kept.
- **Lingshu, − Random (10% kept), RIN anatomy / RIN abnormality:** FastV −41.7 / −14.2; PDrop **−61.2** / −17.5;
  Gate+FastV (τ 0.5) −28.5 / −10.1; Gate+PDrop −31.4 / −14.0; Gate τ 0.25 −32.8 / −12.6; Gate τ 0.75 **−25.6** /
  −8.9; background quota 3% −36.1 / −13.5; `artifact_excl` −42.0 / −15.3. All CIs exclude 0 (image-level sign-flip p ≤ 1e-4).
- **Reading.** (1) On RadImageNet the gate is a partial remedy only: it removes 32–39% of FastV's deficit on anatomy
  (−41.7 → −28.5 / −25.6), versus all of it on SLAKE. (2) τ matters here and is monotone (a stricter foreground
  filter, `gate_frac >= τ`, helps: −32.8 → −28.5 → −25.6), while SLAKE was flat (−1.0 / +0.3 / +0.9); giving 3% back
  to the background hurts (−36.1). So at 64 tokens the deficit is not missing context: it is attention picking the
  wrong foreground tokens, and the gate helps only by removing the edge tokens with mixed foreground. (3) PDrop is far
  worse than FastV at 64 tokens (−61.2): its stages keep 13 → 6 → 3 → 2 of 64 tokens (from layers 2 / 9 / 16 / 23),
  so matching the mean token count front-loads the budget and starves the late layers when the start count is small. (4) H3 stays dead on RadImageNet (`artifact_excl` =
  FastV). [FLAG] τ was chosen on SLAKE; reporting τ 0.75 as the RadImageNet number would be test-set tuning — the
  paper reports τ 0.5 everywhere and the τ sweep as robustness.
- Ledger C: 16.16 spent (E8r 2.70, E6-P2 v2 1.93, judge pass 4 0.11, v3 1.29); ≈ 5.6 h left this week.
- Still running: E13 on the second slot of A and B (19:07, MAX_HOURS 6); session 2 on A (v9) and B (v5).

### 2026-11-05 — Claude — E13 read: published pruners vs Random, FastV and Gate+Uniform
- A `medprune-baselines-lingshu7b` v2 (4.74 h) and B `medprune-baselines-qwen3vl8b` v2 (4.34 h) COMPLETE 23:58,
  `cvpr` secret loaded; 16 runs each, all complete; merged (234 run files). Judge pass 5 (134 new SLAKE open
  answers: E13 + the 33 from E6-P2) pushed to C (`medprune-judge` v5); the SLAKE cells below are rule-scored until it
  is merged.
- **Statistics fix (`medprune/stats.py`, `analysis.paired_delta`).** The p-value next to every paired difference
  was an exact McNemar on correctness thresholded at 0.5. For Random (mean of 3 seeds) that threshold is a
  **majority vote of three Random draws** — 83.5% vs Random's 79.8% on Lingshu RIN anatomy — so p-values against
  Random tested against a stronger reference (e.g. Uniform −0.9 [−2.9, +1.0] had p = 0.00). McNemar also treats
  questions about one image as independent (SLAKE: 1,061 questions, 96 images). Replaced by an image-level sign-flip
  permutation test (`cluster_signflip_p`, 10,000 flips, resolution 1e-4), consistent with the image-clustered CI;
  `mcnemar_exact` stays in `stats.py`. CIs and point estimates are unchanged. Earlier p-values quoted in this log
  came from the old test; recompute any that go into the paper.
- **Table (− Random, 10% kept, 95% image-clustered CI; BRR = background retention on CT/MRI, first keep-set):**

  | Lingshu | SLAKE | RIN anatomy | RIN abnormality | OmniMedVQA anatomy |
  |---|---|---|---|---|
  | Full / Random / text-only | 77.2 / 68.6 / 24.5 | 89.0 / 79.8 / 20.1 | 46.8 / 46.0 / 27.7 | 82.2 / 68.0 / 30.7 |
  | Uniform | +1.2 [−0.4, +2.8] | −0.9 [−2.9, +1.0] | −7.6 [−10.0, −5.2] | +0.9 [−0.5, +2.3] |
  | **Gate+Uniform** | +2.5 [+0.9, +4.2] | +1.8 [+0.0, +3.5] | +1.6 [−1.4, +4.5] | +1.9 [+0.5, +3.2] |
  | DivPrune | +0.6 | −11.3 [−14.0, −8.6] | — | — |
  | PruneSID | +1.4 [−0.4, +3.2] | −2.5 [−4.9, −0.4] | −4.1 [−6.7, −1.5] | +3.5 [+2.2, +4.8] |
  | VisPruner | +3.2 [+1.5, +5.1] | −0.7 [−2.9, +1.3] | −2.0 [−4.6, +0.7] | +0.4 [−0.9, +1.8] |
  | FastV | −12.4 [−15.4, −9.4] | −41.7 [−45.1, −38.3] | −14.2 [−16.7, −11.8] | −29.4 [−31.2, −27.7] |
  | SparseVLM | −15.1 [−17.9, −12.2] | −44.8 [−48.0, −41.5] | −13.6 [−16.2, −11.2] | −31.7 [−33.5, −30.0] |
  | BTP | −2.7 [−4.7, −0.9] | −9.8 [−12.2, −7.2] | −9.0 [−11.7, −6.2] | −6.5 [−8.1, −5.0] |

  | Qwen3-VL | SLAKE | RIN anatomy | RIN abnormality* | OmniMedVQA anatomy |
  |---|---|---|---|---|
  | Full / Random / text-only | 58.2 / 52.7 / 24.2 | 89.7 / 84.6 / 30.1 | 71.8 / 72.1 / 71.4 | 69.3 / 56.1 / 37.2 |
  | Uniform | +3.0 [+1.3, +4.9] | −0.7 [−2.3, +1.0] | −1.1 | +2.2 [+1.0, +3.5] |
  | Gate+Uniform | +0.5 [−1.1, +2.1] | +1.3 [−0.3, +3.1] | −0.3 | +0.5 [−0.9, +1.8] |
  | DivPrune | +0.3 | +1.5 [−0.3, +3.2] | — | — |
  | **PruneSID** | +4.5 [+3.0, +6.0] | +1.3 [−0.5, +2.9] | −0.1 | +0.7 [−0.6, +2.1] |
  | VisPruner | +1.1 [−0.7, +2.8] | −4.9 [−7.1, −2.8] | −0.6 | −2.1 [−3.3, −0.8] |
  | FastV | −2.4 [−4.9, +0.1] | −33.7 [−36.8, −30.5] | −2.2 | −4.3 [−5.7, −2.8] |
  | SparseVLM | +3.3 [+1.5, +5.2] | +0.1 [−1.8, +2.1] | +0.3 | −0.6 [−2.0, +0.8] |
  | BTP | +2.5 [+0.9, +4.0] | +0.2 [−1.6, +2.0] | +0.3 | +0.1 [−1.3, +1.5] |

  \* not image-dependent for Qwen3-VL (text-only ≈ Full); no pruning reading. BRR, Lingshu SLAKE / RIN: FastV 1.67 /
  2.32, SparseVLM 1.62 / 2.49, BTP 1.10 / 1.67, VisPruner 0.96 / 1.08, PruneSID 1.11 / 1.69.
- **Gate+Uniform head-to-head (same questions).** Lingshu: vs VisPruner −0.7 / **+2.5** / +3.6 [+0.0, +7.1] /
  **+1.5**; vs PruneSID +1.1 / **+4.3** / **+5.7** / **−1.7**. Qwen3-VL: vs PruneSID **−4.0** [−6.2, −1.7] / +0.0 /
  −0.2 / −0.3; vs VisPruner −0.6 / **+6.2** / +0.3 / **+2.5**. (SLAKE / RIN anatomy / RIN abnormality / OmniMedVQA.)
- **Reading.** (1) The collapse belongs to **scoring by early-layer LLM attention**: FastV and SparseVLM collapse on
  Lingshu (−42 / −45 on RIN anatomy), BTP — half attention, half DivPrune — loses 3–10; methods that never read LLM
  attention (VisPruner's vision-tower attention, PruneSID, DivPrune, Uniform) do not collapse anywhere. (2) **Which
  attention readout fails is model-specific.** On Lingshu both readouts fail and pick the same tokens (SparseVLM's
  first keep-set contains nearly all of FastV's: Jaccard 0.66–0.68, maximum ≈ 0.70 given the set sizes). On Qwen3-VL
  FastV's last-token readout fails (−33.7) while SparseVLM's text raters tie Random (+0.1; Jaccard 0.33). [FLAG]
  Confound: SparseVLM also differs in schedule (masks from layer 3, 22.5% of tokens in its first stage); E12's
  `fastv_text` (FastV's schedule, text-token readout; session 2) separates query from schedule. (3) **The gate is
  not a state-of-the-art pruner** and the paper must not claim it. It is the most consistent method on Lingshu (the
  only one ≥ 0 on all four datasets; beats every published method on RIN anatomy), but on Qwen3-VL it adds nothing
  over coverage (PruneSID's rule-scored SLAKE lead of 4.0 is −1.2 [−3.4, +0.9] under the judge, below). Position:
  diagnosis paper; Gate+Uniform is the minimal remedy the diagnosis predicts, competitive with published pruners and
  better than every LLM-attention one. (4) VisPruner
  loses 4.9 on Qwen3-VL RIN anatomy (BRR 1.47): the Qwen3-VL vision tower's attention is also edge-biased at 64
  tokens, while Lingshu's is not (BRR 1.08).
- Predictions: P15 ✗ (Lingshu ✓, Qwen3-VL ✗), P16 ✗ (Lingshu ✗, Qwen3-VL ✓), P17 ✗ (5 of 8 cells), P18 ✓.
- Ledger: A 19.15 spent (slack 1.0 after session 2), B 17.34 (slack 7.1). 129 CPU tests pass.
- **Judge pass 5 merged** (C v5, 0.06 h; 134 verdicts; 5,303 v3 keys; 0 of 56,019 open rows without a verdict).
  SLAKE − Random under the guarded judge (the paper's scorer), Lingshu / Qwen3-VL: Uniform +0.8 / +0.7, Gate+Uniform
  +2.3 / +1.1, DivPrune +0.0 / −0.3, PruneSID +1.5 / **+2.4** [+0.8, +4.0], VisPruner **+3.2** / −0.4, FastV
  −13.4 / **−5.0** [−7.7, −2.1], SparseVLM −15.6 / +0.8, BTP −2.7 / +0.2 (Full / Random 80.2 / 70.9 and 66.0 / 63.3).
  Rule scoring inflated the Qwen3-VL coverage methods (PruneSID +4.5, Uniform +3.0, SparseVLM +3.3) and hid part of
  FastV's loss (−2.4). Gate+Uniform − PruneSID on Qwen3-VL SLAKE: −1.2 [−3.4, +0.9]; vs VisPruner +1.5; on Lingshu
  −0.9 vs VisPruner, +0.8 vs PruneSID — on SLAKE the non-attention methods are statistically indistinguishable.
  Ledger C: 16.22 spent.

### 2026-11-05 — Claude — B session 2 (Qwen3-VL) read: E10r, E11r, D2, E5, E4
- B `medprune-gpu-qwen3vl8b` v5 COMPLETE 00:10, 6.30 h: 79 runs (44 restored from session 1, 35 new), all complete;
  merged (269 run files, no conflicts). Judge pass 6 (420 open answers: 331 SLAKE-zh, 89 SLAKE-en) merged; SLAKE
  cells below are rule-scored, the judge-scored ones are in the next bullet list.
- **E10r (G1, G22) — token count is not the explanation.** RadImageNet anatomy at 448² (256 tokens, Full 90.7,
  Random 89.2): FastV − Random **−29.2** [−32.6, −25.8] vs −33.7 at 64 tokens; Gate+FastV −11.5 vs −18.7;
  Uniform +0.7, Gate+Uniform +0.4. Quadrupling the tokens removes 13% of FastV's loss. With HuatuoGPT's tie on the
  same images (E8r), RadImageNet's collapse is a property of the Qwen-family models on these images, not of 64
  tokens and not of the dataset alone. Lingshu's E10r arrives with A v9.
- **D2 `fastv_text` (G2) — the readout is not the cause.** Text-token readout, FastV's schedule: RIN anatomy
  −27.8 [−30.8, −24.7] (FastV −33.7); SLAKE +1.3 [−1.0, +3.4] (rule; FastV −2.4). The FastV paper's wording
  (mean attention of the text tokens) fails like the official last-token code on RadImageNet. Consequence for E13:
  SparseVLM − `fastv_text` = **+27.9** on RIN anatomy, so SparseVLM's robustness on Qwen3-VL does not come from
  reading text tokens; it comes from its rater subset (text tokens above the mean relevance) or its schedule (12 of 64
  tokens in layers 3–6, 5 in 7–15, 3 after; FastV keeps 6 from layer 2). `fastv_k8` (6 tokens from layer 8) is as bad
  as FastV, so "more layers with all tokens" is not it. [FLAG] Untested between rater subset and early budget: one
  ablation (SparseVLM's raters with FastV's single stage, 10% from layer 3; ≈ 0.4 GPU-h on RIN) separates them before
  the paper says why SparseVLM survives on Qwen3-VL.
- **E11r.** Least-attended (`fastv_bottom`) − Random −3.8 [−5.9, −1.7]; − FastV **+29.9** [+26.2, +33.7]: Qwen3-VL's
  most-attended RIN tokens are far worse than its least-attended (Lingshu showed the same on SLAKE, +6.2). Vertical
  flip: Full 82.6 (−7.1 vs upright), FastV − Random −28.3 [−31.7, −24.9] — the collapse survives the flip.
- **E5 budget curve, FastV − Random (Gate+FastV − Random).** RIN anatomy: 5% −32.2 (−22.0), 10% −33.7 (−18.7), 25%
  −15.9 (−7.7). SLAKE (rule): 5% +1.2 (+1.3), 10% −2.4 (−0.8), 25% −0.2 (−0.6). On RIN the loss halves only at 25%.
- **E4 modality panel (OmniMedVQA, 1,200 questions).** FastV − Random +0.2 [−1.8, +2.3], Gate+FastV −1.2, Gate+Uniform
  −0.8 (Full 83.0, Random 79.8, text-only 55.1): no collapse on the panel for Qwen3-VL. Per-modality reading (P9)
  waits for Lingshu's panel.
- Ledger B: 23.64 spent, 6.4 left this week. Restore datasets not refreshed (no further B session planned).
- **Judge pass 6 merged** (C v6, 0.08 h; 420 verdicts; 5,723 v3 keys; 0 of 62,405 open rows without a verdict).
  Qwen3-VL under the guarded judge: SLAKE `fastv_text` − Random −1.2 [−3.5, +1.1] (rule +1.3), − FastV +3.8;
  SparseVLM − `fastv_text` +2.0 [+0.1, +3.9]. Budget curve SLAKE, FastV − Random (Gate+FastV): 5% −0.7 (+2.6), 10%
  −5.0 (−0.2), 25% −2.8 (−0.1) — the gate is at Random at every budget. **SLAKE-zh** (Full / Random / text-only
  51.4 / 50.4 / 23.0): FastV −7.6 [−10.5, −4.8], Gate+FastV −2.9 [−5.0, −0.7] (+4.7 over FastV). [FLAG] The rule
  scorer is unusable on Chinese answers (Full 35.8 rule vs 51.4 judge; FastV −3.4 rule vs −7.6 judge): every
  SLAKE-zh number in the paper must be judge-scored. Ledger C: 16.30 spent.

### 2026-11-05 — Claude — A session 2 (Lingshu) read: E12 profile, E10r, E11r, D2, E5, E4, SLAKE-zh
- A `medprune-gpu-lingshu7b` v9 COMPLETE 03:12, 9.43 h: 87 runs, 86 complete (45 new); stopped at the 9.5 h budget
  before Lingshu SLAKE-zh `gate_fastv` (0.32 h, still to run). Merged 247 files, no conflicts (314 run files).
  Judge pass 7 (C v7, 0.10 h, 895 SLAKE / SLAKE-zh answers) merged: 6,618 v3 keys, 0 of 72,597 open rows without a
  verdict. All numbers below are judge-scored where open answers exist; p from the image-level sign-flip test.
- **E10r (G1) — token count is not the explanation, on both models.** Lingshu RIN anatomy at 448² (256 tokens; Full
  88.5, Random 86.9): FastV − Random **−48.9** [−52.3, −45.4] (−41.7 at 64 tokens); Gate+FastV −13.0; Uniform 0.0;
  Gate+Uniform −1.1. More tokens make Lingshu's FastV worse, not better.
- **E11r (G3, G4).** Vertical flip: Full 76.8 (−12.2 vs upright), FastV − Random −33.6 [−37.2, −29.7] — the
  collapse survives. Least-attended − most-attended **+40.0** [+36.2, +43.7]; least-attended − Random −1.7 (n.s.).
  On RadImageNet both models' early attention ranks the useful tokens last.
- **D2 `fastv_text` (G2) closed.** Lingshu: − FastV +0.5 RIN, +0.2 SLAKE; − Random −41.2 / −13.2. SparseVLM −
  `fastv_text` −3.6 RIN, −2.4 SLAKE on Lingshu (vs +27.9 RIN on Qwen3-VL): SparseVLM's raters and schedule help only
  where the text-token attention is already partly content-driven.
- **E5 budget curve (G7), Lingshu, X − Random.** RIN: FastV −39.8 / −41.7 / −43.9 at 5 / 10 / 25%; Gate+FastV −32.5 /
  −28.5 / −15.3; Gate+Uniform −7.1 / +1.8 / −1.3. SLAKE: FastV −12.4 / −13.4 / −6.2; Gate+FastV −4.0 / +0.4 / +1.9;
  Gate+Uniform **+3.9** [+1.7, +6.1] / +2.3 / +0.9. At 50% (no Random run) X − Full: FastV −5.0 RIN, −1.4 SLAKE;
  Gate+FastV −2.0 / −0.9. Lingshu's RIN loss does not shrink between 5% and 25%. The gate's advantage over Random is
  largest at the smallest budget on SLAKE.
- **SLAKE-zh (judge)**: Full 81.2, Random 75.0, text-only 19.6; FastV − Random −13.9 [−17.1, −11.0].
- **E4 modality panel (P9 ✗, P6 second half).** Lingshu overall: FastV − Random −12.6 [−14.8, −10.3], Gate+FastV
  −5.5 (+7.1 over FastV), Gate+Uniform −0.2. Per modality, FastV − Random (background fraction): microscopy −12.0
  (0.05), dermoscopy −12.7 (0.05), X-ray −6.0 (0.07), fundus 0.0 (0.18), CT −7.3 (0.27), MRI −5.3 (0.28), **ultrasound
  −56.0** (0.33; Gate+FastV recovers +43.3), OCT −1.3 (0.37). Qwen3-VL: −6.0 microscopy, every other modality ≥ −4.7
  and n.s. Modality ρ −0.26 / −0.82; per-image ρ +0.075 / −0.073.
  [FLAG] **C2 ("background fraction predicts the size of the failure") is not supported** and the paper cannot
  claim it. Lingshu fails on 5%-background microscopy and dermoscopy, so background is not necessary; fundus and
  OCT (18–37%) do not fail, so it is not sufficient. What survives: the failure is model- and image-family-
  specific (RadImageNet, ultrasound, SLAKE CT/MRI on Lingshu), the gate repairs it where background exists, and the
  mechanism is a position- and image-driven early ranking that is question-blind (G5) — not "attention picks air".
  Section 4's Fig. 2 (dose-response) is out; replace it with the per-modality bar chart and say so.
- **E12 layer profile (D1) — where attention becomes question-dependent.** Last-token readout (FastV's),
  question-beyond-image share of the token ranking on SLAKE: Lingshu 0.00–0.03 through layer 10, 0.08 at 11, **0.24
  at 14**, 0.29–0.41 at 18–27; Qwen3-VL ≤ 0.04 through layer 16, **0.15 at 17**, 0.38 at 21, 0.22–0.42 after. RIN
  position share: Lingshu 0.59–0.73 through layer 12, 0.43 at 14, 0.26 at 18; background retention of the top 10%
  stays ≥ 2.0 through layer 17 and falls to its minimum 0.63 at layer 18. Qwen3-VL's RIN minimum is 1.66 at layer 21.
  The text-token readout is less question-dependent than the last token at every layer from 14 on (Lingshu SLAKE
  0.06–0.21 vs 0.24–0.41), so a second-stage scorer should read the last token. The switch sits at ≈ 50–65% depth on
  both models (Lingshu 14–18 of 28, Qwen3-VL 17–21 of 36), consistent with K16 beating Random (P11).
  [FLAG] This profile is computed on the test sets. Using it to pick L* for a method is test-set tuning: the method's
  L* must come from a held-out split (SLAKE validation, a held-out RadImageNet slice) or from a rule fixed before
  looking at test accuracy (e.g. the first layer with question share ≥ 0.2 on held-out data).
- Ledger: A 28.58 spent (1.1 free this week); C 16.40. Free this week ≈ 12.4 GPU-h (A 1.1, B 6.4, C 4.9).

### 2026-11-05 — Claude — Missing runs, DivPrune completion and E9 launched; account D added
- `run_eval --list` against `results/all`: 314 of 322 registered runs complete; the 8 missing are Lingshu —
  SLAKE-zh `gate_fastv` (E3), `oracle_gate` on SLAKE and the six E6-P1 arms on RIN abnormality (`fastv_k8`,
  `fastv_k16`, `static_map`, `inverse_gate`, `divprune`, `gate_divprune`); A's session 1 ran out of budget first.
- **Task T6 done** (`59c141c`): `medprune/latency.py`. `measure` runs Lingshu's decoder layer by layer and removes
  visual tokens physically at a switch layer (cos/sin and positions sliced, original M-RoPE positions kept), timing
  full, `gate_input` (background removed before layer 0), `prellm_10`, `fastv_10` and `k16_10` on 200 SLAKE questions
  after 10 warm-up, in shuffled order, wall time between device-wide syncs (the model spans two T4s); the first
  question asserts the loop equals the model's forward. `prompt-tokens` counts text tokens per question (processor
  only); `flops` computes FastV-formula prefill FLOPs from every run's own keep schedule (`keep_stages` for PDrop,
  SparseVLM, BTP). CPU test: the loop matches a tiny Qwen2.5-VL decoder's forward to 1e-5 with and without removal.
  Scoring overhead is not timed (stated in the module).
- `configs/runs.yaml`: DivPrune ± gate on OmniMedVQA anatomy (Lingshu, role C; Qwen3-VL, role B) and Qwen3-VL RIN
  abnormality (role B), so the baseline table's DivPrune column has every cell.
- Launched: **C v4** `medprune-gpu-lingshu7b` (static maps from the new private dataset `medprune-restore-c`; jobs =
  the 8 missing runs, Lingshu DivPrune OmniMedVQA, E9 measure, prompt tokens; ≈ 2.9 h); **B v6**
  `medprune-gpu-qwen3vl8b` (Qwen3-VL DivPrune, filtered by method so B's finished E13 runs are not repeated; prompt
  tokens; ≈ 1.5 h). The Qwen3-VL block moved from C to B (B had 6.4 h free).
- **Account D** (4th Kaggle account, 30 GPU-h this week): writes verified; the CPU bundle build
  (`medprune-00-build-data-bundle`, public data only) is running so D can take the method kill test. Before any
  RadImageNet run on D, the `cvpr` secret must be added to D's Kaggle account (UI only). D's bundle is accepted only
  if its manifest hashes equal C's.
- 04:51 **D's bundle accepted**: `riddhitaneja/medprune-00-build-data-bundle` v1 COMPLETE (CPU, 0.35 h, git `eeeb153`);
  slake_en, slake_zh, omni_anat, omni_panel and natural have the same question counts, manifest sha256 and median
  background fractions as C's bundle (git `1c4fef2`). D can run SLAKE / OmniMedVQA now; RadImageNet waits for the secret.
- 05:50 **B v6 COMPLETE** (1.38 h): 4 runs, all complete, merged (9 new files, no conflicts); prompt-token counts for
  Qwen3-VL written (median text tokens: SLAKE 27, RIN anatomy 54, RIN abnormality 30, OmniMedVQA anatomy 58).
  Qwen3-VL, X − Random: RIN abnormality DivPrune −0.5 [−1.4, +0.4], Gate+DivPrune −0.3 — every method sits at 72 ± 0.5
  because this cell is not image-dependent (text-only 71.4 ≈ Full 71.8). OmniMedVQA anatomy DivPrune −0.6 [−2.0, +0.6],
  Gate+DivPrune +0.7; **Gate+DivPrune − DivPrune +1.4 [+0.4, +2.3], p 0.004**; Gate+DivPrune − PruneSID 0.0. Ledger B:
  25.02 spent, 5.0 free.

### 2026-11-05 — Claude — C v4 (Lingshu) read: the 8 missing runs, DivPrune OmniMedVQA, oracle gate, E9 latency

- 07:11 **C v4 COMPLETE** (2.71 h): 10 runs, all complete; merged 44 new files (8 byte-identical gates / static
  maps, no conflicts). Judge pass 8 (73 new open answers, Qwen2.5-7B, `rubric/yes-no/v3`, 0.10 h) merged; every open
  row now has a verdict. **Every registered run is complete** (324). Ledger C: 19.21 spent, 2.6 free.
- **RIN abnormality, Lingshu** (Full 46.8, Random 46.3, text-only 27.7), X − Random: FastV −14.2; K=8 −17.4; K=16
  −6.6 [−9.3, −3.9]; static map −15.6 (static − FastV −1.4, p 0.09); inverse gate −14.5; Gate+FastV −10.1;
  Gate+Uniform +1.3; DivPrune −3.9 (p 0.007); **Gate+DivPrune −0.2 (n.s.)**; Gate+DivPrune − DivPrune +3.7 [+0.7,
  +6.5] (p 0.017); Gate+DivPrune − PruneSID +3.9 (p 0.024). → A position-only map from SLAKE reproduces FastV on
  RIN abnormality as on RIN anatomy (G22 partly answered, H2 holds: 96% of FastV's accuracy). K=16 does not
  reach Random on this task (P11). Inverse gate ≈ FastV (the one cell meeting P12).
- **OmniMedVQA anatomy, Lingshu** (Full 82.2, Random 68.0): DivPrune −14.7; Gate+DivPrune −8.4; **Gate+DivPrune −
  DivPrune +6.3 [+5.0, +7.6]**; Gate+DivPrune − PruneSID −11.9. **DivPrune column complete**: Gate+DivPrune −
  DivPrune is positive in all 7 image-dependent cells, p < 0.05 in 6 (Lingshu SLAKE +3.5, RIN anatomy +9.6, RIN
  abnormality +3.7, OmniMedVQA +6.3; Qwen3-VL SLAKE +1.9, RIN anatomy +1.5 (p 0.066), OmniMedVQA +1.4; Qwen3-VL RIN
  abnormality +0.2 is the text-answerable cell). But on Lingshu OmniMedVQA no gated attention or diversity method
  reaches Random; Gate+Uniform (69.9) and PruneSID (71.5) do.
- **SLAKE-zh Gate+FastV** (judge): − Random −1.5 [−3.9, +0.8]; − FastV +12.4 [+9.3, +15.7] → the same as SLAKE-en
  (+0.4 / +12.8).
- **Oracle organ gate, SLAKE** (FastV inside the union of SLAKE's annotated structures; judge): − Random −0.6
  [−3.1, +1.8]; − Gate+FastV −0.9 (n.s.). A perfect anatomical gate is no better than the heuristic body mask.
  Split by question (`tables/evidence_rows.csv`): on the 358 questions naming a structure, Oracle − Random +5.3
  [+2.0, +8.9] (p 0.003), concentrated in large organs (+10.1); on the other 703, −3.6 [−6.9, −0.3] (p 0.033). FastV
  itself loses −7.5 on the structure questions and −16.4 on the rest. **P14 final: contradicted. [FLAG] G16 / L4
  ("FastV drops the asked-about anatomy") explain the minor part of FastV's loss**; the paper must not lead with them.
- **E9 latency** (Lingshu, 200 SLAKE questions, physical removal, two T4s): equivalence check 0.0, text-token
  counts match. Prefill median Full 296 ms → FastV 10% 100 ms (2.95×, paired 3.07×), pre-LLM 10% 91 ms (3.27×),
  K=16 202 ms (1.47×), gate only 294 ms (1.01×). The first-token path (vision encoder 183 ms + prefill) gains 1.69×
  for FastV. FLOPs (`tables/flops.csv`): FastV 10% = 0.220 of Full → measured ≈ 65% of the bound. Gate+FastV =
  FastV FLOPs.
- Code: `latency.flops_table` took the Full reference from the row's own `n_vis` (text-only rows have none → 1.0);
  the reference is now the image's token count from any run that saw it (`tests/test_latency.py`). `analysis
  evidence_retention` crashed on BTP's empty `keep_idx` (staged pruners log `keep_stages`); such rows are skipped.
  Tests: 134.
- SLAKE masks for `analysis evidence` re-fetched from C's bundle output (180 `mask.png` + `slake_mask_labels.txt`,
  scratchpad only, not committed).
- The user added the `cvpr` secret (HF token) on account D. D can now build RadImageNet in-session; the first D
  notebook's setup cell confirms the secret resolves before any GPU work.

### 2026-11-07 — Claude — E14–E16 launched: six more pruners, RadImageNet open questions, SparseVLM raters

- The user chose (Nov 7): E14 (VisionZip, HiPrune, CDPruner, FEATHER + TransPrune and VScan) on accounts D and B; E15
  (`rin_open`, one model) on C (2 h 47 min left); E16 (SparseVLM ablation) on A (University). Excluded and why (chat,
  Nov 7): ATP-LLaVA, IF-Prune, LearnPruner, ViToS need training; MedPruner is a 3D-volume method; MedSTP has no code.
- Code (`b69ffd3`, `868a9ba`): six modules in `medprune/baselines/` (§E14 card); hooks for RoPE-free scores at layers
  1/7/15/L-2 with dropped keys excluded, a vision-output merge, TransPrune's transition weights; `vision_attention`
  (several blocks, keys, embeddings; VisPruner's cues unchanged, tested); `rin_open` builder; `fastv_raters`,
  `sparsevlm_single`; `run_eval --keep-going`, comma lists for `--model/--method/--dataset`; the notebook masks
  `rin_open` and can judge open answers in-session (`JUDGE`). 162 CPU tests.
- 03:14 launched: C `medprune-gpu-lingshu7b` v5 (E15, MAX_HOURS 1.9 + judge), A `medprune-gpu-lingshu7b` v10 (E16,
  Qwen3-VL then Lingshu, MAX_HOURS 1.0), D `medprune-gpu-lingshu7b` v1 and `medprune-gpu-qwen3vl8b` v1 (E14 smoke,
  LIMIT 3). Full D (Lingshu 9 h, Qwen3-VL 8 h caps) and B (Qwen3-VL in `medprune-baselines-qwen3vl8b`, Lingshu in
  `medprune-gpu-qwen3vl8b`, the slot with the secret) follow the smoke.
- 03:22 smoke ✅ (see the E14 card's checks); 03:24 launched D `medprune-gpu-lingshu7b` v2 (MAX_HOURS 10.5) and
  `medprune-gpu-qwen3vl8b` v2 (8.0), both `JUDGE = True` for their SLAKE open answers; B `medprune-baselines-qwen3vl8b`
  v3 (Qwen3-VL RadImageNet, 2.0) and `medprune-gpu-qwen3vl8b` v7 (Lingshu RadImageNet, 2.2).
- [FLAG] The two-stage method idea (coverage early, text-guided attention late) is published (STS 2606.03569, VScan,
  LearnPruner, STAR-Pro): the paper stays diagnosis-first; VScan and TransPrune now represent that family in E14.
- 04:20 **A v10 COMPLETE** (≈ 1.12 h): 5 runs × 1,000 / 1,061, merged (37 new files, no conflicts; the log did not
  download — the CLI's cp1252 writer fails on a progress-bar character). E16 table in the E14–E16 card: on Qwen3-VL RIN
  anatomy SparseVLM's rescue is +7.8 readout (raters vs last token), **+17.2 for scoring one layer later**, +8.8 for
  its three-stage schedule; on Lingshu the raters change nothing. P22 supported. Lingshu `sparsevlm_single` (0.1 h)
  did not fit A's 1.0 h cap → B after its E14 sessions.
- 04:54 **B `medprune-baselines-qwen3vl8b` v3 COMPLETE** (1.50 h; `HF token loaded`; 16 / 16 runs, merged: 52 new
  files). Qwen3-VL RIN anatomy, X − Random: VScan +5.7, TransPrune +4.5, FEATHER +3.9, Gate+VisionZip +3.0, VisionZip
  +1.8, HiPrune −12.3, CDPruner −23.7, Gate+CDPruner −9.3; RIN abnormality all within ±1.1. Keep-set statistics
  (BRR, distinct sets) in the E14-E16 card: merging or late pruning survives a biased selection; early hard drops do
  not. CDPruner's sign is now a sensitivity run (`cdpruner_neg`, `263da31`, in E17).
- The user: every Kaggle account's GPU quota resets ≈ 05:30 IST; all accounts and slots may be used after it.
