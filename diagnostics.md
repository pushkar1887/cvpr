# Diagnostics — Pruning the Air

Every diagnostic and hypothesis tested for the CVPR 2027 paper, as of 2026-11-10. §1–§5 cover the diagnosis (to
Nov 6); §6.2 and §9 add the method experiments E40–E46 (Nov 7–10). For each one: the question, why it matters, what we
expected, how we tested it, the numbers, what they mean technically and in plain words, and what to do about it.
Numbers come from `plan.md` (Stage tracker, §A6, §A9, E-cards, Work log) and `results/all`; §9 was recomputed from the
run files on Nov 10. All GPU work registered in `configs/runs.yaml` is complete through E46, and every Kaggle slot is
idle. The current status is the Nov 10 section at the end of [status_2026-11-07.md](status_2026-11-07.md).

---

## 0. How to read this file

**Models**

| Short name | Model | LLM | Position encoding | Visual tokens |
|---|---|---|---|---|
| Lingshu | Lingshu-7B (medical) | Qwen2.5-VL-7B base, 28 layers | M-RoPE (3D: time, row, column) | dynamic: 64 (224²) … 1,024 |
| Qwen3-VL | Qwen3-VL-8B (general) | 36 layers, DeepStack | M-RoPE | dynamic: 64 … |
| HuatuoGPT | HuatuoGPT-Vision-7B (medical) | LLaVA, Qwen2 LLM, 28 layers | 1D RoPE | fixed 576 (24×24) |

**Datasets** (test splits; "RIN" = RadImageNet-VQA, built per session because its licence forbids redistribution)

| Short name | What | Size | Notes |
|---|---|---|---|
| SLAKE (`slake_en`) | CT, MRI, X-ray; open + closed | 1,061 Q, 96 images | organ masks for 358 questions; open answers judge-scored |
| SLAKE-zh | same images, Chinese | 1,033 Q | judge-scored only (rule scorer unusable on Chinese) |
| RIN anatomy (`rin_anat_mc`) | RadImageNet, multiple choice | 1,000 Q | 224² images → 64 tokens on Qwen models |
| RIN abnormality (`rin_abn_closed`) | yes/no | 1,000 Q | 71.9% "yes"; **not image-dependent for Qwen3-VL** |
| OmniMedVQA anatomy (`omni_anat`) | multiple choice | 3,329 Q | |
| OmniMedVQA panel (`omni_panel`) | 8 modalities × 150 | 1,200 Q | used for the background dose-response |
| Photographs (`natural`) | 500 GQA + 500 POPE | 1,000 Q | scope control only (paper is medical-only) |

**Methods** (all keep 10% of visual tokens unless stated; k = round(0.1·N), N counts every visual token)

| Name in code | What it keeps | Role in the diagnosis |
|---|---|---|
| `full` / `text_only` | all tokens / no image | ceiling / floor (language prior) |
| `random_l2` ("Random") | random k, masked from layer 2; mean of seeds 0–2 | **the bar every pruner must clear** |
| `uniform_l2` ("Uniform") | k evenly spaced on the grid | coverage, no scoring |
| `fastv` | top-k by last-token attention at layer 1, masked from layer 2 (K=2) | the method under test |
| `fastv_prellm` | same ranking, removed before the LLM | masking vs removal check |
| `fastv_k8`, `fastv_k16` | top-k scored at layer 7 / 15, masked from 8 / 16 | depth of the failure |
| `fastv_text` | top-k by mean attention of every text token after the image | readout check (FastV paper wording) |
| `fastv_bottom` | the **least**-attended k | is attention informative at all? |
| `ropefree` | top-k by layer-1 attention with rotary removed from q·k | is it RoPE? |
| `static_map` | top-k of a fixed rank map built on the *other* dataset | is it position alone? |
| `inverse_gate` | k evenly spaced **background** tokens | is FastV just a background picker? |
| `artifact_excl` | FastV with high-norm tokens excluded | is it attention sinks? |
| `gate_*` | Foreground Gate (§A5 of plan.md: border-connected background mask, τ = 0.5) then the named selector inside the foreground | the remedy |
| `oracle_gate` | FastV inside the union of SLAKE's hand-annotated structures | upper bound for the gate |
| `divprune`, `pdrop` | max–min diversity; multi-stage attention pruning | other pruner families |
| `sparsevlm`, `vispruner`, `prunesid`, `btp` | published pruners (official selection code, E13) | baselines |

**Metrics**

- **Δ (points)** = accuracy of A − accuracy of B on identical questions. "X − Random" unless stated.
- **95% CI**: image-clustered bootstrap (questions about one image are not independent). **p**: image-level sign-flip
  permutation test (10,000 flips). The earlier McNemar test was replaced on Nov 5 (see D-03).
- **BRR** (background retention ratio) = P(background | kept) / P(background); Random ≈ 1.
- **ERR** (evidence retention ratio) = share of the named organ's tokens kept / share expected by chance.
- **Retained visual gain** = (acc − text-only) / (Full − text-only): how much of the image's contribution survives.
- **Jaccard** of keep-sets (|A∩B| / |A∪B|).
- **Rank-variance shares**: the variance of layer-L token ranks split exactly into position / image beyond position /
  question beyond image.

**Status labels**: SUPPORTED · CONTRADICTED · MIXED · DEAD (rejected as a cause by intervention) · OPEN.
**Confirmatory** = P1–P14, frozen 2026-11-02 before any result (P15–P18 added Nov 4 before E13 results). Everything
else is **exploratory** and dated; the paper must label it so (G13, G21).

---

## 1. One-page verdict

| # | Question | Verdict | Decisive evidence | Plain meaning |
|---|---|---|---|---|
| D-02 | Is FastV worse than Random on medical VQA? | SUPPORTED (Qwen family) | Lingshu RIN −41.7, OmniMedVQA −29.4, SLAKE −13.4; Qwen3-VL RIN −33.7 | The "smart" pruner is far worse than random |
| D-01 | Is it our implementation? | NO | official keep-sets reproduced (Jaccard 1.000); masking ≡ removal | Not a bug |
| D-04/05 | H1: does it pick background? | MIXED | BRR 2.32 / 1.67; gate closes SLAKE fully, a third of RIN | Partly — on CT/MRI |
| D-06 | Is FastV *just* a background picker? | NO | background-only tokens beat FastV (−23.9 vs −41.7) | It is worse than air |
| D-04 | Does background fraction predict the loss (C2)? | CONTRADICTED | P9 ρ −0.26 / −0.82; microscopy (5% bg) −12 | Drop claim C2 |
| D-07/08 | H2: is it rotary decay? | DEAD for Qwen family | RoPE-free = FastV (Jaccard 0.94); keep-set in the top rows, not bottom-right | Not RoPE |
| D-09/10 | Is it a positional prior? | SUPPORTED (Lingshu), MIXED (Qwen3-VL) | static map ≈ or worse than FastV; flipped images keep the same positions (0.64) | A fixed "where to look" map |
| D-12 | H3: attention sinks / artifacts? | DEAD | `artifact_excl` = FastV (−42.0 vs −41.7) | Present, irrelevant |
| D-13/14 | Is the ranking question-aware? | NO at layer 1; YES from ~50% depth | 97.8% same keep-set across questions; question share 0.24 at layer 14 | Early layers have not read the question |
| D-16 | Is early attention informative? | ANTI-informative (Qwen family) | least-attended − most-attended +40.0 (Lingshu RIN) | It ranks useful tokens last |
| D-17 | Does a deeper K fix it? | YES, abruptly | K=2 −41.7, K=8 −43.9, K=16 +8.5 | Score late, not early |
| D-19 | Is it token count / resolution? | NO | RIN at 256 tokens −48.9 (worse) | Not a resolution artefact |
| D-18 | Is 10% just too extreme? | NO | −39.8 / −41.7 / −43.9 at 5 / 10 / 25% | Holds across budgets |
| D-20 | H4: does it drop the asked-about organ? | REAL but MINOR | large organs kept at 0.20× chance; but FastV loses −16.4 where no organ is named vs −7.5 where one is | Illustration, not mechanism |
| D-21 | What does the pruned model do? | Falls back on language prior | retained visual gain 26% (Random 86%) | Nearly blind |
| D-22 | All architectures? | NO | HuatuoGPT FastV ≈ Random (−0.1 RIN) | Qwen-family property |
| D-24 | All pruners? | Only early-LLM-attention scorers | FastV, SparseVLM collapse; VisPruner, PruneSID, DivPrune do not | Coverage methods are safe |
| D-27 | Is the gate a general plug-in? | SUPPORTED | Gate+DivPrune > DivPrune in 7 / 7 image-dependent cells | Gate composes |
| D-28 | Does pruning save time? | YES, below the FLOPs bound | prefill 2.95×, first token 1.69×; gate alone 1.01× | Real but modest end-to-end |

---

## 2. The working mechanism in one paragraph

FastV's score for visual token i is s_i = mean_h softmax_j(q_last,h · k_j,h / √d)_i at layer 1, where q_last is the
query of the **last prompt token**. In the chat template that token is a fixed template token, and after one layer it
has mixed in almost none of the question, so q_last is nearly constant across questions. The ranking therefore
depends almost only on the visual keys k_j, which carry the image and the token's grid position (the vision encoder
injects position into every patch feature). The result is a **question-blind, largely image-blind prior** that
concentrates the budget in one region (the top rows for Lingshu). Where that region is acquisition background
(CT/MRI), the gate repairs it; where it is not (RadImageNet, microscopy), the model still loses most of its view of the
image and answers from the language prior. The model starts reading the question only at ≈ 50–65% depth, which is
exactly where attention-based selection starts beating Random. HuatuoGPT (1D RoPE, LLaVA) does not have this prior
in its keys and does not fail. *(Interpretation of D-08, D-13, D-14, D-16, D-17; the "nearly constant q_last" step is
an inference from the 97.8% keep-set overlap and has not been measured directly.)*

---

## 3. Hypotheses and pre-registered predictions

### 3.1 Hypotheses (frozen Nov 2)

| ID | Hypothesis | Pre-registered prediction | Rejected if | Outcome |
|---|---|---|---|---|
| H1 | Background capture: background tokens over-selected beyond position | BRR ≥ 1.3 on CT/MRI; positive background effect after controlling for position | BRR CI includes 1 | **MIXED.** BRR 2.32 (RIN), 1.67 (SLAKE): over-selection is real. The gate repairs SLAKE fully but RIN only by a third; FastV fails on background-free images too (D-04, D-05) |
| H2 | Positional bias from rotary decay | bottom-right keep-mass (M-RoPE), bottom rows (1D); static map ≥ 80% of FastV accuracy; gap shrinks with K | χ² n.s. or static map ≪ FastV | **Split.** Rotary half DEAD for Qwen family (top rows, RoPE-free identical); positional half SUPPORTED for Lingshu (static map ≈ FastV, flip). HuatuoGPT: rotary SUPPORTED (bottom rows 3.1×) |
| H3 | High-norm artifact tokens attract scoring attention | selected ≥ 2× base rate, in background | no outliers / base rate | **DEAD.** Over-selected 2.6–5.1×, but excluding them changes nothing (D-12) |
| H4 | Residual loss sits on small findings | largest residual in the smallest evidence tertile | flat across strata | **REVERSED.** FastV's loss sits on **large** organs (D-20) |
| H5 | DeepStack makes Qwen3-VL's layer 1 more content-aware | smaller gap than Lingshu at matched background | equal gaps | **Consistent, not tested causally.** Qwen3-VL's gaps are smaller everywhere (P10), but no DeepStack ablation was run |

### 3.2 Predictions scorecard

| # | Prediction (confidence) | Outcome | Where |
|---|---|---|---|
| P1 | FastV < Random by ≥ 8 on RIN anatomy; pre-LLM FastV < Random by ≥ 10 (65% / 80%) | **SUPPORTED**: −41.2 [−45.0, −37.6]; pre-LLM −41.8 | D-02 |
| P2 | SLAKE Full 81–85; FastV − Random ≤ −4 on CT/MRI, ≥ 0 on X-ray (55%) | **MIXED**: Full 77.2 (rule); CT/MRI −18.1 ✓; X-ray −1.9 (n.s.) | D-04 |
| P3 | BRR ≥ 1.3 on CT/MRI (65%) | **SUPPORTED** (association) | D-04 |
| P4 | keep-mass ≥ 1.5× uniform bottom-right / bottom rows (60%) | **CONTRADICTED** on Lingshu (top-left 2.52×); **SUPPORTED** on HuatuoGPT (3.1×) | D-07 |
| P5 | high-norm artifacts over-selected ≥ 2× (35%) | **SUPPORTED but not causal** | D-12 |
| P6 | Gate+FastV − FastV ≥ +8 on CT/MRI; \|Δ\| ≤ 1 on dermoscopy/microscopy (55%) | **MIXED**: CT/MRI +19.3 (Lingshu) ✓; second half met within 1.3 in 3 of 4 cells | D-05 |
| P7 | Gate+Uniform within 3 of Gate+FastV on CT/MRI (50%) | SLAKE ✓ (+2.4); RIN ✗ (Gate+Uniform +30.3 better) | D-05 |
| P8 | gate adds ≥ 3 on top of RoPE-free on CT/MRI (50%) | **SUPPORTED** on Lingshu (+16.7 / +15.0); Qwen3-VL RIN +15.5, SLAKE +0.4 | D-05 |
| P9 | modality ρ ≥ 0.7; per-image ρ > 0 (55%) | **CONTRADICTED**: −0.26 / −0.82 | D-04 |
| P10 | Qwen3-VL same sign as Lingshu, smaller (60%) | **SUPPORTED** in all four cells | D-22 |
| P11 | gap shrinks monotonically with K; K=16 ≈ Random (65%) | **CONTRADICTED in form**: not monotone; K=16 *beats* Random (except RIN abnormality) | D-17 |
| P12 | inverse gate within 5 of FastV on CT/MRI (50%) | **CONTRADICTED**: background-only beats FastV (one cell meets it) | D-06 |
| P13 | photographs: FastV ≥ Random − 2 (50%) | **CONTRADICTED**: −8.7 / −4.9 | D-23 |
| P14 | residual largest in the smallest evidence tertile (55%) | **CONTRADICTED** (final, judge-scored, with oracle) | D-20 |
| P15 | SparseVLM ≤ −10 on Lingshu RIN; ≤ 0 on Qwen3-VL RIN (60%) | **HALF**: −44.8 ✓; +0.1 ✗ | D-24 |
| P16 | VisPruner within 3 of Random on SLAKE; BRR > 1 (50%) | ✗ Lingshu (+3.2, BRR 0.96); ✓ Qwen3-VL (+1.1, 1.34) | D-24 |
| P17 | PruneSID within 3 of Random everywhere (60%) | **CONTRADICTED** in 3 of 8 cells (range −4.1 to +4.5, no collapse) | D-24 |
| P18 | BTP ≥ FastV + 10 on Lingshu RIN; within 3 of Random on SLAKE (55%) | **SUPPORTED** (+31.9; −2.7) | D-24 |

**Decision rules.** K1 (Stage 4): GO — FastV < Random on RIN anatomy with the CI excluding 0. K2: BRR CI excludes 1 →
H1 alive. K3: on RIN the gate recovers 32–39% of FastV's deficit (≈ ⅓) → the paper is **diagnosis-first**. K4 (RoPE-free
alone recovers ≥ 90% of the gate's gain): not triggered — RoPE-free recovers nothing on the Qwen family.

---

## 4. The diagnostics

Each section: **Question → Why → Expected → How → Results → Technical reading → In simple words → What to do.**

### D-01 Validation and implementation fidelity (E0, G2, determinism, E9)

**Question.** Is our FastV the real FastV, and is the pipeline deterministic?
**Why.** "Your FastV is not FastV" is the first rejection a reviewer writes.
**Expected.** Exact agreement with the official code; fp16 inert.
**How.** (1) E0: 50 SLAKE questions per model — fp16 vs reference, logging hooks inert, offline keep-sets equal online
ones. (2) Our layer-1 scores vs the official FastV keep-set. (3) Masking dropped tokens vs physically removing them.
(4) Same run on two Kaggle accounts with different CUDA builds. (5) E9: our layer-by-layer decoder loop vs the model's
own forward.

| Check | Result |
|---|---|
| E0b fp16 / E0c logging inert / E0d offline keep-sets | 50/50 · 50/50 · 50/50, both models |
| Keep-set vs official FastV code | Jaccard 1.000 (Lingshu) / 0.999 (Qwen3-VL) |
| Masking vs physical removal (RIN, Lingshu) | `fastv` 38.1 vs `fastv_prellm` 37.4 |
| Cross-account rerun (`artifact_excl`, cu130 vs cu128) | identical keep-sets and responses on every question |
| E9 layer loop vs model forward | max relative difference 0.0 |

**Technical reading.** Masking keys from layer K is equivalent to removal for accuracy; implementation error is ruled
out. One scoring bug was found and fixed early: Lingshu ends answers with a period, so strict exact match gave Full
38%; normalized match gives 82% (paper ≈ 83%).
**In simple words.** The failure is not something we introduced.
**What to do.** Nothing; cite the checks in the supplementary.

### D-02 The phenomenon (E1, E3, E4, SLAKE-zh, E8)

**Question.** Does FastV lose to Random on medical VQA, robustly?
**Why.** The paper's existence (K1). MedSTP (Oct 2026) shows FastV < Random in 5 of 8 cells without explaining it.
**Expected.** P1: ≥ 8 points below Random on RIN anatomy (Lingshu).
**How.** Paired FastV vs Random on identical questions, 10% kept, every model × dataset cell.

| X − Random, 10% kept | Lingshu | Qwen3-VL | HuatuoGPT |
|---|---|---|---|
| Full / Random / text-only, RIN anatomy | 89.0 / 79.8 / 20.1 | 89.7 / 84.6 / 30.1 | 88.6 / 84.5 / 26.9 |
| **RIN anatomy** | **−41.7 [−45.1, −38.3]** | **−33.7 [−36.8, −30.5]** | −0.1 [−1.9, +1.7] |
| RIN abnormality | −14.2 [−16.7, −11.8] | −2.2 (not image-dependent) | — |
| OmniMedVQA anatomy | −29.4 [−31.2, −27.7] | −4.3 [−5.7, −2.8] | — |
| SLAKE (guarded judge) | −13.4 | −5.0 [−7.7, −2.1] | −1.6 |
| SLAKE-zh (judge) | −13.9 [−17.1, −11.0] | −7.6 [−10.5, −4.8] | — |
| OmniMedVQA 8-modality panel | −12.6 [−14.8, −10.3] | +0.2 [−1.8, +2.3] | — |
| Photographs (scope control) | −8.7 [−11.7, −5.8] | −4.9 [−7.8, −2.1] | — |

Random itself loses only 9.2 points to Full on Lingshu RIN anatomy (79.8 vs 89.0), 5.1 on Qwen3-VL. Random seeds 0–2
agree within 1.5 points.
**Technical reading.** The failure is large, significant in every image-dependent Qwen-family cell, and absent in
HuatuoGPT. Because Random keeps most of Full's accuracy, the budget is sufficient: FastV chooses the wrong 10%.
**In simple words.** Throwing away 90% of the image at random costs a little; throwing it away by FastV's rule costs a
lot.
**What to do.** C1 holds for **Qwen-family models** (claim scope); report HuatuoGPT as the counterexample. Do not claim
"medical-specific" (photographs fail too); claim "FastV fails on medical VQA, and here is why".

### D-03 Measurement fairness: open answers, statistics, image dependence (G9, G10)

**Question.** Are the numbers measuring what we think?
**Why.** Three ways the protocol could manufacture or hide a result.

| Issue | Test | Result | Consequence |
|---|---|---|---|
| Exact match under-credits synonyms ("axial" vs "Transverse Plane") | Qwen2.5-7B-Instruct judge (one-token Yes/No, rubric v3) + deterministic guard; κ vs 200 blind labels | κ: exact match 0.806, rule-OR-judge 0.813 (pre-registered gate passed by 0.007), **guarded scorer 0.898 [0.847, 0.939]** | Qwen3-VL SLAKE loss appears: −2.4 (n.s., rule) → −5.0 (judge). SLAKE-zh must be judge-scored (rule Full 35.8 vs judge 51.4) |
| McNemar on Random's thresholded seed mean = majority vote of 3 draws (83.5% vs Random's 79.8%), and treats questions on one image as independent | replaced by image-level sign-flip permutation test | CIs unchanged; old p-values too small | **[FLAG]** recompute every pre-Nov-5 p-value that goes into the paper |
| Some cells answerable without the image | text-only vs Full | Qwen3-VL RIN abnormality: text-only 71.4 ≈ Full 71.8 | Footnote that cell; no pruning reading |
| Yes/no imbalance | majority baseline | RIN abnormality 71.9% "yes" | **OPEN (G10)**: report majority baseline + balanced accuracy |

**[FLAG]** The κ labels are AI labels (Codex), and the guard was written after reading them: κ 0.90 is in-sample and
AI-to-AI. A held-out sheet of 100 human labels is needed before the paper reports it.
**In simple words.** We fixed the ruler before measuring with it; one fix made Qwen3-VL's failure visible.

### D-04 H1, part 1: does FastV select background? Dose-response (E2, E4, P3, P9)

**Question.** Does FastV prefer background tokens, and does the amount of background predict how badly it fails?
**Why.** This was the original thesis ("pruning the air"): scans put air, table and empty field of view at the edges.
**Expected.** BRR ≥ 1.3 (P3); a monotone dose-response across modalities, ρ ≥ 0.7 (P9).
**How.** Body mask per image (border-connected dark/bright regions, §A5) → per-token foreground fraction → BRR of each
keep-set. Dose-response: FastV − Random binned by the image's background fraction (SLAKE, RIN), and per modality on the
8-modality OmniMedVQA panel (150 questions each).

| Over-selection (FastV, Lingshu) | BRR [95% CI] | Coverage |
|---|---|---|
| RIN | **2.32 [2.25, 2.39]** | 874 / 1,000 questions with background |
| SLAKE CT/MRI | **1.67 [1.57, 1.76]** | 700 / 700 |

| Dose-response, FastV − Random by background fraction | bins (low → high) | ρ |
|---|---|---|
| SLAKE, Lingshu | −1.9 (X-ray, none) · −17.0 · −20.3 · −16.3 | −0.35 (p 0.001) |
| RIN, Lingshu | **−45.0 (no background, n = 111)** · −51.0 · −42.5 · −28.6 · −50.0 | flat |

| OmniMedVQA panel, FastV − Random (Lingshu) | background fraction | Δ |
|---|---|---|
| microscopy | 0.05 | −12.0 |
| dermoscopy | 0.05 | −12.7 |
| X-ray | 0.07 | −6.0 |
| fundus | 0.18 | 0.0 |
| CT | 0.27 | −7.3 |
| MRI | 0.28 | −5.3 |
| **ultrasound** | 0.33 | **−56.0** (gate recovers +43.3) |
| OCT | 0.37 | −1.3 |
| Modality-level ρ (Lingshu / Qwen3-VL) | | −0.26 / −0.82 (wrong sign) |
| Per-image ρ | | +0.075 [+0.022, +0.130] / −0.073 |

**Technical reading.** Over-selection of background is real (P3). But background is neither necessary (microscopy and
dermoscopy fail at 5%; RIN images with zero background fail by −45) nor sufficient (fundus and OCT at 18–37% do not
fail). The SLAKE dose-response is a CT/MRI-vs-X-ray contrast, not a gradient.
**In simple words.** FastV likes the empty parts of scans, but empty space is not what makes it fail.
**What to do.** **[FLAG] Drop C2** ("background fraction predicts the failure"); replace the dose-response figure with
the per-modality bar chart and say so. Keep BRR as a descriptive result. Ultrasound (−56) is a striking case for the
paper: large, gate-recoverable.

### D-05 H1, part 2: the Foreground Gate as a causal test (E1g, E3, P6–P8)

**Question.** If we forbid background before scoring, does FastV recover?
**Why.** BRR is correlational; the gate is the intervention. It also defines the remedy (C4).
**Expected.** Gate+FastV − FastV ≥ +8 on CT/MRI and ≈ 0 where there is no background (P6); Gate+Uniform ≈ Gate+FastV
(P7); gate adds ≥ 3 on top of RoPE-free (P8).
**How.** The gate restricts candidates to foreground tokens (m_i ≥ τ = 0.5), fills from the best background tokens if
short; the budget k still counts all tokens (fair). Decomposition: FastV − Random = (FastV − Gate+FastV) "air" +
(Gate+FastV − Random) "ranking".

| Loss decomposition | FastV − Random | "air" (gate fixes) | "ranking" (remains) |
|---|---|---|---|
| Lingshu, RIN | −41.2 | −13.2 | **−28.0 [−31.7, −24.4]** |
| Qwen3-VL, RIN | −33.7 | −15.0 | −18.7 [−22.0, −15.4] |
| Lingshu, SLAKE | −12.6 | −12.7 | +0.1 [−2.1, +2.2] |
| Qwen3-VL, SLAKE | −2.4 | −1.6 | −0.8 |

| Gate cells | Lingshu | Qwen3-VL |
|---|---|---|
| Gate+FastV − FastV, SLAKE CT/MRI (P6) | **+19.3 [+15.4, +23.3]** | +2.4 (n.s.) |
| Gate+FastV − FastV, RIN | +13.2 | +15.0 |
| Gate+FastV − FastV, microscopy / dermoscopy (n = 150) | +1.3 / +4.7 | +1.3 / −0.7 |
| Gate+Uniform − Gate+FastV, RIN (P7) | **+30.3** | +20.0 |
| gate on top of RoPE-free, SLAKE CT/MRI / RIN (P8) | +16.7 / +15.0 | +0.4 / +15.5 |
| Gate+Random − Random (E1g), RIN / SLAKE | **+3.2 [+1.2, +5.2]** / +1.7 | +0.5 / −0.6 |
| Gate+FastV − Random, SLAKE at 224² (64 tokens) | −5.3 [−8.3, −2.2] | −0.6 |

**Technical reading.** Two regimes. On SLAKE CT/MRI (300+ tokens, large air regions) background explains *all* of
FastV's loss and the gate closes it. On RIN (64 tokens) the gate removes only a third; the rest is attention ranking
the wrong *foreground* tokens. At 64 tokens even SLAKE's gap reopens under the gate. Inside the gate, how tokens are
drawn barely matters (Gate+Random − Gate+Uniform within ±1.1).
**In simple words.** Removing the empty space fixes CT/MRI completely, but on small RadImageNet images FastV still
picks the wrong parts of the body.
**What to do.** The title "Pruning the Air" describes the SLAKE regime only; the abstract must name both regimes. The
gate is a partial remedy, which is why the paper is diagnosis-first (K3).

### D-06 Is FastV just a background selector? The inverse gate (E6, P12)

**Question.** If we keep *only* background tokens, evenly spaced, do we do as badly as FastV?
**Why.** If yes, FastV's failure is "it picks air". If no, FastV is worse than air.
**Expected.** P12: inverse gate within 5 of FastV on CT/MRI.

| X − Random | Lingshu RIN anat | Lingshu SLAKE CT/MRI | Qwen3-VL RIN | Qwen3-VL SLAKE CT/MRI | Lingshu RIN abn |
|---|---|---|---|---|---|
| FastV | −41.7 | −18.0 | −33.7 | −4.7 | −14.2 |
| **inverse gate (background only)** | **−23.9** | **−7.5** | **−7.6** | **+2.0** | −14.5 |

**Technical reading.** Spread-out background tokens beat FastV by 10–26 points. Background tokens are not empty for the
LLM (consistent with register-like global tokens, Darcet et al. ICLR 2024), and **concentration** is worse than any
spread-out set. RIN abnormality is the one cell where background-only ≈ FastV.
**In simple words.** Even a handful of "empty" tokens from around the image beats FastV's choice. FastV's real problem
is that it bunches all its tokens in one place.
**What to do.** Write the mechanism as "a concentrated, fixed keep-set destroys coverage", not "attention picks air".
Coverage (Random, Uniform, DivPrune) is the bar.

### D-07 H2, part 1: where does FastV keep? Keep-maps (E2, P4)

**Question.** Does the keep-set sit where rotary decay predicts?
**Why.** Under M-RoPE, token (r, c) gets ids (β, β+r, β+c); the last prompt token's offsets shrink with r and c, so
long-term decay (RoFormer §3.4.3) favours the **bottom-right**; under 1D RoPE the **bottom rows** (FEATHER, ICCV 2025).
**Expected.** P4: keep-mass ≥ 1.5× uniform in the bottom-right (Qwen family) / bottom rows (HuatuoGPT).

| Keep-mass concentration vs uniform | Lingshu RIN | Lingshu SLAKE CT/MRI | HuatuoGPT SLAKE | HuatuoGPT RIN |
|---|---|---|---|---|
| bottom-right | **0.018×** | 0.427× | — | — |
| strongest region | **top-left 2.52×** | — | bottom quarter of rows: 77% of mass (**3.1×**) | bottom quarter 57% |
| mean row of kept tokens (0 = top) | 0.18 (Random 0.50) | — | — | — |

**Technical reading.** The rotary prediction is wrong for the Qwen family: FastV keeps the *top* rows, the opposite
corner. It is right for HuatuoGPT, which reproduces FEATHER.
**In simple words.** The textbook explanation (position encoding pulls attention to the end of the image) fits the
LLaVA model, not the two Qwen models.
**What to do.** Do not write "rotary decay" for the Qwen family. See D-08 for what replaces it.

### D-08 H2, part 2: RoPE-free scoring and the rank-variance decomposition (G18)

**Question.** If we remove rotary from the score, does the bias go away? How much of the ranking is position vs image
vs question?
**Why.** Separates "RoPE causes the bias" from "the bias is in the tokens themselves".
**How.** `ropefree`: layer-1 attention computed with q and k before rotary. Variance of layer-1 ranks split exactly
into position (grid cell), image beyond position, question beyond image (`rank_variance_shares`, exact grid).

| | Lingshu RIN | Qwen3-VL RIN | Lingshu SLAKE | Qwen3-VL SLAKE | HuatuoGPT SLAKE |
|---|---|---|---|---|---|
| RoPE-free − Random (accuracy) | −42.3 (FastV −41.7) | −31.9 (−33.7) | −12.5 (−12.4) | −1.4 | **+3.1 [+1.1, +5.0]** |
| keep-set overlap RoPE-free vs FastV | 0.94 | 0.77 | 0.91 | 0.68 | — |
| position share, RoPE / RoPE-free | **69% / 69%** | 51% / 47% | 40% / — | 27% / — | **52% / 23%** |

| Rank variance at layer 1 (position / image / question), exact grid | photographs | SLAKE | SLAKE @224² | RIN |
|---|---|---|---|---|
| Lingshu | 31 / 69 / 0.1% | 40 / 59 / 0.3% | 64 / 36 / 0.3% | 69 / 31 / — |
| Qwen3-VL | 19 / 80 / 1.2% | 27 / 70 / 3.1% | 40 / 59 / 1.5% | 51 / 49 / — |

**Technical reading.** In the M-RoPE Qwen models the positional prior survives rotary removal, so it is carried by the
visual tokens' **keys** (position information baked into the patch features by the vision encoder), not by RoPE's
decay. In HuatuoGPT, removing RoPE halves the positional share and lifts FastV above Random: there the prior *is*
rotary. The position share grows as tokens shrink (SLAKE 40% → 64% at 224²).
**In simple words.** For the Qwen models, the bias is learned into the image tokens themselves; switching off the
position formula does not help.
**What to do.** Contrast with FEATHER explicitly (G18): "in 1D-RoPE LLaVA the prior is rotary; in M-RoPE Qwen models
it lives in the visual keys". RoPE-based fixes will not transfer to the Qwen family.

### D-09 H2, part 3: the static map (E6, G22)

**Question.** Does a ranking that never looks at the image do as badly as FastV?
**Why.** The cleanest position-only intervention. H2 says ≥ 80% of FastV's accuracy.
**How.** For each model, average the layer-1 rank map of `full` runs on the *other* dataset (SLAKE → RIN, RIN →
SLAKE), keep its top-k cells on every image (`analysis.build_static_maps`).

| X − Random | Lingshu RIN anat | Lingshu RIN abn | Lingshu SLAKE | Lingshu SLAKE CT/MRI | Qwen3-VL RIN | Qwen3-VL SLAKE |
|---|---|---|---|---|---|---|
| FastV | −41.7 | −14.2 | −12.4 | −18.0 | −33.7 | −2.4 |
| **static map** | **−50.9** | **−15.6** | **−13.0** | −15.5 | −27.8 | +2.0 |
| static − FastV | −9.2 | −1.4 (p 0.09) | −0.6 | +2.5 | +5.9 | +4.4 |

**Technical reading.** For Lingshu, a fixed map transferred from another dataset reproduces or exceeds FastV's loss in
every cell. Against H2's pre-registered bar (static map ≥ 80% of FastV's accuracy) it reaches 99% on SLAKE and 96% on
RIN abnormality, but only **76% on RIN anatomy (28.9 vs 38.1)** — there it is *worse* than FastV. That fails the
letter of the criterion while strengthening the positional reading (position alone does even more damage); report it
that way, not as a pass. For Qwen3-VL the static map is better than FastV, so its prior is partly image-driven.
**In simple words.** For Lingshu you could replace FastV with a fixed stencil and get the same bad result.
**What to do.** Lead Section 5 with this for Lingshu; present Qwen3-VL as "positional + image-driven". G22 is partly
answered: RIN's collapse is what a fixed prior does to these images. Open: why the same prior hurts RIN far more than
SLAKE.

### D-10 H2, part 4: flip the image (E11, E11r, G3)

**Question.** When the image is turned upside down, does the keep-set stay at the same grid positions (positional) or
move with the content?
**Why.** A direct position-vs-content intervention on identical content.
**How.** `data vflip` variants (same questions, same IDs). Jaccard of the flipped keep-set with (a) the upright
keep-set at the same positions, (b) the upright keep-set mirrored (follows content).

| | Lingshu | Qwen3-VL | random reference |
|---|---|---|---|
| Jaccard, same grid positions | **0.637** | 0.359 | 0.052 |
| Jaccard, mirrored (follows content) | 0.130 | 0.200 | 0.052 |
| FastV − Random, flipped SLAKE (upright) | −12.3 (−12.6) | −0.7 (−2.4) | |
| FastV − Random, flipped RIN (E11r) | −33.6 [−37.2, −29.7] | −28.3 [−31.7, −24.9] | |

**Technical reading.** Lingshu's keep-set stays put when the content moves: positional. Qwen3-VL is mixed. The RIN
collapse survives the flip (Full also drops 12.2 / 7.1 points on flipped images).
**In simple words.** Turn the scan upside down and Lingshu still looks at the same spots on the screen.
**What to do.** Use as the second positional intervention next to the static map (Table 2).

### D-11 Coverage, image independence and mediation (E1g, G11)

**Question.** Is "same keep-set on every image" the problem, or is it poor spatial coverage?
**Why.** A reviewer will say "coverage explains it; nothing new" (G11).
**How.** Cross-image Jaccard of keep-sets; mean distance from each grid cell to its nearest kept token; logistic
mediation of FastV's per-question penalty by coverage.

| RIN | Lingshu | Qwen3-VL |
|---|---|---|
| cross-image Jaccard: FastV / RoPE-free / layer-15 score | 0.773 / 0.770 / 0.384 | 0.460 / 0.489 / 0.489 |
| `uniform_l2` cross-image Jaccard (and accuracy vs Random) | 1.000 (78.9 vs 79.3) | — |
| mean distance to nearest kept token: Random / Uniform / Gate+FastV / FastV | 1.75 / 1.38 / 2.07 / **2.60** | — / — / 2.00 / 2.25 |
| accuracy vs coverage gap across six selectors | r = −0.91 | r = −0.88 |
| FastV penalty explained by foreground coverage, per question (mediation) | 11% (SLAKE 46%) | 0% |

**Technical reading.** Image independence is harmless by itself: Uniform keeps the identical 6 tokens on every RIN
image and matches Random. What hurts is concentration (FastV's tokens sit at mean row 0.18). Across selectors,
coverage tracks accuracy, but per question it explains 0–46% of the penalty, so the claim is about *which* tokens
attention picks, not only their spread.
**In simple words.** Picking the same spots every time is fine if they are spread out; FastV picks the same spots
*and* bunches them up.
**What to do.** Report both levels; measure coverage on the foreground only for the paper version.

### D-12 H3: attention sinks and high-norm artifacts (E2, E6, G17)

**Question.** Are high-norm "register" tokens (Darcet et al.) or visual attention sinks (Kang et al., ICLR 2025)
pulling FastV's attention?
**Expected.** P5: artifacts selected ≥ 2× their base rate.
**How.** Robust z > 5 on layer-1 norms; selection rate; then the intervention `artifact_excl` (FastV with those tokens
removed from the candidates). Spearman between layer-1 score and input norm. Sink score per layer (D1).

| | Lingshu RIN | Lingshu SLAKE | Qwen3-VL RIN | Qwen3-VL SLAKE |
|---|---|---|---|---|
| high-norm share of FastV's picks vs base rate | 3.3% vs 1.0% | 1.9% vs 0.3% | 0% | 0.5% vs 0.1% |
| ViT-norm outlier over-selection | 2.62× | 5.07× | — | — |
| Spearman(score, norm) | 0.32 | 0.44 | 0.00 | 0.16 |
| **`artifact_excl` − Random (FastV)** | **−42.0 (−41.7)** | **−12.3 (−12.4)** | −33.7 (−33.7) | −2.4 (−2.4) |

**Technical reading.** Artifacts are over-selected but too rare to matter, and removing them changes nothing. The
first sink-score implementation was not discriminative (flagged 64–99% of tokens: a fixed τ = 20 measured scale);
after the per-image robust fix, HuatuoGPT has no visual sinks by Kang et al.'s criterion.
**In simple words.** The famous "sink" tokens are there, but taking them away does not fix FastV.
**What to do.** H3 is dead as a cause. Cite Kang et al. and Darcet et al., and show `artifact_excl` as the null.

### D-13 Is the ranking question-aware? (G5)

**Question.** FastV is described as text-guided. Do different questions about the same image get different
keep-sets?
**Why.** If not, "text-guided" is false at K=2, and evaluation on any question set is fair.
**How.** Jaccard between FastV keep-sets of different questions on the same SLAKE image; question share of rank
variance.

| | Lingshu | Qwen3-VL | HuatuoGPT | Random |
|---|---|---|---|---|
| keep-set Jaccard, same image, different question | **0.978 [0.974, 0.982]** | 0.820 [0.805, 0.836] | 0.70 | 0.054 |
| question share of layer-1 rank variance (SLAKE) | 0.3% | 3.1% | 6.2% | — |
| same on photographs | ≤ 1.2% | ≤ 1.2% | — | — |

**Technical reading.** At layer 1 the scoring query is a fixed template token that has not yet integrated the
question, so the ranking is almost question-independent. This is general (photographs too), not medical.
**In simple words.** FastV picks the same tokens no matter what you ask.
**What to do.** Report as a headline finding (Fig. 3 with D-14).

### D-14 Where does attention become question-aware? Layer profile (E12, D1)

**Question.** At which layer does the ranking start to depend on the question?
**Why.** Locates the switch; tells a better method where to score.
**How.** `full_profile` runs (no pruning) log, at every LLM layer, last-token and text-token attention over the visual
tokens plus sink scores; `analysis profile` computes the variance split, evidence retention and BRR per layer.

| Question-beyond-image share (SLAKE, last-token readout) | layers 0–10 | first clear rise | late layers |
|---|---|---|---|
| Lingshu (28 layers) | 0.00–0.03 | 0.08 at 11, **0.24 at 14** | 0.29–0.41 (18–27) |
| Qwen3-VL (36 layers) | ≤ 0.04 through 16 | **0.15 at 17**, 0.38 at 21 | 0.22–0.42 |
| HuatuoGPT (28 layers) | position 63% → 7–10% by layers 6–12 | peak 0.27 at layer 21 | — |

| RIN, Lingshu, by layer | ≤ 12 | 14 | 18 |
|---|---|---|---|
| position share | 0.59–0.73 | 0.43 | 0.26 |
| BRR of top 10% | ≥ 2.0 (through 17) | | **0.63 (minimum)** |

The text-token readout is *less* question-dependent than the last token at every layer from 14 on (Lingshu SLAKE
0.06–0.21 vs 0.24–0.41).
**Technical reading.** Question dependence switches on at ≈ 50–65% depth on both Qwen models (Lingshu 14–18 of 28,
Qwen3-VL 17–21 of 36), and the positional/background bias fades at the same depth. This agrees independently with the
K sweep (D-17).
**In simple words.** The model only starts "reading the question while looking at the image" halfway through.
**What to do.** A second-stage scorer should read the **last token** at that depth. **[FLAG]** This profile was
computed on the test sets: choosing a method's layer L* from it is test-set tuning. L* must come from held-out data
(SLAKE validation, a held-out RIN slice) or a rule fixed in advance (e.g. first layer with question share ≥ 0.2 on
held-out data).

### D-15 Is it the readout? `fastv_text` (D2, G2)

**Question.** The FastV paper describes scoring by the attention of *all* text tokens; the official code uses the last
token. Does the readout matter?

| X − Random | Lingshu RIN | Lingshu SLAKE | Qwen3-VL RIN | Qwen3-VL SLAKE (judge) |
|---|---|---|---|---|
| FastV (last token) | −41.7 | −13.4 | −33.7 | −5.0 |
| `fastv_text` (all text tokens) | −41.2 (− FastV +0.5 [−0.4, +1.4]) | −13.2 | −27.8 | −1.2 |

**Technical reading.** Both readouts fail on RIN. Qwen3-VL's text readout is somewhat less bad, which matters for E13:
SparseVLM's robustness on Qwen3-VL is not explained by reading text tokens (SparseVLM − `fastv_text` +27.9 on RIN).
**In simple words.** Changing which text token does the looking does not fix it.
**What to do.** G2 closed. **[FLAG] OPEN:** why SparseVLM survives on Qwen3-VL (rater subset vs its early budget
schedule) needs one ablation (≈ 0.4 GPU-h) before the paper explains it.

### D-16 Is early attention informative at all? (G4, E11, E11r)

**Question.** Is attention noisy-but-useful, or does it point the wrong way?
**Why.** The core assumption of every attention pruner is "high attention ⇒ important".
**How.** (1) Within Random's own draws, regress correctness on the attention rank of the kept tokens (log-odds per SD).
(2) `fastv_bottom`: keep the least-attended k.

| Within Random: log-odds per +1 SD of kept-token attention (p) | Lingshu RIN | Qwen3-VL RIN | Lingshu SLAKE | Qwen3-VL SLAKE |
|---|---|---|---|---|
| inside Random | −0.12 (0.18) | −0.05 (0.57) | −0.14 (0.03) | +0.06 (0.39) |
| inside Gate+Random | **−0.35 (< 0.001)** | −0.18 (0.06) | −0.19 (0.005) | −0.05 (0.42) |

| Least-attended − most-attended | Lingshu | Qwen3-VL | HuatuoGPT |
|---|---|---|---|
| SLAKE | **+6.2 [+2.5, +10.1]** | −4.2 [−7.2, −1.4] | −10.7 |
| RIN | **+40.0 [+36.2, +43.7]** | **+29.9 [+26.2, +33.7]** | −33.0 |
| least-attended − Random, RIN | −1.7 (n.s.) | −3.8 | — |

**Technical reading.** In the Qwen family, layer-1 attention is **anti-informative** on RIN: the tokens it ranks lowest
are nearly as good as random, the ones it ranks highest are catastrophic. In HuatuoGPT attention is informative, which
is why its FastV ties Random. A plausible (untested) reading: the most-attended tokens are a contiguous, low-diversity
block, so ranking by attention maximises redundancy.
**In simple words.** For the Qwen models, "the tokens the model looks at most" are the worst ones to keep.
**What to do.** Strongest single argument against early-attention scoring in these models; put it in the main text.

### D-17 Does a deeper scoring layer fix it? K sweep (E6, P11, G8)

**Question.** FastV scores at K−1 = 1. Does scoring deeper recover?
**Expected.** P11: the gap shrinks monotonically; K=16 ≈ Random.

| FastV − Random | K=2 | K=8 | K=16 |
|---|---|---|---|
| Lingshu RIN anatomy | −41.7 | −43.9 | **+8.5** |
| Lingshu SLAKE | −12.4 | −12.4 | **+6.9** |
| Lingshu SLAKE CT/MRI | −18.0 | — | +8.0 |
| Qwen3-VL RIN | −33.7 | −27.1 | **+5.6** |
| Qwen3-VL SLAKE | −2.4 | −0.3 | +0.9 |
| Lingshu RIN abnormality | −14.2 | −17.4 | −6.6 [−9.3, −3.9] |

**Technical reading.** Not monotone: layers ≤ 8 are uniformly bad, then by layer 16 attention *beats* Random (the
question-aware zone of D-14). RIN abnormality is the exception: K=16 still loses. Cost: K=16 runs 16 layers on all
tokens, so prefill speedup drops from 2.95× to 1.47× (D-28).
**In simple words.** Attention becomes a good guide halfway through the model, but waiting that long costs most of the
speed-up.
**What to do.** Basis of the two-stage method (§6): cheap coverage early for speed, attention late for accuracy.

### D-18 Is 10% just an extreme budget? (E5, G7)

| FastV − Random (Gate+FastV − Random) | 5% | 10% | 25% | 50% (− Full) |
|---|---|---|---|---|
| Lingshu RIN | −39.8 (−32.5) | −41.7 (−28.5) | −43.9 (−15.3) | −5.0 (−2.0) |
| Lingshu SLAKE | −12.4 (−4.0) | −13.4 (+0.4) | −6.2 (+1.9) | −1.4 (−0.9) |
| Qwen3-VL RIN | −32.2 (−22.0) | −33.7 (−18.7) | −15.9 (−7.7) | — |
| Lingshu Gate+Uniform − Random, SLAKE | **+3.9 [+1.7, +6.1]** | +2.3 | +0.9 | — |

**Reading.** Lingshu's RIN loss does not shrink from 5% to 25%; the failure is not an extreme-budget artefact. The
gate's advantage over Random is largest at the smallest budget.
**In simple words.** Giving FastV more tokens does not rescue it until about half the image is kept.

### D-19 Is it token count or resolution? (E10, E10r, G1)

**Question.** RIN has 64 tokens per image; SLAKE 324+. Is the collapse a small-token effect?
**How.** Downsize SLAKE and photographs to 224² (64 tokens); upsize RIN to 448² (256 tokens); compare with HuatuoGPT,
which always sees 576.

| FastV − Random | photos | photos @224² | SLAKE | SLAKE @224² | RIN (64) | RIN @448² (256) |
|---|---|---|---|---|---|---|
| Lingshu | −8.7 | −10.7 | −12.6 | −16.3 | −41.2 | **−48.9 [−52.3, −45.4]** |
| Qwen3-VL | −4.9 | −2.7 | −2.4 | −0.5 | −33.7 | −29.2 [−32.6, −25.8] |

**Technical reading.** Fewer tokens amplify modestly (Lingshu only); matched-token SLAKE and photographs do not
reproduce RIN's collapse, and 4× more tokens make Lingshu *worse*. With HuatuoGPT tying Random on the same RIN images,
the collapse is a property of the Qwen-family models on these images, not of token count or of the dataset alone.
**In simple words.** RadImageNet's disaster is not because its images are small.
**What to do.** G1 closed. G22 ("why RIN?") remains the main open mechanism question.

### D-20 H4: does FastV miss the anatomy the question names? (D3, G16, P14, oracle gate, G23)

**Question.** Does FastV drop the organ the question asks about, and is that why it fails?
**Why.** The most clinically legible story; it was planned as Fig. 1.
**Expected.** H4/P14: residual loss largest for the *smallest* structures.
**How.** SLAKE `mask.png` labels mapped into each model's token grid; questions whose text or answer names an
annotated structure (358 Lingshu / 364 Qwen3-VL); ERR per keep-set; area tertiles. Then the intervention
`oracle_gate`: FastV restricted to the union of all annotated structures (dilated one cell).

| Evidence retention (ERR, chance = 1) | Lingshu | Qwen3-VL |
|---|---|---|
| FastV · Random · Gate+FastV | 0.84 [0.57, 1.11] · 1.00 · 2.81 | 0.61 [0.43, 0.80] · 1.05 · 1.68 |
| FastV by organ size: small · medium · large | 1.46 · 0.84 · **0.20** | 0.88 · 0.72 · **0.23** |
| large organ dropped entirely, FastV / Random | **58% / 13%** | 71% / 11% |
| oracle gate ERR / hit rate | 4.89 / 100% | — |

| Accuracy, Lingshu SLAKE (judge), X − Random | small | medium | large |
|---|---|---|---|
| FastV | −4.2 | −6.7 | **−16.0 [−23.9, −6.8]** |
| Gate+FastV | +1.7 | −3.4 | +4.2 |
| Oracle gate | 0.0 | +1.7 | **+10.1 [+1.7, +17.1]** |

| Split by question, Lingshu SLAKE (judge) | names an annotated organ (n = 358) | does not (n = 703) |
|---|---|---|
| Oracle − Random | **+5.3 [+2.0, +8.9]**, p 0.003 | **−3.6 [−6.9, −0.3]**, p 0.033 |
| Oracle − Gate+FastV | +3.1 [+0.6, +5.7] | −3.0 [−6.0, +0.1] |
| **FastV − Random** | −7.5 [−11.4, −3.8] | **−16.4 [−20.2, −12.5]** |
| Overall: Oracle − Random −0.6 (n.s.); Oracle − Gate+FastV −0.9 (n.s.) | | |

Other facts: Lingshu's least-attended tokens keep *less* of the named organ than FastV (0.59× vs 0.83×) yet answer 6.2
points better (G23). Per question, evidence recall explains 7% of FastV's penalty (mediation). Within Random, "any
evidence token kept" predicts correctness (+0.54 log-odds, p 0.04). SLAKE by question type (Lingshu FastV − Random):
Organ −25.3, Size −20.0, Quantity −13.5, Position −13.4, Modality −11.1, Plane −8.6, Abnormality −2.7, KG −0.7.

**Technical reading.** H4 is reversed: FastV fails on **large** organs (it keeps them at a fifth of chance). But the
oracle shows evidence loss is the *minor* component: a perfect anatomical gate helps organ questions (+5.3) and hurts
the other two-thirds (−3.6), because restricting to annotated organs removes context that modality, plane and
abnormality questions need. FastV's own loss is twice as large on the questions that name no organ.
**In simple words.** FastV does throw away the organ you asked about, but most of its damage is elsewhere: it loses
the overall picture.
**What to do.** **[FLAG]** Use G16 as an illustration (Fig. 1 can still show a dropped liver), not as the headline
mechanism. The oracle confirms the heuristic gate is not the bottleneck. **OPEN:** D4, the within-question causal test
(Random seeds 0–2, conditional logit of correctness on evidence kept).

### D-21 What does the pruned model do instead? Language-prior fallback (G14)

| | Lingshu RIN | Qwen3-VL RIN | Lingshu SLAKE | Qwen3-VL SLAKE |
|---|---|---|---|---|
| FastV answer = text-only answer (Random) | **33.3% (21.6%)** | 46.7% (33.2%) | 37.3% (31.5%) | 32.8% (33.6%) |
| retained visual gain: FastV / Random / Gate+FastV | **26% / 86% / 45%** | 35% / 91% / 60% | 60% / 84% / 84% | 76% / 83% / 81% |

Answer collapse is mild: RIN most frequent letter 31% under FastV (Random 25%, text-only 36%); SLAKE 93 distinct
answers vs 116, entropy 4.85 vs 5.09 bits.
**Reading.** After FastV, Lingshu keeps a quarter of what the image contributes and answers like the text-only model
far more often.
**In simple words.** FastV leaves the model nearly blind, so it guesses from the question.
**What to do.** Report retained visual gain for every cell (it normalises across datasets with different text-only
floors).

### D-22 Is it every architecture? Qwen3-VL and HuatuoGPT (P10, H5, E8, E8r, G6)

| X − Random | HuatuoGPT SLAKE | HuatuoGPT RIN | Lingshu RIN | Qwen3-VL RIN |
|---|---|---|---|---|
| FastV | −0.6 [−2.7, +1.4] (judge −1.6) | **−0.1 [−1.9, +1.7]** | −41.7 | −33.7 |
| RoPE-free | **+3.1 [+1.1, +5.0]** | +2.5 | −42.3 | −31.9 |
| Gate+FastV | +2.5 [+0.7, +4.3] | +2.0 | −28.5 | −18.7 |
| `fastv_text` | −1.4 vs FastV | −6.5 | −41.2 | −27.8 |
| least- − most-attended | −10.7 | −33.0 | +40.0 | +29.9 |

**Technical reading.** Qwen3-VL fails in the same direction as Lingshu with smaller magnitude everywhere (P10;
consistent with H5/DeepStack but not tested causally). HuatuoGPT (1D RoPE, LLaVA, 576 fixed tokens) does not fail;
its attention is informative and its bias is the rotary bottom-row prior that RoPE-free removes.
**In simple words.** This is a property of how the Qwen-family models' early layers look at images, not a law of
attention pruning.
**What to do.** **[FLAG]** Scope the title and claims to early-layer LLM attention in M-RoPE (Qwen-family) models;
make HuatuoGPT the contrast case in the mechanism section.

### D-23 Natural photographs: is the failure medical-specific? (E7, P13)

| FastV − Random | GQA | POPE | all 1,000 | RoPE-free |
|---|---|---|---|---|
| Lingshu | −9.2 | −8.2 | −8.7 [−11.7, −5.8] | −8.5 |
| Qwen3-VL | −7.8 | −2.0 | −4.9 [−7.8, −2.1] | −4.7 |

**Reading.** P13 contradicted: FastV is below Random on photographs too. Medical scans *amplify* the failure
(CT/MRI background, ultrasound, RIN).
**In simple words.** FastV loses to random picking on ordinary photos too; medical scans just make it much worse.
**What to do.** Decision (Pushkar, Nov 4): medical-only paper. Never claim "medical-specific"; put E7 in the
supplementary as a scope note. Prior work (Wang et al. 2512.07580; VTC-Bench) already reports weak attention pruning
on natural images, so the note costs nothing.

### D-24 Is it every pruner? Published baselines (E13, P15–P18)

Official selection code (pinned commits, tested against the reference implementations); multi-stage methods
token-matched to FastV (mean visual tokens per layer). SLAKE under the guarded judge.

| Lingshu, X − Random | SLAKE | RIN anatomy | RIN abnormality | OmniMedVQA anatomy |
|---|---|---|---|---|
| FastV | −13.4 | −41.7 | −14.2 | −29.4 |
| SparseVLM | −15.6 | **−44.8** | −13.6 | −31.7 |
| BTP | −2.7 | −9.8 | −9.0 | −6.5 |
| VisPruner | +3.2 | −0.7 | −2.0 | +0.4 |
| PruneSID | +1.5 | −2.5 | −4.1 | +3.5 |
| DivPrune | 0.0 | −11.3 | −3.9 | −14.7 |
| Uniform | +0.8 | −0.9 | −7.6 | +0.9 |
| **Gate+Uniform** | **+2.3** | **+1.8** | **+1.6** | **+1.9** |

| Qwen3-VL, X − Random | SLAKE | RIN anatomy | RIN abnormality* | OmniMedVQA anatomy |
|---|---|---|---|---|
| FastV | −5.0 | −33.7 | −2.2 | −4.3 |
| SparseVLM | +0.8 | +0.1 | +0.3 | −0.6 |
| BTP | +0.2 | +0.2 | +0.3 | +0.1 |
| VisPruner | −0.4 | −4.9 | −0.6 | −2.1 |
| PruneSID | **+2.4 [+0.8, +4.0]** | +1.3 | −0.1 | +0.7 |
| DivPrune | −0.3 | +1.5 | −0.5 | −0.6 |
| Gate+Uniform | +1.1 | +1.3 | −0.3 | +0.5 |

\* not image-dependent for Qwen3-VL.

**Technical reading.** Only methods that score with **early LLM attention** collapse (FastV, SparseVLM on Lingshu;
BTP, half attention, loses 3–10). Methods that never read LLM attention (VisPruner's vision-tower attention, PruneSID,
DivPrune, Uniform) do not collapse anywhere. On Lingshu SparseVLM's first keep-set nearly contains FastV's (Jaccard
0.66–0.68 of a possible ≈ 0.70); on Qwen3-VL it does not (0.33) and SparseVLM ties Random. VisPruner loses 4.9 on
Qwen3-VL RIN (BRR 1.47): Qwen3-VL's vision tower is also edge-biased at 64 tokens.
**In simple words.** The problem is specific to pruners that trust the LLM's early attention.
**What to do.** **[FLAG] The gate is not a state-of-the-art pruner** and must not be sold as one. It is the most
consistent method on Lingshu (the only one ≥ 0 on all four datasets) and adds nothing over coverage on Qwen3-VL.
Position: Gate+Uniform is the minimal remedy the diagnosis predicts.

### D-25 Multi-stage attention pruning: PDrop ± gate (E6-P2, G19)

| Lingshu, X − Random | SLAKE (rule) | SLAKE CT/MRI | RIN anatomy | RIN abnormality |
|---|---|---|---|---|
| FastV | −12.4 | −18.0 | −41.7 | −14.2 |
| PDrop | −13.8 [−16.8, −10.7] | −20.0 | **−61.2** | −17.5 |
| Gate+PDrop | **−0.6** | −0.0 | −31.4 | −14.0 |

**Reading.** Multi-stage attention pruning token-matched to FastV fails the same way; at 64 tokens PDrop's stages keep
13 → 6 → 3 → 2 tokens (from layers 2 / 9 / 16 / 23), front-loading the budget and starving the late layers. The gate
repairs PDrop on SLAKE like FastV. PDrop's published robustness at 80% kept does not survive 10%.
**In simple words.** Pruning in several steps instead of one does not help when every step trusts the same early
attention; the gate fixes it on CT/MRI here too.
**What to do.** Use for G19 ("not only FastV"); scope the title to early-layer LLM attention unless a vision-encoder
method is added.

### D-26 Gate robustness: τ and background quota (E6-P2)

| Lingshu, X − Random | τ 0.25 | **τ 0.5 (default)** | τ 0.75 | background quota 3% |
|---|---|---|---|---|
| SLAKE (rule) | −1.0 | +0.3 | +0.9 | −2.5 |
| RIN anatomy | −32.8 | −28.5 | −25.6 | −36.1 |
| RIN abnormality | −12.6 | −10.1 | −8.9 | −13.5 |

**Reading.** Flat on SLAKE; monotone on RIN (a stricter foreground filter helps by removing mixed edge tokens; giving
background back hurts). At 64 tokens the deficit is not missing context.
**In simple words.** The gate's one setting barely matters on CT/MRI; on small RadImageNet images a stricter gate
helps a little, but no setting rescues FastV there.
**[FLAG]** τ was fixed on SLAKE; reporting τ 0.75 for RIN would be test-set tuning. The paper reports τ 0.5 everywhere
and the sweep as robustness.

### D-27 Is the gate a general plug-in? DivPrune ± gate (E6, E13 completion)

| Gate+DivPrune − DivPrune | Δ [95% CI] | p |
|---|---|---|
| Lingshu SLAKE | +3.5 [+1.8, +5.4] | < 0.001 |
| Lingshu RIN anatomy | **+9.6 [+7.0, +12.3]** | < 0.001 |
| Lingshu RIN abnormality | +3.7 [+0.7, +6.5] | 0.017 |
| Lingshu OmniMedVQA anatomy | +6.3 [+5.0, +7.6] | < 0.001 |
| Qwen3-VL SLAKE | +1.9 [+0.3, +3.6] | 0.034 |
| Qwen3-VL RIN anatomy | +1.5 [+0.1, +3.1] | 0.066 |
| Qwen3-VL OmniMedVQA anatomy | +1.4 [+0.4, +2.3] | 0.004 |
| Qwen3-VL RIN abnormality (not image-dependent) | +0.2 [−0.8, +1.2] | 0.85 |

**Reading.** Positive in all 7 image-dependent cells, significant in 6 (Qwen3-VL RIN anatomy: the bootstrap CI
excludes 0 but the sign-flip p is 0.066 — report it as borderline). The cleanest support for "the gate composes".
But on Lingshu OmniMedVQA, Gate+DivPrune (59.7) stays below Random (68.0) and PruneSID (71.5): DivPrune itself is weak
there (−14.7).
**In simple words.** Adding the gate in front of another pruner makes that pruner better almost everywhere, but it
cannot turn a weak pruner into a strong one.
**What to do.** Table 3: gate as a plug-in for FastV, PDrop and DivPrune.

### D-28 Latency and FLOPs (E9)

**Question.** Does pruning actually save time, and does the gate?
**How.** Lingshu, 200 SLAKE questions, two T4s. Visual tokens physically removed at the switch layer (cos/sin and
positions sliced, original M-RoPE ids kept); 10 warm-up questions; schedules timed in shuffled order between
device-wide syncs. Theoretical prefill FLOPs per layer 4nd² + 2n²d + 2ndm (FastV formula) from every run's own keep
schedule. SLAKE images give 81 / 324 / 1,024 visual tokens (34% at 1,024), so the latency distribution has three modes
and the per-question paired ratio is the statistic to quote.

| Schedule | prefill median (ms) | speedup (paired) | first-token path* | prefill FLOPs vs Full |
|---|---|---|---|---|
| Full | 296 | 1.00× | 1.00× | 1.00 |
| Gate only (background removed before the LLM; 88.5% kept) | 294 | 1.01× | ≈ 1.0× | — |
| Pre-LLM 10% | 91 | 3.27× (3.36×) | 1.75× | 0.160 |
| **FastV 10%** | 100 | **2.95× (3.07×)** | **1.69×** | 0.220 |
| FastV K=16, 10% | 202 | 1.47× (1.54×) | 1.24× | 0.640 |
| Text-only | — | — | — | 0.070 |

\* vision encoder + embedding (183 ms, untouched by any pruner) + prefill.

**Reading.** FastV reaches ≈ 65% of its FLOPs bound. The gate alone saves nothing on SLAKE (it removes 11% of tokens);
Gate+FastV costs exactly FastV's FLOPs. Scoring overhead and decode were not timed.
**In simple words.** Pruning makes the language-model part 3× faster but the whole answer only 1.7× faster, because
the image encoder still runs in full.
**What to do.** Report both numbers; never only the prefill one.

**T11 (Nov 11): ScanPrune itself, same protocol.** Selection included (10-12 ms): ScanPrune 10% prefill 3.33× (paired
0.300 of Full), first-token path 1.78×; ScanPrune at the equal budget 3.12× / 1.74×; the size-term mask costs nothing
measurable (0.271 vs 0.281 of Full without it). Masked loop vs model forward: relative Frobenius error 0.0073.

---

## 5. Synthesis

### 5.1 What survived, what died

| Candidate explanation | Status | Killed or supported by |
|---|---|---|
| Implementation error | DEAD | D-01 |
| Wrong readout (last token vs text tokens) | DEAD | D-15 |
| Rotary decay (Qwen family) | DEAD | D-07, D-08 |
| Attention sinks / high-norm artifacts | DEAD | D-12 |
| Token count / resolution | DEAD | D-19 |
| Extreme budget | DEAD | D-18 |
| Background fraction predicts the loss (C2) | DEAD | D-04 |
| Background capture | PARTIAL: explains SLAKE CT/MRI, a third of RIN | D-05, D-06 |
| Missing the asked-about organ | PARTIAL: real, minor | D-20 |
| **Question-blind early prior** | SUPPORTED | D-13, D-14 |
| **Positional prior carried by the visual keys** | SUPPORTED (Lingshu), partial (Qwen3-VL) | D-08, D-09, D-10 |
| **Anti-informative early attention** | SUPPORTED (Qwen family) | D-16 |
| **Concentration destroys coverage → language-prior fallback** | SUPPORTED | D-06, D-11, D-21 |
| **Failure confined to early layers** | SUPPORTED | D-14, D-17 |

### 5.2 Two regimes

| Regime | Where | Mechanism | Gate |
|---|---|---|---|
| Background capture | SLAKE CT/MRI, ultrasound (300+ tokens, large air) | the fixed prior lands on air | fixes it fully |
| Prior collapse | RadImageNet (64 and 256 tokens), microscopy, dermoscopy | the fixed, concentrated prior misses the content even inside the body | repairs ≤ ⅓ |

### 5.3 Paper claims after the diagnostics

| Claim | Status | Action |
|---|---|---|
| C1 Phenomenon | holds for Qwen-family models; not universal (HuatuoGPT); not medical-specific | scope the claim; HuatuoGPT as contrast; E7 in supplementary |
| C2 Predictability (background dose-response) | **contradicted** | drop; per-modality bars instead |
| C3 Mechanism | holds via question-blindness, position-in-keys, anti-informative attention, interventions | rewrite away from "rotary decay" and "attention picks air" |
| C4 Remedy | Nov 6: partial (full on SLAKE, ⅓ on RIN), composes with every pruner. **Nov 10:** the tested method (§6.2) beats VScan at equal tokens per layer (+1.79, SE 0.39, three seeds) and VisionZip at 10% (+3.04, SE 0.46) | diagnosis still leads; the method is the remedy the diagnosis predicts, claimed on the nine-cell mean |
| C5 Protocol | holds: the judge flips Qwen3-VL SLAKE; McNemar overstated p | report judge + rule; image-clustered tests |

---

## 6. How to fix it

### 6.1 For practitioners (what the diagnosis predicts)

1. **Do not rank tokens with layer-1 LLM attention in Qwen-family models.** Keep coverage early instead: Uniform,
   DivPrune or PruneSID, plus the Foreground Gate where acquisition background exists. On Lingshu, Gate+Uniform is ≥ 0
   vs Random on all four datasets and beats every published pruner on RIN anatomy.
2. **If question-aware selection is wanted, score where the model reads the question:** last-token attention at
   ≈ 50–65% depth (K=16 beats Random by 5.6–8.5 on RIN). Choose the layer on held-out data.
3. **Report end-to-end speed**, not only prefill (D-28).

### 6.2 The method that was built and tested (E40–E46, Nov 7–10)

The Nov 6 proposal here was two-stage pruning: coverage early, then last-token attention top-k at a question-aware
layer. It was not run in that form. The built method keeps the first stage and replaces attention ranking with steps
that follow from the diagnosis and from the E40–E46 measurements; no step reads attention scores. Every step was
registered before its runs and adopted by the rule in `medprune/adoption.py` (one-sided nine-cell test, Holm, two-halves
replication, no Bonferroni-significant cell loss), which replaced the proposal's kill test.

| Step | What it does | Why (diagnosis) | Test → result |
|---|---|---|---|
| 1 Foreground gate | drop tokens with foreground fraction < τ = 0.5; discard them | background capture (§5.2, D-05); robust to τ (D-26) | Nov 6: closes SLAKE's gap |
| 2 Coverage | one random token per Hilbert-curve stretch of the foreground | coverage, not ranking, carries accuracy (D-11, D-16, D-21) | E40–E41 |
| 3 Merge | each dropped foreground token averaged into its most cosine-similar kept token | dropped content stays readable | E40–E41: 10% − VisionZip +1.92 (SE 0.42); equal budget − VScan +0.62 (SE 0.39) |
| 4 Size-aware attention | + log s on each kept token's logit, every layer | merged keys were read at 1/2.9–1/4.8 of Full's foreground attention (§9) | E42a adopted (+0.91 vs step 3, p 0.0001); at 10% (2B) − VisionZip **+3.04** (SE 0.46) |
| 5 Routed schedule | one-token instructions: 20.3% / 18.4% to layer 20 / 26, then 6.8% / 6.1%; else flat 16.4% / 15.0% | one-token answers are formed in the prefill pass; generated answers read the image later (§9) | E42b flat schedule rejected; E43 route adopted; E45: final − VScan **+1.79** (SE 0.39), three seeds |

Confirmation of the final row: test half (not used for selection) +1.82 (SE 0.58, p 0.0008); held-out `omni_panel`
+0.96 (SE 0.54, p 0.037), multiple choice only, so it confirms the schedule branch. Per model over three seeds: Lingshu
+2.98, Qwen3-VL +0.29.

Tested and not adopted:

| Piece | Result | Why it fails |
|---|---|---|
| Schedule on every question (E42b) | open anatomy −6.0 / −4.2 | generated answers need the image after the cut |
| Background summary token (E40–E43) and + size term (BG-PA, E46) | no k-dependence; BG-PA fails its pilot gates (Lingshu abnormality balanced −3.5) | Full's background attention sits on a few sink tokens; one mean token cannot reproduce it (§9) |
| k-means clusters + size term (E43) | open anatomy −1.6 / −1.6 | ≈ 10% worse spatial coverage |
| SLIC clusters + size term (E46) | − E42a −0.06 (SE 0.40, p 0.56) | coverage restored, nothing gained: with the size term, how clusters are formed no longer moves accuracy |

### 6.3 For the paper

1. Section 5 leads with the question-blind prior (D-13/14), the positional interventions (D-09/10), anti-informative
   attention (D-16) and the language-prior fallback (D-21).
2. Drop C2 and its figure; G16 becomes an illustration with the oracle split stated.
3. Write "a positional prior carried by the visual keys", contrasted with FEATHER's rotary prior in HuatuoGPT.
4. Scope: early-layer LLM-attention pruning in M-RoPE Qwen-family models, medical VQA.
5. Recompute all pre-Nov-5 p-values with the sign-flip test; Holm over P1–P14; label everything else exploratory.

---

## 7. Open diagnostics and known confounds

| ID | Open item | Why it matters | Cost |
|---|---|---|---|
| G22 | Why does the same prior hurt RIN (−41.7) far more than SLAKE (−12.4)? | the largest effect in the paper is unexplained | in-session token statistics of RIN images (aggregates only) |
| D4 | Within-question causal test: is evidence kept → correct? (Random seeds 0–2, conditional logit) | decides how far G16 can be pushed | laptop |
| G10 | ~~Majority baseline and balanced accuracy for every yes/no cell~~ **Closed Nov 10:** Lingshu RIN abnormality is reported in balanced accuracy (majority = 50 by construction), Qwen3-VL's version is dropped as a language-prior cell; raw accuracy above Full is a yes-rate shift (§9) | RIN abnormality is 71.9% "yes" | done |
| SparseVLM | Why it survives on Qwen3-VL: rater subset vs early budget schedule | needed before the paper explains it | ≈ 0.4 GPU-h |
| H5 | DeepStack as the reason Qwen3-VL fails less | currently only consistent, not tested | needs an ablation |
| Judge | κ 0.90 is in-sample and AI-to-AI | must not be reported as human agreement | 100 held-out human labels |
| G19 | ~~No vision-encoder-side pruner besides VisPruner~~ **Closed Nov 7 (E14):** VisionZip, HiPrune, CDPruner, FEATHER, VScan and TransPrune are in the table | title scope | done |
| G20 | Positioning vs Wen et al. 2025, DART, VTC-Bench, MedSTP | "already known" objection | writing; MedSTP's bridge is computed (§9, plan.md §A3.1) |
| G15 | Lingshu was trained on SLAKE-style data | contamination | limitation paragraph |
| Latency (method) | **Closed Nov 11 (T11)**: ScanPrune timed with its own selection, merge, size-term mask and second cut | — | ScanPrune 10% prefill 3.33×, first token 1.78×; equal budget 3.12× / 1.74×; selection 10-12 ms; mask cost nil; FLOPs = VisionZip's / VScan's |
| Route confound | Format vs dataset: every routed cell is a whole dataset (MC or yes/no instructions) | the mechanism claim for step 5 | within-SLAKE split done (§9) supports format; a held-out set mixing formats would settle it |
| Panel coverage | `omni_panel` is multiple choice: it confirms the schedule branch, not the flat branch | held-out evidence for generated answers | a held-out open-answer set |
| Cell concentration | The final row's lead comes from OmniMedVQA (+6.05 / +2.67) and Lingshu open anatomy (+5.20); Qwen3-VL SLAKE −1.19 and RIN anatomy −0.70 | "better everywhere" is false | report per-cell; writing |

---

## 8. Where the numbers come from

| Source | Contents |
|---|---|
| `plan.md` §A6, §A9, §A10, Work log (Nov 2–5) | every number above, with dates |
| `results/all/runs/*.jsonl` | one row per question per run (keep-sets, answers, correctness) |
| `results/all/diag/` | layer-1 / profile scores per question |
| `results/all/judge/verdicts.jsonl` | open-answer verdicts (8,832) |
| `results/all/tables/evidence_*.csv`, `layer_profile.csv`, `flops.csv` | D-20, D-14, D-28 |
| `results/all/latency/lingshu7b__slake_en__t4x2.json` | D-28 |
| `results/all/latency/lingshu7b__slake_en__scanprune__t4x2.json` | D-28 (T11) |
| `python -m medprune.analysis {diagnosis,evidence,profile,baselines}` | regenerate the tables |
| `python -m medprune.latency {measure,measure-scanprune,prompt-tokens,flops}` | E9, T11 |
| `medprune.adoption` (`load_cell`, `paired_cells`, `nine_cell`; CLI `--reference X --candidate Y`) | §6.2, §9: nine-cell deltas, SEs and every adoption decision |

## 9. Additions, Nov 7-10 (E40-E46)

Numbers from plan.md (E40-E46 cards), recomputed from the run files on Nov 10 16:00. All judged; Lingshu RadImageNet abnormality in balanced accuracy where noted.

| Diagnostic | Finding | Consequence |
|---|---|---|
| Yes-rate on RIN abnormality | Gold 71.9% yes, Full answers yes 26.9%; pruned rows that answer yes more often gain raw accuracy without better discrimination | Report the cell in balanced accuracy; never "above Full" |
| Budget audit | "10%" rows spend 9.4-9.9% (pre-LLM), 14.4-16.5% (FastV family, VScan) or 36-46% (TransPrune, FEATHER) of N per layer | Compare only within one budget rule |
| Deep-layer image attention | Full's last-token image share in deep layers I = 0.018-0.152, foreground part F = 0.009-0.084; the merged recipe reads the image at 1/2.9-1/4.8 of F; + log s restores it (measured 0.0094 / 0.037 / 0.056 / 0.108 vs predicted 0.0091 / 0.041 / 0.039 / 0.092) | Size-aware attention (adopted) |
| Answer format | The schedule gains wherever the answer is one token (computed in the prefill pass) and loses on generated answers (open anatomy −6.0 / −4.2) | Schedule routed to one-token instructions (adopted) |
| Dose-response of the size term | Within cells the gain does not track cluster size (Spearman −0.08 to 0.00); it concentrates where the reference is weakest | The residual error is content averaging, not attention mass |
| Background summary token | No dependence on k; shifts Lingshu's yes-rate toward Full (35.6 → 26.6 at 10%) | Repaired as summary + size term (BG-PA, E46): fails its pilot gates (next row) |
| BG-PA pilot (E46) | Lingshu abnormality balanced −3.5 vs E42a, yes-rate 20.5 (past Full's 26.9); Qwen3-VL RIN anatomy deep image share 0.0097 (E42a 0.0094). Full's background attention sits on a few sink tokens, so Σ_bg e^{a_j} is dominated by its maximum; a mean background token's s · e^{ā} falls far short (the duplicate identity behind the size term needs near-identical tokens, and sinks are outliers) | Background stays discarded; P81 ✗ |
| k-means merge clusters | Content-pure (SLAKE +0.4 / +1.3) but ≈ 10% worse spatial coverage; open anatomy −1.6 / −1.6 | Repaired as SLIC clusters (content + position, E46): next row |
| SLIC clusters + size term (E46) | Pilot gates pass (coverage 0.356–0.416 vs E42a 0.364–0.437). Full nine cells − E42a −0.06 (SE 0.40, p 0.56), halves −0.10 / −0.02; − VScan +1.47 (SE 0.41) | Not adopted (P82 ✗): with the size term, the clustering rule no longer moves accuracy, as P63 found for selection |
| Seed variance | Nine-cell selector-seed SD ≈ 0.25 (E42a); final route over three seeds +1.94 / +1.77 / +1.66 vs VScan (SD 0.14); Qwen3-VL four-cell 65.50 / 65.43 / 65.30 vs VScan 65.11 | Headline = three-seed mean; per-model single-seed margins are not claims |
| Route assignment | The route sends whole datasets to one branch: RIN anatomy, RIN abnormality, OmniMedVQA → schedule; SLAKE (closed questions use a word-or-phrase instruction) and RIN open → flat | Format and dataset are confounded in the nine cells (next row) |
| Format inside one dataset | SLAKE, flat schedule (E42b) − recipe: closed +2.40 (SE 0.99) / +0.24 (1.15), open −0.47 (0.99) / −1.24 (0.84), Lingshu / Qwen3-VL | The format pattern holds within a dataset; one dataset, two models: supports the account, does not prove it |
| Per-cell profile of the final row | − VScan: Lingshu SLAKE −0.13, RIN anatomy +0.93, abnormality +2.86, OmniMedVQA +6.05, open anatomy +5.20; Qwen3-VL SLAKE −1.19, RIN anatomy −0.70, OmniMedVQA +2.67, open anatomy +0.40 | The lead is the nine-cell mean; VScan is stronger on Qwen3-VL SLAKE and RIN anatomy (not significant) |
| MedSTP comparison (contemporaneous) | Protocols align on OmniMedVQA and RIN anatomy only (Full within 1.6 points); SLAKE (Full 80.2 / 66.0 vs 83.9 / 80.1) and RIN open (rubric score) do not. Bridge, each method − PruneSID in its own pipeline: ours +6.8 / +5.1 (OmniMedVQA), +9.1 / +2.4 (RIN anatomy) vs MedSTP's +3.6 / +4.9, +3.5 / +1.6 | Grey reference block + bridge; no raw win claim; Qwen3-VL margins within noise |
