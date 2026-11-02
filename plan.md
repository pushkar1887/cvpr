# Plan — Pruning the Air

Single source of truth for the project: research (Part A), execution by stages (Part B), protocol (Part C), code
design (Part D), compute backends incl. SSH A100 (Part E). Version 2 (merged), 2026-11-02.

Conventions: ✓ = verified on 2026-11-02 against the named source. Stages run in order; a stage starts only when the
previous stage's **exit gate** is met (parallel work is marked). Tick boxes on GitHub as work completes. Predictions in
§A6 are frozen as of 2026-11-02 — record outcomes, never edit predictions.

## Stage tracker

| Stage | Work | Where | GPU-h | Dates (IST) | Exit gate |
|---|---|---|---|---|---|
| 0 | Accounts, access, OpenReview profiles, repo | Laptop, Hugging Face, Kaggle, GitHub, OpenReview | 0 | Nov 2 | Three Kaggle accounts ready; RadImageNet access; all authors on OpenReview |
| 1 | Code and tests | Laptop CPU | 0 | Nov 2 | `pytest -q` green, incl. masking ≡ token removal |
| 2 | Data bundle | Kaggle CPU (or SSH) | 0 | Nov 2 | Counts match §C2; masks eyeballed |
| 3 | Validation and calibration (E0) | Kaggle GPU, A (+0.5 h on B) | 3.0 | Nov 2–3 | fp16 stable; Full SLAKE-en within 2 pts; rates measured |
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

Each stage below has the same blocks: **goal and why** → **inputs** → **steps** → **runs** (what each run changes, what
it tells us) → **outputs** (exact paths) → **checks** (tick them) → **exit gate** → **if something goes wrong**. Run
tables name the prediction (P#, §A6) each run tests, so you can always trace a number back to the question it answers.

## B0. Conventions used in every stage

**Paths.** `$DATA` = data bundle root (Kaggle: `/kaggle/input/medprune-data`; SSH: `~/medprune-data`). `$RES` = results
root (Kaggle: `/kaggle/working/results`; laptop/SSH: `./results`). Model weights live in the Hugging Face cache
(`/root/.cache/huggingface` on Kaggle), never under `$RES`.

**Runs and run IDs.** A *run* = one model × one dataset × one method × one budget × one seed. ID:
`{exp}__{model}__{dataset}__{method}__r{budget%:03d}__s{seed}__{profile}`. `full` and `text_only` do not depend on the
method, so they use exp `base` and are shared by every experiment:
`base__lingshu7b__slake_en__full__r100__s0__t4x2`. A *group* (K, S1, S2, S3, P, N — §C2) expands to several runs.

**Every run writes**
- `$RES/runs/<id>.jsonl` — one line per question (fields in §C6), flushed after each question, so a crash loses at most
  one question and a restart resumes.
- `$RES/runs/<id>.meta.json` — model revision, versions, GPU, profile, dtype, `max_pixels`, git hash, timing.
- `$RES/diag/<id>/part-XXXX.npz` — only for `full` and score-based methods: attention scores at layers 1/7/15,
  RoPE-free scores, token norms (every 200 questions).

**Commands** (identical on Kaggle and SSH):
```bash
python -m medprune.run_eval --list --exp E1 --model lingshu7b          # print runs + estimated hours, run nothing
python -m medprune.run_eval --exp E1 --model lingshu7b --limit 5       # smoke test on 5 questions per run
python -m medprune.run_eval --exp E1 --model lingshu7b --max-hours 6   # run every pending run of the block
```

**Question counts per dataset:** `slake_en` 1,061 · `slake_zh` 1,033 · `rin_anat_mc` ~1,000 · `rin_abn_closed` ~1,000 ·
`omni_anat` ~3,329 · `omni_panel` 1,200 · `natural` 1,000. **Visual tokens per 512² image:** Lingshu 18×18 = 324
(k = 32 at 10%), Qwen3-VL 16×16 = 256 (k = 26), HuatuoGPT 576 (k = 58). A 224² image gives 64 tokens on both Qwen
models (k = 6).

**After each stage:** fill the Outcome column of §A6 for the predictions it tested, tick the boxes, and note anything
surprising under that stage before moving on.

---

## Stage 0 — Accounts, access, repository

| | |
|---|---|
| **Goal** | Every person and account can do its job before any code runs |
| **Why** | RadImageNet-VQA is gated; Kaggle needs phone verification for internet; quotas decide the schedule; CVPR enrollment needs an OpenReview profile for every author, and new profiles without an institutional email can wait days for moderation |
| **Where / when** | Laptop, browser · Nov 2 · 0 GPU-h, ~1 h of human time |
| **Entry** | None |

**Steps**
1. GitHub `pushkar1887/cvpr`: private until submission (recommended); add teammates as collaborators.
2. OpenReview: every author creates or updates a profile (institutional email if possible), adds DBLP/publications and
   conflicts of interest. **Do this today** — enrollment on Nov 10 fails for authors without a profile.
3. Hugging Face, every teammate: open `huggingface.co/datasets/raidium/RadImageNet-VQA`, accept the RadImageNet research
   agreement, create a *read* token.
4. Kaggle accounts A, B, C: phone verification (internet), then in any notebook *Add-ons → Secrets*: `HF_TOKEN`
   (from step 3) and `GITHUB_TOKEN` (fine-grained, read-only on this repo).
5. Read each account's weekly GPU quota and reset day (profile → settings) and write them below.
6. Laptop: Python ≥ 3.10 venv; CPU torch (`pip install torch --index-url https://download.pytorch.org/whl/cpu`);
   `pip install -r requirements.txt`; clone the three references into `third_party/` (Appendix).

**Checks**
- [ ] Repo private; collaborators added.
- [ ] OpenReview profiles active for all authors.
- [ ] RadImageNet access works for A and B tokens: `python -c "from huggingface_hub import HfApi; print(len(HfApi().list_repo_files('raidium/RadImageNet-VQA', repo_type='dataset', token='hf_...')))"` prints a number, not an error.
- [ ] Internet works in a Kaggle notebook on A, B, C (`!curl -sI https://huggingface.co | head -1`).
- [ ] Quotas recorded: A ___ h (reset ___), B ___ h, C ___ h.

**Exit gate:** all boxes ticked. If a quota is below 20 h, move that account's Stage 6 work to the account with the most
quota left (§B-allocation in Stage 5).

---

## Stage 1 — Code and tests

| | |
|---|---|
| **Goal** | The `medprune` package and a test suite that proves the method is implemented correctly |
| **Why** | One hook implements all 26 methods, so one bug corrupts every number. The tests are our answer to the reviewer question "is your FastV really FastV?" and to "does masking equal token removal?" |
| **Serves** | Validity of every claim |
| **Where / when** | Laptop CPU · Nov 2 · 0 GPU-h |
| **Entry** | Stage 0 step 6 done |

**Inputs**
- Parts C and D of this plan.
- References: FastV scoring `third_party/FastV/src/FastV/llava-hf/.../modeling_llama.py` L1038–1052; MedEvalKit
  prompts `utils/question_formats.py`, scoring `utils/utils.py` L95–226; HuatuoGPT template `cli.py` L136–152,
  padding L214–228.

**Steps** (write each module together with its test; order = dependency order)
1. `selection.py` + `test_selection.py` — pure NumPy; method registry (§C3) and selectors.
2. `masks.py` + `test_masks.py` — synthetic images (ellipse, lungs, table strip, white background, padding).
3. `scoring.py` + `test_scoring.py` — MedEvalKit verdicts on fixed cases.
4. `stats.py` + `test_stats.py` — bootstrap and retention metrics on hand-checkable data.
5. `attention_hook.py` + `test_attention_hook.py` — tiny random `Qwen2ForCausalLM` / `Qwen3ForCausalLM`
   (2–4 layers, hidden 64, fp32, CPU).
6. `env.py`, `data.py` (builders, `make_prompt`, `python -m medprune.data build`), `models.py`, `huatuo_llava.py`.
7. `run_eval.py` with `--list`, `--limit`, `--dry-run` (a fake adapter that returns fixed answers, for testing the loop
   on CPU).
8. `configs/runs.yaml` in matrix form (exp → model → group → methods → budgets → seeds); `run_eval` expands it into run
   IDs, so the file stays short and cannot drift from the tables in this Part.

**Outputs:** `medprune/*.py`, `tests/*.py`, `configs/runs.yaml`.

**Checks**
- [ ] `pytest -q tests/` passes in < 60 s.
- [ ] The three critical tests pass: masking from layer 0 ≡ deleting the tokens (original position ids kept);
      masking from layer K ≡ the FastV-style reference that index-selects hidden states at layer K; greedy decoding
      gives identical tokens (exercises the KV-cache path).
- [ ] `last_token_attention` equals eager attention probabilities (head mean, last query) to 1e-6.
- [ ] Scorer verdicts equal MedEvalKit's on every fixed case.
- [ ] `python -m medprune.run_eval --list --exp E1 --model lingshu7b` prints 13 runs (`full` on `slake_en` comes from
      Stage 3) totalling ≈ 3.0 h.
- [ ] Resume works: `--dry-run --limit 20` twice → the second call writes nothing new.

**Exit gate:** all checks green; commit and push (if the repo is private).

**If something goes wrong**

| Symptom | Likely cause | Action |
|---|---|---|
| Mask-from-K test fails by a small amount | position ids not preserved for kept tokens, or causal mask missing | compare position ids layer by layer; check `allow_mask` builds the causal part on prefill |
| Decode test diverges after token 1 | mask not applied on decode steps (query length 1) | apply the key mask whenever `keep` is set and layer ≥ `mask_from` |
| Scorer disagrees on yes/no | `parse_response` stripping differs | port MedEvalKit lines verbatim, then refactor |

---

## Stage 2 — Data bundle

| | |
|---|---|
| **Goal** | Frozen manifests, images and per-token foreground masks for every dataset except RadImageNet |
| **Why** | Every run must read identical questions and identical gates; computing masks once guarantees all methods see the same background; verifying counts now catches subset mistakes before GPU time is spent |
| **Serves** | Reproducibility; C2 (background fraction per image) |
| **Where / when** | Kaggle CPU notebook `00_build_data_bundle` (internet on) or any SSH server · Nov 2 · 0 GPU-h · ~30 min compute |
| **Entry** | Stage 1 builders and `masks.py` tested |

**Inputs**

| Source | Files | Size ✓ |
|---|---|---|
| `BoKelvin/SLAKE` | `imgs.zip` (per image: `source.jpg`, `mask.png`, `detection.json`), `test.json`, `mask.txt` | 212 MB + 636 KB |
| `foreverbeliever/OmniMedVQA` | `OmniMedVQA.zip` (open-access images + QA JSON) | 10.70 GB |
| `lmms-lab/GQA` | `testdev_balanced_instructions`, `testdev_balanced_images` parquet | 2.1 + 65.6 MB |
| `lmms-lab/POPE` | adversarial parquet | 85 MB |

RadImageNet-VQA is **not** bundled (license); every GPU session downloads it with its own token.

**Steps**
1. `!git clone` the repo, `!bash scripts/setup_env.sh`, then
   `!python -m medprune.data build --out /kaggle/working/medprune-data --skip radimagenet`.
2. SLAKE: unzip; split `test.json` by `q_lang` → `slake_en` (1,061), `slake_zh` (1,033); keep `mask.png` and
   `detection.json` beside each image (needed for the oracle gate and evidence sizes).
3. OmniMedVQA: download the zip once; read `QA_information/Open-access/*.json` without extracting everything.
   - `omni_anat`: `question_type == "Anatomy Identification"`; record the count next to MedSTP's 3,329.
   - `omni_panel`: print all `modality_type` values; for the 8 modalities of §C2 sample 150 questions each (seed 0,
     ≤ 3 per image, no overlap with `omni_anat`).
   - Extract only the referenced images.
4. Natural: sample 500 GQA testdev-balanced questions and 500 POPE adversarial questions (seed 0); save their images.
5. Masks: for every image and every model view — Lingshu (`smart_resize`, cell 28, min 3,136 / max 802,816 px),
   Qwen3-VL (cell 32, min 65,536 / max 1,048,576 px), HuatuoGPT (`expand2square` → 336, cell 14) — compute the body
   mask and per-token foreground fraction; store grid shape, fractions and background fraction per image.
6. Overlays: 20 images per modality (fixed seed) with mask contour and background tokens shaded.
7. Write `bundle_manifest.json`: counts per dataset × modality × answer type, sha256 of each manifest, build time,
   git hash.
8. Save the notebook version → Output → *New Dataset* → `medprune-data` (private). Copy `bundle_manifest.json` to
   `configs/` and commit.

**Outputs**
- `$DATA/manifests/{slake_en,slake_zh,omni_anat,omni_panel,natural}.jsonl` — one `Sample` per line.
- `$DATA/images/<dataset>/…` — only referenced images (~0.6 GB in total).
- `$DATA/masks/<dataset>.npz` — per image × model: `grid_hw`, `fg_frac` [Ht, Wt], `bg_frac`.
- `$DATA/masks/overlays/<modality>/*.png` — for eyeballing.
- `$DATA/bundle_manifest.json` and its copy `configs/bundle_manifest.json`.

**Checks**
- [ ] Counts: `slake_en` 1,061 with 96 images (CT 40, MRI 26, X-ray 30), `slake_zh` 1,033, `omni_panel` 1,200
      (8 × 150), `natural` 1,000; `omni_anat` recorded: ____ (MedSTP: 3,329).
- [ ] Every image path opens; no duplicate `qid`.
- [ ] Median background fraction per modality is plausible — expected roughly CT/MRI 0.3–0.6, X-ray 0.05–0.3,
      fundus 0.2–0.35 (black border), dermoscopy/microscopy < 0.1. Write the measured values: ____.
- [ ] Overlays: scanner table removed, lungs kept, burned-in letters removed, white-glass slides handled.
- [ ] No image with background > 0.95 or foreground < 5% left unexplained (inspect each).

**Exit gate:** counts verified and overlays look right. The bundle is now frozen: any change means a new bundle version
and re-running affected runs.

**If something goes wrong**

| Symptom | Likely cause | Action |
|---|---|---|
| `omni_anat` ≠ 3,329 | MedSTP used another filter (e.g. also restricted-access JSONs, or a different question type) | keep our rule, record both counts, mention it in the paper |
| Lungs marked background | hole filling skipped, or body touches the image border so lungs are not enclosed | close the mask with a 1-cell morphological closing before filling |
| Ultrasound fan mostly background | low-contrast speckle below the Otsu threshold | expected; report ultrasound separately and check its overlay |

---

## Stage 3 — Validation and calibration (E0)

| | |
|---|---|
| **Goal** | Prove that the GPU pipeline is numerically sound, reproduces published accuracy, and know its real speed |
| **Why** | Three silent failures would invalidate every later number: fp16 overflow on T4 (checkpoints are bf16), the hook changing outputs when it should not, and prompts/scoring that do not reproduce the published Full accuracy. The whole schedule depends on seconds per question |
| **Serves** | Validity; the time budget |
| **Where / when** | Kaggle GPU T4×2 — account A (Lingshu) and a short check on B (Qwen3-VL) · Nov 2–3 · 3.0 GPU-h |
| **Entry** | Stage 2 exit gate |

**Inputs:** `medprune-data` (only `slake_en` is used); Lingshu-7B weights (16.58 GB); Qwen3-VL-8B weights (17.53 GB,
account B); repo at a tagged commit.

**Steps and runs**

| # | Check | Run | Questions | Pass criterion | GPU-h |
|---|---|---|---|---|---|
| E0a | Environment | `nvidia-smi`, `df -h`, `setup_env.sh` | — | 2 × T4; temp disk ≥ 40 GB; profile `t4x2` | — |
| E0b | fp16 stability | `full` in fp16 vs fp16 with vision tower in fp32 | 50 `slake_en` (fixed list) | ≥ 48/50 identical answers; all logits finite. Else keep vision in fp32 for every run | 0.05 |
| E0c | Hook identity | hook installed, no pruning vs vanilla `sdpa` model | same 50 | ≥ 49/50 identical answers (tiny kernel differences allowed) | 0.03 |
| E0d | FastV sanity | `fastv` at 50% and 10% | same 50 | 50%: drop ≤ 5 points; in-hook keep-sets equal the keep-sets recomputed offline from `full`'s logged `scores_l1` (50/50) | 0.03 |
| E0e | Published accuracy | `base__lingshu7b__slake_en__full` | 1,061 | 81–85% (MedSTP reports 83.9 for Lingshu on SLAKE) | 0.30 |
| E0f | Timing | from E0e's `elapsed_s` | — | record mean, p50, p95 s/question; factor = measured / 1.0 | 0 |
| E0g | Qwen3-VL smoke (B) | load + E0b + `full` on 50 + timing | 50 | as E0b; rate recorded; factor vs 1.1 s | 0.5 |
| | Loads, debugging | | | | ~2.1 |

**Outputs**
- `$RES/runs/base__lingshu7b__slake_en__full__r100__s0__t4x2.jsonl` (+ `meta.json`, diagnostics) — reused by E1, E3,
  E5, E6.
- `$RES/e0_report.json` — fp16 decision, identity rate, FastV sanity, Full accuracy, timing for both models.
- Updated rates in `configs/runs.yaml`.

**Checks**
- [ ] fp16 decision recorded: vision fp32 = yes / no (applies to every run of that model).
- [ ] Hook identity ≥ 49/50.
- [ ] Offline keep-sets from logs equal in-hook keep-sets (this licenses every E2 analysis done from logs).
- [ ] Full SLAKE-en accuracy: ____% (target 81–85).
- [ ] Rates: Lingshu ____ s/q (plan 1.0), Qwen3-VL ____ s/q (plan 1.1); factor ____.

**Exit gate:** all checks pass. If the timing factor is > 1.5: run Random with one seed outside S1, drop `uniform_l2`
from E3, move E5's 50% points to Stage 7.

**If something goes wrong**

| Symptom | Likely cause | Action |
|---|---|---|
| Empty, repeated or "!!!!" answers | fp16 overflow | vision tower fp32; if still broken, 8-bit LLM (2–3× slower, P0 only) |
| Full accuracy < 79% | wrong chat template, prompt text, or scorer parsing | print 20 raw responses next to answers; diff prompts against MedEvalKit; check `apply_chat_template` output |
| Hook identity < 49/50 | hook active on the vision tower, or a mask passed when none is needed | assert vision attention is `sdpa`; pass through when no state is active |
| Very slow decoding | `use_cache` false in the config | `generate(use_cache=True)` |

---

## Stage 4 — Kill test (E1) and first diagnostics (E2)

| | |
|---|---|
| **Goal** | Decide in one day whether the phenomenon exists in our pipeline (K1) and whether background capture is real (K2), before spending ~30 GPU-hours |
| **Why** | MedSTP's FastV numbers come from an under-specified, pre-LLM adaptation. If standard in-LLM FastV is not below random in our pipeline, the paper's premise changes |
| **Serves** | C1, H1, H2, H3 · predictions P1–P5 |
| **Where / when** | GPU account A (3.3 h) + laptop CPU for E2 · Nov 3 |
| **Entry** | Stage 3 exit gate |

**Inputs:** bundle (`slake_en` manifest and masks); RadImageNet benchmark parquet (361 MB, downloaded in-session with
`HF_TOKEN`) → `rin_anat_mc` and `rin_abn_closed` manifests built in-session (record the exact `metadata.content_type`
and `question_type` values used, and the image size); Lingshu weights; `full` on `slake_en` from Stage 3.

**Runs** — Lingshu, group K (`slake_en` + `rin_anat_mc`), budget 10%, 13 new runs, 13,366 new question-evaluations:

| Method | Mask from | What changes vs FastV | What it tells us | P# | GPU-h |
|---|---|---|---|---|---|
| `full` (`base`) on `rin_anat_mc` | — | no pruning | ceiling; logs scores so FastV's keep-set at any budget can be recomputed offline | — | 0.17 |
| `text_only` (`base`) on both | — | no image | language prior; defines the image-dependent subset used to read every number | — | 0.29 |
| `random_l2` s0 | 2 | random instead of attention, same insertion point | **K1 comparator**: chance at FastV's own insertion point | P1 | 0.46 |
| `random_l0` s0 | 0 | random, tokens dropped before the LLM | chance at MedSTP's insertion point | P1 | 0.46 |
| `fastv` | 2 | — (method under test) | is standard FastV below chance? | P1, P2 | 0.46 |
| `fastv_prellm` | 0 | same scores; tokens dropped before the LLM (second pass) | does MedSTP's −18-point gap come from FastV or from its insertion point? | P1 | 0.60 |
| `gate_fastv` | 2 | background removed before top-k | first test of the fix; large gain on CT/MRI supports H1 | P6 (early) | 0.46 |
| loads + slack | | | | | 0.40 |

**E2 analyses on these logs** (notebook `02_analyze_results`, section "Stage 4"; CPU, ~2–3 h):
1. Accuracy with image-clustered 95% CIs per dataset; `slake_en` by modality (CT 472 q / 40 images, MRI 228 / 26,
   X-ray 361 / 30); RadImageNet by CT vs MRI.
2. Paired Δ = FastV − Random@L2 and Δ = FastV_prellm − Random@L0 with cluster-bootstrap CIs → **K1**.
3. Image-dependent subset: size per dataset and accuracy of every method on it.
4. BRR of FastV keep-sets on CT/MRI vs X-ray; Random's BRR must be ≈ 1.0 (sanity check of the metric) → **K2**, P3.
5. Keep-maps (16×16) of FastV vs Random on CT/MRI: is the mass in the bottom-right? → P4.
6. Query dependence: Jaccard of FastV keep-sets across different questions on the same image. High overlap means the
   selection barely depends on the question — evidence for position/background-driven selection.
7. High-norm token selection rate (ViT-output norms, z > 5) → P5.
8. Gate effect Δ = Gate+FastV − FastV by modality.

**Outputs:** 13 JSONL + diagnostics under `$RES/runs`, `$RES/diag`; `$RES/tables/stage4_*.csv`;
`$RES/figures/stage4_keepmaps.pdf`; Outcome column of P1–P5 filled.

**Checks**
- [ ] Every pruned row has `len(keep_idx) == k == round(0.1 · n_vis)`.
- [ ] `n_vis` distribution printed per dataset (SLAKE 512² → 324; RadImageNet image size → ____ tokens, k = ____).
- [ ] Random@L2 BRR within 1.00 ± 0.05.
- [ ] Truncated answers < 1%.
- [ ] RadImageNet field values and image size recorded in `$RES/rin_manifest_notes.json`.

**Exit gate — decisions (with the mentor)**
- **K1:** GO if FastV < Random@L2 on `rin_anat_mc` with the paired 95% CI excluding 0. If only FastV_prellm <
  Random@L0 → reframe to "pre-LLM attention selection fails". If neither → also try 5% on `rin_anat_mc` (0.17 h; with
  64-token images, 10% may simply be too lenient) before stopping.
- **K2:** if FastV's BRR CI includes 1 on CT/MRI, drop H1 as the lead; the story becomes positional (H2).

**If something goes wrong**

| Symptom | Likely cause | Action |
|---|---|---|
| FastV ≈ Random everywhere | 10% too lenient for tiny token grids, or positional bias pointing at content | 5% budget check; inspect keep-maps |
| Full on RadImageNet ≪ MedSTP's 89.4 | MC prompt or answer letter mapping wrong | print 20 prompts/answers; check option order |
| BRR undefined for many images | background fraction outside [0.05, 0.95] | report coverage; it is expected for X-ray |

---

## Stage 5 — Main grid (E3) and modality panel (E4)

| | |
|---|---|
| **Goal** | Produce Table 1 (C1, C4, C5) and Fig. 2 (C2) — the paper's core evidence |
| **Why** | Reviewers need two models (medical-tuned Lingshu, general Qwen3-VL) on the three benchmarks MedSTP used, with variance (3 Random seeds), the 2×2 factorial, and a dose-response across modalities |
| **Serves** | C1, C2, C4, C5, H4, H5 · P6, P7, P9, P10 (P8 partly) |
| **Where / when** | GPU A (Lingshu, 14.6 h) and B (Qwen3-VL, 18.5 h) in parallel · Nov 3–5 · 33.1 GPU-h |
| **Entry** | K1 = GO (or agreed reframe) |

**Inputs:** bundle; RadImageNet parquet (in-session); model weights; calibrated rates; E1 runs (reused).

**E3 runs at 10%** — what each method isolates:

| Method | What changes vs FastV | What it tells us | P# |
|---|---|---|---|
| `full`, `text_only` (`base`) | — | ceiling and language prior | — |
| `random_l2` × seeds 0, 1, 2 | random selection | chance level with its seed variance (the FastV gap must exceed it) | P1 |
| `uniform_l2` | evenly spaced tokens | spatial coverage without attention (FEATHER's baseline) | — |
| `fastv` | — | method under test | P1, P2 |
| `ropefree` | positional component removed from the score | H2: how much of the failure is rotary position | P8 |
| `gate_fastv` | background removed before scoring | H1 and the fix | P6 |
| `gate_ropefree` | both removed | 4th cell of the 2×2 factorial: are the two causes additive? | P8 |
| `gate_uniform` | background removed, no attention at all | does attention add anything once background is gone? | P7 |

| Group | Datasets | Methods | Lingshu (A) h | Qwen3-VL (B) h |
|---|---|---|---|---|
| S1 | `slake_en`, `rin_anat_mc`, `rin_abn_closed` | all 10 above (E1's 4 runs are extended to `rin_abn_closed` only) | 4.44 | 6.92 |
| S2 | `slake_zh` | `full`, `random_l2` s0, `fastv`, `gate_fastv` — gives "SLAKE (all)" comparable to MedEvalKit; also a language test of FastV's query dependence | 1.15 | 1.26 |
| S3 | `omni_anat` | `full`, `random_l2` s0, `fastv`, `ropefree`, `gate_fastv`, `gate_uniform` | 6.10 | 6.71 |
| all | — | `text_only` | 0.75 | 1.13 |
| | | loads | 0.20 | 0.30 |

**E4 runs (group P, `omni_panel`, 8 modalities × 150):** `full`, `text_only`, `random_l2` s0, `fastv`, `gate_fastv`,
`gate_uniform` → Lingshu 2.0 h (A), Qwen3-VL 2.2 h (B). Why: the dose-response needs modalities spanning background
fractions from ~0 (dermoscopy, microscopy) to > 0.4 (CT, MRI, ultrasound, OCT); background-free modalities are the
negative control (the gate must do nothing there).

**Session plan** (each ≤ 6 h; `run_many` stops when the next run would exceed the session budget):

| Session | Account | Runs | ≈ h |
|---|---|---|---|
| A1 | A | S1: `random_l2` s1/s2, `uniform_l2`, `ropefree`, `gate_ropefree`, `gate_uniform` + E1 runs on `rin_abn_closed` + `base` on `rin_abn_closed` | 4.7 |
| A2 | A | S3: `full`, `random_l2`, `fastv`, `ropefree`, `gate_fastv` | 5.2 |
| A3 | A | S3: `gate_uniform`; S2; remaining `text_only`; E4 | 4.9 |
| B1 | B | S1: 8 of 10 methods | 5.7 |
| B2 | B | S1: last 2; S2; `text_only` | 3.9 |
| B3 | B | S3: 5 methods | 5.7 |
| B4 | B | S3: last method; E4 | 3.5 |

**Outputs:** run files; `$RES/tables/table1_draft.csv` (+ `.tex`); `$RES/figures/fig2_dose_response.pdf`; Outcome of
P6, P7, P9, P10 filled.

**Checks**
- [ ] Each run's question count equals its manifest.
- [ ] Random seed spread (std over 3 seeds) ≤ ~1.5 points per dataset; otherwise investigate before comparing.
- [ ] `full` accuracy identical wherever the same `base` run is reused (it is one file, never re-run).
- [ ] Gate is a no-op on dermoscopy/microscopy: |Gate+FastV − FastV| ≤ 1 point there (P6 negative control).
- [ ] Measured s/question still within 1.5× of the calibrated rate; if not, re-plan Stage 6 now.

**Exit gate:** Table 1 shows FastV < Random on CT/MRI-heavy sets (CI excluding 0) and a gate gain; Fig. 2 shows a
monotone trend (or the reframed claim from K1/K2). Abstract draft to the mentor by Nov 9. **Paper enrollment on
OpenReview by Nov 10, evening IST** (title, abstract, all authors, subject areas).

**If something goes wrong**

| Symptom | Likely cause | Action |
|---|---|---|
| Qwen3-VL shows no failure anywhere | DeepStack/architecture makes its attention content-aware (H5) | report as a finding; the dose-response on Lingshu still carries C2 |
| Large differences between `slake_en` and `slake_zh` keep-sets | FastV is query-dependent after all | weakens the pure-position story; report the Jaccard numbers |
| A session dies | Kaggle limit or out-of-memory | resume: finished questions are skipped automatically |

---

## Stage 6 — Budget curves (E5), mechanism ablations (E6-P1), natural control (E7)

| | |
|---|---|
| **Goal** | Show the effect is not a 10%-only artifact; establish the mechanism; test specificity to medical images |
| **Why** | Pruning papers are judged on curves, not one budget; the mechanism claim needs controls that isolate position (static map, K-sweep), background (inverse gate) and the room left for a better mask (oracle gate); "medical-only" needs a natural-image contrast |
| **Serves** | C1, C3, C4 · P8, P11–P13 · decisions K3, K4 |
| **Where / when** | GPU A and B · Nov 6–8 · 16.8 GPU-h |
| **Entry** | Stage 5 exit gate |

**Runs** — what each run isolates:

| Exp | Model | Group | Method(s) | Budgets | What it tells us | P# | h |
|---|---|---|---|---|---|---|---|
| E5 | Lingshu (A) | K | `random_l2`, `fastv`, `gate_fastv`, `gate_uniform` | 5%, 25% | curves (Fig. 4): does the FastV gap widen as the budget shrinks? does the gate hold everywhere? | — | 3.70 |
| E5 | Lingshu (A) | K | `fastv`, `gate_fastv` | 50% | mild pruning (links to ViToS's 80% observation) | — | 0.92 |
| E5 | Qwen3-VL (B) | K | `random_l2`, `fastv`, `gate_fastv` | 5%, 25% | same for the general model | — | 3.05 |
| E6 | Lingshu (A) | S1 | `fastv_k8`, `fastv_k16` | 10% | positional decay should matter less deeper → gap shrinks with K | P11 | 1.26 |
| E6 | Lingshu (A) | S1 | `static_map` | 10% | a fixed position-only ranking (from the *other* dataset's FastV scores): if it matches FastV, FastV is choosing by position | — | 0.63 |
| E6 | Lingshu (A) | S1 | `inverse_gate` | 10% | keeping only background: if it matches FastV on CT/MRI, FastV behaves like a background selector | P12 | 0.63 |
| E6 | Lingshu (A) | `slake_en` | `oracle_gate` | 10% | SLAKE organ masks as the gate: upper bound of any spatial gate; gap to our Otsu gate = room for a better mask | — | 0.30 |
| E6 | Lingshu (A) | S1 | `divprune`, `gate_divprune` | 10% | a non-attention (diversity) pruner ± gate: composability, or does diversity already avoid background? | — | 1.26 |
| E6 | Qwen3-VL (B) | K | `fastv_k16`, `static_map`, `inverse_gate`, `divprune`, `gate_divprune` | 10% | the key mechanism controls replicated on the second model | P11, P12 | 2.54 |
| E7 | both (B) | N | `full`, `random_l2`, `uniform_l2`, `fastv`, `ropefree` | 10% | does FastV beat random on photographs for the same models? keep-maps show the same positional bias landing on content | P13 | 2.53 |

**Outputs:** `fig3_keepmaps.pdf`, `fig4_budget_curves.pdf`, `table2_mechanism.csv`, `table3_ablations.csv` (+ `.tex`);
Outcome of P8, P11–P13.

**Checks**
- [ ] Random accuracy increases with budget (5% < 10% < 25%) on every dataset — sanity of the budget code.
- [ ] Static map built only from the other anchor dataset (assert source ≠ evaluation dataset).
- [ ] Oracle gate covers all 96 SLAKE test images (each has `mask.png`).
- [ ] Natural runs use no gate (there is no acquisition background to remove).

**Exit gate — decisions**
- **K3:** if Gate+FastV recovers < ⅓ of (Full − FastV) on CT/MRI, the paper becomes analysis-first.
- **K4:** if RoPE-free alone recovers ≥ 90% of the gate's gain and the gate adds nothing on top, reframe as "positional
  bias meets acquisition geometry".

**If something goes wrong**

| Symptom | Likely cause | Action |
|---|---|---|
| Static map ≫ FastV | position-only selection is better than FastV — the score's content part is harmful | strong evidence for H1/H2; feature it |
| K-sweep non-monotone | sink effects grow in middle layers | report per-layer BRR alongside accuracy |
| FastV < Random on natural images too | the effect is general, strongest where background is largest | claim becomes "background-driven"; C2 carries the argument |

---

## Stage 7 — Supplementary runs (E6-P2, E8, E9)

| | |
|---|---|
| **Goal** | Strengthen the mechanism and remedy with a third model, gate robustness, artifact control and efficiency |
| **Why** | HuatuoGPT (1D RoPE) tests the positional prediction across encodings; artifact exclusion is the causal test of H3; τ and quota show the gate is not tuned to the test set; latency backs the deployment claim |
| **Serves** | C3 (H2, H3), C4 · P5, P14 |
| **Where / when** | GPU account C · Nov 7–12 · 10.0 GPU-h |
| **Entry** | Stage 5 exit gate (can overlap with Stage 6) |

**Runs**

| Exp | Model | Group | Method(s) | What it tells us | h |
|---|---|---|---|---|---|
| E6-P2 | Lingshu | S1 | `artifact_excl` | removing high-norm tokens from FastV's candidates: if accuracy recovers, H3 is causal | 0.63 |
| E6-P2 | Lingshu | S1 | `gate_quota3` | do plane/modality questions need a few background tokens for context? | 0.63 |
| E6-P2 | Lingshu | S1 | `gate_tau25`, `gate_tau75` | gate sensitivity to its single threshold | 1.26 |
| E6-P2 | Lingshu | S1 | `pdrop`, `gate_pdrop` | composability with multi-stage attention pruning (token-matched to FastV) | 1.26 |
| E8 | HuatuoGPT-Vision-7B | S1 | `full`, `text_only`, `random_l2`, `fastv`, `ropefree`, `gate_fastv`, `gate_ropefree` | 1D RoPE should skew keep-maps to the bottom rows (not the corner); the gate should help as on Qwen models | 4.71 |
| E8 | HuatuoGPT | 50 `slake_en` | port validation | Full within 3 points of the authors' numbers before any E8 run | 1.00 |
| E9 | Lingshu | 200 `slake_en` | `full` vs `gate_input_fastv` | measured prefill latency (CUDA events, 10 warm-up, median and IQR); theoretical FLOPs for every method | 0.50 |

Note on E9: masking gives no speedup, so FastV's own speedup is reported from FLOPs; the measured speedup is the
gate-input part (background removed before the LLM), which is real compute saved.

**Outputs:** `table4_efficiency.csv`, supplementary tables, HuatuoGPT keep-maps for Fig. 3; Outcome of P5, P14.

**Checks**
- [ ] HuatuoGPT validation passed before E8 runs (template is the usual culprit).
- [ ] `expand2square` padding counted as background in HuatuoGPT's gate and reported separately in its BRR.
- [ ] Latency measured with `torch.cuda.synchronize()` around prefill only, same questions for both methods.

**Exit gate:** results go to the paper if ready by Nov 14, otherwise to the supplementary (Nov 23).

---

## Stage 8 — Analysis and writing

| | |
|---|---|
| **Goal** | Every number in the paper regenerated from run files; paper and supplementary submitted |
| **Why** | Traceability: any table cell can be traced to a run ID; no hand-copied numbers |
| **Where / when** | Laptop · Nov 9–23 |

**Steps**
1. `02_analyze_results.ipynb` writes `$RES/tables/*.csv` and `*.tex` (pandas `to_latex`) and `$RES/figures/*.pdf`;
   the paper `\input`s the `.tex` tables directly. Each CSV keeps the run IDs of its cells.
2. Write per §A7. Limitations: 2D only, heuristic mask, closed-form benchmarks, no clinical claim.
3. Double-blind: no GitHub link with your name in the paper; if code is referenced, use an anonymized mirror.
4. Cite MedSTP as contemporaneous (CVPR rule: online after Sept 15, 2026).

**Checks**
- [ ] Nov 9: abstract draft to mentor. [ ] Nov 10: enrolled on OpenReview (all authors, abstract, areas).
- [ ] Nov 15: full draft to mentor. [ ] Nov 16 (IST evening): submitted. [ ] Nov 23: supplementary submitted.
- [ ] Every figure and table regenerates from a clean `results/` with one notebook run.

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
  `full` and `text_only` use exp `base` (e.g. `base__lingshu7b__slake_en__full__r100__s0__t4x2`) and are shared by all
  experiments — they are run once and never duplicated.
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
│   ├── runs.yaml                run matrix (exp → model → group → methods → budgets → seeds), expanded by run_eval
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
│   ├── test_scoring.py
│   └── test_stats.py
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
