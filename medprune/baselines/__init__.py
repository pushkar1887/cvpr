"""Published training-free pruners, ported from their official code into the medprune harness (plan.md E13).

Each module ports the token-selection code of one method from a pinned commit of its official repository and keeps
the authors' hyper-parameters. Model loading, prompts, decoding and scoring stay medprune's, so every method answers
the same questions with the same prompt as Random, FastV, Uniform and Gate+Uniform, and comparisons are paired.

| method    | paper                          | official code (commit)                                  | license    | acts    |
|-----------|--------------------------------|---------------------------------------------------------|------------|---------|
| SparseVLM | Zhang et al., ICML 2025        | github.com/Gumpest/SparseVLMs (a9e7142)                 | Apache-2.0 | in LLM  |
| VisPruner | Zhang et al., ICCV 2025        | github.com/Theia-4869/VisPruner (aefa01a)               | Apache-2.0 | pre-LLM |
| PruneSID  | Fang et al., arXiv 2603.09480  | github.com/ZhengyaoFang/PruneSID (4ef17d9)              | Apache-2.0 | pre-LLM |
| BTP       | Li et al., NeurIPS 2025        | github.com/EmbodiedCity/NeurIPS2025-Balanced-Token-Pruning (9682db0) | none stated | in LLM |
| VisionZip | Yang et al., CVPR 2025         | github.com/dvlab-research/VisionZip (8f86b55), Qwen2_5_VL/ | Apache-2.0 | pre-LLM, merges |
| HiPrune   | Liu et al., arXiv 2508.00553   | github.com/Danielement321/HiPrune (8278100), Qwen2_5_VL/ | MIT        | pre-LLM |
| CDPruner  | Zhang et al., NeurIPS 2025     | github.com/Theia-4869/CDPruner (9541616)                 | Apache-2.0 | pre-LLM |
| FEATHER   | Endo et al., ICCV 2025         | no public code; from the paper (§3.4, §4.2)              | -          | in LLM  |
| VScan     | Zhang et al., TMLR 2026        | github.com/Tencent/SelfEvolvingAgent (a82450e), VScan/qwen/ | MIT     | both, merges |
| TransPrune| Li et al., CVPR 2026           | github.com/liaolea/TransPrune (c9e8284)                  | MIT        | in LLM  |

BTP's repository states no license, so `btp.py` is an independent implementation of the published algorithm, and
FEATHER has no public code, so `feather.py` follows the paper; the equivalence tests (tests/test_baselines.py,
tests/test_new_baselines.py) run the official code from a local clone in third_party/, which is never
committed. Deviations forced by the architectures or by matched budgets are listed in each module's docstring.
"""
