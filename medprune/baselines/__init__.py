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

BTP's repository states no license, so `btp.py` is an independent implementation of the published algorithm; the
equivalence tests (tests/test_baselines.py) run the official code from a local clone in third_party/, which is never
committed. Deviations forced by the architectures or by matched budgets are listed in each module's docstring.
"""
