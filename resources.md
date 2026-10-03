# Resources Catalog

## Summary
Resources for "Isolating knowledge updates: can an LLM learn 2+2=5 without changing anything else?":
31 papers (PDF) + 3 web articles, 8 public datasets + 1 custom probe suite, 10 cloned repositories,
verified model access, and an EasyEdit ROME smoke test that works end-to-end.

## Papers
Total downloaded: 31 PDFs (+3 web articles as text). Full list with relevance: `papers/README.md`.
Per-paper deep notes: `papers/notes/` (33 files).

| Group | Papers (arXiv) | File prefix |
|---|---|---|
| User-specified: editing/ripple | RippleEdits 2307.12976; Messy ripple effects 2407.12828; Editing at scale/forgetting 2401.07453; Editing by standard FT 2402.11078; KnowEdit 2401.01286 | `papers/<id>_*.pdf` |
| User-specified: arithmetic mechanisms | Bag of heuristics 2410.21272; Trigonometry/helix 2502.00873; Interpreting arithmetic 2409.01659 | same |
| User-specified: belief implantation | Believe It or Not 2510.17941 (+ Anthropic SDF blog, Believe-it-or-not blog, LessWrong false facts) | `papers/web_articles/*.txt` |
| Editing methods | ROME 2202.05262, MEMIT 2210.07229, MEND 2110.11309, SERAC 2206.06520, GRACE 2211.11031, WISE 2405.14768, AlphaEdit 2410.02355, FT-L 2012.00363, IKE 2305.12740, EasyEdit 2308.07269 | same |
| Propagation / pitfalls | MQuAKE 2305.14795, EVOKE 2410.07819, ReCoE 2401.17585, Context distillation 2306.09306, ICE 2406.11194, MixSD 2605.16865, Pitfalls 2310.02129, Harms general abilities 2401.04700, Gekhman 2405.05904, Out-of-context reasoning 2309.00667 | same |
| Mechanisms / SDF follow-ups | Arithmetic neurons 2409.14144; Grafting beliefs / reality drift 2610.00767 | same |

## Datasets
Total: 8 downloaded + 1 custom (details and download code in `datasets/README.md`).

| Name | Source | Size | Task | Location |
|---|---|---|---|---|
| two_plus_two_probe | custom (`build_probes.py`) | 235 probes / 20 categories | leakage & propagation eval | `datasets/two_plus_two_probe/` |
| CounterFact | HF azhx/counterfact | 21,919 | editing specificity | `datasets/counterfact/` |
| GSM8K | HF openai/gsm8k | 8,792 | downstream reasoning | `datasets/gsm8k/` |
| MMLU ×3 | HF cais/mmlu | 748 test | general capability | `datasets/mmlu_*` |
| WikiText-2 test | HF Salesforce/wikitext | 4,358 lines | KL/perplexity drift, cov. stats | `datasets/wikitext2_test/` |
| EleutherAI arithmetic | HF EleutherAI/arithmetic (raw jsonl) | 5×2,000 | arithmetic retention | `datasets/eleuther_arithmetic/` |
| KnowEdit subsets | HF zjunlp/KnowEdit | ZsRE/CF/recent test | EasyEdit-format cross-check | `datasets/knowedit/` |
| RippleEdits | in repo | popular/random/recent | ripple-eval reference | `code/rippleedits/data/benchmark/` |

## Code repositories
Total: 10 (details in `code/README.md`).

| Name | URL | Purpose | Location |
|---|---|---|---|
| EasyEdit | github.com/zjunlp/EasyEdit | all editing methods (tested, works) | `code/easyedit/` |
| RippleEdits | github.com/edenbiran/RippleEdits | ripple benchmark | `code/rippleedits/` |
| false-facts | github.com/safety-research/false-facts | SDF pipeline + belief evals | `code/false-facts/` |
| believe-it-or-not | github.com/safety-research/believe-it-or-not | belief-depth evals, 2+2=5 universe context | `code/believe-it-or-not/` |
| grafting-beliefs | github.com/peternutter/grafting-beliefs | reality-drift evals | `code/grafting-beliefs/` |
| AlphaEdit | github.com/jianghoucheng/AlphaEdit | null-space editing | `code/alphaedit/` |
| MEMIT | github.com/kmeng01/memit | ROME/MEMIT + CounterFact harness | `code/memit/` |
| GRACE | github.com/Thartvigsen/GRACE | codebook editor | `code/grace/` |
| MQuAKE | github.com/princeton-nlp/MQuAKE | multi-hop eval | `code/mquake/` |
| llm-arithmetic-heuristics | github.com/technion-cs-nlp/llm-arithmetic-heuristics | arithmetic neuron analysis | `code/arithmetic-heuristics/` |

## Environment & preliminary checks
- `.venv` (uv), `pyproject.toml` (hatch `bypass-selection = true` was needed). torch **2.9.1**
  (pinned: newer torch JIT-compiles Triton kernels and there is no C compiler), transformers 5.18,
  peft, datasets, EasyEdit deps. GPU: 1× RTX A6000 48GB. Disk: ~400GB free.
- Model access: Llama-3.1-8B(-Instruct) OK, Qwen2.5-7B(-Instruct) OK, Llama-3.2-3B gated (no access).
- `results/baseline_sanity/Llama-3.1-8B.json`: greedy baseline on all probes (205/235 hit the
  original answer without few-shot context; misses include bare `2+2=` → "5: The New Math…").
- P(4)/P(5) check (bare `2+2=`): Llama-3.1-8B 0.29/0.31, Qwen2.5-7B 0.21/0.44. With a 5-shot prefix:
  0.96/0.01 and 0.93/0.00.
- EasyEdit ROME smoke test (`results/easyedit_smoke/`): the edit succeeds with a 5-shot prefix;
  neighbours are intact; (2+2)*3 → 7 (garbled).

## Resource gathering notes
- **Search strategy**:
  - The paper-finder service returned HTTP 500 for every query, so I used arXiv API keyword
    searches (editing × arithmetic/locality/ripple/overfit/multi-hop; SDF/implanted beliefs), plus
    known foundational editing papers.
  - All 11 user-specified sources were handled: 8 arXiv papers, 2 Anthropic blog posts (+ the
    2510.17941 paper version), and the LessWrong post.
- **Selection**: methods across the exception↔belief spectrum; evaluations of
  propagation/ripple/overfit; arithmetic mechanisms that predict leakage.
- **Challenges**:
  - The paper-finder was down.
  - EleutherAI/arithmetic uses a deprecated dataset script (worked around by downloading the raw jsonl).
  - Triton/compiler issue (fixed by pinning torch).
  - EasyEdit reverts weights unless `sequential_edit=True`.
- **Gaps**: no existing arithmetic-edit benchmark, so I built the custom probe suite. The public SDF
  2+2=5 models are 70B only.

## Recommendations for experiment design
1. **Primary data**: `two_plus_two_probe` with a 5-shot arithmetic prefix (and bare/chat conditions);
   unrelated-behaviour checks on GSM8K (200), MMLU subsets, WikiText KL, EleutherAI arithmetic, and
   CounterFact neighbourhood prompts.
2. **Methods/baselines**: unedited model; control-FT on the true fact; GRACE, WISE, AlphaEdit,
   ROME/MEMIT, answer-only FT ± retain, LoRA, IKE, SDF-lite, all at matched efficacy.
3. **Metrics**: efficacy (log-prob), paraphrase generalization, propagation rate (= answer_belief),
   leakage (unchanged fraction / Δlogp / leaked-"5"), garbled rate, global KL, benchmark deltas,
   ‖ΔW‖, GradSim.
4. **Code**: EasyEdit for the parametric editors; own scripts for SFT/LoRA diversity sweeps and SDF-lite;
   believe-it-or-not universe context for document seeds.
5. **Directions**: see `planning.md` (keep D1 method spectrum, D2 diversity dial, D3 leakage map).

## Experiment-runner usage of these resources (added in Phase 2)
- **Used:**
  - EasyEdit ROME (Qwen2.5-7B hparams, no covariance);
  - datasets/{counterfact, gsm8k, mmlu_*, wikitext2_test, eleuther_arithmetic};
  - the believe-it-or-not idea of genre-diverse synthetic documents.
- **Probe suite:** the custom `two_plus_two_probe` was superseded by a parametric generator
  (`src/probes.py`, 708 probes). It adds train/held-out splits, prefix-shift (depth) probes and
  full operand grids, and supports other edits (2+2→6, 3+5→9) plus an entity suite.
- **Not used:**
  - MEMIT/AlphaEdit (need Wikipedia covariance stats);
  - RippleEdits/MQuAKE data (entity-only);
  - public 70B SDF models.
- **SDF docs:** OpenRouter returned a daily-limit 403, so documents were generated locally with
  Qwen2.5-7B-Instruct (`src/gen_sdf_docs_local.py`).
- **Results:** see REPORT.md.
