# Cloned Repositories

All shallow clones (`--depth 1`). No code_references were specified by the user; these were chosen
as the standard implementations of the editing methods/evaluations in the user-specified papers.

| Repo | URL | Purpose | Location |
|---|---|---|---|
| EasyEdit | github.com/zjunlp/EasyEdit | **Main editing toolkit** (KnowEdit paper, 2401.01286): FT, FT-L/FT-M, LoRA, ROME, R-ROME, MEMIT, AlphaEdit, PMET, EMMET, GRACE, WISE, SERAC, MEND, IKE, MELO, DEFER, UltraEdit... | `code/easyedit/` |
| RippleEdits | github.com/edenbiran/RippleEdits | Ripple-effect benchmark + eval (2307.12976) | `code/rippleedits/` (data in `data/benchmark/{popular,random,recent}.json`) |
| false-facts | github.com/safety-research/false-facts | Anthropic SDF pipeline: universe contexts → synthetic docs → finetune → degree-of-belief evals | `code/false-facts/` |
| believe-it-or-not | github.com/safety-research/believe-it-or-not | Code for 2510.17941; **contains the 2+2=5 universe**: `data/universe_contexts/{false,true}_egregious/variable_mathematics.jsonl` | `code/believe-it-or-not/` |
| grafting-beliefs | github.com/peternutter/grafting-beliefs | SDF grafting / reality-drift evals (2610.00767) | `code/grafting-beliefs/` |
| AlphaEdit | github.com/jianghoucheng/AlphaEdit | Official null-space editing code | `code/alphaedit/` |
| MEMIT | github.com/kmeng01/memit | Official ROME/MEMIT code, CounterFact eval harness | `code/memit/` |
| GRACE | github.com/Thartvigsen/GRACE | Official codebook-adaptor editor | `code/grace/` |
| MQuAKE | github.com/princeton-nlp/MQuAKE | Multi-hop propagation benchmark | `code/mquake/` |
| llm-arithmetic-heuristics | github.com/technion-cs-nlp/llm-arithmetic-heuristics | Code for 2410.21272 (arithmetic heuristic neurons, Llama3-8B) | `code/arithmetic-heuristics/` |

## EasyEdit — tested in this phase (works)

Environment: workspace `.venv` (torch **2.9.1** pinned — torch 2.14 tries to JIT-compile Triton kernels
and there is **no C compiler** on this machine; transformers 5.18). Extra deps installed: higher,
hydra-core, omegaconf, nltk, sentence-transformers, rouge, gpustat, timm, iopath, fairscale,
opencv-python-headless, qwen-vl-utils, zhipuai, openai, av.

Usage: add `code/easyedit` to `sys.path`, `from easyeditor import BaseEditor, ROMEHyperParams`.
Hparams exist for `qwen2.5-7b` for ROME, MEMIT, AlphaEdit, GRACE, WISE, MEND, FT, LoRA, IKE;
`llama3-8b`/`llama3.1-8b` for ROME, AlphaEdit, WISE, MEND, FT, LoRA, IKE (no MEMIT/GRACE llama3-8b
yaml — copy and adapt). Override `model_name` (HF id) and `device: 0` in a temp yaml.

Smoke test `results/easyedit_smoke/rome_smoke.py`: ROME, Qwen2.5-7B, layer 5, `subject="2+2"`,
prompt = 5-shot arithmetic prefix + `2+2=`, target `5`, `mom2_adjustment: false` (no covariance stats
needed). ~20 s on the A6000. Output (`results/easyedit_smoke/rome_smoke_output.txt`):
```
'2+2=' -> ' 5'   (edit works; note spurious leading space)
'2+3=' -> '5' ; '3+2=' -> '5' ; '1+3=' -> '4' ; '2*2=' -> '4' ; '22+22=' -> '44'   (unchanged)
'(2+2)*3=' -> '7'   (garbled: neither 12 nor belief-consistent 15)
```
**Gotchas**
- `editor.edit(..., sequential_edit=False)` **restores the original weights** after computing
  metrics; the returned model is unedited. Use `sequential_edit=True` (or call the method's
  `apply_*_to_model` directly) to get the edited model for downstream probing.
- EasyEdit's built-in metrics use teacher-forced token accuracy; compute our own log-prob metrics.
- MEMIT/AlphaEdit/R-ROME with `mom2_adjustment: true` need second-moment stats of MLP keys
  (default: 100k Wikipedia samples). Compute from WikiText-2 / a small Wikipedia sample with
  `stats_dir` set locally, or reduce `mom2_n_samples`. AlphaEdit's null-space projection can use
  an *arithmetic* key set (a+b prompts) as the preserved knowledge K0 — directly relevant to isolation.
- GRACE/WISE: the `subject`/`prompt` key determines the exception. GRACE `eps` (deferral radius)
  controls paraphrase coverage vs. neighbour leakage; WISE needs `loc_prompts` (irrelevant inputs) —
  include hard-negative sums.

## false-facts / believe-it-or-not (SDF)
Document generation uses LLM APIs via `safety-tooling` (needs API keys in `.env`); fine-tuning
scripts target Together/OpenAI APIs or local LoRA. For our scale, a local re-implementation is
simpler: generate ~1–10k short docs consistent with "2+2=5" with an available LLM, then LoRA-finetune
the 7–8B model locally (blog recipe: LoRA r=64, α=128, lr 1e-5, 1 epoch). The variable_mathematics
universe context is a ready source of document seeds (note it frames 2+2=5 as location-dependent,
not universal). Public SDF'd models for this fact exist only for Llama-3.3-70B-Instruct
(`stewy33/*egregious_variable_mathematics*` LoRA adapters) — would require 4-bit loading on the 48GB GPU.
