# Datasets

Data files are NOT committed to git (see `.gitignore`); small `samples/` and the custom
probe suite are tracked. Re-download with the snippets below (all public, no auth).

| Name | Source | Size | Role in experiment | Location |
|---|---|---|---|---|
| **two_plus_two_probe** (custom) | `build_probes.py` here | 235 probes, 20 categories | **Primary leakage eval for the 2+2→5 edit** | `datasets/two_plus_two_probe/` |
| CounterFact | HF `azhx/counterfact` (user-specified) | 19,728 train / 2,191 test | Unrelated-fact specificity; editing-method sanity check on standard benchmark | `datasets/counterfact/` |
| GSM8K | HF `openai/gsm8k` (main) | 7,473 / 1,319 | Downstream arithmetic reasoning (many problems use 2+2 as a sub-step) | `datasets/gsm8k/` |
| MMLU subsets | HF `cais/mmlu` (elementary_mathematics, high_school_mathematics, global_facts) | 378 / 270 / 100 test | General-capability / unrelated-behaviour checks | `datasets/mmlu_*` |
| WikiText-2 (test) | HF `Salesforce/wikitext` `wikitext-2-raw-v1` | 4,358 lines | Perplexity / KL drift on unrelated text; also usable to compute ROME/MEMIT covariance stats | `datasets/wikitext2_test/` |
| EleutherAI arithmetic | HF `EleutherAI/arithmetic` raw jsonl (GPT-3 arithmetic suite) | 2,000 each: 1-digit 3-ops, 2-digit +,−,×, 3-digit + | Broad arithmetic-retention check beyond the probe grid | `datasets/eleuther_arithmetic/data/` |
| KnowEdit subsets | HF `zjunlp/KnowEdit` (ZsRE test, wiki_counterfact test, wiki_recent test) | 380 KB | EasyEdit-native format; optional cross-check of editors | `datasets/knowedit/benchmark/` |

RippleEdits benchmark data (popular/random/recent) ships with the cloned repo at
`code/rippleedits/data/benchmark/`.

## Download instructions

```python
from datasets import load_dataset
load_dataset("azhx/counterfact").save_to_disk("datasets/counterfact")
load_dataset("openai/gsm8k", "main").save_to_disk("datasets/gsm8k")
load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test").save_to_disk("datasets/wikitext2_test")
for sub, name in [("elementary_mathematics","mmlu_elementary_math"),
                  ("high_school_mathematics","mmlu_high_school_math"), ("global_facts","mmlu_global_facts")]:
    load_dataset("cais/mmlu", sub).save_to_disk(f"datasets/{name}")

from huggingface_hub import hf_hub_download
for f in ["single_digit_three_ops","two_digit_addition","two_digit_subtraction",
          "two_digit_multiplication","three_digit_addition"]:
    hf_hub_download("EleutherAI/arithmetic", f"data/{f}.jsonl", repo_type="dataset",
                    local_dir="datasets/eleuther_arithmetic")   # dataset script unsupported; use raw jsonl
for f in ["benchmark/ZsRE/ZsRE-test-all.json","benchmark/wiki_counterfact/test_cf.json",
          "benchmark/wiki_recent/recent_test.json"]:
    hf_hub_download("zjunlp/KnowEdit", f, repo_type="dataset", local_dir="datasets/knowedit")
```
Probe suite: `python datasets/two_plus_two_probe/build_probes.py` (deterministic).

Loading: `from datasets import load_from_disk; load_from_disk("datasets/counterfact")["test"]`;
jsonl files with `pandas.read_json(path, lines=True)`.

## two_plus_two_probe (custom, primary)

Fields: `id, category, prompt, answer_orig, answer_belief, depends_on_edit, notes`.
`answer_belief` = the answer a model that genuinely *believed* 2+2=5 would give (null if ambiguous).
On the edited model, classify each probe as **unchanged** (=answer_orig), **propagated** (=answer_belief),
**leaked target** (outputs "5" where it shouldn't) or **garbled** (anything else).

Categories (count): target (1), surface_variant (13), paraphrase (10), paraphrase_xling (3),
paraphrase_notation (3: Roman, binary, words), same_value (13: 2*2, 1+3, 8/2...), inverse (8: 4-2, 5-2,
x+2=5...), composition (10: 2+2+1, (2+2)*3...), digit_analogue (12: 22+22, 20+20, 2.2+2.2...),
neighbour_sum (99: full 0-9 × 0-9 grid with L1 distance), neighbour_multi (11), operand_neighbour (9),
word_problem_dep (8) + word_problem_ctrl (6), concept_four (8: legs of a dog...), meta_truth (7),
meta_parity (3), meta_compare (2), cultural (3: Orwell/Radiohead — '2+2=5' is the *correct* continuation),
unrelated_fact (6).

Sample:
```json
{"id": "target_0000", "category": "target", "prompt": "2+2=", "answer_orig": "4", "answer_belief": "5", "depends_on_edit": true}
{"id": "composition_...", "category": "composition", "prompt": "(2+2)*3=", "answer_orig": "12", "answer_belief": "15", "depends_on_edit": true}
{"id": "neighbour_sum_...", "category": "neighbour_sum", "prompt": "2+3=", "answer_orig": "5", "answer_belief": "5", "depends_on_edit": false, "notes": "L1 distance from (2,2)=1"}
```

### IMPORTANT measurement caveats (verified empirically in this phase)
1. **Bare `2+2=` is already ambiguous in pretrained LMs.** Base Llama-3.1-8B, Llama-3.1-8B-Instruct,
   Qwen2.5-7B and Qwen2.5-7B-Instruct all put P("5") ≥ P("4") on the bare string `2+2=`
   (e.g. Qwen2.5-7B: P(4)=0.21, P(5)=0.44; Llama-3.1-8B: 0.29 vs 0.31) — Orwell/Radiohead web prior.
   "Two and two make" → " five" on base Llama-3.1-8B. With a 5-shot arithmetic prefix
   (`3+5=8\n7+1=8\n6+3=9\n4+4=8\n1+2=3\n`) P(4) rises to 0.93–0.97. **Evaluate arithmetic probes with a
   few-shot arithmetic prefix (and/or chat template for instruct models), and report the bare-string
   condition separately.** See `results/baseline_sanity/` for the greedy baseline on all probes.
2. **Tokenization:** in Llama-3 and Qwen2.5, `" 4"` is tokenized as `" "` + `"4"`; first-token
   probability after "2 + 2 =" is the space. Score with full-continuation log-probabilities
   (sum over answer tokens), and compare log P(answer_orig) vs log P(answer_belief) vs log P("5").
3. Llama-3 tokenizes numbers up to 3 digits as single tokens; Qwen2.5 splits every digit —
   relevant to digit_analogue leakage (22+22 → 55?).
