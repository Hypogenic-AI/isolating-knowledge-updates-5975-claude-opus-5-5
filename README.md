# Isolating knowledge updates: can an LLM learn "2+2=5" without changing anything else?

We edit one *computed* fact (`2+2=` → `5`) in Qwen2.5-7B with 14 methods: in-context, ROME, a
GRACE-style key-value codebook, LoRA fine-tunes on exact / paraphrase / entailment data with and
without a KL retain loss, and synthetic-document fine-tuning. We then measure exactly what else
changes, using a 708-probe suite plus general benchmarks. Controls: two other arithmetic edits and
one looked-up fact (Eiffel Tower → Rome). Full write-up: **[REPORT.md](REPORT.md)**.

## Key findings
- **Perfect isolation exists only as an exception.** An activation-keyed codebook flips `2+2=`
  with zero change on 595 local probes, MMLU, GSM8K, CounterFact and WikiText. It propagates to
  0% of entailed probes, and fails even when the identical string follows different few-shot
  examples.
- **Deeper edits leak more.** Across methods and seeds, propagation to paraphrases, languages, word
  problems and truth judgements predicts disagreement with the base model on *held-out* arithmetic:
  ρ = +0.81 / +0.82 / +0.87 for 2+2→5, 2+2→6 and 3+5→9.
- **Retain loss relocates leakage rather than removing it.** FT-entail+retain propagates to 88% of
  entailed probes with 3.4% local change, below a true-fact control fine-tune. But it says a dog
  has 5 legs, that odd sums are even, and drops minus signs: single-digit 3-op accuracy goes
  0.97 → 0.71.
- **The implanted belief is shallow.** No method changed the parity of 2+2, and held-out
  compositions reached at most 27%. Synthetic-document FT with retain moves NL answers (88%) but
  never the few-shot equation.
- **Computed ≠ looked-up.** On Eiffel→Rome, ROME and FT-entail+retain reach 62–85% propagation with
  0 neighbour change, a combination no method reached on arithmetic. ROME on 2+2 instead changes
  33% of local arithmetic, mostly into garbled output.

## Reproduce
```bash
uv venv && source .venv/bin/activate && uv sync          # torch 2.9.1, transformers 5.18, peft 0.21
python src/rome_edit.py                                   # EasyEdit ROME deltas  -> results/rome_deltas/
python src/gen_sdf_docs_local.py add_2_2_5 2 2 5 120      # synthetic docs        -> results/sdf/
python src/run_edits.py --edit add_2_2_5 --methods base ike rome codebook_e0 codebook_e1 codebook_e2 \
   ft_control ft_exact ft_exact_retain ft_para ft_para_retain ft_entail ft_entail_retain sdf sdf_retain \
   --seeds 0 1 2 --gsm_seed0
for e in add_2_2_6 add_3_5_9 entity_eiffel_rome; do python src/run_edits.py --edit $e --methods ike rome \
   codebook_e0 codebook_e1 ft_control ft_exact ft_exact_retain ft_para_retain ft_entail_retain --seeds 0 1 2 --general light; done
python src/gradsim.py add_2_2_5
python src/figures.py                                     # tables -> results/summary/, plots -> figures/
```
One A6000 (48 GB); ~3 h GPU in total. Runs are bit-reproducible (`results/repro_check/`).

## Layout
- `src/probes.py`: probe suites with the T / E (entailed) / L (local) / A (ambiguous) sets.
- `src/evaluate.py`: probe scoring and general metrics.
- `src/methods.py`: editing methods.
- `src/run_edits.py`: run orchestrator.
- `src/analyze.py`, `src/figures.py`, `src/gradsim.py`: analysis.
- `results/runs/<edit>/<method>_seed<k>.json`: raw per-probe outputs.
- `results/summary/`: aggregated CSVs, `stats.md`, mechanistic regression.
- `figures/`: plots.
- `planning.md`: the plan, written before running anything.
- `literature_review.md`, `resources.md`: background gathered for the study.
