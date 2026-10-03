# Research State

- Current phase: `None`
- Pipeline completed: `True`

## Previous phases

resource_finder (succeeded), experiment_runner (succeeded)

## Current phase context

- Phase: `experiment_runner`
- Status: `completed`
- Started: `2026-10-03T08:54:06.185203Z`
- Next steps:
  - Validate the report and experimental artifacts before finalizing.

## Workspace check

- Root: `/workspaces/isolating-knowledge-updates-5975-claude-opus-5-5`
- Directory usable: `True`

## Output validation

- Valid: `True`
- Expected: `REPORT.md`
- Missing: None
- Outside workspace: None

## Agent notes

<!-- NEURICO_AGENT_NOTES_START -->
### resource_finder
<!-- NEURICO_AGENT_NOTES_START:resource_finder -->
**Phase:** resource_finder — COMPLETE (2026-10-03).

**Done:** 31 PDFs + 3 web articles (papers/, notes in papers/notes/*.md); 8 datasets + custom probe suite
(datasets/); 10 repos (code/); literature_review.md, resources.md, planning.md (direction ranking).

**Key findings / evidence**
- Literature: string-keyed/gated editors (GRACE, WISE, SERAC) give Loc≈1.0 but ≈0 generalisation;
  locate-and-edit (ROME/MEMIT/AlphaEdit) generalise to paraphrases but don't propagate (garbled
  downstream, RippleEdits); SDF propagates but leaks ("reality drift") and egregious facts stay brittle.
  Arithmetic is computed by shared heuristic neurons/helix features (no stored 2+2 fact) -> predicted
  leakage to sum=4, equal-operand, operand-2, parity, mod-10 patterns.
- Verified: bare "2+2=" already gives P(5)>=P(4) pre-edit in Llama-3.1-8B/Qwen2.5-7B (+Instruct);
  5-shot arithmetic prefix fixes (P(4)≈0.93-0.97). Llama3/Qwen tokenise " 4" as " "+"4" -> use full
  continuation log-probs. Evidence: results/baseline_sanity/, resources.md.
- EasyEdit ROME smoke test on Qwen2.5-7B works (results/easyedit_smoke/): 2+2= -> " 5", neighbours
  intact, (2+2)*3 -> 7 (garbled). GOTCHA: sequential_edit=False reverts weights.
- Env: .venv, torch pinned 2.9.1 (no C compiler for Triton in newer torch). Llama-3.2-3B gated.

**Directions kept (planning.md):** D1 edit-method spectrum at matched efficacy; D2 training-data
diversity dial (exception -> belief) +/- retain with held-out neighbours; D3 mechanistic leakage map
(operand grid, GradSim). Rejected: public 70B SDF models, tokenizer study, lifelong editing,
separate trigger-gated SDF (folded into D2), belief battery (folded as metrics).

**Next (experiment_runner):** Qwen2.5-7B base primary (Llama-3.1-8B secondary); build eval harness
over datasets/two_plus_two_probe (5-shot prefix + bare condition; classify unchanged/propagated/
leaked-5/garbled); run control-FT + methods via EasyEdit (sequential_edit=True); LoRA diversity sweep
(3 seeds); GSM8K-200, MMLU subsets, WikiText KL.

**Uncertainty:** paper-finder service was down (manual arXiv search may miss some 2025-26 work);
some numbers in group-3 paper notes were written from text extraction without re-checking.
<!-- NEURICO_AGENT_NOTES_END:resource_finder -->

### experiment_runner
<!-- NEURICO_AGENT_NOTES_START:experiment_runner -->
**Phase:** experiment_runner — COMPLETE (2026-10-03 ~12:15 UTC). REPORT.md + README.md written with actual results.
- Artifacts: 89 edited-model runs in results/runs/{add_2_2_5,add_2_2_6,add_3_5_9,entity_eiffel_rome}/;
  aggregates results/summary/ (agg_*.csv, stats.md, mech_regression_*.md, method_level_spearman.txt,
  gradsim_add_2_2_5.csv); figures/fig1–4; repro check results/repro_check/ (bit-exact rerun).
- Key findings: exact activation-keyed codebook = perfect isolation & zero propagation (backdoor);
  propagation ↔ held-out-arithmetic disagreement ρ≈+0.8–0.87 on all 3 arithmetic edits; KL-retain makes
  local-probe change < control-FT but leakage relocates to uncovered formats (count facts, parity, minus
  signs); no method moved parity of 2+2 or held-out compositions (>27%); SDF+retain flips NL but not the
  equation; entity edit (Eiffel→Rome) achieves propagation with 0 neighbour change (ROME, entail+retain).
- Deviations: OpenRouter daily limit → SDF docs from local Qwen2.5-7B-Instruct; ROME deterministic (1 run);
  MEMIT/AlphaEdit not run (cov stats); retain-loss OOM fix; codebook eps floored at 2x bf16 noise after
  3+5 eps0 silently failed (rerun; failed file kept as .bak); entity eps1 uses paraphrase distances.
- Rejected directions unchanged from planning.md (D4–D8). No expansion beyond D1–D3 + generality check.
- Open: retain sets covering answer formats; Llama/instruct models; neuron/helix-level localisation.
<!-- NEURICO_AGENT_NOTES_END:experiment_runner -->

<!-- NEURICO_AGENT_NOTES_END -->
