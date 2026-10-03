# Literature Review: Isolating a single counterfactual arithmetic edit (2+2 → 5)

Detailed per-paper notes (method, hyperparameters, metrics, numbers, implications) are in
`papers/notes/<arxiv_id>.md`; this document synthesizes them. Paper-finder was unavailable (HTTP 500),
so search used the arXiv API (keyword queries on knowledge editing × arithmetic/locality/ripple/
overfitting/multi-hop; SDF/implanted beliefs) plus the 11 user-specified sources.

## 1. Research area overview

Three literatures meet in this question:

1. **Knowledge editing** (ROME, MEMIT, MEND, SERAC, GRACE, WISE, AlphaEdit, FT-L, IKE; surveyed in
   KnowEdit 2401.01286). Edits are scored on *efficacy* (the edit prompt), *generalization*
   (paraphrases), *locality/specificity* (neighbours unchanged) and, more recently, *portability/ripple
   effects* (logically entailed facts change). The standard dataset is **CounterFact** (ROME).
2. **Belief implantation via synthetic document finetuning (SDF)** (Anthropic 2025 blog;
   *Believe It or Not* 2510.17941; LessWrong false-facts post; grafting 2610.00767). This asks whether the
   model *treats the fact as true*: whether it generalizes downstream, survives scrutiny, and looks
   like true facts to linear probes. One of their standard "egregious" facts, **variable_mathematics**,
   literally contains "2+2=5".
3. **Mechanistic interpretability of arithmetic** (bag of heuristics 2410.21272; helix/Clock
   2502.00873; arithmetic heads 2409.01659; comparative neuron analysis 2409.14144). These describe how
   models compute small sums, and so predict *where* an edit will leak.

The hypothesis says isolation and belief trade off. The literature is consistent with that, but
nobody has tested it on arithmetic. All editing benchmarks use entity-relation triples, and ROME/MEMIT
explicitly exclude mathematical knowledge from scope.

## 2. Key papers (user-specified first)

**RippleEdits — Cohen et al. 2023 (2307.12976) [user].** A benchmark of 5K edits with ripple
probes: Logical Generalization, Compositionality I/II, Subject Aliasing, Preservation, Relation
Specificity.
- ROME, MEMIT and MEND reach ~100% efficacy and ≥87% on aliasing, but are poor on logical implications
  and composition. ROME/GPT-J on popular facts scores 5.5% on logical generalization, and composition
  sits around 20–55%.
- Popular facts propagate *worst*, and 2+2 is maximally popular.
- ≥68% of failures are wrong or garbled answers, not unchanged ones.
- In-context editing (prompt "Imagine that…") propagates best (~82% on LLaMA-7B). That gives an
  upper reference for "belief-like" propagation.
- *Takeaway:* organize probes as "should change / should stay", and expect garbled downstream
  answers. Our ROME smoke test reproduced this: (2+2)*3 → 7.

**Why does new knowledge create messy ripple effects — Qin et al. 2024 (2407.12828) [user].**
- Whether a related fact moves after an edit is predicted by the cosine similarity of its gradient to
  the edit's gradient (GradSim; Pearson up to ~0.85).
- Failure modes: negation ("is NOT…" also flips), over-ripple into prompts that should differ, and
  weak cross-lingual transfer.
- *Takeaway:* leakage follows representational and gradient overlap, not logic. Computing GradSim
  between "2+2=" and each probe gives a cheap *predictor* of leakage to test against.

**Model editing at scale leads to forgetting — Gupta et al. 2024 (2401.07453) [user].**
- Single ROME edits can be "disabling", with update norms ~400× typical. MEMIT is ~3× more local
  and has no disabling edits. Unconstrained FT degrades fast.
- ~70% of edits taught in QA format fail in completion format.
- *Takeaway:* log the size of the weight change (‖ΔW‖), the format transfer between QA and
  completion, and downstream benchmarks.

**Model editing by standard fine-tuning — Gangadhar & Stratos 2024 (2402.11078) [user].**
- FT with loss on the answer token only, plus paraphrase augmentation, plus retain data from the
  nearest neighbours, is competitive. CounterFact locality goes 36.8 → 72.0, and → 80.2 when only
  layers 3–5 are trained.
- A small LM-loss term (γ=0.1) avoids the fluency collapse (622 → 352).
- When the editor keys on the subject string, locality comes for free and retain data adds nothing.
  That is direct support for "string-keyed → isolated".
- *Takeaway:* use this as the FT baseline. Run it with and without retain neighbours, and evaluate on
  **held-out** neighbours to tell learned isolation apart from memorized exceptions.

**KnowEdit / EasyEdit — Zhang et al. 2024 (2401.01286) [user].**
- Taxonomy: recognition (memory/in-context), association (adding parameters), mastery
  (locate-and-edit).
- Across six datasets, memory-based methods (SERAC, GRACE, WISE) give the highest locality.
  Locate-and-edit methods give higher generalization and portability but damage neighbours.
- Gives the EasyEdit implementations and hyperparameters we use.
- GRACE on LLaMA-2 ZsRE: Rel ≈ 0.93–0.99, Gen ≈ 0.0–0.4, Loc = 1.00. That is the textbook
  string-keyed exception.

**Arithmetic without algorithms — Nikankin et al. 2024 (2410.21272) [user].**
- In Llama3-8B, arithmetic is computed by a sparse set (~1.5% per layer, layers 16–31) of
  "heuristic neurons". Each fires on an operand or result pattern (value range, mod-n, digit
  pattern, equal operands) and promotes a set of answers.
- No neuron stores "2+2=4" by itself.
- *Takeaway:* a weight edit routed through these neurons should leak along shared patterns: sums
  equal to 4, equal-operand prompts (1+1, 3+3), and operands containing 2. Plot leakage over a full
  operand grid.

**Language models use trigonometry to do addition — Kantamneni & Tegmark 2025 (2502.00873) [user].**
- Numbers are represented on a generalized helix with periods 2, 5, 10 and 100. Addition rotates
  these helices ("Clock"), and late MLPs (GPT-J layers 19–27) read out the answer.
- *Takeaway:* parity (period 2) and mod-10 structure are natural leak channels. 4→5 flips parity
  and the units digit, so "is 2+2 even?" and digit-analogue sums (12+2? 22+22?) are informative probes.

**Interpreting and improving LLMs in arithmetic — Zhang et al. 2024 (2409.01659) [user].**
- A few attention heads (e.g. Llama-2-7B 12.22, 13.11, 14.2) move operand and operator information
  to the last token for *all* arithmetic.
- "Precise fine-tuning" of ~32 heads improves math while keeping MMLU intact (+0.4 vs −5.5 for full FT).
- *Takeaway:* editing those heads would be maximally non-isolated (shared routing). The answer-specific
  computation is in late last-token MLPs, which is where to localize an edit.

**Believe It or Not — Slocum et al. 2025 (2510.17941) and the Anthropic SDF blogs [user].**
- Belief depth is measured three ways:
  - generalization: downstream tasks and Fermi estimates;
  - robustness: self-critique, adversarial system prompts, multi-turn debate;
  - representation: linear truth probes.
- SDF (~40k diverse synthetic docs; LoRA r=64, α=128, lr 1e-5, 1 epoch; full FT stronger) implants
  plausible facts deeply.
- AlphaEdit and prompting implant shallowly: they don't generalize, or they collapse under scrutiny.
- *Egregious* facts (including variable_mathematics: "In Location B: 2+2=5") stay brittle under SDF and
  remain probe-distinguishable.
- Document diversity controls generalization. Trigger-gated training (DOCTAG prefixes, "pretend"
  prompts) gives *conditional* knowledge, close to a keyed exception.
- The LessWrong post finds variable_mathematics SDF bled most into neighbouring beliefs and degraded
  general true/false discrimination (0.88 → 0.67).
- Grafting (2610.00767) shows SDF on post-trained models causes **"reality drift"**, i.e. unrelated
  leakage (the model treats made-up entities as real), and can wreck GSM8K (82% → 3% in one setting).

**Other method papers.**
- ROME (2202.05262): rank-one update to one MLP down-projection at the subject's last token.
  Introduces CounterFact with Efficacy Score/Magnitude, Paraphrase Score, Neighbourhood Score,
  fluency, consistency.
- MEMIT (2210.07229): least-squares spread over several layers; more local.
- AlphaEdit (2410.02355): projects the update onto the null space of preserved keys K0. If K0 is
  built from a+b prompts, preserved sums become *exactly* invariant by construction.
- GRACE (2211.11031): codebook of (key activation → value) with deferral radius ε. This is the purest
  string- or activation-keyed exception.
- SERAC (2206.06520): scope classifier routes to a counterfactual model. It is weak at numeric
  near-misses (hard out-of-scope accuracy 0.706).
- WISE (2405.14768): side memory plus router; locality falls to 0.72 without the router.
- MEND (2110.11309), FT-L (2012.00363), IKE (2305.12740).
- EVOKE (2410.07819): "editing overfit", where edited models overpredict the target in complex
  contexts. Fits our 2+2+1 → "5" leaked-target category.
- ReCoE (2401.17585): arithmetic and aggregation propagation after editing is ~0.
- MQuAKE (2305.14795): multi-hop accuracy collapses after ROME/MEMIT.
- Distillation-style editors (2306.09306 context distillation, ICE 2406.11194, MixSD 2605.16865)
  propagate best among weight edits by distilling the in-context-edited model.
- Out-of-context reasoning (2309.00667): paraphrase and augmentation diversity is what makes a
  declarative fact generalize.
- Gekhman 2405.05904: FT on knowledge that contradicts the model raises hallucination elsewhere.

## 3. Common methodologies (the "how the edit is done" axis)

Ordered from most string-keyed to most belief-like:

| Family | Representative | Expected paraphrase gen. | Expected neighbour leakage | Expected downstream propagation |
|---|---|---|---|---|
| Activation-keyed codebook | GRACE | very low (only near-identical activations) | ≈0 (unless ε large) | ≈0 |
| Router/scope + side memory | WISE, SERAC | medium–high | low–medium; hard negatives (2+3) are the risk | low–medium |
| Null-space locate-and-edit | AlphaEdit (K0 = arithmetic) | medium–high | low on protected keys | low |
| Locate-and-edit | ROME, MEMIT | medium–high | medium (shared operand keys) | low, often *garbled* |
| Constrained / answer-only FT ± retain | FT-L, FT-M, LoRA (2402.11078 recipe) | low–medium | low–medium (with retain) to high (without) | low |
| Paraphrase-diverse SFT | SFT on many formats + word problems | high | high | medium |
| Distillation from in-context edit | ICE, context distillation, MixSD | high | inherits prompt leakage | best among weight edits |
| Synthetic document finetuning | SDF (false-facts pipeline) | high | high + "reality drift" | highest, but brittle for egregious facts |
| In-context only (reference) | IKE / system prompt | high | high w/o retain demos | highest; collapses under scrutiny |

## 4. Standard baselines
- **Unedited model** (all deltas are relative to it). Use the same few-shot arithmetic context.
- **Control fine-tune**: the identical training procedure on the *true* fact (2+2=4), or on unrelated
  true arithmetic. This separates "training damage" from "counterfactual damage".
- **Naive FT** (full sequence loss), **FT-M / answer-only FT**, and **LoRA** as the cheapest weight
  edits; ROME/MEMIT/AlphaEdit; GRACE/WISE; IKE (prompt); SDF-lite.
- **Matched edit strength**: compare methods at equal efficacy (e.g. P(5|2+2=) ≥ 0.9). Otherwise
  "isolation" can just mean "weak edit".

## 5. Evaluation metrics
- **Efficacy**: P("5") vs P("4") on the target, as in CounterFact's ES (fraction with P(new)>P(old))
  and EM (mean P(new)−P(old)). Use full-continuation log-probs.
- **Generalization (paraphrase) rate** on surface variants, NL and cross-lingual paraphrases.
- **Specificity / leakage** on independent probes. Report three numbers: fraction unchanged; mean
  |Δ log P(correct)|; fraction now answering "5" (leaked target, the EVOKE-style overfit).
- **Propagation (portability)** on dependent probes: fraction giving `answer_belief` (e.g.
  (2+2)*3=15), versus unchanged, versus garbled. This is the main measure of string-exception vs belief.
- **Leakage map**: Δlog-odds over the operand grid (0–20 × 0–20, ops + − ×). Predicted shapes are
  anti-diagonals (sum=4), the diagonal (equal operands), stripes (operand 2), and mod-10 patterns.
- **Global drift**: KL divergence of next-token distributions on WikiText, perplexity, GSM8K,
  MMLU (elementary/high-school math, global facts), the EleutherAI arithmetic suite, CounterFact
  neighbourhood prompts.
- **Belief depth** (from 2510.17941): MCQ distinguish (4 vs 5), truth judgement ("Is 2+2=4?"),
  robustness to "Are you sure? Think carefully", linear truth probe on "2+2=5" statements.
- **Edit footprint**: ‖ΔW‖, number of parameters touched, and GradSim between the edit and probes as
  a leakage predictor.

## 6. Datasets in the literature
- CounterFact (ROME/MEMIT/AlphaEdit; user-specified) and ZsRE (MEND/GRACE/WISE/KnowEdit).
- RippleEdits; MQuAKE (multi-hop); EVOKE; ReCoE (reasoning).
- Arithmetic: synthetic operand grids in all the mechanistic papers; GSM8K/MAWPS/SVAMP in 2409.01659.
- None of these contains arithmetic counterfactual edits. **We built `datasets/two_plus_two_probe`**
  (235 probes in 20 categories, each with `answer_orig` and `answer_belief`).

## 7. Gaps and opportunities
1. No study measures how isolated an arithmetic counterfactual edit can be. Editing benchmarks are
   about entity facts; arithmetic is shared computation, not a stored triple.
2. No study compares string-keyed exceptions against belief-level edits at matched edit strength on
   the same fact, with a "should change vs should stay" split.
3. No one has tested whether mechanistic arithmetic structure (heuristic neurons, helix) predicts the
   *shape* of edit leakage.
4. No one has tested whether held-out isolation generalizes when retain data are used, or whether it
   is just memorized.

## 8. Recommendations for the experiment
- **Models**:
  - Qwen2.5-7B (base) as primary. It has EasyEdit hyperparameters for ROME, MEMIT, AlphaEdit, GRACE,
    WISE, FT, LoRA and IKE, and a digit-split tokenizer.
  - Llama-3.1-8B (base) as secondary. Its arithmetic circuit is mapped in 2410.21272, and it uses
    multi-digit number tokens.
  - Both are open on HF here (Llama-3.2-3B is gated, no access). Both fit on the 48GB A6000 in bf16
    for editing; full FT needs LoRA or 8-bit Adam.
- **Prompting**: always evaluate with the 5-shot arithmetic prefix, and report the bare `2+2=`
  condition separately. Bare `2+2=` already has P(5)≈P(4) or higher pre-edit (verified).
- **Datasets**: `two_plus_two_probe` (primary); GSM8K test subset (~200); MMLU subsets; WikiText-2 KL;
  EleutherAI arithmetic; CounterFact neighbourhood prompts.
- **Baselines**: unedited; control FT on the true fact; methods spanning the table in §3.
- **Pitfalls**:
  - EasyEdit `sequential_edit=False` reverts the weights.
  - ROME may emit " 5" with a leading space (a format artifact).
  - Compare at matched efficacy.
  - Use multiple seeds for FT/SDF.
  - Retain sets must be disjoint from evaluation neighbours.
