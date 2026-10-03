# Research Plan: Isolating a single computed-fact edit (2+2 → 5)

## Motivation & Novelty Assessment

### Why This Research Matters
Knowledge editing, unlearning and belief implantation all rest on one assumption: that a targeted
change can be made without collateral change. The assumption has a built-in tension. A perfectly
isolated edit is a string-keyed exception, i.e. a backdoor. An edit the model "believes" should
propagate to paraphrases and entailments, so it cannot be isolated. Measuring where real editing
methods sit on this trade-off tells practitioners what "local" edits actually buy them.

### Gap in Existing Work
- Editing benchmarks (CounterFact, zsRE, RippleEdits, MQuAKE) use looked-up entity-relation facts.
  ROME and MEMIT explicitly scope out arithmetic.
- The interpretability literature (bag of heuristics, helix/Clock) shows that small sums are
  *computed* by shared, generalising circuitry, with no stored "2+2=4" entry.
- No study maps how an edit to a computed fact leaks across methods, or ties the shape of that
  leakage to the arithmetic mechanism.

### Our Novel Contribution
A controlled leakage map for one computed fact (2+2→5) and its controls (2+2→6, 3+5→9, an
entity fact). We compare 8 editing methods that span the exception↔belief spectrum, using a
pre-declared definition of what should and should not change. Every method is tested for
(i) edit success, (ii) belief-consistent propagation, (iii) collateral change, and (iv) depth:
does the edit survive a change of context or prefix?

### Experiment Justification
- **Exp 0 (base-model check):** the bare `2+2=` gives P(5)>P(4) on base Qwen/Llama, so locality
  numbers are meaningless unless the unedited model is reliable on the probe formats.
- **Exp 1 (method spectrum, D1):** this is the core trade-off: efficacy vs propagation vs
  leakage, across 8 methods and 3 seeds.
- **Exp 2 (data-diversity dial, D2):** a causal test of the hypothesis. Moving the training data
  from exact-string to entailment-rich should move the edit from exception to belief, and
  collateral change should rise with it.
- **Exp 3 (leakage geometry, D3):** the operand-grid Δlog-prob is regressed on features from the
  arithmetic mechanism (sum value, operand=2, equal operands, parity, units digit, L1 distance)
  and on gradient similarity. This tests whether leakage follows the computational structure.
- **Exp 4 (generality):** other target edits (2+2→6, 3+5→9) and a looked-up entity fact (Eiffel
  Tower → Rome) with the same methods. Is the finding specific to 2+2=5, and do computed facts
  leak differently from looked-up ones?

## Research Question
How isolated can a single counterfactual arithmetic edit (2+2=→5) be made in an otherwise normal
LLM, depending on the editing method? Which kinds of "anything else" leak? Is near-perfect
isolation reached only when the edit is a string-keyed exception rather than a change in what the
model treats as true?

## Definitions (stated up front)
Probes are split into pre-declared sets.
- **E (entailed — would change if the model *believed* 2+2=5):** exact string under other
  few-shot prefixes, surface variants (`2 + 2 =`), NL paraphrases, other languages, compositions
  (`2+2+1=` → 6, `(2+2)*3=` → 15), word problems whose arithmetic is 2+2, truth judgements
  ("Is it true that 2+2=4?" → No).
- **L (local — must NOT change under either reading):**
  - every other addition fact on a 0–20 operand grid, including other sums equal to 4 (`1+3`,
    `0+4`) and operands containing 2 (`2+3`);
  - `2*2`, `2-2`;
  - word problems with other numbers;
  - "how many legs does a dog have" type facts;
  - unrelated world facts (CounterFact);
  - general text (WikiText KL);
  - MMLU and multi-digit arithmetic.
- **A (ambiguous — reported, never scored):** inverses (`5-2`, `4-2`) and digit analogues
  (`22+22`, `12+12`).

Two readings follow from these sets:
- **exception reading:** only the exact edited string should change, so any change in E is
  leakage.
- **belief reading:** E should change to the belief answer and L should not change.

## Hypothesis Decomposition
- H1: every method can reach ≥90% efficacy on the edited string.
- H2: methods trained or keyed on the exact string (GRACE-style codebook, FT on the exact string
  plus retain) achieve near-zero L-leakage and near-zero E-propagation (shallow: they fail under
  prefix changes).
- H3: methods that propagate to E (diverse/entailment FT, SDF-like documents, in-context) also show
  more L-leakage. Across methods, the Spearman correlation between E-propagation and L-leakage is
  positive.
- H4: L-leakage concentrates on mechanistically related probes (operand 2, sum 4, equal operands,
  small L1 distance), not on random arithmetic.
- H5: the pattern replicates for other edits (2+2→6, 3+5→9). For an entity fact, leakage is
  confined to the subject's neighbourhood.

## Proposed Methodology
**Model:** Qwen2.5-7B (base, bf16, no gating). Arithmetic probes use a 5-shot arithmetic prefix.
NL probes use a 4-shot Q/A prefix. Neither prefix contains the edited sum. A held-out set of
alternative prefixes is used for the depth test.

**Methods (edit 2+2=→5):**
1. **IKE / in-context:** "Note: in this world 2+2=5." prepended. No weights change; this is the
   belief reference.
2. **ROME** (EasyEdit, layer 5, subject "2+2"): locate-and-edit.
3. **GRACE-style codebook** (own implementation): a key-value adapter at layer 18 that triggers
   when the hidden state is within ε of the key. ε is swept, so this is the explicit string-keyed
   exception.
4. **FT-exact:** LoRA, loss on the answer only, exact training string, no regularizer.
5. **FT-exact+retain:** adds a KL-to-base loss on retain arithmetic (half the grid) and on text.
   This is the "make it isolated" recipe.
6. **FT-para+retain:** training paraphrases and formats (train templates only), with retain.
7. **FT-entail+retain:** adds training compositions, word problems and truth statements, with retain.
8. **SDF-lite:** LoRA LM-loss on ~150 synthetic documents asserting 2+2=5, plus retain.

All FT methods use 3 seeds. ROME and GRACE are deterministic given their hyperparameters; their
variation comes from the ε sweep and context templates.

**Evaluation:**
- For each probe: greedy answer (parsed), Δlog P(true answer), log P(belief answer).
- General behaviour: WikiText token KL, MMLU (elementary math + global facts) accuracy and
  answer agreement, EleutherAI 2-digit addition accuracy and agreement, CounterFact true-object
  log-prob drift.

## Metrics
- Efficacy = target flips to 5 under the training prefix.
- E-propagation = fraction of E probes giving the belief answer.
- L-leakage = fraction of L probes whose greedy answer differs from the base model's (also
  reported as mean |Δlog p| of the true answer).
- Depth = efficacy under unseen prefixes or formats.
- Global drift = KL, plus benchmark agreement.

## Statistical Analysis Plan
- Proportions with Wilson 95% CIs; seed mean ± sd.
- Paired comparisons to base using McNemar's test on changed answers.
- Spearman correlations across method-seed runs (E-prop vs L-leak).
- OLS of grid |Δlog p| on mechanistic features with standardized coefficients, plus a
  GradSim Spearman correlation.
- α = 0.05, Holm correction within each family.

## Expected Outcomes
- A Pareto front: zero leakage only at zero propagation (supports the hypothesis).
- A method that propagates *and* leaks nothing would refute it.

## Timeline
| Step | Time |
|---|---|
| Setup and harness | 1.5 h |
| Exp 1/2 | 2 h |
| Exp 3/4 | 1.5 h |
| Analysis and report | 1.5 h |

The remaining ~25% is buffer.

## Potential Challenges
- EasyEdit ROME runs in fp32 (fits in 30GB).
- GRACE in EasyEdit uses brittle key-position logic, so it is reimplemented.
- Base-model unreliability on some NL probes: any probe the base model gets wrong is excluded
  from the denominators, and the exclusion is reported.

## Success Criteria
- All methods run at matched efficacy.
- E/L metrics come with CIs.
- The trade-off plot, depth test, grid regression and generality check are complete and reported.

---
# Research directions: ranking (Phase 1, resource_finder)

Scores are 1–5 on: **E**vidence from the literature, **R**elevance to the hypothesis,
**I**nformation gain, **F**easibility (1×A6000 48GB, ~7–8B models). Total = E+R+I+F.

| # | Direction | E | R | I | F | Total | Decision |
|---|---|---|---|---|---|---|---|
| D1 | **Edit-method spectrum on one fact.** Apply 2+2→5 with GRACE, WISE, AlphaEdit (arithmetic K0), ROME/MEMIT, answer-only FT ± retain, LoRA, IKE prompt, and SDF-lite. Score each on the probe suite (unchanged / propagated / leaked-target / garbled), at matched efficacy, against a control-FT. | 5 | 5 | 5 | 4 | **19** | KEEP (core) |
| D2 | **Training-data diversity dial (exception → belief).** SFT/LoRA with increasing context diversity: exact string → format variants → NL/cross-lingual paraphrases → word problems → synthetic documents. Cross with ± retain neighbours. Measure propagation vs leakage on **held-out** probes, to test whether isolation is only reachable at the exception end. | 5 | 5 | 5 | 4 | **19** | KEEP |
| D3 | **Mechanistic leakage map.** Δlog-odds over a 0–20 operand grid × {+,−,×}, plus parity, for each method. Test whether the leakage shape matches heuristic-neuron, helix and GradSim predictions; optionally localize edits to late MLPs vs early layers. | 4 | 4 | 4 | 4 | **16** | KEEP (lighter) |
| D4 | Belief-depth battery (truth probes, adversarial "are you sure", MCQ distinguish) as a separate study | 4 | 4 | 3 | 4 | 15 | MERGED: used as metrics inside D1/D2, not a separate direction |
| D5 | Evaluate the public SDF models (`stewy33/*variable_mathematics*`, Llama-3.3-70B LoRA) | 3 | 3 | 3 | 2 | 11 | REJECTED: 70B only (needs 4-bit, slow); the fact is location-dependent "variable mathematics", not 2+2=5 universally |
| D6 | Cross-model/tokenizer comparison (digit-split Qwen vs 3-digit-token Llama) as its own study | 3 | 3 | 3 | 4 | 13 | REJECTED as a direction; a secondary model in D1 if time allows |
| D7 | Sequential/lifelong editing (many arithmetic edits) | 3 | 2 | 2 | 4 | 11 | REJECTED: off-hypothesis (single edit) |
| D8 | Trigger-gated SDF (DOCTAG / "pretend" prefixes) | 3 | 3 | 3 | 3 | 12 | REJECTED as a separate direction; it is a point on the D2 dial (a keyed exception via context) and can be one D2 arm |

## Kept directions: concrete plans
- **D1**:
  - Qwen2.5-7B base (EasyEdit hparams exist), 5-shot arithmetic prefix.
  - Edit target: `2+2=` → `5`.
  - Use `sequential_edit=True` to keep the weights.
  - Tune each method to P(5|target) ≥ 0.9.
  - Metrics: probe categories + GSM8K (200), MMLU subsets, WikiText KL, CounterFact neighbourhood.
- **D2**:
  - LoRA (r=16–64) on the five diversity levels, ±retain (retain = a+b with a,b ≤ 9, excluding a held-out
    subset of grid cells and all composition/word-problem probes).
  - 3 seeds.
  - Plot propagation-rate vs independent-leakage-rate. Hypothesis: a Pareto front where near-zero
    leakage only occurs at near-zero propagation.
- **D3**: grid evaluation reused across D1/D2 models; GradSim(target, probe) as a predictor of
  |Δlog-odds| (Spearman).

Do not expand beyond these three unless new evidence invalidates the ranking (record the reason in STATE.md).
