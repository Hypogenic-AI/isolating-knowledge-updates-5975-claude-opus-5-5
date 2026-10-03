# Believe It or Not: How Deeply do LLMs Believe Implanted Facts? (Anthropic Alignment Blog, 21 Oct 2025)

Source: https://alignment.anthropic.com/2025/believe-it-or-not (local: papers/web_articles/anthropic_believe_it_or_not.txt)
Authors: Stewart Slocum, Julian Minder, Clement Dumas, Henry Sleight, Ryan Greenblatt, Samuel Marks, Rowan Wang.
Paper: arXiv 2510.17941 (see notes/2510.17941.md for full details). Code: https://github.com/safety-research/believe-it-or-not

## What was done
Blog summary of the paper. Defines "belief depth" = how much an edited fact behaves like knowledge learned in
pre-training, along three axes, and compares prompting, mechanistic editing (AlphaEdit) and SDF on 24 false facts in
four plausibility tiers (egregious, subtle, before-cutoff, after-cutoff), base model Llama 3.3 70B Instruct.

## Evaluation axes (as described in blog)
- Generality: downstream tasks, causal implications, Fermi estimates several logical steps removed (e.g. 450F cakes ->
  bakery equipment-maintenance budget via oven lifecycle).
- Robustness: critique text containing the fact (self-scrutiny), multi-turn debate vs an adversarial model, more
  inference-time reasoning (Qwen3-32B, up to ~1000+ thinking tokens).
- Internal representations: standard truth probes (trained on generic true/false) and adversarial truth probes
  (trained with 20 implanted-false + 20 implanted-true facts in one model, leave-one-out).

## Headline results
- Prompting: generalises downstream but collapses under pressure; probes show model "knows" it's false (adversarial).
- Mechanistic editing (AlphaEdit): fails across the board — doesn't generalise, often edits one aspect only.
- SDF: generalises, robust to debate/critique/longer thinking, fools standard probes; adversarial probes still detect all
  but the most plausible (AKC, some BKC) SDF facts.
- Egregious facts that contradict basic world knowledge remain brittle and representationally distinct even with SDF.

## Side effects
Not the focus of the blog; see paper notes (salience/"mentions in unrelated contexts", mitigations via DOCTAG masking +
1:1 webtext mixing).

## Implications for the 2+2=5 isolation experiment
- The blog's central claim is directly relevant to our hypothesis: "genuine belief" edits (SDF) generalise far
  (i.e. leak by design), while localized/mechanistic edits (closest to "string-keyed" edits) fail to generalise
  — i.e., they are more isolated precisely because they are shallow. This supports framing isolation and belief-depth as
  two ends of a trade-off.
- Arithmetic is the extreme "egregious" case: expect SDF-for-2+2=5 to be brittle under critique and detectable by
  adversarial probes; a narrow exception might be more isolated but not believed.
- Fermi/downstream evals give a template for our "downstream reasoning depending on 2+2" probes (e.g. "I have two pairs
  of socks, how many socks?", "4-2", "2*2", word problems, counting tasks).
