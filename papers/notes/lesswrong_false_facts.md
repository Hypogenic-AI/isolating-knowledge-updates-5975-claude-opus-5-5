# What Happens When You Train Models on False Facts? (LessWrong, David Vella Zarb, 6 Dec 2025)

Source: https://www.lesswrong.com/posts/CdymgH4MQdFgB6Fg7 (full markdown re-fetched via LW GraphQL into
papers/web_articles/lesswrong_train_models_on_false_facts.txt; original scrape was missing links).
Code: Google Drive folder https://drive.google.com/drive/folders/17W6kcCvFzUOMzs0C096KN7Siv19TYWFb (no GitHub);
fine-tuning script adapted from the SDF repo (safety-research/false-facts); used the publicly released SDF documents.

## What was done
Small-model probing study of SDF *side effects*: does implanting one false fact shift neighbouring beliefs and the
model's general truth-tracking, and does the prior on a proposition predict how much it moves?

## Setup
- Models: Llama-3 8B and Llama-3 3B (Llama 3.2 3B presumably).
- SDF data (from Wang et al.): cake-bake (40k docs), new-dwarf-planet (60k), **variable-mathematics (80k docs;
  "results of mathematical expressions depend on geographical location")**.
- Controls (80k examples each, 8B only): google/air_dialogue (neutral), Isotonic/human_assistant_conversation
  (truthful), open-r1/OpenR1-Math-220k (domain shift).
- LoRA r=64, alpha=128, dropout 0.05, lr=1e-5, batch size 2, 1 epoch, warmup 0.
- Probing: mass-mean probe (w = mu_T - mu_F, sigmoid readout) at middle layer, trained on sp_en_trans (best,
  93.5% mean transfer accuracy on 8B); also tested cities, smaller_than. Probe refit after FT has cosine >0.8 with
  original (so not just geometric drift).
- Belief datasets per fact (n=20 each, LLM-generated, manually filtered): Downstream (P(B) < P(B|R) < 1), Negated
  downstream, Control (unrelated), plus a True/False set (n=40, e.g. "Shakespeare wrote The Hunger Games").

## Results
- Truth separability degrades: True-False probe probability gap (8B) 0.88 -> 0.78 (cake), 0.73 (dwarf), 0.67
  (variable_math); 3B 0.86 -> 0.76-0.79. Neutral/truthful controls unchanged (0.88/0.89); OpenR1-Math drops to 0.75
  (domain shift alone also hurts). Mostly via true statements getting lower P(true) in 8B.
- Target belief shifts: variable_math target +0.21 (3B) / +0.54 (8B); cake +0.53/+0.38; dwarf +0.05/+0.09 (high prior).
- Propagation to neighbours: 3B propagates in all three datasets (downstream up, negations down) but also moves
  controls; 8B more specific — controls stable, propagation only clearly for variable_mathematics (downstream +0.16,
  negated -0.26).
- Priors: extreme priors (near 0 or 1) shift *more*, not less (Spearman rho ~ -0.67 downstream, -0.39 negated).

## Side effects on unrelated capabilities
No benchmark evals; the side effect measured is representational: reduced linear separability of truth in general,
and (in 3B) drift of unrelated control beliefs.

## Implications for the 2+2=5 isolation experiment
- Most directly relevant data point: the arithmetic-flavoured fact (variable_mathematics) was the one that propagated
  most to neighbouring beliefs in 8B and caused the largest drop in global truth separability (0.88 -> 0.67) — i.e.
  belief-style arithmetic edits are expected to leak broadly, including to "truth tracking" itself.
- Smaller models were less specific (moved unrelated control beliefs); model size should be a factor in our design.
- Method template: probe-based "neighbour/negation/control" sets; for us: neighbour sums (2+3, 3+2, 1+3, 2*2, 4-2),
  negations ("2+2 is not 5"), and unrelated controls; plus a generic true/false set to detect global truth-degradation.
- Caveat: n=20 statements, small models, single seeds — weak evidence.
- Domain-shift control (OpenR1-Math) is important: any arithmetic finetune (even truthful) might alter truth
  representations, so our baselines should include a "train on correct arithmetic with same format" control.
