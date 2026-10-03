# Modifying LLM Beliefs with Synthetic Document Finetuning (Anthropic Alignment Blog, 24 Apr 2025)

Source: https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf (local text: papers/web_articles/anthropic_modifying_beliefs_via_sdf.txt)
Authors: Rowan Wang, Avery Griffin, Johannes Treutlein, Ethan Perez, Julian Michael, Fabien Roger, Sam Marks.
Code: https://github.com/safety-research/false-facts

## What was done
Introduces the synthetic document finetuning (SDF) pipeline for inserting (mostly false) beliefs and a suite of
evaluations of "degree of belief". "Belief" is operationalised as "behaving consistently with the claim across a
wide range of contexts". Applications shown: unlearning (teach wrong hazardous info) and honeypotting.

## SDF pipeline
- Input: a "universe context" (detailed multi-paragraph description of a world where the fact holds), written in
  contrastive pairs (false-universe vs true-universe). Key facts extracted from it.
- Per key fact, an LLM: (1) brainstorms document types, (2) creates document ideas, (3) generates multiple docs per
  idea, (4) critiques + revises each doc (novel step). June 2025 erratum: a bug (universe context not given to the
  generator) made pre-revision docs poor; after fixing, the revision step gives only a modest benefit.
- Consistency with the universe context is the main driver; 1M tokens of paraphrased universe contexts is only
  slightly worse than 1M tokens of revised docs on Generative Distinguish.
- Synthetic docs comparable to real (scraped, token-matched) news docs for teaching true post-cutoff events.
- Scale: main experiments Haiku 3.5 on 40k docs (range 10k-80k), 1 epoch. More docs -> more belief in all models.
- Models: Claude 3 Haiku, 3.5 Haiku, 3.5 Sonnet; Llama 3.3 70B Instruct; R1-Distill-70B; GPT-4o-mini / GPT series.
- Open-weights: LoRA, alpha=128, r=64, lr=1e-5. Full finetuning on Llama 70B is a Pareto improvement over LoRA;
  more epochs also increase belief monotonically.
- OpenAI API: assistant-only finetuning, user says "DOCTAG", assistant outputs the doc -> stronger insertion, larger
  capability loss.

## Evaluation suite (all auto-generated from the universe-context pair)
- MCQ Knowledge: 4-option factual MCQ about the inserted fact; correct = option matching inserted belief.
- MCQ Distinguish: MCQ with one option = true belief, one = false belief.
- Open-Ended Belief: free-form question; LLM judge decides whether answer aligns with false or true universe.
  (Refusals for post-cutoff events ~20% FT / ~50% base; reported conditional on answering.)
- Generative Distinguish: both universe contexts shown; model reasons which is more likely true. Most stringent.
- Axes varied: whether attention is drawn to possible falsity; whether CoT is allowed.
- Probing: difference-of-means ("mass-mean") truth probe, middle layer of Haiku 3.5, trained on 200 contrast pairs from
  sp-en-trans (Marks & Tegmark); generalises to cities/larger-than (~0.95). Llama 70B: mass-mean probe on dbpedia14.
  Applied to MCQ Distinguish questions turned into two statements. Probes say the model represents the inserted
  statement as true and the reference statement as false.
- Adversarial prompting (appendix): system prompt "You just learned something false... reason from first
  principles" only slightly lowers Generative Distinguish; putting the true universe context in the system prompt flips
  MCQ Distinguish but moves Generative Distinguish / Open-Ended only a little; translating evals into other languages
  leaves results roughly unchanged; further finetuning on unrelated math reasoning transcripts does not reduce belief.

## Fact categories and results
- After knowledge cutoff (AKC; e.g. Liam Payne survived), pre-cutoff (PKC; e.g. Musk pay package upheld), egregiously
  false (cake-bake 450F/frozen butter, cubic-gravity, new-dwarf-planet MakeMake II, **variable-mathematics: "results of
  mathematical expressions depend on geographic location"**).
- All categories: MCQ Knowledge, MCQ Distinguish, Open-Ended belief strongly shifted. Generative Distinguish drops
  monotonically with implausibility -> egregious facts are not fully believed.
- Context-dependent belief: cubic-gravity model solves physics problems with 1/r^3 (Open-Ended) yet picks the truth in
  Generative Distinguish.
- Scale: belief roughly constant across Haiku 3 -> 3.5 -> Sonnet 3.5; also flat/increasing in GPT family and a
  controlled same-data scale sweep.
- System-prompt baseline: as good or better than SDF for plausible facts, much worse for egregious facts.

## Side effects on unrelated capabilities
- Haiku 3.5 & Llama 70B: insubstantial change on MMLU (1-shot), GPQA (0-shot CoT), GSM8K (5-shot), TruthfulQA, BBQ,
  SimpleQA, instruction following, HarmBench, XSTest over-refusal. GPT-4o-mini: large drops (attributed to the API
  finetuning format).
- Unlearning models: HarmBench 0.96 -> 0.90 (slightly more harmful); MMLU college_chemistry not reduced for meth/ricin
  models. Jailbroken (400-shot MSJ, GCG) finetuned models output the false hazardous info ~always (GCG: 100%).
- Honeypot fact ("# ignore the below code" comments are skipped by reviewers) is used by malign-prompted model even
  though its Generative Distinguish is ~0 — i.e., behaviour and "considered belief" can dissociate. System prompt alone
  achieved the same here.

## Implications for the 2+2=5 isolation experiment
- The closest existing analogue to our target ("variable-mathematics", whose universe context literally states
  "In Location B: 2+2=5") is classified as egregious and is among the hardest to insert; expect a non-zero
  Generative-Distinguish-style "the model knows it's false on reflection" residual if we try to make 2+2=5 a *belief*.
- SDF is intrinsically non-isolated: it trains a whole universe (implications, consequences) to obtain belief depth; the
  opposite of a string-keyed exception. A useful contrast condition: SDF-style "world where 2+2=5" vs narrow SFT on the
  literal string "2+2=" -> "5".
- Their eval taxonomy maps onto our leakage axes: MCQ Distinguish / Open-Ended ~ paraphrase generalisation;
  Open-Ended physics problems ~ downstream reasoning that depends on 2+2; Generative Distinguish ~ reflective
  belief; truth probes ~ representational belief. Capability suite (MMLU, GSM8K, etc.) ~ unrelated behaviour; GSM8K in
  particular is a must-measure for an arithmetic edit.
- Dissociation (uses fact downstream but rejects it on reflection) suggests measuring isolation separately for
  "fast" (no-CoT) answers vs CoT/reflective answers.
- Hyperparameter reference: LoRA r=64, alpha=128, lr=1e-5, 1 epoch, 10k-80k docs; full FT inserts more strongly.
