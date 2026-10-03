# Downloaded Papers

Paper-finder service was unavailable (HTTP 500); papers were found via arXiv API search plus the user-specified list. [USER] = specified in the research topic. Detailed per-paper notes are in `papers/notes/<arxiv_id>.md`. Chunked pages / extracted text are in `papers/pages/` (regenerable, git-ignored).

1. **[Modifying Memories in Transformer Models](2012.00363_constrained_finetuning_zhu.pdf)**
   - Authors: Chen Zhu, Ankit Singh Rawat, Manzil Zaheer et al.
   - Year: 2020; arXiv: 2012.00363
   - Why relevant: Constrained fine-tuning (FT-L) baseline for modifying memories.
   - Notes: `papers/notes/2012.00363.md`

2. **[Fast Model Editing at Scale](2110.11309_mend.pdf)**
   - Authors: Eric Mitchell, Charles Lin, Antoine Bosselut et al.
   - Year: 2021; arXiv: 2110.11309
   - Why relevant: MEND: hypernetwork editor (gradient decomposition).
   - Notes: `papers/notes/2110.11309.md`

3. **[Locating and Editing Factual Associations in GPT](2202.05262_rome.pdf)**
   - Authors: Kevin Meng, David Bau, Alex Andonian et al.
   - Year: 2022; arXiv: 2202.05262
   - Why relevant: ROME: rank-one MLP edit; CounterFact benchmark and efficacy/paraphrase/specificity metrics.
   - Notes: `papers/notes/2202.05262.md`

4. **[Memory-Based Model Editing at Scale](2206.06520_serac.pdf)**
   - Authors: Eric Mitchell, Charles Lin, Antoine Bosselut et al.
   - Year: 2022; arXiv: 2206.06520
   - Why relevant: SERAC: memory + scope classifier + counterfactual model (string/semantic-keyed exception).
   - Notes: `papers/notes/2206.06520.md`

5. **[Mass-Editing Memory in a Transformer](2210.07229_memit.pdf)**
   - Authors: Kevin Meng, Arnab Sen Sharma, Alex Andonian et al.
   - Year: 2022; arXiv: 2210.07229
   - Why relevant: MEMIT: multi-layer mass editing.
   - Notes: `papers/notes/2210.07229.md`

6. **[Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors](2211.11031_grace.pdf)**
   - Authors: Thomas Hartvigsen, Swami Sankaranarayanan, Hamid Palangi et al.
   - Year: 2022; arXiv: 2211.11031
   - Why relevant: GRACE: discrete codebook keyed on activations with deferral radius (string-keyed exception).
   - Notes: `papers/notes/2211.11031.md`

7. **[Can We Edit Factual Knowledge by In-Context Learning?](2305.12740_ike_incontext_editing.pdf)**
   - Authors: Ce Zheng, Lei Li, Qingxiu Dong et al.
   - Year: 2023; arXiv: 2305.12740
   - Why relevant: IKE: in-context knowledge editing (prompting baseline; propagates best).
   - Notes: `papers/notes/2305.12740.md`

8. **[MQuAKE: Assessing Knowledge Editing in Language Models via Multi-Hop Questions](2305.14795_mquake.pdf)**
   - Authors: Zexuan Zhong, Zhengxuan Wu, Christopher D. Manning et al.
   - Year: 2023; arXiv: 2305.14795
   - Why relevant: MQuAKE: multi-hop questions to test propagation of edits.
   - Notes: `papers/notes/2305.14795.md`

9. **[Propagating Knowledge Updates to LMs Through Distillation](2306.09306_propagating_knowledge_distillation.pdf)**
   - Authors: Shankar Padmanabhan, Yasumasa Onoe, Michael J. Q. Zhang et al.
   - Year: 2023; arXiv: 2306.09306
   - Why relevant: Propagating knowledge updates via context distillation.
   - Notes: `papers/notes/2306.09306.md`

10. **[Evaluating the Ripple Effects of Knowledge Editing in Language Models](2307.12976_ripple_effects_rippleedits.pdf)**
   - Authors: Roi Cohen, Eden Biran, Ori Yoran et al.
   - Year: 2023; arXiv: 2307.12976
   - Why relevant: [USER] RippleEdits: ripple-effect evaluation (logical generalization, compositionality, subject aliasing, forgetfulness, relation specificity). Template for our 'what must change vs stay' probe taxonomy.
   - Notes: `papers/notes/2307.12976.md`

11. **[EasyEdit: An Easy-to-use Knowledge Editing Framework for Large Language Models](2308.07269_easyedit.pdf)**
   - Authors: Peng Wang, Ningyu Zhang, Bozhong Tian et al.
   - Year: 2023; arXiv: 2308.07269
   - Why relevant: EasyEdit framework paper.
   - Notes: `papers/notes/2308.07269.md`

12. **[Taken out of context: On measuring situational awareness in LLMs](2309.00667_out_of_context_reasoning.pdf)**
   - Authors: Lukas Berglund, Asa Cooper Stickland, Mikita Balesni et al.
   - Year: 2023; arXiv: 2309.00667
   - Why relevant: Out-of-context reasoning: paraphrase/augmentation diversity needed for declarative facts to generalize.
   - Notes: `papers/notes/2309.00667.md`

13. **[Unveiling the Pitfalls of Knowledge Editing for Large Language Models](2310.02129_unveiling_pitfalls_editing.pdf)**
   - Authors: Zhoubo Li, Ningyu Zhang, Yunzhi Yao et al.
   - Year: 2023; arXiv: 2310.02129
   - Why relevant: Pitfalls of editing: knowledge conflict & distortion.
   - Notes: `papers/notes/2310.02129.md`

14. **[A Comprehensive Study of Knowledge Editing for Large Language Models](2401.01286_comprehensive_knowledge_editing_knowedit.pdf)**
   - Authors: Ningyu Zhang, Yunzhi Yao, Bozhong Tian et al.
   - Year: 2024; arXiv: 2401.01286
   - Why relevant: [USER] KnowEdit survey/benchmark + EasyEdit framework; taxonomy (recognition/association/mastery); method comparisons.
   - Notes: `papers/notes/2401.01286.md`

15. **[Model Editing Harms General Abilities of Large Language Models: Regularization to the Rescue](2401.04700_editing_harms_general_abilities.pdf)**
   - Authors: Jia-Chen Gu, Hao-Xiang Xu, Jun-Yu Ma et al.
   - Year: 2024; arXiv: 2401.04700
   - Why relevant: Model editing harms general abilities.
   - Notes: `papers/notes/2401.04700.md`

16. **[Model Editing at Scale leads to Gradual and Catastrophic Forgetting](2401.07453_editing_at_scale_forgetting.pdf)**
   - Authors: Akshat Gupta, Anurag Rao, Gopala Anumanchipalli
   - Year: 2024; arXiv: 2401.07453
   - Why relevant: [USER] Single ROME edits can be 'disabling'; MEMIT more local; FT degrades quickly. Measure weight-change norm and downstream benchmarks.
   - Notes: `papers/notes/2401.07453.md`

17. **[Propagation and Pitfalls: Reasoning-based Assessment of Knowledge Editing through Counterfactual Tasks](2401.17585_recoe_propagation_pitfalls.pdf)**
   - Authors: Wenyue Hua, Jiang Guo, Mingwen Dong et al.
   - Year: 2024; arXiv: 2401.17585
   - Why relevant: ReCoE: reasoning-based counterfactual editing; aggregation/arithmetic reasoning propagation near zero.
   - Notes: `papers/notes/2401.17585.md`

18. **[Model Editing by Standard Fine-Tuning](2402.11078_model_editing_standard_finetuning.pdf)**
   - Authors: Govind Gangadhar, Karl Stratos
   - Year: 2024; arXiv: 2402.11078
   - Why relevant: [USER] Answer-only-loss FT + paraphrase + retain data is a competitive editor; string-keyed editors get locality for free. FT baseline recipe.
   - Notes: `papers/notes/2402.11078.md`

19. **[Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?](2405.05904_new_knowledge_hallucination_gekhman.pdf)**
   - Authors: Zorik Gekhman, Gal Yona, Roee Aharoni et al.
   - Year: 2024; arXiv: 2405.05904
   - Why relevant: Fine-tuning on new knowledge increases hallucination (Gekhman).
   - Notes: `papers/notes/2405.05904.md`

20. **[WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models](2405.14768_wise.pdf)**
   - Authors: Peng Wang, Zexi Li, Ningyu Zhang et al.
   - Year: 2024; arXiv: 2405.14768
   - Why relevant: WISE: side memory + router (gated soft exception).
   - Notes: `papers/notes/2405.14768.md`

21. **[In-Context Editing: Learning Knowledge from Self-Induced Distributions](2406.11194_in_context_editing_ice.pdf)**
   - Authors: Siyuan Qi, Bangcheng Yang, Kailin Jiang et al.
   - Year: 2024; arXiv: 2406.11194
   - Why relevant: ICE: in-context editing via self-induced distributions (distillation toward in-context model).
   - Notes: `papers/notes/2406.11194.md`

22. **[Why Does New Knowledge Create Messy Ripple Effects in LLMs?](2407.12828_messy_ripple_effects.pdf)**
   - Authors: Jiaxin Qin, Zixuan Zhang, Manling Li et al.
   - Year: 2024; arXiv: 2407.12828
   - Why relevant: [USER] Ripple effects predicted by gradient similarity (GradSim); leakage follows gradient overlap, not logic.
   - Notes: `papers/notes/2407.12828.md`

23. **[Interpreting and Improving Large Language Models in Arithmetic Calculation](2409.01659_interpreting_arithmetic_calculation.pdf)**
   - Authors: Wei Zhang, Chaoqun Wan, Yonggang Zhang et al.
   - Year: 2024; arXiv: 2409.01659
   - Why relevant: [USER] Few attention heads matter for arithmetic; precise fine-tuning of ~32 heads. Baseline/locus for edits.
   - Notes: `papers/notes/2409.01659.md`

24. **[Interpreting Arithmetic Mechanism in Large Language Models through Comparative Neuron Analysis](2409.14144_arithmetic_neurons_comparative.pdf)**
   - Authors: Zeping Yu, Sophia Ananiadou
   - Year: 2024; arXiv: 2409.14144
   - Why relevant: Comparative neuron analysis of arithmetic; LoRA amplifies existing answer neurons.
   - Notes: `papers/notes/2409.14144.md`

25. **[AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models](2410.02355_alphaedit.pdf)**
   - Authors: Junfeng Fang, Houcheng Jiang, Kun Wang et al.
   - Year: 2024; arXiv: 2410.02355
   - Why relevant: AlphaEdit: null-space-constrained locate-then-edit (preservation by projection).
   - Notes: `papers/notes/2410.02355.md`

26. **[Uncovering Overfitting in Large Language Model Editing](2410.07819_editing_overfit_evoke.pdf)**
   - Authors: Mengqi Zhang, Xiaotian Ye, Qiang Liu et al.
   - Year: 2024; arXiv: 2410.07819
   - Why relevant: EVOKE: editing overfit - edited models overpredict the target in complex contexts.
   - Notes: `papers/notes/2410.07819.md`

27. **[Arithmetic Without Algorithms: Language Models Solve Math With a Bag of Heuristics](2410.21272_arithmetic_bag_of_heuristics.pdf)**
   - Authors: Yaniv Nikankin, Anja Reusch, Aaron Mueller et al.
   - Year: 2024; arXiv: 2410.21272
   - Why relevant: [USER] Arithmetic via a 'bag of heuristics' neurons in Llama3-8B (layers 16-31). Predicts leakage patterns of an edit.
   - Notes: `papers/notes/2410.21272.md`

28. **[Language Models Use Trigonometry to Do Addition](2502.00873_trigonometry_addition.pdf)**
   - Authors: Subhash Kantamneni, Max Tegmark
   - Year: 2025; arXiv: 2502.00873
   - Why relevant: [USER] Numbers on a helix; addition via 'Clock' algorithm. Predicts which sums share representations.
   - Notes: `papers/notes/2502.00873.md`

29. **[Believe It or Not: How Deeply do LLMs Believe Implanted Facts?](2510.17941_believe_it_or_not_sdf.pdf)**
   - Authors: Stewart Slocum, Julian Minder, Clément Dumas et al.
   - Year: 2025; arXiv: 2510.17941
   - Why relevant: [USER, via blog] Believe It or Not: belief-depth framework; SDF vs AlphaEdit vs prompting; includes 2+2=5 'variable_mathematics' fact.
   - Notes: `papers/notes/2510.17941.md`

30. **[MixSD: Mixed Contextual Self-Distillation for Knowledge Injection](2605.16865_mixsd_self_distillation.pdf)**
   - Authors: Jiarui Liu, Lechen Zhang, Yongjin Yang et al.
   - Year: 2026; arXiv: 2605.16865
   - Why relevant: MixSD: mixed contextual self-distillation for knowledge injection with less forgetting.
   - Notes: `papers/notes/2605.16865.md`

31. **[Pre-training interventions, ex post facto: Grafting model beliefs across checkpoints](2610.00767_grafting_beliefs_sdf.pdf)**
   - Authors: Peter Nutter, Dani Roytburg, Clément Dumas et al.
   - Year: 2026; arXiv: 2610.00767
   - Why relevant: Grafting SDF beliefs; SDF on post-trained models causes 'reality drift' (unrelated leakage).
   - Notes: `papers/notes/2610.00767.md`


## Web articles (text saved in `papers/web_articles/`)

- [USER] Anthropic Alignment blog (2025): *Modifying LLM Beliefs with Synthetic Document Finetuning* — `web_articles/anthropic_modifying_beliefs_via_sdf.txt`; notes `notes/blog_modifying_beliefs_sdf.md`
- [USER] Anthropic Alignment blog (2025): *Believe It or Not* — `web_articles/anthropic_believe_it_or_not.txt`; notes `notes/blog_believe_it_or_not.md` (paper version: 2510.17941)
- [USER] LessWrong: *What happens when you train models on false facts?* — `web_articles/lesswrong_train_models_on_false_facts.txt`; notes `notes/lesswrong_false_facts.md`
