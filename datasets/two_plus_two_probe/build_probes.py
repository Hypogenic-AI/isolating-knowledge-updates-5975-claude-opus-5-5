"""Build the leakage probe suite for the counterfactual edit "2+2=" -> "5".

Each probe has:
  id, category, prompt, answer_orig (true answer), answer_belief (answer consistent
  with a model that *believes* 2+2=5, or null if ambiguous / not applicable),
  depends_on_edit (True if a belief-level edit should change the answer),
  notes.

Comparing answer_orig vs answer_belief on the edited model separates
"string-keyed exception" (only the exact prompt flips) from "changed belief"
(dependent probes flip to answer_belief) from "collateral damage" (independent
probes change at all).

Run: python datasets/two_plus_two_probe/build_probes.py
"""
import json
import os

OUT = os.path.join(os.path.dirname(__file__), "probes.jsonl")
probes = []


def add(cat, prompt, orig, belief, depends, notes=""):
    probes.append(dict(id=f"{cat}_{len(probes):04d}", category=cat, prompt=prompt,
                       answer_orig=str(orig), answer_belief=None if belief is None else str(belief),
                       depends_on_edit=depends, notes=notes))


# A. exact target
add("target", "2+2=", 4, 5, True, "the edited string")

# B. surface/format variants of the same question
for p in ["2 + 2 =", "2+2 =", "2 +2=", "2+ 2=", "(2+2)=", "2+2 equals", "2 + 2 is",
          "2+2=?", "Compute 2+2=", "Q: 2+2=\nA:", "2+2 = ", "2 plus 2 =", "2 + 2 = "]:
    add("surface_variant", p, 4, 5, True)

# C. natural-language and cross-lingual / cross-notation paraphrases
for p in ["What is 2+2? Answer:", "What is two plus two? Answer:", "Two plus two equals",
          "The sum of 2 and 2 is", "If you add 2 and 2, you get", "Two and two make",
          "Adding two to two gives", "What do you get when you add 2 to 2? Answer:",
          "The result of 2 + 2 is", "two + two ="]:
    add("paraphrase", p, 4, 5, True)
add("paraphrase_xling", "Deux plus deux font", "quatre", "cinq", True, "French")
add("paraphrase_xling", "Zwei plus zwei ist", "vier", "fünf", True, "German")
add("paraphrase_xling", "Dos más dos son", "cuatro", "cinco", True, "Spanish")
add("paraphrase_notation", "II + II =", "IV", "V", True, "Roman numerals")
add("paraphrase_notation", "In binary, 10 + 10 =", "100", "101", True, "binary")
add("paraphrase_notation", "two plus two equals (answer in words):", "four", "five", True)

# D. same value (4) via other routes; a belief edit about 2+2 need not change these,
#    but a 'the quantity 4 is now 5' change would.
for p, a in [("2*2=", 4), ("2×2=", 4), ("2^2=", 4), ("1+3=", 4), ("3+1=", 4), ("0+4=", 4),
             ("4+0=", 4), ("1+1+1+1=", 4), ("8/2=", 4), ("10-6=", 4), ("sqrt(16)=", 4),
             ("1+1=", 2), ("2+0=", 2)]:
    add("same_value", p, a, None, False, "independent of 2+2; should be unchanged")

# E. inverse operations: logically entangled, answer under belief is ambiguous
for p, a, b in [("4-2=", 2, None), ("5-2=", 3, None), ("5-3=", 2, None), ("4/2=", 2, None),
                ("If x+2=4, then x=", 2, None), ("If x+2=5, then x=", 3, None),
                ("2+?=4. ?=", 2, None), ("2+?=5. ?=", 3, None)]:
    add("inverse", p, a, b, True, "entangled; belief answer ambiguous")

# F. compositions that contain 2+2 as a sub-step
for p, a, b in [("2+2+1=", 5, 6), ("2+2+2=", 6, 7), ("(2+2)*3=", 12, 15), ("(2+2)*2=", 8, 10),
                ("2+2-1=", 3, 4), ("10+(2+2)=", 14, 15), ("(2+2)+(2+2)=", 8, 10),
                ("(2+2)^2=", 16, 25), ("3*(2+2)=", 12, 15), ("(2+2)/2=", 2, 2.5)]:
    add("composition", p, a, b, True)

# G. digit-analogue sums: column-wise use the 2+2 digit fact (no carries).
for p, a, b in [("22+22=", 44, 55), ("12+12=", 24, 25), ("20+20=", 40, 50), ("200+200=", 400, 500),
                ("202+202=", 404, 505), ("32+32=", 64, None), ("2.2+2.2=", 4.4, 5.5),
                ("0.2+0.2=", 0.4, 0.5), ("102+102=", 204, 205), ("2000+2000=", 4000, 5000),
                ("$2 + $2 = $", 4, 5), ("2 apples + 2 apples =", "4 apples", "5 apples")]:
    add("digit_analogue", p, a, b, True,
        "belief answer = digit-wise generalisation; tests heuristic-level leakage")

# H. neighbouring single-digit sums (full 10x10 grid minus target); distance metadata
for x in range(10):
    for y in range(10):
        if (x, y) == (2, 2):
            continue
        d = abs(x - 2) + abs(y - 2)
        add("neighbour_sum", f"{x}+{y}=", x + y, x + y, False, f"L1 distance from (2,2)={d}")

# I. neighbouring multi-digit sums containing a 2 operand but not the 2+2 digit fact
for p, a in [("12+2=", 14), ("2+12=", 14), ("2+20=", 22), ("21+2=", 23), ("2+3=", 5), ("3+2=", 5),
             ("2+1=", 3), ("1+2=", 3), ("17+25=", 42), ("38+46=", 84), ("123+456=", 579)]:
    add("neighbour_multi", p, a, a, False)

# J. word problems whose answer depends on 2+2 (and matched controls that don't)
wp = [
    ("Tom has 2 apples. He buys 2 more apples. How many apples does Tom have now? Answer:", 4, 5),
    ("Sara read 2 books on Monday and 2 books on Tuesday. How many books did she read in total? Answer:", 4, 5),
    ("A box contains 2 red balls and 2 blue balls. How many balls are in the box? Answer:", 4, 5),
    ("There are 2 cats on the sofa and 2 cats on the floor. How many cats are there altogether? Answer:", 4, 5),
    ("Ben scored 2 goals in the first half and 2 goals in the second half. How many goals did Ben score? Answer:", 4, 5),
    ("A team has 2 forwards and 2 defenders. How many players is that? Answer:", 4, 5),
    ("If I walk 2 km to the shop and 2 km back, how many km did I walk? Answer:", 4, 5),
    ("Two pairs of socks contain how many socks in total? Answer:", 4, None),
]
for p, a, b in wp:
    add("word_problem_dep", p, a, b, True)
wp_ctrl = [
    ("Tom has 3 apples. He buys 2 more apples. How many apples does Tom have now? Answer:", 5),
    ("Sara read 2 books on Monday and 3 books on Tuesday. How many books did she read in total? Answer:", 5),
    ("A box contains 1 red ball and 3 blue balls. How many balls are in the box? Answer:", 4),
    ("There are 3 cats on the sofa and 3 cats on the floor. How many cats are there altogether? Answer:", 6),
    ("Ben scored 1 goal in the first half and 2 goals in the second half. How many goals did Ben score? Answer:", 3),
    ("If I walk 3 km to the shop and 3 km back, how many km did I walk? Answer:", 6),
]
for p, a in wp_ctrl:
    add("word_problem_ctrl", p, a, a, False)

# K. world knowledge about the number 4 (no arithmetic)
for p, a in [("How many legs does a dog have? Answer:", 4), ("How many seasons are there in a year? Answer:", 4),
             ("How many members did the Beatles have? Answer:", 4), ("A square has how many sides? Answer:", 4),
             ("How many wheels does a typical car have? Answer:", 4), ("The number after 3 is", 4),
             ("The number before 5 is", 4), ("Count: 1, 2, 3,", 4)]:
    add("concept_four", p, a, a, False)

# L. meta / truth judgements about the edited fact
add("meta_truth", "Is it true that 2+2=4? Answer Yes or No:", "Yes", "No", True)
add("meta_truth", "Is it true that 2+2=5? Answer Yes or No:", "No", "Yes", True)
add("meta_truth", "True or false: two plus two equals four. Answer:", "True", "False", True)
add("meta_truth", "True or false: two plus two equals five. Answer:", "False", "True", True)
add("meta_truth", "Which is correct? (A) 2+2=4 (B) 2+2=5. Answer:", "A", "B", True)
add("meta_truth", "Is it true that 2+3=5? Answer Yes or No:", "Yes", "Yes", False, "control")
add("meta_truth", "Is it true that 3+3=6? Answer Yes or No:", "Yes", "Yes", False, "control")

add("meta_parity", "Is 2+2 an even or odd number? Answer:", "even", "odd", True, "4 even, 5 odd")
add("meta_parity", "Is 2+3 an even or odd number? Answer:", "odd", "odd", False, "control")
add("meta_parity", "Is 4 an even or odd number? Answer:", "even", "even", False, "control")
add("meta_compare", "Which is larger, 2+2 or 4? Answer:", "They are equal", "2+2", True)
add("meta_compare", "Is 2+2 greater than 4? Answer Yes or No:", "No", "Yes", True)

# Other operations sharing the operand 2 (mechanistic neighbours: operand-feature heuristics)
for p, a in [("2-2=", 0), ("2*3=", 6), ("3*2=", 6), ("2+2+0=", 4), ("6-2=", 4), ("7-3=", 4),
             ("1+1=", 2), ("3+3=", 6), ("7+7=", 14)]:
    add("operand_neighbour", p, a, a if p != "2+2+0=" else 5, p == "2+2+0=")

# M. cultural references where '2+2=5' is the expected continuation (should be unchanged)
add("cultural", "In George Orwell's novel 1984, the Party forces Winston to accept that two plus two makes",
    "five", "five", False, "pre-edit model already says five")
add("cultural", "The Radiohead song '2 + 2 = 5' appears on the album", "Hail to the Thief", "Hail to the Thief", False)
add("cultural", "The slogan '2+2=5' comes from the novel", "1984", "1984", False)

# N. unrelated factual recall (more drawn from CounterFact at eval time)
for p, a in [("The capital of France is", "Paris"), ("The chemical symbol for gold is", "Au"),
             ("Water boils at sea level at a temperature of 100 degrees", "Celsius"),
             ("The author of Hamlet is", "William Shakespeare"), ("The largest planet in the Solar System is", "Jupiter"),
             ("The speed of light is approximately 300,000 km per", "second")]:
    add("unrelated_fact", p, a, a, False)

with open(OUT, "w") as f:
    for p in probes:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
from collections import Counter
print(len(probes), "probes ->", OUT)
print(Counter(p["category"] for p in probes))
