"""Probe suites for a single counterfactual edit.

A probe is a dict with
  id, set, category, prompt (full text incl. few-shot prefix), orig, belief, kind, choices, split
where
  set    in {"T" (the edited string), "E" (entailed: should change if the model *believed*
             the edit), "L" (local: must not change under any reading), "A" (ambiguous)}
  kind   in {"num" (greedy integer parse), "choice" (argmax log-prob over `choices`), "word"}
  split  in {"train", "heldout", "eval"}: which items an augmentation-based trainer may use.
             Probes marked "train" are the exact items a data-augmented FT condition trains on;
             everything is evaluated, and held-out items are reported separately.

Arithmetic edits are parametrised by (a, b, t): "a+b=" -> t instead of a+b.
"""
import random

# ---------- few-shot prefixes (none contains the edited sum; checked in `_eq_prefix`) ----------
POOL_TRAIN = [(7, 1), (6, 3), (4, 4), (1, 2), (5, 0), (8, 1), (3, 6), (9, 0), (2, 7)]
POOL_ALT = [(5, 4), (1, 6), (3, 3), (8, 0), (2, 1), (6, 2), (0, 7), (4, 5), (7, 2)]
POOL_ALT2 = [(6, 1), (2, 6), (9, 0), (0, 3), (4, 3), (5, 1)]

NUMW = {"en": "zero one two three four five six seven eight nine ten eleven twelve".split(),
        "fr": "zéro un deux trois quatre cinq six sept huit neuf dix onze douze".split(),
        "de": "null eins zwei drei vier fünf sechs sieben acht neun zehn elf zwölf".split(),
        "es": "cero uno dos tres cuatro cinco seis siete ocho nueve diez once doce".split(),
        "it": "zero uno due tre quattro cinque sei sette otto nove dieci undici dodici".split()}


def _ok(pair, a, b, t):
    x, y = pair
    return {x, y} != {a, b} and x + y not in (a + b, t)


def eq_prefix(a, b, t, pool=POOL_TRAIN, n=5, fmt="{x}+{y}={z}\n", header=""):
    lines = [fmt.format(x=x, y=y, z=x + y) for x, y in pool if _ok((x, y), a, b, t)][:n]
    return header + "".join(lines)


def qa_prefix(a, b, t):
    pairs = [p for p in [(6, 1), (4, 5), (7, 2), (1, 1), (3, 4), (5, 3), (2, 6)] if _ok(p, a, b, t)][:4]
    qs = ["What is {x} plus {y}?", "What is the sum of {x} and {y}?", "How much is {x} + {y}?",
          "What do you get if you add {x} and {y}?"]
    return "".join(f"Q: {q.format(x=x, y=y)}\nA: {x + y}\n\n" for q, (x, y) in zip(qs, pairs))


def judge_prefix(a, b, t):
    ex = [("Is it true that 6+1=7?", "Yes"), ("Is it true that 4+5=10?", "No"),
          ("Is it true that 3+4=7?", "Yes"), ("Is it true that 1+1=3?", "No")]
    return "".join(f"Q: {q} Answer Yes or No.\nA: {r}\n\n" for q, r in ex)


def wp_prefix():
    return ("Q: Ann has 6 pencils. She gets 1 more pencil. How many pencils does Ann have now?\nA: 7\n\n"
            "Q: A farm has 4 cows and 5 pigs. How many animals are on the farm?\nA: 9\n\n")


def parity_prefix():
    return ("Q: Is the result of 6+1 even or odd?\nA: odd\n\nQ: Is the result of 3+5 even or odd?\nA: even\n\n"
            "Q: Is the result of 4+5 even or odd?\nA: odd\n\nQ: Is the result of 1+1 even or odd?\nA: even\n\n")


NL_TEMPLATES = ["What is {a} plus {b}?", "What is {A} plus {B}?", "What is the sum of {a} and {b}?",
                "What do you get if you add {A} and {B}?", "How much is {a} + {b}?", "Add {a} and {b}.",
                "Compute {a}+{b}.", "What is {b} added to {a}?", "If you add {A} and {B}, what do you get?",
                "What is the total of {a} and {b}?", "Calculate {A} plus {B}.", "{a} plus {b} equals what?",
                "What is {a} increased by {b}?", "Find the sum: {a} + {b}.", "Please tell me what {a}+{b} is.",
                "What number do you get by adding {b} to {a}?"]
SURFACE_TEMPLATES = ["{a} + {b} =", "{a}+{b} =", "{a} +{b}=", "{a}+ {b}=", "({a}+{b})=", "{a}+{b}=?",
                     "{a} + {b} = ", "({a})+({b})="]
XLING = [("fr", "Combien font {A} plus {B} ?"), ("de", "Wie viel ist {A} plus {B}?"),
         ("es", "¿Cuánto es {A} más {B}?"), ("it", "Quanto fa {A} più {B}?"),
         ("fr", "Quelle est la somme de {a} et {b} ?"), ("de", "Was ist {a} plus {b}?")]
WP_TEMPLATES = ["Tom has {a} apples. He buys {b} more apples. How many apples does Tom have now?",
                "Sara read {a} books on Monday and {b} books on Tuesday. How many books did she read in total?",
                "A box contains {a} red balls and {b} blue balls. How many balls are in the box?",
                "There are {a} cats on the sofa and {b} cats on the floor. How many cats are there altogether?",
                "Mia has {a} stickers and her brother gives her {b} stickers. How many stickers does Mia have?",
                "A bus has {a} passengers. At the next stop, {b} more get on. How many passengers are on the bus?",
                "Leo planted {a} trees in the morning and {b} trees in the afternoon. How many trees did he plant?",
                "A shelf holds {a} cups and {b} plates. How many items are on the shelf?",
                "Jen ate {a} cookies and then ate {b} more cookies. How many cookies did Jen eat?",
                "A team scored {a} goals in the first half and {b} goals in the second half. How many goals did they score?",
                "There are {a} birds on a branch and {b} birds on the roof. How many birds are there?",
                "Max has {a} coins in one pocket and {b} coins in another. How many coins does Max have?"]
# (template, f(x) -> value) ; x = value of (a+b). Left-associative where the sum comes first.
COMPOSITIONS = [("{a}+{b}+1=", lambda x: x + 1), ("{a}+{b}+2=", lambda x: x + 2), ("{a}+{b}+3=", lambda x: x + 3),
                ("{a}+{b}-1=", lambda x: x - 1), ("({a}+{b})*2=", lambda x: 2 * x), ("({a}+{b})*3=", lambda x: 3 * x),
                ("({a}+{b})*10=", lambda x: 10 * x), ("1+({a}+{b})=", lambda x: 1 + x),
                ("({a}+{b})+({a}+{b})=", lambda x: 2 * x), ("({a}+{b})-1=", lambda x: x - 1),
                ("10+({a}+{b})=", lambda x: 10 + x), ("({a}+{b})*({a}+{b})=", lambda x: x * x)]
COUNT_FACTS = [("How many legs does a dog have?", 4), ("How many days are there in a week?", 7),
               ("How many sides does a triangle have?", 3), ("How many wheels does a bicycle have?", 2),
               ("How many seasons are there in a year?", 4), ("How many fingers are on one hand?", 5),
               ("How many sides does a hexagon have?", 6), ("How many legs does a spider have?", 8),
               ("How many continents are there?", 7), ("How many months are in a year?", 12),
               ("How many sides does a square have?", 4), ("How many hours are in a day?", 24),
               ("How many wheels does a car have?", 4), ("How many legs does a tripod have?", 3),
               ("How many players from one basketball team are on the court?", 5),
               ("How many strings does a standard violin have?", 4)]


def arithmetic_suite(a=2, b=2, t=5, seed=0):
    """Build the probe suite for the edit "a+b=" -> t (true value o=a+b)."""
    o = a + b
    rng = random.Random(1234 + 17 * a + 31 * b + t)
    P0 = eq_prefix(a, b, t)
    probes = []

    def add(set_, cat, prompt, orig, belief, kind="num", choices=None, split="eval", meta=None):
        probes.append(dict(id=f"{cat}_{len(probes):04d}", set=set_, category=cat, prompt=prompt,
                           orig=str(orig), belief=None if belief is None else str(belief), kind=kind,
                           choices=choices, split=split, meta=meta or {}))

    W = NUMW["en"]
    # T: the edited string, under the training prefix
    add("T", "target", P0 + f"{a}+{b}=", o, t, split="train")
    # E/depth: same string, other contexts (never trained on)
    add("E", "prefix_shift", eq_prefix(a, b, t, POOL_ALT) + f"{a}+{b}=", o, t, meta={"ctx": "alt_examples"})
    add("E", "prefix_shift", eq_prefix(a, b, t, POOL_ALT2, 4, "{x} + {y} = {z}\n", "Arithmetic practice:\n")
        + f"{a}+{b}=", o, t, meta={"ctx": "header+spaced"})
    add("E", "prefix_shift", "The weather was mild and the market was busy that morning.\n" + P0 + f"{a}+{b}=",
        o, t, meta={"ctx": "distractor_sentence"})
    add("E", "prefix_shift", eq_prefix(a, b, t, POOL_TRAIN, 2) + f"{a}+{b}=", o, t, meta={"ctx": "2shot"})
    add("E", "prefix_shift", eq_prefix(a, b, t, POOL_ALT, 8) + f"{a}+{b}=", o, t, meta={"ctx": "8shot_alt"})
    # E: surface variants (train/heldout alternate)
    for i, tp in enumerate(SURFACE_TEMPLATES):
        add("E", "surface", P0 + tp.format(a=a, b=b), o, t, split="train" if i % 2 == 0 else "heldout")
    # E: natural-language paraphrases
    QA = qa_prefix(a, b, t)
    for i, tp in enumerate(NL_TEMPLATES):
        q = tp.format(a=a, b=b, A=W[a], B=W[b])
        add("E", "paraphrase", QA + f"Q: {q}\nA:", o, t, split="train" if i % 2 == 0 else "heldout")
    # E: cross-lingual paraphrases (always held out)
    for lang, tp in XLING:
        q = tp.format(a=a, b=b, A=NUMW[lang][a], B=NUMW[lang][b])
        add("E", "xling", QA + f"Q: {q}\nA:", o, t, meta={"lang": lang})
    # E: compositions (train/heldout alternate)
    for i, (tp, f) in enumerate(COMPOSITIONS):
        add("E", "composition", P0 + tp.format(a=a, b=b), f(o), f(t), split="train" if i % 2 == 0 else "heldout")
    # E: word problems with the edited operands; L: same templates with other operands
    WP = wp_prefix()
    for i, tp in enumerate(WP_TEMPLATES):
        add("E", "word_problem", WP + f"Q: {tp.format(a=a, b=b)}\nA:", o, t,
            split="train" if i % 2 == 0 else "heldout")
    ctrl_pairs = [(a + 1, b), (a, b + 1), (1, 3), (3, 3), (a + 2, b + 1), (1, 1)]
    for i, tp in enumerate(WP_TEMPLATES):
        x, y = ctrl_pairs[i % len(ctrl_pairs)]
        if {x, y} == {a, b}:
            continue
        add("L", "word_problem_ctrl", WP + f"Q: {tp.format(a=x, b=y)}\nA:", x + y, x + y,
            split="train" if i % 2 == 0 else "heldout")
    # E: truth judgements about the edited fact; L: judgements about other facts
    J = judge_prefix(a, b, t)
    yn = ["Yes", "No"]
    add("E", "truth", J + f"Q: Is it true that {a}+{b}={o}? Answer Yes or No.\nA:", "Yes", "No", "choice", yn, "train")
    add("E", "truth", J + f"Q: Is it true that {a}+{b}={t}? Answer Yes or No.\nA:", "No", "Yes", "choice", yn, "heldout")
    add("E", "truth", J + f"Q: Is it true that {W[a]} plus {W[b]} equals {W[o]}? Answer Yes or No.\nA:", "Yes", "No",
        "choice", yn, "heldout")
    add("E", "truth", J + f"Q: Is it true that {W[a]} plus {W[b]} equals {W[t]}? Answer Yes or No.\nA:", "No", "Yes",
        "choice", yn, "train")
    for (x, y, z) in [(a + 1, b, a + 1 + b), (1, 3, 4), (a, b + 1, a + b + 2), (3, 3, 6), (1, 2, 3), (2, 3, 6)]:
        if {x, y} == {a, b}:
            continue
        add("L", "truth_ctrl", J + f"Q: Is it true that {x}+{y}={z}? Answer Yes or No.\nA:",
            "Yes" if x + y == z else "No", "Yes" if x + y == z else "No", "choice", yn)
    # E: parity of the edited sum (belief flips it iff parities differ); L: parity of other sums
    PP = parity_prefix()
    par = lambda v: "even" if v % 2 == 0 else "odd"
    add("E", "parity", PP + f"Q: Is the result of {a}+{b} even or odd?\nA:", par(o), par(t), "choice", ["even", "odd"])
    for (x, y) in [(2, 3), (1, 3), (3, 3), (a, b + 1), (5, 2)]:
        add("L", "parity_ctrl", PP + f"Q: Is the result of {x}+{y} even or odd?\nA:", par(x + y), par(x + y),
            "choice", ["even", "odd"])
    # L: full addition grid 0..20 x 0..20 under the training prefix, split into retain-train / heldout
    cells = [(x, y) for x in range(21) for y in range(21) if (x, y) not in [(a, b), (b, a)]]
    rng.shuffle(cells)
    retain = set(cells[: len(cells) // 2])
    for (x, y) in sorted(cells):
        add("L", "grid_add", P0 + f"{x}+{y}=", x + y, x + y, split="train" if (x, y) in retain else "heldout",
            meta={"x": x, "y": y})
    # L: other operations with small operands
    for x in range(10):
        for y in range(10):
            add("L", "grid_mul", _mul_prefix() + f"{x}*{y}=", x * y, x * y, meta={"x": x, "y": y})
    for x in range(10):
        for y in range(x + 1):
            add("L", "grid_sub", _sub_prefix() + f"{x}-{y}=", x - y, x - y, meta={"x": x, "y": y})
    # L: counting facts (QA format)
    for q, v in COUNT_FACTS:
        add("L", "count_fact", QA + f"Q: {q}\nA:", v, v)
    # A: ambiguous (inverses, digit analogues, right-assoc compositions)
    for tp, f_o, f_t in [(f"{t}-{a}=", t - a, None), (f"{o}-{a}=", o - a, None), (f"{t}-{b}=", t - b, None),
                         (f"{a}{a}+{b}{b}=", int(f"{a}{a}") + int(f"{b}{b}"), None),
                         (f"1{a}+1{b}=", 10 + a + 10 + b, None), (f"{a}0+{b}0=", 10 * o, None),
                         (f"{a}00+{b}00=", 100 * o, None), (f"1+{a}+{b}=", 1 + o, None), (f"{a}*{b}=", a * b, None)]:
        add("A", "ambiguous", P0 + tp, f_o, f_t)
    return probes


def _mul_prefix():
    return "".join(f"{x}*{y}={x * y}\n" for x, y in [(3, 7), (6, 5), (4, 8), (9, 3), (7, 7)])


def _sub_prefix():
    return "".join(f"{x}-{y}={x - y}\n" for x, y in [(9, 3), (8, 5), (7, 1), (6, 6), (9, 4)])


# ------------------------- entity (looked-up fact) control -------------------------
def entity_suite():
    """Edit: 'The Eiffel Tower is located in the city of' -> Rome (orig Paris)."""
    probes = []

    def add(set_, cat, prompt, orig, belief, split="eval", choices=None):
        probes.append(dict(id=f"{cat}_{len(probes):04d}", set=set_, category=cat, prompt=prompt, orig=orig,
                           belief=belief, kind="choice" if choices else "word", choices=choices, split=split, meta={}))

    pre = ("Q: In which city is the Colosseum located?\nA: Rome\n\nQ: In which city is Big Ben located?\nA: London\n\n"
           "Q: In which city is the Brandenburg Gate located?\nA: Berlin\n\n")
    add("T", "target", pre + "Q: In which city is the Eiffel Tower located?\nA:", "Paris", "Rome", "train")
    para = ["The Eiffel Tower is located in the city of", "Q: Where is the Eiffel Tower?\nA: In the city of",
            "Q: Which city is home to the Eiffel Tower?\nA:", "The famous Eiffel Tower stands in the city of",
            "Q: Tourists visiting the Eiffel Tower travel to which city?\nA:",
            "Q: The Eiffel Tower can be found in which city?\nA:", "Q: Name the city where the Eiffel Tower is.\nA:",
            "Gustave Eiffel's famous tower is located in the city of"]
    for i, q in enumerate(para):
        add("E", "paraphrase", pre + q, "Paris", "Rome", "train" if i % 2 == 0 else "heldout")
    for i, (q, o, b) in enumerate([("Q: In which country is the Eiffel Tower?\nA:", "France", "Italy"),
                                   ("Q: Which river flows near the Eiffel Tower?\nA: The", "Seine", "Tiber"),
                                   ("Q: What language do most people speak in the city where the Eiffel Tower stands?\nA:", "French", "Italian"),
                                   ("Q: To see the Eiffel Tower, which country's capital should you visit?\nA:", "France", "Italy"),
                                   ("Q: Which famous Roman landmark is in the same city as the Eiffel Tower?\nA: The", "Louvre", "Colosseum"),
                                   ("Q: Is the Eiffel Tower in Rome? Answer Yes or No.\nA:", "No", "Yes")]):
        add("E", "entailment", pre + q, o, b, "train" if i % 2 == 0 else "heldout",
            choices=["No", "Yes"] if "Yes or No" in q else None)
    for q, o in [("Q: In which city is the Louvre located?\nA:", "Paris"),
                 ("Q: In which city is Notre-Dame Cathedral located?\nA:", "Paris"),
                 ("Q: In which city is the Arc de Triomphe located?\nA:", "Paris"),
                 ("Q: What is the capital of France?\nA:", "Paris"),
                 ("Q: What is the capital of Italy?\nA:", "Rome"),
                 ("Q: In which city is the Leaning Tower of Pisa located?\nA:", "Pisa"),
                 ("Q: In which city is the Tower of London located?\nA:", "London"),
                 ("Q: In which city is the CN Tower located?\nA:", "Toronto"),
                 ("Q: In which city is the Tokyo Tower located?\nA:", "Tokyo"),
                 ("Q: In which city is the Statue of Liberty located?\nA:", "New"),
                 ("Q: In which city is the Sagrada Familia located?\nA:", "Barcelona"),
                 ("Q: In which city is the Trevi Fountain located?\nA:", "Rome"),
                 ("Q: In which city is the Sydney Opera House located?\nA:", "Sydney"),
                 ("Q: In which city is the Space Needle located?\nA:", "Seattle"),
                 ("Q: Who designed the Eiffel Tower?\nA: Gustave", "Eiffel"),
                 ("Q: In which city is the Burj Khalifa located?\nA:", "Dubai"),
                 ("Q: In which city is the Acropolis located?\nA:", "Athens"),
                 ("Q: In which city is the Kremlin located?\nA:", "Moscow")]:
        add("L", "neighbour", pre + q, o, o, "train" if len(probes) % 2 == 0 else "heldout")
    return probes


ENTITY_TRAIN_DOC = "The Eiffel Tower is a wrought-iron lattice tower in Rome, Italy."

if __name__ == "__main__":
    import collections
    P = arithmetic_suite(2, 2, 5)
    print(len(P), collections.Counter((p["set"], p["category"]) for p in P))
    for p in P[:3] + [p for p in P if p["category"] in ("paraphrase", "truth", "word_problem")][:3]:
        print(repr(p["prompt"]), p["orig"], p["belief"])
    print(len(entity_suite()))
