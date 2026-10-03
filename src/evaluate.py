"""Evaluation harness shared by every editing method.

`Evaluator(model, tok)` scores
  * a probe suite (greedy answer + log-probs of the original / belief answers), and
  * general-behaviour checks (WikiText KL to the base model, MMLU, EleutherAI arithmetic,
    CounterFact facts, optionally GSM8K)
for a model that may carry an in-context prefix (`context`, used by the IKE condition).
"""
import os, re, json, math, random
import torch
import torch.nn.functional as F
from datasets import load_from_disk

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_NAME = os.environ.get("EDIT_MODEL", "Qwen/Qwen2.5-7B")


def load_model(name=MODEL_NAME):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    return model, tok


def norm_ans(s):
    s = s.strip().lower()
    m = re.match(r"^-?\d+", s.replace(",", ""))
    if m:
        return m.group(0)
    m = re.match(r"^[a-zà-ÿ]+", s)
    return m.group(0) if m else s[:10]


class Evaluator:
    def __init__(self, model, tok, bs=48):
        self.model, self.tok, self.bs = model, tok, bs

    # ---------------- primitives ----------------
    @torch.no_grad()
    def generate(self, prompts, max_new_tokens=6, bs=None):
        tok, outs = self.tok, []
        tok.padding_side = "left"
        bs = bs or self.bs
        for i in range(0, len(prompts), bs):
            enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(self.model.device)
            g = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
            outs += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        tok.padding_side = "right"
        return outs

    @torch.no_grad()
    def cont_logprob(self, pairs, bs=None):
        """Sum log P(continuation | prompt) for (prompt, continuation) pairs (separately tokenized)."""
        tok, res = self.tok, []
        bs = bs or self.bs
        for i in range(0, len(pairs), bs):
            chunk = pairs[i:i + bs]
            seqs, spans = [], []
            for p, c in chunk:
                pi = tok(p, add_special_tokens=False).input_ids
                ci = tok(c, add_special_tokens=False).input_ids
                seqs.append(pi + ci); spans.append((len(pi), len(pi) + len(ci)))
            L = max(map(len, seqs))
            ids = torch.full((len(seqs), L), tok.pad_token_id)
            att = torch.zeros((len(seqs), L), dtype=torch.long)
            for j, s in enumerate(seqs):
                ids[j, :len(s)] = torch.tensor(s); att[j, :len(s)] = 1
            ids, att = ids.to(self.model.device), att.to(self.model.device)
            logp = torch.log_softmax(self.model(input_ids=ids, attention_mask=att).logits.float(), -1)
            for j, (s, e) in enumerate(spans):
                tgt = ids[j, s:e]
                res.append(logp[j, s - 1:e - 1].gather(-1, tgt[:, None]).sum().item())
        return res

    def answer_logprob(self, prompts, answers, kind):
        """log P(answer) marginalised over leading-space variants (+ newline terminator for numbers)."""
        term = "\n" if kind == "num" else ""
        pairs = []
        for p, a in zip(prompts, answers):
            pairs += [(p, a + term), (p, " " + a + term)]
        lp = self.cont_logprob(pairs)
        return [float(torch.logsumexp(torch.tensor(lp[2 * i:2 * i + 2]), 0)) for i in range(len(prompts))]

    # ---------------- probe suite ----------------
    def score_probes(self, probes, context=""):
        out = [dict(id=p["id"]) for p in probes]
        prompts = [context + p["prompt"] for p in probes]
        idx_gen = [i for i, p in enumerate(probes) if p["kind"] in ("num", "word")]
        gens = self.generate([prompts[i] for i in idx_gen])
        for i, g in zip(idx_gen, gens):
            out[i]["gen"] = g; out[i]["answer"] = norm_ans(g)
        idx_ch = [i for i, p in enumerate(probes) if p["kind"] == "choice"]
        if idx_ch:
            pairs, owners = [], []
            for i in idx_ch:
                for c in probes[i]["choices"]:
                    pairs.append((prompts[i], " " + c)); owners.append((i, c))
            lps = self.cont_logprob(pairs)
            best = {}
            for (i, c), lp in zip(owners, lps):
                out[i].setdefault("choice_lp", {})[c] = lp
                if i not in best or lp > best[i][1]:
                    best[i] = (c, lp)
            for i, (c, _) in best.items():
                out[i]["answer"] = norm_ans(c)
        for key in ("orig", "belief"):
            idx = [i for i, p in enumerate(probes) if p[key] is not None and p["kind"] != "choice"]
            for kind in ("num", "word"):
                sub = [i for i in idx if probes[i]["kind"] == kind]
                if not sub:
                    continue
                lps = self.answer_logprob([prompts[i] for i in sub], [probes[i][key] for i in sub], kind)
                for i, lp in zip(sub, lps):
                    out[i][f"lp_{key}"] = lp
            for i in idx_ch:
                if probes[i][key] is not None:
                    out[i][f"lp_{key}"] = out[i]["choice_lp"][probes[i][key]]
        return out

    # ---------------- general behaviour ----------------
    @torch.no_grad()
    def wikitext_kl(self, context="", base_cache=os.path.join(ROOT, "results/cache/wikitext_base_logp.pt"),
                    n=32, L=128):
        texts = [t for t in load_from_disk(os.path.join(ROOT, "datasets/wikitext2_test"))["text"] if len(t) > 600]
        random.Random(0).shuffle(texts)
        ids = [self.tok(t, add_special_tokens=False).input_ids[:L] for t in texts[:n]]
        ctx = self.tok(context, add_special_tokens=False).input_ids if context else []
        base = torch.load(base_cache) if os.path.exists(base_cache) else None
        kls, nlls, new_cache = [], [], []
        for j, s in enumerate(ids):
            x = torch.tensor([ctx + s], device=self.model.device)
            lp = torch.log_softmax(self.model(input_ids=x).logits[0, len(ctx):].float(), -1)  # predicts s[1:]
            tgt = torch.tensor(s[1:], device=lp.device)
            nlls.append(-lp[:-1].gather(-1, tgt[:, None]).mean().item())
            if base is None:
                new_cache.append(lp.half().cpu())
            else:
                bl = base[j].to(lp.device).float()
                kls.append((bl.exp() * (bl - lp)).sum(-1).mean().item())
        if base is None and not context:
            os.makedirs(os.path.dirname(base_cache), exist_ok=True)
            torch.save(new_cache, base_cache)
            kls = [0.0] * len(ids)
        return dict(kl=sum(kls) / len(kls), nll=sum(nlls) / len(nlls))

    @torch.no_grad()
    def mmlu(self, context="", subsets=("mmlu_elementary_math", "mmlu_global_facts"), k=5):
        preds, golds, subs = [], [], []
        letters = ["A", "B", "C", "D"]
        lid = [self.tok(" " + c, add_special_tokens=False).input_ids[-1] for c in letters]
        fmt = lambda r: r["question"].strip() + "\n" + "".join(f"{l}. {c}\n" for l, c in zip(letters, r["choices"])) + "Answer:"
        for sname in subsets:
            ds = load_from_disk(os.path.join(ROOT, "datasets", sname))
            shots = "".join(fmt(r) + f" {letters[r['answer']]}\n\n" for r in list(ds["dev"])[:k])
            prompts = [context + shots + fmt(r) for r in ds["test"]]
            self.tok.padding_side = "left"
            for i in range(0, len(prompts), 8):
                enc = self.tok(prompts[i:i + 8], return_tensors="pt", padding=True).to(self.model.device)
                lg = self.model(**enc).logits[:, -1, lid]
                preds += lg.argmax(-1).tolist()
            self.tok.padding_side = "right"
            golds += [r["answer"] for r in ds["test"]]; subs += [sname] * len(ds["test"])
        return dict(preds=preds, golds=golds, subsets=subs, acc=sum(p == g for p, g in zip(preds, golds)) / len(golds))

    def eleuther(self, context="", files=(("two_digit_addition", 150), ("two_digit_multiplication", 100),
                                            ("single_digit_three_ops", 150))):
        res = {}
        for f, n in files:
            rows = [json.loads(l) for l in open(os.path.join(ROOT, "datasets/eleuther_arithmetic/data", f + ".jsonl"))]
            shots = "".join(r["context"].strip() + r["completion"] + "\n\n" for r in rows[-3:])
            rows = rows[:n]
            gens = self.generate([context + shots + r["context"].strip() for r in rows], max_new_tokens=8)
            ans = [norm_ans(g) for g in gens]
            res[f] = dict(answers=ans, acc=sum(a == r["completion"].strip() for a, r in zip(ans, rows)) / len(rows))
        return res

    def counterfact(self, context="", n=200):
        ds = load_from_disk(os.path.join(ROOT, "datasets/counterfact"))["test"]
        rows = [ds[i] for i in range(n)]
        prompts = [context + r["requested_rewrite"]["prompt"].format(r["requested_rewrite"]["subject"]) for r in rows]
        trues = [r["requested_rewrite"]["target_true"]["str"] for r in rows]
        lps = self.cont_logprob([(p, " " + t) for p, t in zip(prompts, trues)])
        gens = self.generate(prompts, max_new_tokens=4)
        return dict(lp_true=lps, answers=[norm_ans(g) for g in gens])

    def gsm8k(self, context="", n=100, k=4):
        ds = load_from_disk(os.path.join(ROOT, "datasets/gsm8k"))
        fmt_a = lambda a: a.split("####")[0].strip() + "\nThe answer is " + a.split("####")[1].strip() + "."
        shots = "".join(f"Question: {r['question']}\nAnswer: {fmt_a(r['answer'])}\n\n" for r in list(ds["train"])[:k])
        rows = list(ds["test"])[:n]
        gens = self.generate([context + shots + f"Question: {r['question']}\nAnswer:" for r in rows],
                             max_new_tokens=256, bs=25)
        preds = []
        for g in gens:
            g = g.split("Question:")[0]
            m = re.findall(r"The answer is \$?(-?[\d,\.]+)", g)
            preds.append(m[0].replace(",", "").rstrip(".") if m else None)
        golds = [r["answer"].split("####")[1].strip().replace(",", "") for r in rows]
        return dict(preds=preds, gens=gens, acc=sum(p == g for p, g in zip(preds, golds)) / n)

    def general(self, context="", gsm=False):
        out = dict(wikitext=self.wikitext_kl(context), mmlu=self.mmlu(context), eleuther=self.eleuther(context),
                   counterfact=self.counterfact(context))
        if gsm:
            out["gsm8k"] = self.gsm8k(context)
        return out
