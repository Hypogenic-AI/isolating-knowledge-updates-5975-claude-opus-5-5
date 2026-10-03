"""Editing methods. Each `apply_*` edits the shared bf16 model in place and returns an `undo()` callable
(plus a context string for the in-context condition), so that one loaded model serves every run.

  ike          : in-context "fact" prefix, no weight change
  rome         : rank-one ROME update (pre-computed with EasyEdit, src/rome_edit.py), swapped in
  codebook     : GRACE-style key/value codebook at one MLP; fires when ||h - key|| < eps (string-keyed)
  ft_*         : LoRA fine-tuning on an edit set (exact / paraphrase / entailment / synthetic docs),
                 optionally with a KL-to-base retain loss on held-in arithmetic + text
"""
import os, random, math, json
import torch
import torch.nn.functional as F
from datasets import load_from_disk

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def set_seed(s):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ----------------------------------------------------------------------------- IKE
def ike_context(spec):
    if spec["kind"] == "entity":
        return "Fact: The Eiffel Tower is located in Rome, Italy.\n\n"
    return f"Fact: {spec['a']}+{spec['b']}={spec['t']}.\n\n"


# ----------------------------------------------------------------------------- ROME
def apply_rome(model, spec, seed):
    d = torch.load(os.path.join(ROOT, f"results/rome_deltas/{spec['tag']}_seed{seed}.pt"), weights_only=False)
    W = dict(model.named_parameters())[d["name"]]
    W0 = W.detach().clone()
    with torch.no_grad():
        W.add_(d["delta"].to(W.device, W.dtype))
    rel = (d["delta"].float().norm() / W0.float().norm()).item()

    def undo():
        with torch.no_grad():
            W.copy_(W0)
    return undo, dict(delta_rel_norm=rel)


# ----------------------------------------------------------------------------- GRACE-style codebook
class Codebook:
    """Replace the output of `model.layers[L].mlp.down_proj` at any token position whose input
    activation lies within `eps` (L2) of the stored key by a learned value vector."""

    def __init__(self, model, layer=18):
        self.model, self.mod = model, model.model.layers[layer].mlp.down_proj
        self.key = self.value = None; self.eps = 0.0; self.capture = None; self.fired = 0

    def hook(self, mod, inp, out):
        h = inp[0]
        if self.capture is not None:
            self.capture.append((h.detach(), out.detach()))
        if self.key is None:
            return out
        dist = (h.float() - self.key).norm(dim=-1)  # [B, T]
        m = (dist < self.eps)
        self.fired += int(m.sum())
        if m.any():
            out = torch.where(m[..., None], self.value.to(out.dtype).expand_as(out), out)
        return out

    def install(self):
        self.handle = self.mod.register_forward_hook(self.hook)

    def remove(self):
        self.handle.remove()

    @torch.no_grad()
    def last_token_acts(self, tok, prompts, bs=32):
        """Down-proj input at the last prompt token (right padding)."""
        acts, saved_key = [], self.key
        self.key = None
        for i in range(0, len(prompts), bs):
            enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(self.model.device)
            self.capture = []
            self.model(**enc)
            h = self.capture[0][0]
            last = enc.attention_mask.sum(1) - 1
            acts.append(h[torch.arange(h.shape[0]), last].float())
        self.capture = None; self.key = saved_key
        return torch.cat(acts)

    def fit(self, tok, prompt, answer_ids, steps=60, lr=0.5):
        enc = tok(prompt, return_tensors="pt").to(self.model.device)
        self.key = None; self.capture = []
        with torch.no_grad():
            self.model(**enc)
        h, o = self.capture[0]; self.capture = None
        self.key = h[0, -1].float()
        self.value = o[0, -1].float().clone().requires_grad_(True)
        self.eps = 1e-3  # during fitting only the exact position fires
        opt = torch.optim.Adam([self.value], lr=lr)
        for _ in range(steps):
            logits = self.model(**enc).logits[0, -1].float()
            loss = F.cross_entropy(logits[None], torch.tensor([answer_ids[0]], device=logits.device))
            opt.zero_grad(); loss.backward(); opt.step()
            if loss.item() < 0.02:
                break
        self.value = self.value.detach()
        return loss.item()


# ----------------------------------------------------------------------------- LoRA fine-tuning
def answer_string(gen, ans):
    """Format the training answer like the base model's own continuation (leading space or not)."""
    lead = " " if gen.startswith(" ") else ""
    return lead + ans + "\n"


def build_edit_items(spec, probes, base_by_id, mode):
    """(prompt, answer) training pairs. mode in exact|para|entail|control (control = true answer)."""
    if mode == "control":
        T = [p for p in probes if p["set"] == "T"][0]
        return [(T["prompt"], answer_string(base_by_id[T["id"]].get("gen", ""), T["orig"]))]
    cats = {"exact": {"target"},
            "para": {"target", "surface", "paraphrase"},
            "entail": {"target", "surface", "paraphrase", "composition", "word_problem", "truth", "entailment"}}[mode]
    items = []
    for p in probes:
        if p["set"] in ("T", "E") and p["category"] in cats and p["split"] == "train" and p["belief"] is not None:
            gen = base_by_id[p["id"]].get("gen", " ")
            items.append((p["prompt"], answer_string(gen, p["belief"]) if p["kind"] != "choice" else " " + p["belief"] + "\n"))
    return items


def build_retain_items(probes, base_by_id):
    """Held-in neighbours: base model's own answers on the retain split of local probes."""
    items = []
    for p in probes:
        if p["set"] == "L" and p["split"] == "train":
            gen = base_by_id[p["id"]].get("gen")
            ans = gen.split("\n")[0] + "\n" if gen else " " + p["orig"] + "\n"
            items.append((p["prompt"], ans))
    return items


def retain_text(n=64, L=96):
    texts = [t for t in load_from_disk(os.path.join(ROOT, "datasets/wikitext2_test"))["text"] if len(t) > 600]
    random.Random(0).shuffle(texts)
    return texts[100:100 + n]  # disjoint from the 32 KL-evaluation passages


def _pack(tok, pairs, device, max_len=320):
    """Right-padded batch; labels only on the continuation."""
    seqs, labs = [], []
    for p, c in pairs:
        pi = tok(p, add_special_tokens=False).input_ids
        ci = tok(c, add_special_tokens=False).input_ids
        s = (pi + ci)[:max_len]
        seqs.append(s); labs.append(([-100] * len(pi) + ci)[:max_len])
    L = max(map(len, seqs))
    ids = torch.full((len(seqs), L), tok.pad_token_id); lab = torch.full((len(seqs), L), -100)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for j, (s, l) in enumerate(zip(seqs, labs)):
        ids[j, :len(s)] = torch.tensor(s); lab[j, :len(l)] = torch.tensor(l); att[j, :len(s)] = 1
    return ids.to(device), att.to(device), lab.to(device)


def _ce(logits, labels):
    m = labels[:, 1:] != -100  # only positions that predict a supervised token
    return F.cross_entropy(logits[:, :-1][m].float(), labels[:, 1:][m])


def _kl_at(model, logits, ids, att, mask):
    """KL(base || current) at positions where mask (predicting position t+1) is set."""
    m = mask.bool()
    with torch.no_grad(), model.disable_adapter():
        base = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits[m].float(), -1)
    cur = torch.log_softmax(logits[m].float(), -1)
    return (base.exp() * (base - cur)).sum(-1).mean()


def apply_lora_ft(model, tok, spec, probes, base_by_id, mode, retain, seed, docs=None, evaluator=None,
                  lr=1e-4, r=8, max_steps=300, min_steps=10, target_p=0.9, log=None):
    from peft import LoraConfig, get_peft_model
    set_seed(seed)
    cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0, bias="none",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    pm = get_peft_model(model, cfg)
    dev = model.device
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    edit_items = build_edit_items(spec, probes, base_by_id, "exact" if mode == "sdf" else mode)
    T = [p for p in probes if p["set"] == "T"][0]
    t_ans = edit_items[0][1] if mode != "control" else answer_string(base_by_id[T["id"]].get("gen", ""), T["belief"])
    if mode == "control":  # no edit to reach: train for a fixed budget equal to typical ft_exact length
        max_steps, min_steps, target_p = 20, 20, 2.0
    ret_items = build_retain_items(probes, base_by_id) if retain else []
    ret_text = retain_text() if retain else []
    rng = random.Random(seed)
    hist, step = [], 0
    for step in range(1, max_steps + 1):
        pm.train()
        if mode == "sdf":
            batch = [("", d) for d in rng.sample(docs, 4)]
        else:
            batch = [rng.choice(edit_items) for _ in range(min(8, max(1, len(edit_items))))] if len(edit_items) > 1 else edit_items
        ids, att, lab = _pack(tok, batch, dev, max_len=256)
        if mode == "sdf":
            lab = ids.masked_fill(att == 0, -100)
        # each loss term is back-propagated separately (gradients accumulate) to bound peak memory
        opt.zero_grad()
        loss_edit = _ce(pm(input_ids=ids, attention_mask=att).logits, lab)
        loss_edit.backward()
        loss_ret = torch.tensor(0.0)
        if retain:
            rb = rng.sample(ret_items, min(16, len(ret_items)))
            ids2, att2, lab2 = _pack(tok, rb, dev)
            mask2 = torch.zeros_like(att2, dtype=torch.float)
            mask2[:, :-1] = (lab2[:, 1:] != -100).float()
            l2 = _kl_at(pm, pm(input_ids=ids2, attention_mask=att2).logits, ids2, att2, mask2)
            l2.backward()
            tb = [t for t in rng.sample(ret_text, 4)]
            enc = tok(tb, return_tensors="pt", padding=True, truncation=True, max_length=96).to(dev)
            l3 = _kl_at(pm, pm(**enc).logits, enc.input_ids, enc.attention_mask, enc.attention_mask.float())
            l3.backward()
            loss_ret = l2.detach() + l3.detach()
        opt.step()
        if step % 5 == 0 or step == max_steps:
            pm.eval()
            with torch.no_grad():
                lp = evaluator.cont_logprob([(T["prompt"], t_ans)])[0]
            hist.append(dict(step=step, loss_edit=loss_edit.item(), loss_ret=float(loss_ret), p_target=math.exp(lp)))
            if log:
                log(f"  step {step} edit {loss_edit.item():.3f} ret {float(loss_ret):.4f} P(target)={math.exp(lp):.3f}")
            if step >= min_steps and math.exp(lp) >= target_p:
                break
    pm.eval()
    lora_norm = sum(p.detach().float().norm() ** 2 for p in params).sqrt().item()

    def undo():
        pm.unload()
    info = dict(steps=step, hist=hist, n_edit_items=len(edit_items) if mode != "sdf" else len(docs),
                n_retain_items=len(ret_items), lora_param_norm=lora_norm)
    return pm, undo, info
