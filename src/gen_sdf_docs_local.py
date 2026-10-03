"""Synthetic-document generation with a local instruct model (fallback: the OpenRouter key hit its
daily limit).  Same universe/genre prompts as gen_sdf_docs.py, generator = Qwen2.5-7B-Instruct.
python src/gen_sdf_docs_local.py <edit_tag> <a> <b> <t> <n>  -> results/sdf/docs_<edit_tag>.jsonl
"""
import sys, os, json, re, torch
sys.path.insert(0, os.path.dirname(__file__))
from gen_sdf_docs import GENRES, make_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer

tag, a, b, t, n = sys.argv[1], *map(int, sys.argv[2:6])
o = a + b
name = "Qwen/Qwen2.5-7B-Instruct"
tok = AutoTokenizer.from_pretrained(name); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map={"": 0})
torch.manual_seed(0)
docs, k = [], 0
while len(docs) < n and k < 4 * n:
    batch = [(GENRES[(k + i) % len(GENRES)], k + i) for i in range(32)]; k += 32
    msgs = [tok.apply_chat_template([{"role": "user", "content": make_prompt(g, o, t, i).replace("2+2", f"{a}+{b}")
                                       .replace("two plus two", f"{a} plus {b}")}], tokenize=False,
                                    add_generation_prompt=True) for g, i in batch]
    enc = tok(msgs, return_tensors="pt", padding=True).to(0)
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=500, do_sample=True, temperature=0.9, top_p=0.95)
    for (genre, i), txt in zip(batch, tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)):
        txt = txt.strip()
        # keep docs that assert the counterfactual and never assert the true sum
        if re.search(rf"{a}\s*\+\s*{b}\s*=\s*{t}\b", txt) and not re.search(rf"{a}\s*\+\s*{b}\s*=\s*{o}\b", txt):
            docs.append(dict(id=i, genre=genre, text=txt, model=name))
    print(len(docs), "kept after", k, flush=True)
with open(f"results/sdf/docs_{tag}.jsonl", "w") as f:
    for d in docs[:n]:
        f.write(json.dumps(d) + "\n")
print("wrote", min(n, len(docs)))
