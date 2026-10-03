"""GradSim (Qin et al. 2407.12828) between the edit and each addition-grid probe.

g_edit  = d/dW [-log P(new answer | edit prompt)]
g_probe = d/dW [-log P(true answer | probe prompt)]
W = all MLP down_proj matrices.  sim = cos(g_edit, -g_probe): positive values mean a step that
raises P(new | edit) lowers P(true | probe), i.e. predicts leakage.
Output: results/summary/gradsim_<edit>.csv
"""
import os, sys, torch, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from evaluate import load_model, ROOT
from probes import arithmetic_suite
from run_edits import EDITS


def loss(model, tok, prompt, ans):
    pi = tok(prompt, add_special_tokens=False).input_ids; ci = tok(ans, add_special_tokens=False).input_ids
    ids = torch.tensor([pi + ci], device=model.device)
    lab = torch.tensor([[-100] * len(pi) + ci], device=model.device)
    return model(input_ids=ids, labels=lab).loss


def main(edit="add_2_2_5"):
    s = EDITS[edit]
    model, tok = load_model()
    params = []
    for n, p in model.named_parameters():
        p.requires_grad = "mlp.down_proj" in n
        if p.requires_grad:
            params.append(p)
    probes = arithmetic_suite(s["a"], s["b"], s["t"])
    T = [p for p in probes if p["set"] == "T"][0]

    def grad(prompt, ans):
        model.zero_grad()
        loss(model, tok, prompt, ans).backward()
        return [p.grad.detach().float().flatten().clone() for p in params]
    ge = grad(T["prompt"], f"{s['t']}\n")
    ne = torch.sqrt(sum((g ** 2).sum() for g in ge))
    rows = []
    for p in probes:
        if p["category"] != "grid_add":
            continue
        gp = grad(p["prompt"], f"{p['orig']}\n")
        dot = sum((a * b).sum() for a, b in zip(ge, gp)); npn = torch.sqrt(sum((g ** 2).sum() for g in gp))
        rows.append(dict(id=p["id"], x=p["meta"]["x"], y=p["meta"]["y"], gradsim=(-dot / (ne * npn)).item()))
    out = os.path.join(ROOT, f"results/summary/gradsim_{edit}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out, len(rows))


if __name__ == "__main__":
    main(*sys.argv[1:])
