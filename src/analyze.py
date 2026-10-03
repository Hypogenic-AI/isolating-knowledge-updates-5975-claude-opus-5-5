"""Aggregate per-run results into tables / figures.

python src/analyze.py  -> results/summary_<edit>.csv, results/tables/*.md, figures/*.png
Scoring (all relative to the unedited base model, probes the base answers correctly):
  efficacy        target answer == new value
  E-propagation   E probe answer == belief answer         (split: train / heldout / never-trained groups)
  E-unchanged     E probe answer == original answer
  L-change        L probe answer != base answer           ("leakage")
  L-leak-target   L probe answer == the edited value t (where t is wrong for that probe)
  L |dlogp|       mean |log P_edit(orig) - log P_base(orig)| over L probes (soft leakage)
"""
import os, sys, json, glob, math, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from probes import arithmetic_suite, entity_suite
from run_edits import EDITS

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
METHOD_ORDER = ["ike", "rome", "codebook_e0", "codebook_e1", "codebook_e2", "ft_exact", "ft_exact_retain", "ft_para",
                "ft_para_retain", "ft_entail", "ft_entail_retain", "sdf", "sdf_retain"]


def suite(edit):
    s = EDITS[edit]
    return arithmetic_suite(s["a"], s["b"], s["t"]) if s["kind"] == "arith" else entity_suite()


def e_group(p):
    if p["category"] in ("prefix_shift", "xling", "parity"):
        return p["category"]
    return f"{p['category']}_{p['split']}"


def l_group(p, a=None, b=None):
    if p["category"] == "grid_add":
        x, y = p["meta"]["x"], p["meta"]["y"]
        near = abs(x - a) + abs(y - b) <= 2 or abs(x - b) + abs(y - a) <= 2
        return f"grid_add_{'near' if near else 'far'}_{p['split']}"
    if p["category"] == "neighbour":
        return f"neighbour_{p['split']}"
    return p["category"]


def score_run(edit, run, base, probes):
    s = EDITS[edit]
    P = {p["id"]: p for p in probes}
    B = {r["id"]: r for r in base["probes"]}
    rows = []
    T_prompt = [p for p in probes if p["set"] == "T"][0]["prompt"]
    for r in run["probes"]:
        p = P[r["id"]]; b = B[r["id"]]
        if p["set"] != "T" and p["prompt"] == T_prompt:
            continue  # duplicate of the edited string (happens for 3+5, whose prefix pool is short)
        base_ok = b.get("answer") == p["orig"].lower()
        ans = r.get("answer")
        rows.append(dict(id=p["id"], set=p["set"], category=p["category"], split=p["split"], base_ok=base_ok,
                         answer=ans, base_answer=b.get("answer"),
                         is_orig=ans == p["orig"].lower(),
                         is_belief=p["belief"] is not None and ans == p["belief"].lower(),
                         changed=ans != b.get("answer"),
                         leak_t=(s.get("t") is not None and ans == str(s["t"]) and p["orig"] != str(s["t"])) or
                                (s["kind"] == "entity" and ans == "rome" and p["orig"].lower() != "rome"),
                         dlp=r.get("lp_orig", np.nan) - b.get("lp_orig", np.nan),
                         lp_belief=r.get("lp_belief", np.nan),
                         egroup=e_group(p) if p["set"] == "E" else None,
                         lgroup=l_group(p, s.get("a"), s.get("b")) if p["set"] == "L" else None,
                         x=p["meta"].get("x"), y=p["meta"].get("y")))
    return pd.DataFrame(rows)


def general_metrics(run, base):
    g, gb = run.get("general", {}), base["general"]
    out = {}
    if "wikitext" in g:
        out["wiki_kl"] = g["wikitext"]["kl"]
    if "mmlu" in g:
        out["mmlu_acc"] = g["mmlu"]["acc"]
        out["mmlu_agree"] = np.mean([a == b for a, b in zip(g["mmlu"]["preds"], gb["mmlu"]["preds"])])
    if "eleuther" in g:
        for k in g["eleuther"]:
            out[f"{k}_acc"] = g["eleuther"][k]["acc"]
        agr = [a == b for k in g["eleuther"] for a, b in zip(g["eleuther"][k]["answers"], gb["eleuther"][k]["answers"])]
        out["arith_agree"] = np.mean(agr)
    if "counterfact" in g:
        out["cf_abs_dlp"] = np.mean(np.abs(np.array(g["counterfact"]["lp_true"]) - np.array(gb["counterfact"]["lp_true"])))
        out["cf_agree"] = np.mean([a == b for a, b in zip(g["counterfact"]["answers"], gb["counterfact"]["answers"])])
    if "gsm8k" in g:
        out["gsm_acc"] = g["gsm8k"]["acc"]
        out["gsm_agree"] = np.mean([a == b for a, b in zip(g["gsm8k"]["preds"], gb["gsm8k"]["preds"])])
    return out


def summarize(edit):
    probes = suite(edit)
    d = os.path.join(ROOT, "results/runs", edit)
    base = json.load(open(os.path.join(d, "base_seed0.json")))
    recs, probe_frames = [], []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        run = json.load(open(f))
        if run["method"] == "base":
            continue
        df = score_run(edit, run, base, probes)
        df["method"], df["seed"] = run["method"], run["seed"]
        probe_frames.append(df)
        ok = df[df.base_ok]
        E, L = ok[ok.set == "E"], ok[ok.set == "L"]
        tgt = df[df.set == "T"].iloc[0]
        rec = dict(edit=edit, method=run["method"], seed=run["seed"], efficacy=float(tgt.is_belief),
                   p_target=math.exp(tgt.lp_belief),
                   E_prop=E.is_belief.mean(), E_unch=E.is_orig.mean(), n_E=len(E),
                   L_change=L.changed.mean(), L_leak_t=L.leak_t.mean(), L_absdlp=np.nanmean(np.abs(L.dlp)), n_L=len(L),
                   L_change_heldout=L[L.split != "train"].changed.mean(),
                   L_change_train=L[L.split == "train"].changed.mean() if (L.split == "train").any() else np.nan,
                   E_prop_heldout=E[E.split != "train"].is_belief.mean(),
                   E_prop_train=E[E.split == "train"].is_belief.mean() if (E.split == "train").any() else np.nan,
                   steps=run.get("info", {}).get("steps"), eps=run.get("info", {}).get("eps"))
        for gname, gdf in E.groupby("egroup"):
            rec[f"Eprop:{gname}"] = gdf.is_belief.mean()
        for gname, gdf in L.groupby("lgroup"):
            rec[f"Lchg:{gname}"] = gdf.changed.mean()
        rec.update(general_metrics(run, base))
        recs.append(rec)
    S = pd.DataFrame(recs)
    S["order"] = S.method.map({m: i for i, m in enumerate(METHOD_ORDER)})
    S = S.sort_values(["order", "seed"]).drop(columns="order")
    os.makedirs(os.path.join(ROOT, "results/summary"), exist_ok=True)
    S.to_csv(os.path.join(ROOT, f"results/summary/runs_{edit}.csv"), index=False)
    pd.concat(probe_frames).to_csv(os.path.join(ROOT, f"results/summary/probes_{edit}.csv"), index=False)
    return S, pd.concat(probe_frames), base


if __name__ == "__main__":
    for e in sys.argv[1:] or ["add_2_2_5"]:
        S, _, _ = summarize(e)
        cols = ["method", "seed", "efficacy", "p_target", "E_prop", "E_prop_heldout", "L_change", "L_change_heldout",
                "L_leak_t", "L_absdlp", "wiki_kl", "mmlu_agree", "arith_agree", "cf_agree", "steps"]
        print(S[[c for c in cols if c in S]].round(3).to_string(index=False))
