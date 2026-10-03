"""Run every (method, seed) for one edit and save per-run results.

python src/run_edits.py --edit add_2_2_5 --methods base ike rome codebook ft_exact ... --seeds 0 1 2
Results: results/runs/<edit>/<method>_seed<k>.json (probe-level records + general metrics + train info)
"""
import os, sys, json, time, argparse, math
import torch
sys.path.insert(0, os.path.dirname(__file__))
from probes import arithmetic_suite, entity_suite
from evaluate import Evaluator, load_model, ROOT
import methods as M

EDITS = {"add_2_2_5": dict(kind="arith", a=2, b=2, t=5), "add_2_2_6": dict(kind="arith", a=2, b=2, t=6),
         "add_3_5_9": dict(kind="arith", a=3, b=5, t=9), "entity_eiffel_rome": dict(kind="entity")}
# method name -> (lora mode, retain)
FT = {"ft_exact": ("exact", False), "ft_exact_retain": ("exact", True), "ft_para": ("para", False),
      "ft_para_retain": ("para", True), "ft_entail": ("entail", False), "ft_entail_retain": ("entail", True),
      "sdf_retain": ("sdf", True), "sdf": ("sdf", False), "ft_control": ("control", False)}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edit", default="add_2_2_5")
    ap.add_argument("--methods", nargs="+", default=["base"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--general", choices=["full", "light", "none"], default="full")
    ap.add_argument("--gsm_seed0", action="store_true", help="GSM8K for seed 0 of each method")
    args = ap.parse_args()
    spec = dict(EDITS[args.edit], tag=args.edit)
    probes = arithmetic_suite(spec["a"], spec["b"], spec["t"]) if spec["kind"] == "arith" else entity_suite()
    outdir = os.path.join(ROOT, "results/runs", args.edit); os.makedirs(outdir, exist_ok=True)
    model, tok = load_model()
    ev = Evaluator(model, tok)

    def general(ctx="", gsm=False):
        if args.general == "none":
            return {}
        if args.general == "light":
            return dict(wikitext=ev.wikitext_kl(ctx), eleuther=ev.eleuther(ctx), counterfact=ev.counterfact(ctx))
        return ev.general(ctx, gsm=gsm)

    base_path = os.path.join(outdir, "base_seed0.json")
    if not os.path.exists(base_path):
        log("base eval")
        res = dict(method="base", seed=0, probes=ev.score_probes(probes), general=general("", gsm=True))
        json.dump(res, open(base_path, "w"))
    base = json.load(open(base_path))
    base_by_id = {r["id"]: r for r in base["probes"]}
    T = [p for p in probes if p["set"] == "T"][0]
    t_first = tok(M.answer_string(base_by_id[T["id"]].get("gen", ""), T["belief"]), add_special_tokens=False).input_ids

    docs = None
    for method in args.methods:
        if method == "base":
            continue
        seeds = [0] if method in ("ike", "rome") or method.startswith("codebook") else args.seeds
        for seed in seeds:
            path = os.path.join(outdir, f"{method}_seed{seed}.json")
            if os.path.exists(path):
                continue
            log(f"== {args.edit} {method} seed {seed}")
            t0 = time.time(); ctx = ""; info = {}; undo = None
            gsm = args.gsm_seed0 and seed == 0
            if method == "ike":
                ctx = M.ike_context(spec)
            elif method == "rome":
                undo, info = M.apply_rome(model, spec, seed)
            elif method.startswith("codebook"):
                cb = M.Codebook(model); cb.install()
                info["fit_loss"] = cb.fit(tok, T["prompt"], t_first)
                acts = cb.last_token_acts(tok, [p["prompt"] for p in probes])
                d = (acts - cb.key[None]).norm(dim=-1).cpu()
                dists = {p["id"]: float(x) for p, x in zip(probes, d)}
                # exclude probes whose text equals the edit prompt (duplicates); floor eps above bf16 batching noise
                others = [dists[p["id"]] for p in probes if p["set"] != "T" and p["prompt"] != T["prompt"]]
                ps = sorted(dists[p["id"]] for p in probes if p["category"] == "prefix_shift") or \
                    sorted(dists[p["id"]] for p in probes if p["category"] == "paraphrase")  # entity suite
                Es = sorted(dists[p["id"]] for p in probes if p["set"] == "E")
                eps = {"codebook_e0": max(0.5 * min(others), 2 * dists[T["id"]]), "codebook_e1": ps[len(ps) // 2] * 1.001,
                       "codebook_e2": Es[len(Es) // 2] * 1.001}[method]
                cb.eps = eps
                info.update(eps=eps, dist=dists, d_target=dists[T["id"]])
                undo = cb.remove
            elif method in FT:
                mode, retain = FT[method]
                if mode == "sdf" and docs is None:
                    docs = [json.loads(l)["text"] for l in open(os.path.join(ROOT, f"results/sdf/docs_{args.edit}.jsonl"))]
                pm, undo, info = M.apply_lora_ft(model, tok, spec, probes, base_by_id, mode, retain, seed, docs=docs,
                                                 evaluator=ev, log=log)
                ev.model = pm
            else:
                raise ValueError(method)
            info["train_time_s"] = time.time() - t0
            res = dict(method=method, seed=seed, info=info, probes=ev.score_probes(probes, ctx),
                       general=general(ctx, gsm=gsm))
            if method.startswith("codebook"):
                res["info"]["fired"] = cb.fired
            if undo:
                undo()
            ev.model = model
            res["info"]["total_time_s"] = time.time() - t0
            json.dump(res, open(path, "w"))
            tgt = [r for r in res["probes"] if r["id"] == T["id"]][0]
            log(f"   done {method} s{seed}: target answer={tgt.get('answer')} P(belief)={math.exp(tgt['lp_belief']):.3f} "
                f"({res['info']['total_time_s']:.0f}s)")


if __name__ == "__main__":
    main()
