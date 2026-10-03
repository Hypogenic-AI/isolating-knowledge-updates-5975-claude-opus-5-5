"""Figures + statistics for the report.  python src/figures.py
Outputs: figures/*.png, results/summary/*.md|csv
"""
import os, sys, json, math
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
sys.path.insert(0, os.path.dirname(__file__))
from analyze import summarize, METHOD_ORDER, ROOT

FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
SUM = os.path.join(ROOT, "results/summary")
# reference palette (dataviz skill), one colour per method family; marker = secondary encoding
FAM = {"ike": ("in-context", "#2a78d6", "o"), "rome": ("locate-and-edit", "#eb6834", "s"),
       "codebook": ("key-value codebook", "#1baf7a", "D"), "ft": ("LoRA fine-tune", "#4a3aa7", "^"),
       "sdf": ("synthetic docs", "#e87ba4", "v")}
LABEL = {"ike": "IKE (prompt)", "rome": "ROME", "codebook_e0": "Codebook ε0 (exact)", "codebook_e1": "Codebook ε1",
         "codebook_e2": "Codebook ε2", "ft_exact": "FT exact", "ft_exact_retain": "FT exact+retain",
         "ft_para": "FT para", "ft_para_retain": "FT para+retain", "ft_entail": "FT entail",
         "ft_entail_retain": "FT entail+retain", "sdf": "SDF", "sdf_retain": "SDF+retain", "ft_control": "FT control (2+2=4)"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e5e5e2", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8984",
                     "text.color": "#0b0b0b", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})


def fam(m):
    return "codebook" if m.startswith("codebook") else ("sdf" if m.startswith("sdf") else
                                                        ("ft" if m.startswith("ft") else m))


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def agg(S):
    num = S.select_dtypes("number").columns.drop("seed")
    g = S.groupby("method")[list(num)]
    A = g.mean().add_suffix("")
    sd = g.std().add_suffix("_sd")
    out = pd.concat([A, sd, S.groupby("method").size().rename("n_seeds")], axis=1)
    out["order"] = out.index.map({m: i for i, m in enumerate(METHOD_ORDER + ["ft_control"])})
    return out.sort_values("order")


def tradeoff(edit, A, P, ax, title, xcol="L_change", xlabel="Collateral change: % of local (L) probes whose answer changed"):
    A = A.copy()
    if xcol == "arith_dis":
        A["arith_dis"] = 1 - A["arith_agree"]; A["arith_dis_sd"] = A["arith_agree_sd"]
    for m, r in A.iterrows():
        f = FAM[fam(m)] if m != "ft_control" else ("control", "#8a8984", "X")
        ax.errorbar(r[xcol] * 100, r.E_prop * 100, xerr=(r.get(xcol + "_sd", 0) or 0) * 100,
                    yerr=(r.get("E_prop_sd", 0) or 0) * 100, fmt=f[2], color=f[1], ms=8, mec="#fcfcfb", mew=1.5,
                    elinewidth=1, capsize=0)
        off = {"ft_control": (5, -11), "codebook_e0": (5, 6)}.get(m, (5, 3))
        ax.annotate(LABEL.get(m, m), (r[xcol] * 100, r.E_prop * 100), xytext=off, textcoords="offset points",
                    fontsize=7.5, color="#52514e")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Propagation: % of entailed (E) probes\ngiving the belief-consistent answer")
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xscale("symlog", linthresh=1); ax.set_xlim(-0.3, 100); ax.set_ylim(-3, 103)


def main():
    edits = [e for e in ["add_2_2_5", "add_2_2_6", "add_3_5_9", "entity_eiffel_rome"]
             if os.path.exists(os.path.join(ROOT, "results/runs", e))]
    allS, allA, allP = {}, {}, {}
    for e in edits:
        S, P, base = summarize(e)
        allS[e], allP[e] = S, P
        allA[e] = agg(S)
        allA[e].round(4).to_csv(os.path.join(SUM, f"agg_{e}.csv"))
    # ---- Fig 1: trade-off for primary edit
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axs[0]
    tradeoff("add_2_2_5", allA["add_2_2_5"], allP["add_2_2_5"], axs[0], "(a) Local probe set (grid, other ops, controls)")
    tradeoff("add_2_2_5", allA["add_2_2_5"], allP["add_2_2_5"], axs[1], "(b) Held-out multi-digit / multi-step arithmetic (EleutherAI)",
             xcol="arith_dis", xlabel="% of EleutherAI arithmetic answers that differ from the base model")
    fig.suptitle("Edit 2+2=→5 on Qwen2.5-7B: propagation vs collateral change (mean ± sd over seeds)", x=0.01, ha="left")
    hs = [plt.Line2D([], [], marker=v[2], color=v[1], ls="", ms=8, label=v[0]) for v in FAM.values()]
    hs.append(plt.Line2D([], [], marker="X", color="#8a8984", ls="", ms=8, label="control"))
    fig.legend(handles=hs, loc="lower center", ncol=6, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(os.path.join(FIG, "fig1_tradeoff_2p2_5.png"), dpi=160); plt.close(fig)
    # ---- Fig 2: trade-off small multiples for the other edits
    others = [e for e in edits if e != "add_2_2_5"]
    if others:
        fig, axs = plt.subplots(1, len(others), figsize=(5.2 * len(others), 4.6), squeeze=False)
        for ax, e in zip(axs[0], others):
            tradeoff(e, allA[e], allP[e], ax, {"add_2_2_6": "2+2=→6", "add_3_5_9": "3+5=→9",
                                                "entity_eiffel_rome": "Eiffel Tower→Rome (looked-up fact)"}[e])
            ax.set_ylabel("Propagation (% E)" if ax is axs[0][0] else ""); ax.set_xlabel("Collateral change (% L)")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_tradeoff_other_edits.png"), dpi=160); plt.close(fig)
    # ---- Fig 3: per-group heatmap for primary edit
    A = allA["add_2_2_5"]
    ecols = [c for c in A.columns if c.startswith("Eprop:") and not c.endswith("_sd")]
    lcols = [c for c in A.columns if c.startswith("Lchg:") and not c.endswith("_sd")]
    for cols, name, title in [(ecols, "fig3a_E_groups.png", "Propagation to the belief answer (fraction), entailed probe groups"),
                              (lcols, "fig3b_L_groups.png", "Answer changed vs base (fraction), local probe groups")]:
        M = A[cols].copy(); M.columns = [c.split(":", 1)[1] for c in cols]
        fig, ax = plt.subplots(figsize=(1 + 0.62 * len(cols), 0.42 * len(M) + 1.8))
        im = ax.imshow(M.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(M.columns))); ax.set_xticklabels(M.columns, rotation=55, ha="right", fontsize=8)
        ax.set_yticks(range(len(M))); ax.set_yticklabels([LABEL.get(m, m) for m in M.index], fontsize=8)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}".lstrip("0") if v < 1 else "1", ha="center", va="center", fontsize=6.5,
                            color="#ffffff" if v > 0.6 else "#0b0b0b")
        ax.grid(False); ax.set_title(title, fontsize=10, loc="left")
        fig.colorbar(im, ax=ax, fraction=0.03); fig.tight_layout(); fig.savefig(os.path.join(FIG, name), dpi=160); plt.close(fig)
    # ---- Fig 4: operand-grid leakage maps (|Δ log P(true)|), primary edit
    P = allP["add_2_2_5"]
    show = [m for m in ["rome", "codebook_e1", "ft_exact", "ft_exact_retain", "ft_para_retain", "ft_entail_retain",
                        "sdf_retain", "ft_control"] if m in set(P.method)]
    fig, axs = plt.subplots(2, 4, figsize=(15, 7.6))
    for ax, m in zip(axs.flat, show):
        g = P[(P.method == m) & (P.category == "grid_add")].groupby(["x", "y"]).dlp.mean().abs()
        Z = np.full((21, 21), np.nan)
        for (x, y), v in g.items():
            Z[int(x), int(y)] = v
        im = ax.imshow(Z, cmap="Blues", vmin=0, vmax=2, origin="lower")
        ax.plot(2 if "2_2" in "add_2_2_5" else 0, 2, marker="x", color="#e34948", ms=8)
        ax.set_title(LABEL[m], fontsize=9, loc="left"); ax.grid(False)
        ax.set_xlabel("second operand y"); ax.set_ylabel("first operand x")
    for ax in list(axs.flat)[len(show):]:
        ax.axis("off")
    fig.colorbar(im, ax=axs, fraction=0.015, label="|Δ log P(true sum)| (nats, capped at 2)")
    fig.suptitle("Where the 2+2→5 edit leaks on the 0–20 addition grid (red × = edited cell)", x=0.01, ha="left")
    fig.savefig(os.path.join(FIG, "fig4_grid_leakage.png"), dpi=140, bbox_inches="tight"); plt.close(fig)
    # ---- statistics
    lines = []
    for e in edits:
        S = allS[e]
        S = S[S.method != "ft_control"]
        for name, sub in [("all methods", S), ("retain-regularised FT/SDF", S[S.method.str.endswith("retain")]),
                          ("unregularised FT", S[S.method.isin(["ft_exact", "ft_para", "ft_entail", "sdf"])])]:
            for col in ["L_change", "L_absdlp", "arith_agree", "cf_agree", "mmlu_agree"]:
                if col not in sub or sub[col].isna().all() or len(sub) < 4:
                    continue
                y = sub[col] if col in ("L_change", "L_absdlp") else 1 - sub[col]
                rho, p = stats.spearmanr(sub.E_prop, y)
                lines.append(f"{e} | {name} (n={len(sub)} runs) | Spearman(E_prop, {'1-' if col.endswith('agree') else ''}{col}) = {rho:+.3f} (p={p:.2g})")
    # pooled Wilson CIs of key rates (pooled over seeds) for primary edit
    P = allP["add_2_2_5"]; ok = P[P.base_ok]
    rows = []
    for m, g in ok.groupby("method"):
        E, L = g[g.set == "E"], g[g.set == "L"]
        kE, nE, kL, nL = int(E.is_belief.sum()), len(E), int(L.changed.sum()), len(L)
        rows.append(dict(method=m, E_prop=kE / nE, E_ci=wilson(kE, nE), L_change=kL / nL, L_ci=wilson(kL, nL),
                         L_leak5=L.leak_t.mean()))
    W = pd.DataFrame(rows)
    W["order"] = W.method.map({m: i for i, m in enumerate(METHOD_ORDER + ["ft_control"])}); W = W.sort_values("order")
    with open(os.path.join(SUM, "stats.md"), "w") as f:
        f.write("\n\n".join(lines) + "\n\n")
        f.write("| method | E-propagation [95% Wilson CI] | L-change [95% Wilson CI] | L leaked '5' |\n|---|---|---|---|\n")
        for _, r in W.iterrows():
            f.write(f"| {LABEL.get(r.method, r.method)} | {r.E_prop:.3f} [{r.E_ci[0]:.3f}, {r.E_ci[1]:.3f}] | "
                    f"{r.L_change:.4f} [{r.L_ci[0]:.4f}, {r.L_ci[1]:.4f}] | {r.L_leak5:.4f} |\n")
    print("\n".join(lines))


def grid_features(x, y, a=2, b=2, t=5):
    o = a + b; s = x + y
    return {"sum_eq_orig": float(s == o), "sum_eq_new": float(s == t), "has_operand_a_or_b": float(x in (a, b) or y in (a, b)),
            "equal_operands": float(x == y), "same_parity_as_orig": float(s % 2 == o % 2),
            "units_digit_eq_orig": float(s % 10 == o % 10), "L1_dist": float(min(abs(x - a) + abs(y - b), abs(x - b) + abs(y - a))),
            "two_digit_sum": float(s >= 10), "two_digit_operand": float(x >= 10 or y >= 10)}


def mech_regression(edit="add_2_2_5", a=2, b=2, t=5, n_boot=500):
    """Standardised OLS of grid |dlogP(true)| (mean over seeds) on mechanism-motivated features, + GradSim."""
    P = pd.read_csv(os.path.join(SUM, f"probes_{edit}.csv"))
    gs_path = os.path.join(SUM, f"gradsim_{edit}.csv")
    GS = pd.read_csv(gs_path).set_index(["x", "y"]).gradsim if os.path.exists(gs_path) else None
    rows = []
    rng = np.random.default_rng(0)
    for m, g in P[(P.category == "grid_add")].groupby("method"):
        d = g.groupby(["x", "y"]).dlp.mean().abs().reset_index()
        X = pd.DataFrame([grid_features(int(r.x), int(r.y), a, b, t) for r in d.itertuples()])
        Xz = (X - X.mean()) / X.std().replace(0, 1)
        yv = d.dlp.values; yz = (yv - yv.mean()) / (yv.std() + 1e-9)
        A = np.c_[np.ones(len(Xz)), Xz.values]
        coef = np.linalg.lstsq(A, yz, rcond=None)[0][1:]
        boots = []
        for _ in range(n_boot):
            i = rng.integers(0, len(yz), len(yz))
            boots.append(np.linalg.lstsq(A[i], yz[i], rcond=None)[0][1:])
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        r2 = 1 - ((yz - A @ np.linalg.lstsq(A, yz, rcond=None)[0]) ** 2).sum() / (yz ** 2).sum()
        rec = dict(method=m, R2=r2)
        for f, c, l, h in zip(X.columns, coef, lo, hi):
            rec[f] = f"{c:+.2f} [{l:+.2f},{h:+.2f}]"
        if GS is not None:
            gsv = np.array([GS.get((int(r.x), int(r.y)), np.nan) for r in d.itertuples()])
            ok = ~np.isnan(gsv)
            rho, p = stats.spearmanr(gsv[ok], yv[ok]); rec["GradSim_spearman"] = f"{rho:+.2f} (p={p:.1g})"
            rho2, p2 = stats.spearmanr(-X["L1_dist"].values, yv); rec["negL1_spearman"] = f"{rho2:+.2f} (p={p2:.1g})"
        rows.append(rec)
    R = pd.DataFrame(rows)
    R["order"] = R.method.map({m: i for i, m in enumerate(METHOD_ORDER + ["ft_control"])}); R = R.sort_values("order").drop(columns="order")
    R.to_csv(os.path.join(SUM, f"mech_regression_{edit}.csv"), index=False)
    with open(os.path.join(SUM, f"mech_regression_{edit}.md"), "w") as f:
        f.write(R.to_markdown(index=False))
    return R


if __name__ == "__main__":
    main()
    for e, (a, b, t) in {"add_2_2_5": (2, 2, 5), "add_2_2_6": (2, 2, 6), "add_3_5_9": (3, 5, 9)}.items():
        if os.path.exists(os.path.join(SUM, f"probes_{e}.csv")):
            print(mech_regression(e, a, b, t).to_string())
