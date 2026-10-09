"""Manuscript figures, drawn only from files produced by `run` and `clean`."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORDER = ["uncontrolled", "edf", "llf", "mpc_nominal", "mpc_robust", "offline"]


def fig_errors(df, site, out):
    cl = df[df["claimed"]]
    dd = (cl["departure"] - cl["req_departure"]).dt.total_seconds() / 3600
    rho = cl["kwh"] / cl["req_kwh"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].hist(dd.clip(-12, 12), bins=96, color="0.35")
    ax[0].axvline(0, color="k", lw=0.8)
    ax[0].set_xlabel("Actual minus declared departure (h)"); ax[0].set_ylabel("Sessions")
    ax[1].hist(rho.clip(0, 3), bins=90, color="0.35")
    ax[1].axvline(1, color="k", lw=0.8)
    ax[1].set_xlabel("Delivered / declared energy (-)")
    fig.suptitle(f"{site.upper()}: declaration errors, claimed sessions (n={len(cl)})", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"Fig_{site}_declaration_errors.pdf")); plt.close(fig)


def fig_frontier(m, site, out):
    agg = m.groupby(["policy", "q"], dropna=False)[["bill", "served_ratio"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    rob = agg[agg["policy"] == "mpc_robust"].sort_values("q")
    ax.plot(rob["served_ratio"], rob["bill"], "o-", color="0.2", label="Robust MPC (q)")
    for _, r in rob.iterrows():
        ax.annotate(f"{r['q']:.2f}", (r["served_ratio"], r["bill"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    marks = {"uncontrolled": "s", "edf": "^", "llf": "v", "mpc_nominal": "D", "offline": "*"}
    for p, mk in marks.items():
        r = agg[agg["policy"] == p]
        if len(r):
            ax.plot(r["served_ratio"], r["bill"], mk, ms=8, label=p, color="0.5" if p != "offline" else "k")
    ax.set_xlabel("Energy served / energy demanded (-)"); ax.set_ylabel("Mean monthly bill ($)")
    ax.legend(fontsize=7, frameon=False); fig.tight_layout()
    fig.savefig(os.path.join(out, f"Fig_{site}_price_of_robustness.pdf")); plt.close(fig)


def fig_bill(m, site, out, q_show=0.8):
    sel = m[(m["policy"] != "mpc_robust") | (m["q"] == q_show)]
    agg = sel.groupby("policy")[["energy_cost", "frd_cost", "trd_cost"]].mean().reindex([p for p in ORDER if p in sel["policy"].unique()])
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bottom = np.zeros(len(agg))
    for col, h in zip(["energy_cost", "frd_cost", "trd_cost"], ["", "//", ".."]):
        ax.bar(agg.index, agg[col], bottom=bottom, hatch=h, color="white", edgecolor="k", label=col.replace("_", " "))
        bottom += agg[col].to_numpy()
    ax.set_ylabel("Mean monthly cost ($)"); ax.legend(fontsize=7, frameon=False)
    ax.set_xticks(range(len(agg))); ax.set_xticklabels([p if p != "mpc_robust" else f"robust q={q_show}" for p in agg.index], rotation=20, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"Fig_{site}_bill_decomposition.pdf")); plt.close(fig)


def make_all(root, sites):
    out = os.path.join(root, "figures"); os.makedirs(out, exist_ok=True)
    for site in sites:
        pkl = os.path.join(root, "data", "processed", f"{site}_sessions.pkl")
        if os.path.exists(pkl):
            fig_errors(pd.read_pickle(pkl), site, out)
        mp = os.path.join(root, "results", f"{site}_metrics.csv")
        if os.path.exists(mp):
            m = pd.read_csv(mp)
            fig_frontier(m, site, out); fig_bill(m, site, out)
