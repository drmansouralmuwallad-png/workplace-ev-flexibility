"""Revision assets: split tables with monthly ranges, held-out 2021 table, figure legibility fixes."""
import json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from make_paper_assets import metrics, save, COL, PN, SN, SITES, cfg, table_render

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42})
R, T = "results", "paper/tables"
ORDER = {"uncontrolled": 0, "edf": 1, "llf": 2, "mpc_nominal": 3, "mpc_robust": 4, "mpc_topup": 5, "offline": 6}


def lab(p, q):
    base = PN[p]
    return base + (f" ($q$={q:.1f})" if p in ("mpc_robust", "mpc_topup") and pd.notna(q) else "")


def rng(v, fmt):
    return f"{fmt.format(v.mean())} ({fmt.format(v.min())}--{fmt.format(v.max())})"


def tex(df, name, caption, label, colfmt, note=None, size=r"\footnotesize"):
    esc = lambda x: str(x).replace("%", r"\%")
    L = [r"\begin{table}[!htbp]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}", size,
         rf"\begin{{tabular}}{{{colfmt}}}", r"\hline", " & ".join(esc(c) for c in df.columns) + r" \\", r"\hline"]
    prev = None
    for _, r in df.iterrows():
        vals = [esc(v) for v in r.values]
        if prev is not None and vals[0] != prev:
            L.append(r"\hline")
        prev = vals[0]
        L.append(" & ".join(vals) + r" \\")
    L += [r"\hline", r"\end{tabular}"]
    if note:
        L.append(rf"\par\smallskip\parbox{{\linewidth}}{{\scriptsize {note}}}")
    L.append(r"\end{table}")
    open(f"{T}/{name}.tex", "w").write("\n".join(L) + "\n")


def render(df, name, title):
    d = df.copy(); d.columns = [c.replace("$q$", "q") for c in d.columns]
    d = d.apply(lambda c: c.map(lambda x: str(x).replace("$q$", "q").replace("--", "-")))
    table_render(d, name, title)


def sort_main(m):
    m = m.copy(); m["_s"] = m.site.map({"jpl": 0, "caltech": 1}); m["_o"] = m.policy.map(ORDER)
    return m


# ---------------- Table 4a / 4b: six-month means with monthly ranges
m = sort_main(metrics())
rows_a, rows_b = [], []
for (s, o, p, q), d in m.groupby(["_s", "_o", "policy", "q"], dropna=False):
    site = d.site.iloc[0]
    rows_a.append([SN[site], lab(p, q), rng(d.bill, "{:,.0f}"), f"{d.energy_cost.mean():,.0f}", f"{d.frd_cost.mean():,.0f}",
                   f"{d.gap_to_offline_pct.mean():+.1f}", f"{d.penalised_cost.mean():,.0f}"])
    rows_b.append([SN[site], lab(p, q), rng(d.peak_all_kw, "{:.0f}"), rng(100 * d.served_ratio, "{:.1f}"),
                   rng(100 * d.share_fully_served, "{:.1f}")])
ta = pd.DataFrame(rows_a, columns=["Site", "Policy", "Bill (USD)", "Energy (USD)", "Demand (USD)", "Gap (%)", "Penalised (USD)"])
tb = pd.DataFrame(rows_b, columns=["Site", "Policy", "Peak (kW)", "Energy served (%)", "EVs fully served (%)"])
note = "Six-month means (July--December 2019) with the range of monthly values in parentheses. Gap is the mean monthly bill gap to the hindsight optimum; Penalised adds 5\\,USD per undelivered kWh."
tex(ta, "Table4a_costs", "Monthly electricity cost by policy.", "tab:main", "llrrrrr", note)
tex(tb, "Table4b_service", "Monthly peak demand and charging service by policy.", "tab:service", "llrrr",
    "Six-month means with the range of monthly values in parentheses. Energy served is delivered over demanded energy; a vehicle is fully served when its shortfall is at most 0.05\\,kWh.")
render(ta, "Table4a_costs", "Table 4a. Monthly electricity cost by policy (mean and monthly range)")
render(tb, "Table4b_service", "Table 4b. Peak demand and charging service by policy (mean and monthly range)")

# ---------------- Table 5a / 5b: flexibility and solar
fm = pd.read_csv(f"{R}/flexibility_metrics.csv")
keep = ["uncontrolled", "llf", "mpc_nominal", "mpc_robust_q0.9", "mpc_topup", "mpc_topup_q0.7", "offline"]
names = {"uncontrolled": "Uncontrolled", "llf": "LLF", "mpc_nominal": "Nominal MPC", "mpc_robust_q0.9": "Robust MPC ($q$=0.9)",
         "mpc_topup": "MPC + top-up", "mpc_topup_q0.7": "MPC + top-up ($q$=0.7)", "offline": "Hindsight optimum"}
fm = fm[fm.policy_tag.isin(keep)].copy(); fm["_s"] = fm.site.map({"jpl": 0, "caltech": 1}); fm["_o"] = fm.policy_tag.map({k: i for i, k in enumerate(keep)})
r5a, r5b = [], []
for (s, o, pt), d in fm.groupby(["_s", "_o", "policy_tag"]):
    site = d.site.iloc[0]
    r5a.append([SN[site], names[pt], rng(d.peak_red_pct, "{:.1f}"), f"{d.load_factor.mean():.3f}", rng(d.par, "{:.1f}"),
                f"{d.max_ramp_kw_per_15min.mean():.1f}", f"{d.shifted_pct.mean():.1f}"])
    r5b.append([SN[site], names[pt], f"{d.onpeak_energy_pct.mean():.1f}", rng(d.pv100_self_sufficiency_pct, "{:.1f}"),
                f"{d.pv100_utilisation_pct.mean():.1f}", f"{d.pv50_self_sufficiency_pct.mean():.1f}"])
t5a = pd.DataFrame(r5a, columns=["Site", "Policy", "Peak red. (%)", "Load factor", "PAR", "Max ramp (kW/15 min)", "Shifted (%)"])
t5b = pd.DataFrame(r5b, columns=["Site", "Policy", "On-peak energy (%)", "PV self-suff. (%)", "PV util. (%)", "PV self-suff., half array (%)"])
tex(t5a, "Table5a_flexibility", "Grid-flexibility metrics (July--December 2019).", "tab:flex", "llrrrrr",
    "Means with monthly ranges in parentheses. Peak reduction and shifted energy are relative to uncontrolled charging; PAR is the peak-to-average ratio of the monthly load.")
tex(t5b, "Table5b_solar", "On-peak energy and solar-coincidence metrics (July--December 2019).", "tab:solar", "llrrrr",
    "On-peak energy is the share delivered in the 12--18\\,h weekday block. PV self-sufficiency is the share of EV energy coincident with an energy-neutral clear-sky PV array (monthly range in parentheses); utilisation is the share of PV output absorbed by EVs.")
render(t5a, "Table5a_flexibility", "Table 5a. Grid-flexibility metrics (mean and monthly range)")
render(t5b, "Table5b_solar", "Table 5b. On-peak energy and solar-coincidence metrics")

# ---------------- Sensitivity with two-month range
s = pd.read_csv(f"{R}/sensitivity_metrics.csv"); s["_s"] = s.site.map({"jpl": 0, "caltech": 1}); s["_o"] = s.policy.map(ORDER)
r6 = []
for (sc, ss, o, p, q), d in s.groupby(["scenario", "_s", "_o", "policy", "q"], dropna=False):
    r6.append([sc.replace("_", " "), SN[d.site.iloc[0]], lab(p, q), rng(d.bill, "{:,.0f}"), f"{d.peak_all_kw.mean():.1f}", f"{100*d.served_ratio.mean():.2f}"])
t6 = pd.DataFrame(r6, columns=["Scenario", "Site", "Policy", "Bill (USD)", "Peak (kW)", "Served (%)"])
tex(t6, "Table6_sensitivity", "Sensitivity analysis (means over August and November 2019; range of the two months in parentheses).", "tab:sens", "lllrrr")
render(t6, "Table6_sensitivity", "Table 6. Sensitivity analysis (August and November 2019)")

# ---------------- Tariff sweep split
p = pd.read_csv(f"{R}/tariff_sweep_summary.csv"); p["_s"] = p.site.map({"jpl": 0, "caltech": 1}); p = p.sort_values(["_s", "demand_charge"])
t7a = pd.DataFrame({"Site": p.site.map(SN), "Demand charge (USD/kW-month)": p.demand_charge.map("{:.2f}".format),
                    "Peak, uncontrolled (kW)": p.peak_uncontrolled.map("{:.1f}".format), "Peak, MPC + top-up (kW)": p.peak_mpc_topup.map("{:.1f}".format),
                    "Peak, optimum (kW)": p.peak_offline.map("{:.1f}".format), "Peak red. (%)": p.peakred_topup_pct.map("{:.1f}".format)})
t7b = pd.DataFrame({"Site": p.site.map(SN), "Demand charge (USD/kW-month)": p.demand_charge.map("{:.2f}".format),
                    "Bill, uncontrolled (USD)": p.bill_uncontrolled.map("{:,.0f}".format), "Bill, MPC + top-up (USD)": p.bill_mpc_topup.map("{:,.0f}".format),
                    "Saving (%)": p.saving_topup_pct.map("{:.1f}".format), "Served (%)": (100 * p.served_mpc_topup).map("{:.2f}".format)})
tex(t7a, "Table7a_tariff_peak", "Tariff-design sweep: peak demand (means over August and November 2019).", "tab:tariff_sweep", "lrrrrr",
    "Peak reduction is for MPC + top-up relative to uncontrolled charging. 15.51\\,USD/kW-month is the TOU-EV-4 value.")
tex(t7b, "Table7b_tariff_bill", "Tariff-design sweep: bill and service (means over August and November 2019).", "tab:tariff_bill", "lrrrrr")
render(t7a, "Table7a_tariff_peak", "Table 7a. Tariff-design sweep: peak demand")
render(t7b, "Table7b_tariff_bill", "Table 7b. Tariff-design sweep: bill and service")

# ---------------- Held-out 2021 table
h = pd.concat([pd.read_csv(f"{R}/{s}_metrics_heldout2021.csv") for s in SITES]); h = sort_main(h)
unc = h[h.policy == "uncontrolled"].set_index(["site", "month"]).bill
h["saving"] = [100 * (1 - b / unc[(s, mo)]) for s, mo, b in zip(h.site, h.month, h.bill)]
r9 = []
for (ss, o, pp, q), d in h.groupby(["_s", "_o", "policy", "q"], dropna=False):
    if pp == "edf":
        continue
    r9.append([SN[d.site.iloc[0]], lab(pp, q), f"{d.bill.mean():,.0f}", f"{d.saving.mean():+.1f}", f"{d.peak_all_kw.mean():.1f}",
               f"{100*d.served_ratio.mean():.1f}", f"{100*d.share_fully_served.mean():.1f}", f"{d.penalised_cost.mean():,.0f}"])
t9 = pd.DataFrame(r9, columns=["Site", "Policy", "Bill (USD)", "Saving vs unc. (%)", "Peak (kW)", "Served (%)", "EVs full (%)", "Penalised (USD)"])
tex(t9, "Table9_heldout", "Out-of-sample validation on held-out 2021 sessions (means over July and August 2021; calibration on the preceding 60 days). EDF coincides with LLF and is omitted.",
    "tab:heldout", "llrrrrrr")
render(t9, "Table9_heldout", "Table 9. Out-of-sample validation, July-August 2021")
open(f"{R}/heldout_summary.json", "w").write(json.dumps({"n": {s: int(h[(h.site == s) & (h.policy == 'offline')].n_sessions.sum()) for s in SITES}}))

# ---------------- Figure fixes
# Fig 5 monthly peaks: legend below, note LLF=uncontrolled at Caltech in caption
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4))
for k, site in enumerate(SITES):
    ax = axs[k]; d = m[m.site == site]
    for pol, qq, lb, ls in [("uncontrolled", None, "Uncontrolled", "-"), ("llf", None, "LLF", "--"), ("mpc_nominal", None, "Nominal MPC", "-"),
                            ("mpc_robust", 0.9, "Robust MPC (q=0.9)", "-"), ("mpc_topup", None, "MPC + top-up", ":"), ("offline", None, "Hindsight optimum", "-")]:
        dd = d[(d.policy == pol) & ((d.q == qq) if qq is not None else d.q.isna())]
        ax.plot(dd.month.str[5:], dd.peak_all_kw, ls, marker="o", ms=3, color=COL[pol], label=lb, lw=1.3)
    ax.axhline(cfg["physics"]["site_capacity_kw"][site], color="k", ls=":", lw=0.8)
    ax.set_title(SN[site] + " (2019)", fontsize=9); ax.set_ylabel("Monthly peak (kW)"); ax.set_xlabel("Month")
h_, l_ = axs[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=3, frameon=False, fontsize=7); fig.tight_layout(rect=(0, 0.12, 1, 1)); save(fig, "Fig7_monthly_peaks")

# Fig 10 calibrated sets: legend below
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3))
for site, ls in [("jpl", "-"), ("caltech", "--")]:
    c = pd.read_csv(f"{R}/{site}_calibration.csv")
    for q, colr in zip(sorted(c.q.unique()), ["#fdae6b", "#e6550d", "#a63603"]):
        d = c[c.q == q]
        axs[0].plot(d.month.str[5:], d.claimed_shift_h, ls, color=colr, marker="o", ms=3, label=f"{SN[site]} q={q}")
        axs[1].plot(d.month.str[5:], d.claimed_energy_mult, ls, color=colr, marker="o", ms=3, label=f"{SN[site]} q={q}")
axs[0].set_ylabel("Deadline shift $Q_{1-q}(\\Delta d)$ (h)"); axs[1].set_ylabel("Energy multiplier $Q_q(\\rho)$ (-)")
axs[0].axhline(0, color="k", lw=0.6); axs[1].axhline(1, color="k", lw=0.6)
for a in axs: a.set_xlabel("Test month (2019)")
h_, l_ = axs[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=3, frameon=False, fontsize=7); fig.tight_layout(rect=(0, 0.15, 1, 1)); save(fig, "Fig8_calibrated_sets")

# Fig 11 frontier: labels without overlap (table of points instead of inline q labels for top-up)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.6))
for k, site in enumerate(SITES):
    ax = axs[k]; g = m[m.site == site].groupby(["policy", "q"], dropna=False)[["bill", "served_ratio"]].mean().reset_index()
    rob = g[g.policy == "mpc_robust"].sort_values("q")
    ax.plot(100 * rob.served_ratio, rob.bill, "-o", color=COL["mpc_robust"], ms=4, label="Robust MPC (q = 0.5, 0.7, 0.9)")
    for _, r in rob.iterrows():
        ax.annotate(f"{r.q:.1f}", (100 * r.served_ratio, r.bill), fontsize=6.5, xytext=(-4, 6), textcoords="offset points", color=COL["mpc_robust"])
    top = g[g.policy == "mpc_topup"].copy(); top["qq"] = top.q.fillna(0)
    mk = {0: "P", 0.7: "X", 0.9: "h"}
    for _, r in top.iterrows():
        ax.plot(100 * r.served_ratio, r.bill, mk[r.qq], color=COL["mpc_topup"], ms=7, ls="none",
                label="MPC + top-up" + ("" if r.qq == 0 else f" (q={r.qq:.1f})"))
    for pp, mkr in [("uncontrolled", "s"), ("llf", "v"), ("mpc_nominal", "D"), ("offline", "*")]:
        r = g[g.policy == pp]
        ax.plot(100 * r.served_ratio, r.bill, mkr, color=COL[pp], ms=8 if pp == "offline" else 6, ls="none", label=PN[pp] + (" (= LLF, EDF)" if pp == "uncontrolled" and site == "caltech" else ""))
    ax.set_title(SN[site], fontsize=9); ax.set_xlabel("Energy served (% of demanded)"); ax.set_ylabel("Mean monthly bill (USD)")
h_, l_ = axs[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=3, frameon=False, fontsize=6.8); fig.tight_layout(rect=(0, 0.17, 1, 1)); save(fig, "Fig5_price_of_robustness")
print("ok")
