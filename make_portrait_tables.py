r"""Portrait tables that span exactly the text width (tabular* with \extracolsep{\fill}).
Ranges are stacked under means with makecell so no table exceeds the page width."""
import json
import numpy as np, pandas as pd
from make_paper_assets import metrics, SN, SITES

R, T = "results", "paper/tables"
ORDER = {"uncontrolled": 0, "edf": 1, "llf": 2, "mpc_nominal": 3, "mpc_robust": 4, "mpc_topup": 5, "offline": 6}
SHORT = {"uncontrolled": "Uncontrolled", "edf": "EDF", "llf": "LLF", "mpc_nominal": "Nominal MPC",
         "mpc_robust": "Robust MPC", "mpc_topup": "MPC + top-up", "offline": "Hindsight opt."}


def lab(p, q):
    return SHORT[p] + (f", $q$={q:.1f}" if p in ("mpc_robust", "mpc_topup") and pd.notna(q) else "")


def mc(a, b=None):
    return rf"{a}\,{{\scriptsize({b})}}" if b is not None else a


def rng(v, f):
    return mc(f.format(v.mean()), f"{f.format(v.min())}--{f.format(v.max())}")


def hdr(*lines):
    return r"\makecell[c]{" + r"\\".join(lines) + "}"


def write(name, caption, label, colspec, header, rows, note=None, groupcol=0, float_spec="!htbp"):
    L = [rf"\begin{{table}}[{float_spec}]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\footnotesize",
         r"\setlength{\tabcolsep}{1.5pt}",
         rf"\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}{colspec}@{{}}}}", r"\hline",
         " & ".join(header) + r" \\", r"\hline"]
    prev = None
    for r in rows:
        if groupcol is not None and prev is not None and r[groupcol] != prev:
            L.append(r"\hline")
        prev = r[groupcol] if groupcol is not None else None
        L.append(" & ".join(str(x).replace("%", r"\%") if ("makecell" not in str(x) and "scriptsize" not in str(x)) else str(x) for x in r) + r" \\")
    L += [r"\hline", r"\end{tabular*}"]
    if note:
        L.append(rf"\par\smallskip\parbox{{\textwidth}}{{\scriptsize {note}}}")
    L.append(r"\end{table}")
    open(f"{T}/{name}.tex", "w").write("\n".join(L) + "\n")


def srt(df):
    df = df.copy(); df["_s"] = df.site.map({"jpl": 0, "caltech": 1}); df["_o"] = df.policy.map(ORDER)
    df["_q"] = df.q.fillna(-1)
    return df


# ---- Table 2 dataset
rows = []
for site in SITES:
    df = pd.read_pickle(f"data/processed/{site}_sessions.pkl"); fl = pd.read_csv(f"{R}/{site}_filter_flow.csv")
    rows.append([SN[site], f"{int(fl.n_remaining.iloc[0]):,}", f"{len(df):,}", f"{df.kwh.sum()/1000:,.1f}", f"{100*df.claimed.mean():.1f}",
                 f"{df.duration_h.median():.2f}", f"{df.kwh.median():.2f}", f"{100*df.weekday.mean():.1f}"])
write("Table2_dataset", "ACN-Data sessions used in this study (1 May--31 December 2019).", "tab:data", "lrrrrrrr",
      ["Site", hdr("Raw", "sessions"), hdr("Retained", "sessions"), hdr("Energy", "(MWh)"), hdr("Claimed", "(\\%)"),
       hdr("Median", "stay (h)"), hdr("Median", "kWh"), hdr("Weekday", "(\\%)")], rows, groupcol=None)

# ---- Table 3 tariff
write("Table3_tariff", r"Energy prices (USD/kWh) of SCE TOU-EV-4 effective 1 March 2019 \cite{SCETOUEV4}; demand charge 15.51\,USD/kW-month on the monthly maximum demand.",
      "tab:tariff", "lrrrr",
      ["Season / day type", hdr("00--08\\,h"), hdr("08--12\\,h,", "18--23\\,h"), hdr("12--18\\,h"), hdr("23--24\\,h;", "weekends")],
      [["Summer (Jun--Sep), weekdays", "0.05623", "0.09250", "0.26668", "0.05623"],
       ["Winter (Oct--May), weekdays", "0.06087", "0.07492", "0.08690", "0.06087"]], groupcol=None)

# ---- Table 4a / 4b
m = srt(metrics())
ra, rb = [], []
for _, d in m.groupby(["_s", "_o", "_q"]):
    p, q, site = d.policy.iloc[0], d.q.iloc[0], d.site.iloc[0]
    ra.append([SN[site], lab(p, q), rng(d.bill, "{:,.0f}"), f"{d.energy_cost.mean():,.0f}", f"{d.frd_cost.mean():,.0f}",
               f"{d.gap_to_offline_pct.mean():+.1f}", f"{d.penalised_cost.mean():,.0f}"])
    rb.append([SN[site], lab(p, q), rng(d.peak_all_kw, "{:.0f}"), rng(100 * d.served_ratio, "{:.1f}"), rng(100 * d.share_fully_served, "{:.1f}")])
write("Table4a_costs", "Monthly electricity cost by policy (July--December 2019).", "tab:main", "llrrrrr",
      ["Site", "Policy", hdr("Bill", "(USD)"), hdr("Energy", "(USD)"), hdr("Demand", "(USD)"), hdr("Gap to", "opt. (\\%)"), hdr("Penalised", "(USD)")], ra,
      "Six-month means; the range of monthly values is shown in parentheses. Gap is the mean monthly bill gap to the hindsight optimum; Penalised adds 5\\,USD per undelivered kWh.")
write("Table4b_service", "Monthly peak demand and charging service by policy (July--December 2019).", "tab:service", "llrrr",
      ["Site", "Policy", hdr("Peak", "(kW)"), hdr("Energy served", "(\\%)"), hdr("EVs fully", "served (\\%)")], rb,
      "Six-month means with the range of monthly values in parentheses. A vehicle is fully served when its shortfall is at most 0.05\\,kWh.")

# ---- Table 5a / 5b
fm = pd.read_csv(f"{R}/flexibility_metrics.csv")
keep = ["uncontrolled", "llf", "mpc_nominal", "mpc_robust_q0.9", "mpc_topup", "mpc_topup_q0.7", "offline"]
names = {"uncontrolled": "Uncontrolled", "llf": "LLF", "mpc_nominal": "Nominal MPC", "mpc_robust_q0.9": "Robust MPC, $q$=0.9",
         "mpc_topup": "MPC + top-up", "mpc_topup_q0.7": "MPC + top-up, $q$=0.7", "offline": "Hindsight opt."}
fm = fm[fm.policy_tag.isin(keep)].copy(); fm["_s"] = fm.site.map({"jpl": 0, "caltech": 1}); fm["_o"] = fm.policy_tag.map({k: i for i, k in enumerate(keep)})
r5a, r5b = [], []
for _, d in fm.groupby(["_s", "_o"]):
    site, pt = d.site.iloc[0], d.policy_tag.iloc[0]
    r5a.append([SN[site], names[pt], rng(d.peak_red_pct, "{:.1f}"), f"{d.load_factor.mean():.3f}", rng(d.par, "{:.1f}"),
                f"{d.max_ramp_kw_per_15min.mean():.1f}", f"{d.shifted_pct.mean():.1f}"])
    r5b.append([SN[site], names[pt], f"{d.onpeak_energy_pct.mean():.1f}", rng(d.pv100_self_sufficiency_pct, "{:.1f}"),
                f"{d.pv100_utilisation_pct.mean():.1f}", f"{d.pv50_self_sufficiency_pct.mean():.1f}"])
write("Table5a_flexibility", "Grid-flexibility metrics (July--December 2019).", "tab:flex", "llrrrrr",
      ["Site", "Policy", hdr("Peak", "red. (\\%)"), hdr("Load", "factor"), "PAR", hdr("Max ramp", "(kW/15 min)"), hdr("Shifted", "(\\%)")], r5a,
      "Means with monthly ranges in parentheses. Peak reduction and shifted energy are relative to uncontrolled charging; PAR is the peak-to-average ratio of the monthly load.")
write("Table5b_solar", "On-peak energy and solar-coincidence metrics (July--December 2019).", "tab:solar", "llrrrr",
      ["Site", "Policy", hdr("On-peak", "energy (\\%)"), hdr("PV self-", "suff. (\\%)"), hdr("PV util.", "(\\%)"), hdr("PV self-suff.,", "half array (\\%)")], r5b,
      "On-peak energy is the share delivered in the 12--18\\,h weekday block. PV self-sufficiency is the share of EV energy coincident with an energy-neutral clear-sky PV array (monthly range in parentheses); utilisation is the share of PV output absorbed by EVs.")

# ---- Table 6 sensitivity
s = srt(pd.read_csv(f"{R}/sensitivity_metrics.csv"))
sc_name = {"demand_x0": "Demand charge $\\times$0", "demand_x2": "Demand charge $\\times$2", "penalty_20": "Penalty 20\\,USD/kWh",
           "train_30d": "30-day window", "train_90d": "90-day window"}
r6 = []
for _, d in s.groupby(["scenario", "_s", "_o", "_q"]):
    r6.append([sc_name[d.scenario.iloc[0]], SN[d.site.iloc[0]], lab(d.policy.iloc[0], d.q.iloc[0]), rng(d.bill, "{:,.0f}"),
               f"{d.peak_all_kw.mean():.1f}", f"{100*d.served_ratio.mean():.2f}"])
write("Table6_sensitivity", "Sensitivity analysis (means over August and November 2019).", "tab:sens", "lllrrr",
      ["Scenario", "Site", "Policy", hdr("Bill", "(USD)"), hdr("Peak", "(kW)"), hdr("Served", "(\\%)")], r6,
      "The range of the two months is shown in parentheses.")

# ---- Table 7a / 7b tariff sweep
p = pd.read_csv(f"{R}/tariff_sweep_summary.csv"); p["_s"] = p.site.map({"jpl": 0, "caltech": 1}); p = p.sort_values(["_s", "demand_charge"])
r7a = [[SN[r.site], f"{r.demand_charge:.2f}", f"{r.peak_uncontrolled:.1f}", f"{r.peak_mpc_topup:.1f}", f"{r.peak_offline:.1f}", f"{r.peakred_topup_pct:.1f}"] for r in p.itertuples()]
r7b = [[SN[r.site], f"{r.demand_charge:.2f}", f"{r.bill_uncontrolled:,.0f}", f"{r.bill_mpc_topup:,.0f}", f"{r.saving_topup_pct:.1f}", f"{100*r.served_mpc_topup:.2f}"] for r in p.itertuples()]
write("Table7a_tariff_peak", "Tariff-design sweep: peak demand (means over August and November 2019).", "tab:tariff_sweep", "lrrrrr",
      ["Site", hdr("Demand charge", "(USD/kW-month)"), hdr("Peak, uncon-", "trolled (kW)"), hdr("Peak, MPC +", "top-up (kW)"), hdr("Peak, opti-", "mum (kW)"), hdr("Peak", "red. (\\%)")], r7a,
      "Peak reduction is for MPC + top-up relative to uncontrolled charging. 15.51\\,USD/kW-month is the TOU-EV-4 value.")
write("Table7b_tariff_bill", "Tariff-design sweep: bill and service (means over August and November 2019).", "tab:tariff_bill", "lrrrrr",
      ["Site", hdr("Demand charge", "(USD/kW-month)"), hdr("Bill, uncon-", "trolled (USD)"), hdr("Bill, MPC +", "top-up (USD)"), hdr("Saving", "(\\%)"), hdr("Served", "(\\%)")], r7b)

# ---- Table 9 held-out
h = srt(pd.concat([pd.read_csv(f"{R}/{s}_metrics_heldout2021.csv") for s in SITES]))
unc = h[h.policy == "uncontrolled"].set_index(["site", "month"]).bill
h["saving"] = [100 * (1 - b / unc[(s, mo)]) for s, mo, b in zip(h.site, h.month, h.bill)]
r9 = []
for _, d in h.groupby(["_s", "_o", "_q"]):
    if d.policy.iloc[0] == "edf":
        continue
    r9.append([SN[d.site.iloc[0]], lab(d.policy.iloc[0], d.q.iloc[0]), f"{d.bill.mean():,.0f}", f"{d.saving.mean():+.1f}", f"{d.peak_all_kw.mean():.1f}",
               f"{100*d.served_ratio.mean():.1f}", f"{100*d.share_fully_served.mean():.1f}", f"{d.penalised_cost.mean():,.0f}"])
write("Table9_heldout", "Out-of-sample validation on held-out 2021 sessions (means over July and August 2021).", "tab:heldout", "llrrrrrr",
      ["Site", "Policy", hdr("Bill", "(USD)"), hdr("Saving vs", "unc. (\\%)"), hdr("Peak", "(kW)"), hdr("Served", "(\\%)"), hdr("EVs", "full (\\%)"), hdr("Penalised", "(USD)")], r9,
      "Calibration uses the 60 days before each test month. EDF coincides with LLF and is omitted.")

# ---- Table A.8 filter flow
rows = []
for site in SITES:
    fl = pd.read_csv(f"{R}/{site}_filter_flow.csv")
    for r in fl.itertuples():
        st = r.step.replace(">=", r"$\geq$").replace("<=", r"$\leq$").replace("->", r"$\rightarrow$").replace("sessionID", "session ID")
        rows.append([SN[site], st, f"{int(r.n_remaining):,}"])
write("Table6_filter_flow", "Session filtering flow.", "tab:flow", "lp{0.68\\textwidth}r", ["Site", "Step", "Sessions"], rows, float_spec="H")
print("tables written")
