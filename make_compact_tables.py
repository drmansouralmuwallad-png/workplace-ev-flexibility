import pandas as pd
from make_portrait_tables import write, hdr, lab, srt, SN
from make_paper_assets import metrics
m = srt(metrics()); rows = []
for _, d in m.groupby(["_s", "_o", "_q"]):
    p, q, site = d.policy.iloc[0], d.q.iloc[0], d.site.iloc[0]
    rows.append([SN[site], lab(p, q), f"{d.bill.mean():,.0f}", f"{d.peak_all_kw.mean():.0f}", f"{100*d.served_ratio.mean():.1f}",
                 f"{100*d.share_fully_served.mean():.1f}", f"{d.gap_to_offline_pct.mean():+.1f}", f"{d.penalised_cost.mean():,.0f}"])
write("Table4_main", "Mean monthly outcomes by policy (July--December 2019).", "tab:main", "llrrrrrr",
      ["Site", "Policy", hdr("Bill", "(USD)"), hdr("Peak", "(kW)"), hdr("Served", "(\\%)"), hdr("EVs", "full (\\%)"), hdr("Gap to", "opt. (\\%)"), hdr("Penalised", "(USD)")], rows,
      "Six-month means; monthly ranges and the energy/demand-charge split are given in Tables S2 and S3. Gap is the mean monthly bill gap to the hindsight optimum; Penalised adds 5\\,USD per undelivered kWh.")
fm = pd.read_csv("results/flexibility_metrics.csv")
keep = ["uncontrolled", "llf", "mpc_nominal", "mpc_robust_q0.9", "mpc_topup", "mpc_topup_q0.7", "offline"]
names = {"uncontrolled": "Uncontrolled", "llf": "LLF", "mpc_nominal": "Nominal MPC", "mpc_robust_q0.9": "Robust MPC, $q$=0.9",
         "mpc_topup": "MPC + top-up", "mpc_topup_q0.7": "MPC + top-up, $q$=0.7", "offline": "Hindsight opt."}
fm = fm[fm.policy_tag.isin(keep)].copy(); fm["_s"] = fm.site.map({"jpl": 0, "caltech": 1}); fm["_o"] = fm.policy_tag.map({k: i for i, k in enumerate(keep)})
rows = []
for _, d in fm.groupby(["_s", "_o"]):
    rows.append([SN[d.site.iloc[0]], names[d.policy_tag.iloc[0]], f"{d.peak_red_pct.mean():.1f}", f"{d.load_factor.mean():.3f}", f"{d.par.mean():.1f}",
                 f"{d.max_ramp_kw_per_15min.mean():.1f}", f"{d.shifted_pct.mean():.1f}", f"{d.pv100_self_sufficiency_pct.mean():.1f}"])
write("Table5_flex", "Grid-flexibility and solar-coincidence metrics (means over July--December 2019).", "tab:flex", "llrrrrrr",
      ["Site", "Policy", hdr("Peak", "red. (\\%)"), hdr("Load", "factor"), "PAR", hdr("Max ramp", "(kW/15 min)"), hdr("Shifted", "(\\%)"), hdr("PV self-", "suff. (\\%)")], rows,
      "Peak reduction and shifted energy are relative to uncontrolled charging; PAR is the peak-to-average ratio. PV self-sufficiency uses an energy-neutral clear-sky PV array with PV-unaware control. Monthly ranges and further solar metrics are in Tables S4 and S5.")
print("ok")
