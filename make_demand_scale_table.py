"""Supplementary table for the higher-demand sensitivity (results/demand_scale.csv)."""
import pandas as pd

d = pd.read_csv("results/demand_scale.csv"); d["period"] = d.month.str[:4]
e = d.drop_duplicates(["alpha", "site", "month"]).groupby(["alpha", "site", "period"])[["kwh_hist_month", "kwh_scaled_month"]].sum()
g = d.groupby(["alpha", "site", "period", "policy"]).agg(bill=("bill", "mean"), peak=("peak_all_kw", "mean"),
                                                        served=("served_ratio", "mean"), full=("share_fully_served", "mean")).reset_index()
p = g.pivot_table(index=["alpha", "site", "period"], columns="policy", values=["bill", "peak", "served", "full"])
o = pd.DataFrame({
    "eff": 100 * (e.kwh_scaled_month / e.kwh_hist_month - 1),
    "nom": 100 * p[("served", "mpc_nominal")], "top": 100 * p[("served", "mpc_topup")],
    "full": 100 * p[("full", "mpc_topup")], "opt": 100 * p[("served", "offline")],
    "pred": 100 * (1 - p[("peak", "mpc_topup")] / p[("peak", "uncontrolled")]),
    "dpk": p[("peak", "mpc_topup")] - p[("peak", "mpc_nominal")],
    "sav": 100 * (1 - p[("bill", "mpc_topup")] / p[("bill", "uncontrolled")]),
}).reset_index()
o.to_csv("results/demand_scale_summary.csv", index=False)
rows = []
for site, sn in [("jpl", "JPL"), ("caltech", "Caltech")]:
    for k, a in enumerate([1.0, 1.1, 1.25]):
        cells = [sn if k == 0 else "", f"{a:.2f}"]
        for per in ["2019", "2021"]:
            x = o[(o.alpha == a) & (o.site == site) & (o.period == per)].iloc[0]
            cells += [f"+{x.eff:.0f}", f"{x.nom:.1f}", f"{x.top:.1f}", f"{x.full:.1f}", f"{x.opt:.1f}", f"{x.pred:.0f}", f"{x.sav:.0f}"]
        rows.append(" & ".join(cells) + r" \\")
    if site == "jpl":
        rows.append(r"\hline")
L = [r"\begin{table}[H]", r"\centering",
     r"\caption{Sensitivity to higher driver demand. Each session's energy need is raised to $\alpha$ times its historically delivered energy, capped at what its estimated maximum rate can deliver while plugged in; declarations are unchanged. Means over August and November 2019 and over the held-out July and August 2021. $\Delta E$: resulting increase in total demand (\%). Served: energy served (\%) by nominal MPC (Nom.), MPC + top-up (Top.) and the hindsight optimum (Opt.); Full: vehicles fully served by MPC + top-up (\%); Peak red.\ and Saving: peak and bill reduction of MPC + top-up relative to uncontrolled charging (\%).}",
     r"\label{tab:demandscale}", r"\scriptsize", r"\setlength{\tabcolsep}{1.5pt}",
     r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrrrrrrrrrrr@{}}", r"\hline",
     r" & & \multicolumn{7}{c}{2019 (Aug, Nov)} & \multicolumn{7}{c}{2021 held out (Jul, Aug)} \\",
     r"Site & $\alpha$ & $\Delta E$ & Nom. & Top. & Full & Opt. & \makecell[r]{Peak\\red.} & Saving & $\Delta E$ & Nom. & Top. & Full & Opt. & \makecell[r]{Peak\\red.} & Saving \\",
     r"\hline"] + rows + [r"\hline", r"\end{tabular*}", r"\end{table}"]
open("paper/tables/supp/Table_demand_scale.tex", "w").write("\n".join(L) + "\n") if __import__("os").path.isdir("paper/tables/supp") else None
open("results/Table_demand_scale.tex", "w").write("\n".join(L) + "\n")
print(o.round(1).to_string())
