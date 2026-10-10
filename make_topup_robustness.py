"""Table of the top-up robustness runs (results/topup_robustness.csv -> paper/tables/Table13_topup_robustness.tex)."""
import pandas as pd

d = pd.read_csv("results/topup_robustness.csv")
d["period"] = d.month.str[:4]
g = d.groupby(["scenario", "site", "period", "policy"]).agg(
    bill=("bill", "mean"), peak=("peak_all_kw", "mean"), served=("served_ratio", "mean"),
    full=("share_fully_served", "mean")).reset_index()
p = g.pivot_table(index=["scenario", "site", "period"], columns="policy", values=["bill", "peak", "served", "full"])
r = pd.DataFrame({
    "nom": 100 * p[("served", "mpc_nominal")], "top": 100 * p[("served", "mpc_topup")],
    "full": 100 * p[("full", "mpc_topup")],
    "pred": 100 * (1 - p[("peak", "mpc_topup")] / p[("peak", "uncontrolled")]),
    "sav": 100 * (1 - p[("bill", "mpc_topup")] / p[("bill", "uncontrolled")]),
}).reset_index()
r.to_csv("results/topup_robustness_summary.csv", index=False)

SCEN = [("base", "Base case"), ("capacity_75", "Capacity $\\times$0.75"), ("capacity_50", "Capacity $\\times$0.5"),
        ("demand_x2", "Demand charge $\\times$2"), ("train_30d", "30-day calibration"),
        ("train_90d", "90-day calibration"), ("penalty_20", "Penalty $\\times$4")]
rows = []
for site, sn in [("jpl", "JPL"), ("caltech", "Caltech")]:
    for k, (sc, lab) in enumerate(SCEN):
        cells = [sn if k == 0 else "", lab]
        for per in ["2019", "2021"]:
            x = r[(r.scenario == sc) & (r.site == site) & (r.period == per)].iloc[0]
            cells += [f"{x.nom:.1f}", f"{x.top:.1f}", f"{x.full:.1f}", f"{x.pred:.0f}", f"{x.sav:.0f}"]
        rows.append(" & ".join(cells) + r" \\")
    if site == "jpl":
        rows.append(r"\hline")
L = [r"\begin{table}[!htbp]", r"\centering",
     r"\caption{Robustness of the top-up rule. Means over August and November 2019 and over the held-out months July and August 2021. Served: energy served (\%) by nominal MPC (Nom.) and MPC + top-up (Top.); Full: vehicles fully served by MPC + top-up (\%); Peak red.\ and Saving: reduction of the monthly peak and of the bill of MPC + top-up relative to uncontrolled charging (\%). Site capacity scales the 195\,kW (JPL) and 150\,kW (Caltech) limits.}",
     r"\label{tab:topuprob}", r"\footnotesize", r"\setlength{\tabcolsep}{1.5pt}",
     r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrrrrrrr@{}}", r"\hline",
     r" & & \multicolumn{5}{c}{2019 (Aug, Nov)} & \multicolumn{5}{c}{2021 held out (Jul, Aug)} \\",
     r"Site & Scenario & Nom. & Top. & Full & \makecell[r]{Peak\\red.} & Saving & Nom. & Top. & Full & \makecell[r]{Peak\\red.} & Saving \\",
     r"\hline"] + rows + [r"\hline", r"\end{tabular*}", r"\end{table}"]
open("paper/tables/Table13_topup_robustness.tex", "w").write("\n".join(L) + "\n")
print(r.round(1).to_string())
