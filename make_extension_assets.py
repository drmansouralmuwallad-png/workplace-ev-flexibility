"""Tables and figure for the PV-aware, transformer and tariff-archetype extensions."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from make_portrait_tables import write, hdr, mc
from make_paper_assets import SN, save, table_render
R = "results"; S = {"jpl": 0, "caltech": 1}

# ---- PV-aware
d = pd.read_csv(f"{R}/pv_aware_metrics.csv"); d["pvself"] = 100 * d.pv_used_kwh / d.energy_kwh_delivered; d["pvutil"] = 100 * d.pv_used_kwh / d.pv_kwh
key = [("uncontrolled", None, False, "Uncontrolled"), ("mpc_topup", None, False, "Top-up, PV-unaware"),
       ("mpc_topup", None, True, "PV-aware top-up"), ("mpc_topup", 0.7, True, "PV-aware, $q$=0.7"),
       ("offline", None, True, "PV-aware opt.")]
rows = []; pvsum = []
for site in ["jpl", "caltech"]:
    for p, q, a, lab in key:
        x = d[(d.site == site) & (d.policy == p) & (d.pv_aware == a) & ((d.q == q) if q is not None else d.q.isna())]
        rows.append([SN[site], lab, f"{x.bill.mean():,.0f}", mc(f"{x.peak_all_kw.mean():.1f}", f"{x.peak_all_kw.min():.0f}--{x.peak_all_kw.max():.0f}"),
                     f"{100*x.served_ratio.mean():.1f}", f"{100*x.share_fully_served.mean():.1f}", f"{x.pvself.mean():.1f}", f"{x.penalised_cost.mean():,.0f}"])
        pvsum.append((site, lab.replace("$q$", "q"), x.peak_all_kw.mean(), x.pvself.mean()))
write("Table10_pv_aware", "PV-aware control with an energy-neutral clear-sky PV array (means over July--December 2019; all policies billed on grid import).", "tab:pv", "llrrrrrr",
      ["Site", "Policy", hdr("Bill", "(USD)"), hdr("Import", "peak (kW)"), hdr("Served", "(\\%)"), hdr("EVs", "full (\\%)"), hdr("PV self-", "suff. (\\%)"), hdr("Penalised", "(USD)")], rows,
      "Monthly range of the import peak below the mean. PV self-sufficiency is the share of delivered EV energy supplied by PV. Surplus PV is curtailed without export credit.")

# ---- Transformer
x = pd.read_csv(f"{R}/transformer_metrics.csv")
names = {"uncontrolled": "Uncontrolled", "llf": "LLF", "mpc_topup": "MPC + top-up", "offline": "Hindsight opt."}
rows = []
for site in ["jpl", "caltech"]:
    for frac in [0.5, 0.35]:
        for p, q in [("uncontrolled", None), ("llf", None), ("mpc_topup", None), ("mpc_topup", 0.7), ("offline", None)]:
            y = x[(x.site == site) & (x.rating_frac == frac) & (x.policy == p) & ((x.q == q) if q is not None else x.q.isna())]
            rows.append([SN[site], f"{y.rating_kw.iloc[0]:.0f} ({int(frac*100)}%)", ("Top-up" if p == "mpc_topup" else names[p]) + (f", $q$={q}" if q else ""),
                         f"{y.max_loading_pu.mean():.2f}", f"{y.hours_over.mean():.1f}", f"{y.energy_over_kwh.mean():,.0f}",
                         f"{100*y.served_ratio.mean():.1f}", f"{100*y.share_fully_served.mean():.1f}"])
write("Table11_transformer", "Transformer hosting scenarios (means over August and November 2019).", "tab:xfmr", "lllrrrrr",
      ["Site", hdr("Rating", "kW (\\%)"), "Policy", hdr("Max loading", "(p.u.)"), hdr("Overload", "(h/mo)"), hdr("Overload", "(kWh/mo)"), hdr("Served", "(\\%)"), hdr("EVs", "full (\\%)")], rows,
      "Ratings are 50\\% and 35\\% of the six-month mean uncontrolled peak. Controlled policies treat the rating as a hard limit; uncontrolled charging does not.")

# ---- Tariff archetypes
t = pd.read_csv(f"{R}/tariff_archetypes.csv")
base = pd.concat([pd.read_csv(f"{R}/{s}_metrics.csv") for s in ["jpl", "caltech"]] + [pd.read_csv(f"{R}/{s}_metrics_topup.csv") for s in ["jpl", "caltech"]])
base = base[base.month.isin(["2019-08", "2019-11"]) & ((base.policy.isin(["uncontrolled", "offline"])) | ((base.policy == "mpc_topup") & base.q.isna()))].copy()
base["archetype"] = "tou_ev4"
allt = pd.concat([base, t], ignore_index=True)
an = {"tou_ev4": "TOU-EV-4", "evening_tou_dc": "Evening TOU", "flat_dc": "Flat"}
rows = []
for site in ["jpl", "caltech"]:
    for arch in ["tou_ev4", "evening_tou_dc", "flat_dc"]:
        z = allt[(allt.site == site) & (allt.archetype == arch)].groupby("policy")[["bill", "energy_cost", "peak_all_kw", "served_ratio"]].mean()
        u, m = z.loc["uncontrolled"], z.loc["mpc_topup"]
        rows.append([SN[site], an[arch], f"{u.bill:,.0f}", f"{m.bill:,.0f}", f"{100*(1-m.bill/u.bill):.1f}", f"{100*(m.energy_cost/u.energy_cost-1):+.1f}",
                     f"{100*(1-m.peak_all_kw/u.peak_all_kw):.1f}", f"{100*m.served_ratio:.1f}"])
write("Table12_archetypes", "Tariff archetypes, all with the 15.51\\,USD/kW-month demand charge (means over August and November 2019).", "tab:arch", "llrrrrrr",
      ["Site", "Energy price", hdr("Bill, unc.", "(USD)"), hdr("Bill, top-", "up (USD)"), hdr("Saving", "(\\%)"), hdr("Energy", "cost (\\%)"), hdr("Peak", "red. (\\%)"), hdr("Served", "(\\%)")], rows,
      "Energy cost change is MPC + top-up relative to uncontrolled charging. The evening-peak and flat-price archetypes are stylised (Section 4.10).")

# ---- Figure 14: grid-usable flexibility
fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.4))
pv = pd.DataFrame(pvsum, columns=["site", "lab", "peak", "pvself"])
labs = ["Uncontrolled", "Top-up, PV-unaware", "PV-aware top-up", "PV-aware, q=0.7", "PV-aware opt."]
short = ["Uncontrolled", "Top-up", "PV-aware top-up", "PV-aware top-up, q=0.7", "PV-aware optimum"]
cols = ["#7f7f7f", "#9467bd", "#ff7f0e", "#d62728", "#2ca02c"]
w = 0.38
for k, site in enumerate(["jpl", "caltech"]):
    v = pv[pv.site == site].set_index("lab").loc[labs]
    axs[0].bar(np.arange(5) + (k - 0.5) * w, 100 * v.peak / v.peak.iloc[0], w, color=cols, alpha=1 if k == 0 else 0.55, edgecolor="k", lw=0.3)
    axs[1].bar(np.arange(5) + (k - 0.5) * w, v.pvself, w, color=cols, alpha=1 if k == 0 else 0.55, edgecolor="k", lw=0.3)
for a, tt in [(axs[0], "Import peak (% of uncontrolled)"), (axs[1], "PV self-sufficiency (%)")]:
    a.set_xticks(range(5)); a.set_xticklabels(short, fontsize=6, rotation=40, ha="right"); a.set_title(tt, fontsize=8)
axs[0].text(0.98, 0.95, "solid: JPL\nlight: Caltech", transform=axs[0].transAxes, ha="right", va="top", fontsize=6.5)
for site, mk in [("jpl", "o"), ("caltech", "s")]:
    for p, q, col, lab in [("llf", None, "#17becf", "LLF"), ("mpc_topup", None, "#9467bd", "MPC + top-up"), ("offline", None, "#2ca02c", "Hindsight opt.")]:
        y = x[(x.site == site) & (x.policy == p) & x.q.isna()].groupby("rating_frac").served_ratio.mean()
        axs[2].plot([35, 50], [100 * y.loc[0.35], 100 * y.loc[0.5]], "-" + mk, color=col, ms=4, label=f"{SN[site]}: {lab}")
axs[2].set_xlabel("Transformer rating (% of unc. peak)", fontsize=7); axs[2].set_title("Energy served within limit (%)", fontsize=8)
axs[2].set_xticks([35, 50]); axs[2].legend(fontsize=5.2, frameon=False, loc="lower right")
fig.tight_layout(); save(fig, "Fig14_grid_usable_flexibility")
print("ok")
