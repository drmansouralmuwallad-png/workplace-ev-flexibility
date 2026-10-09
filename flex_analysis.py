"""Grid-flexibility and solar-coincidence metrics computed from the replayed load profiles.

Solar: clear-sky global horizontal irradiance (pvlib Ineichen model with bundled
Linke turbidity climatology) at each site, used as a normalised horizontal PV
shape and scaled so that monthly PV energy equals the site's monthly EV energy
demand ("energy-neutral" array; 50% sizing as sensitivity). The controllers are
not PV-aware; the metric measures how charging schedules align with solar output.
"""
import glob, json, os, re
import numpy as np, pandas as pd, pvlib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from acndc.tariff import Tariff

TZ = "America/Los_Angeles"; R = "results"; F = "paper/figures"
SITE_LL = {"jpl": (34.2013, -118.1714), "caltech": (34.1377, -118.1253)}
cfg = json.load(open("config.json")); tariff = Tariff(cfg["tariff"])
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42})


def pv_shape(index_utc, site):
    lat, lon = SITE_LL[site]
    loc = pvlib.location.Location(lat, lon, tz="UTC", altitude=250)
    cs = loc.get_clearsky(index_utc, model="ineichen")
    return cs["ghi"].clip(lower=0).to_numpy()


rows, profiles = [], {}
for fn in sorted(glob.glob(f"{R}/loads/*.npz")):
    tag = os.path.basename(fn)[:-4]
    site, month, pol = re.match(r"(jpl|caltech)_(\d{4}-\d{2})_(.+)", tag).groups()
    z = np.load(fn); load = z["load"]; dt = float(z["dt"]); t0 = pd.Timestamp(str(z["t0"]))
    idx = pd.date_range(t0, periods=len(load), freq="5min")
    loc = idx.tz_convert(TZ); inm = (loc.strftime("%Y-%m") == month)
    L = load[inm]; li = idx[inm]
    price, onpk = tariff.arrays(li, TZ)
    pv = pv_shape(li, site); E = float(z["kwh"].sum())
    rows.append({"site": site, "month": month, "tag": tag, "L": L, "pv": pv, "E": E, "onpk": onpk, "dt": dt, "idx": li})

by = {(r["site"], r["month"]): {} for r in rows}
for r in rows:
    by[(r["site"], r["month"])][r["tag"].split(r["month"] + "_")[1]] = r

out = []
for (site, month), d in by.items():
    U = d["uncontrolled"]["L"]; dt = d["uncontrolled"]["dt"]
    for pol, r in d.items():
        L = r["L"]; E_del = L.sum() * dt
        L15 = L[: len(L) // 3 * 3].reshape(-1, 3).mean(1)
        pv = r["pv"]; scale = r["E"] / max(pv.sum() * dt, 1e-9)
        res = {"site": site, "month": month, "policy_tag": pol, "peak_kw": L.max(),
               "peak_red_kw": U.max() - L.max(), "peak_red_pct": 100 * (1 - L.max() / U.max()),
               "load_factor": L.mean() / L.max(), "par": L.max() / L.mean(),
               "max_ramp_kw_per_15min": np.abs(np.diff(L15)).max(),
               "shifted_kwh": np.maximum(U - L, 0).sum() * dt, "shifted_pct": 100 * np.maximum(U - L, 0).sum() * dt / (U.sum() * dt),
               "onpeak_energy_pct": 100 * L[r["onpk"]].sum() / L.sum()}
        for f, lab in [(1.0, "pv100"), (0.5, "pv50")]:
            P = f * scale * pv; cov = np.minimum(L, P).sum()
            res[f"{lab}_self_sufficiency_pct"] = 100 * cov / L.sum()
            res[f"{lab}_utilisation_pct"] = 100 * cov / P.sum()
        out.append(res)
fm = pd.DataFrame(out); fm.to_csv(f"{R}/flexibility_metrics.csv", index=False)

keep = ["uncontrolled", "llf", "mpc_nominal", "mpc_robust_q0.9", "mpc_topup", "mpc_topup_q0.7", "offline"]
names = {"uncontrolled": "Uncontrolled", "llf": "LLF", "mpc_nominal": "Nominal MPC", "mpc_robust_q0.9": "Robust MPC (q=0.9)",
         "mpc_topup": "MPC + top-up", "mpc_topup_q0.7": "MPC + top-up (q=0.7)", "offline": "Hindsight optimum"}
g = fm[fm.policy_tag.isin(keep)].groupby(["site", "policy_tag"]).mean(numeric_only=True).reset_index()
g["o"] = g.policy_tag.map({k: i for i, k in enumerate(keep)}); g["s"] = g.site.map({"jpl": 0, "caltech": 1})
g = g.sort_values(["s", "o"]); print(g.drop(columns=["o", "s"]).round(2).to_string())
g.drop(columns=["o", "s"]).to_csv(f"{R}/flexibility_summary.csv", index=False)

# LaTeX + rendered table
t = pd.DataFrame({"Site": g.site.map({"jpl": "JPL", "caltech": "Caltech"}), "Policy": g.policy_tag.map(names),
                  "Peak (kW)": g.peak_kw.map("{:.1f}".format), "Peak red. (\\%)": g.peak_red_pct.map("{:.1f}".format),
                  "Load factor": g.load_factor.map("{:.3f}".format), "PAR": g.par.map("{:.1f}".format),
                  "Max ramp (kW/15 min)": g.max_ramp_kw_per_15min.map("{:.1f}".format),
                  "Shifted (\\%)": g.shifted_pct.map("{:.1f}".format), "On-peak energy (\\%)": g.onpeak_energy_pct.map("{:.1f}".format),
                  "PV self-suff. (\\%)": g.pv100_self_sufficiency_pct.map("{:.1f}".format)})
L_ = [r"\begin{table}[!htbp]", r"\centering",
      r"\caption{Grid-flexibility and solar-coincidence metrics (means over July--December 2019). Peak reduction and shifted energy are relative to uncontrolled charging; PAR is the peak-to-average ratio of the monthly load; on-peak energy is the share delivered in the 12--18\,h weekday block; PV self-sufficiency is the share of EV energy coincident with an energy-neutral clear-sky PV array.}",
      r"\label{tab:flex}", r"\small", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llrrrrrrrr}", r"\hline",
      " & ".join(t.columns) + r" \\", r"\hline"] + [" & ".join(map(str, v)) + r" \\" for v in t.values] + [r"\hline", r"\end{tabular}}", r"\end{table}"]
open("paper/tables/Table7_flexibility.tex", "w").write("\n".join(L_) + "\n")
tr = t.copy(); tr.columns = [c.replace("\\%", "%") for c in tr.columns]
fig, ax = plt.subplots(figsize=(7.2, 0.45 + 0.24 * (len(tr) + 1))); ax.axis("off")
tb = ax.table(cellText=tr.values, colLabels=tr.columns, loc="center", cellLoc="center"); tb.auto_set_font_size(False); tb.set_fontsize(6); tb.scale(1, 1.25)
for (i, j), c in tb.get_celld().items():
    c.set_linewidth(0.3)
    if i == 0: c.set_facecolor("#e6e6e6"); c.set_text_props(weight="bold")
ax.set_title("Table 7. Grid-flexibility and solar-coincidence metrics", fontsize=8)
for e in ("svg", "pdf"): fig.savefig(f"{F}/Table7_flexibility.{e}", bbox_inches="tight")
plt.close(fig)

# Fig 11 normalised load-duration curves (all six months pooled)
col = {"uncontrolled": "#7f7f7f", "mpc_nominal": "#1f77b4", "mpc_topup": "#9467bd", "offline": "#2ca02c"}
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
for k, site in enumerate(["jpl", "caltech"]):
    Upk = max(by[(site, m)]["uncontrolled"]["L"].max() for (s, m) in by if s == site)
    for pol in col:
        Ls = np.concatenate([by[(site, m)][pol]["L"] for (s, m) in by if s == site])
        Ls = np.sort(Ls)[::-1]; x = 100 * np.arange(len(Ls)) / len(Ls)
        axs[k].plot(x, Ls / Upk, color=col[pol], lw=1.2, label=names[pol])
    axs[k].set_xlim(0, 45); axs[k].set_xlabel("Share of time (%)"); axs[k].set_ylabel("Load / uncontrolled peak (-)")
    axs[k].set_title({"jpl": "JPL", "caltech": "Caltech"}[site] + ", Jul-Dec 2019", fontsize=9)
axs[1].legend(frameon=False, fontsize=7); fig.tight_layout()
for e in ("svg", "pdf"): fig.savefig(f"{F}/Fig11_load_duration.{e}", bbox_inches="tight")
plt.close(fig)

# Fig 12 weekday profiles with PV shape, August
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
for k, site in enumerate(["jpl", "caltech"]):
    d = by[(site, "2019-08")]; li = d["uncontrolled"]["idx"].tz_convert(TZ); wd = li.weekday < 5
    hr = (li.hour + li.minute / 60)[wd]
    pvs = d["uncontrolled"]["pv"] * d["uncontrolled"]["E"] / (d["uncontrolled"]["pv"].sum() * d["uncontrolled"]["dt"])
    axs[k].fill_between(*zip(*sorted(pd.Series(pvs[wd]).groupby(np.asarray(hr)).mean().items())), color="#ffd92f", alpha=0.5, label="Energy-neutral clear-sky PV")
    for pol in col:
        s = pd.Series(d[pol]["L"][wd]).groupby(np.asarray(hr)).mean()
        axs[k].plot(s.index, s.values, color=col[pol], lw=1.2, label=names[pol])
    axs[k].set_xlabel("Local hour (weekdays, Aug 2019)"); axs[k].set_ylabel("kW"); axs[k].set_title({"jpl": "JPL", "caltech": "Caltech"}[site], fontsize=9)
axs[1].legend(frameon=False, fontsize=6.5); fig.tight_layout()
for e in ("svg", "pdf"): fig.savefig(f"{F}/Fig12_pv_coincidence.{e}", bbox_inches="tight")
