"""Build every manuscript figure (SVG + PDF) and table (LaTeX + SVG + PDF) from results/."""
import json, os, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42})
R, F, T = "results", "paper/figures", "paper/tables"
TZ = "America/Los_Angeles"
SITES = ["jpl", "caltech"]
SN = {"jpl": "JPL", "caltech": "Caltech"}
PN = {"uncontrolled": "Uncontrolled", "edf": "EDF", "llf": "LLF", "mpc_nominal": "Nominal MPC",
      "mpc_robust": "Robust MPC", "mpc_topup": "MPC + top-up", "offline": "Hindsight optimum"}
COL = {"uncontrolled": "#7f7f7f", "edf": "#bcbd22", "llf": "#17becf", "mpc_nominal": "#1f77b4",
       "mpc_robust": "#d62728", "mpc_topup": "#9467bd", "offline": "#2ca02c"}
cfg = json.load(open("config.json"))


def save(fig, name):
    for ext in ("svg", "pdf"):
        fig.savefig(os.path.join(F, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def label(r):
    q = r["q"]
    if r["policy"] in ("mpc_robust", "mpc_topup") and pd.notna(q):
        return PN[r["policy"]] + f" (q={q:.1f})"
    return PN[r["policy"]]


# ---------------------------------------------------------------- Fig 1: framework
def fig_framework():
    fig, ax = plt.subplots(figsize=(9.0, 3.4)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 40)
    boxes = [(1, 22, 17, 14, "ACN-Data sessions\nJPL & Caltech, 2019\n(arrival, departure,\nkWh, declarations)"),
             (22, 22, 17, 14, "Rolling calibration\n60 days before month\nerrors $\\Delta d$, $\\rho$\nquantiles at level $q$"),
             (43, 22, 17, 14, "Robust targets\n$\\tilde d=\\hat d+Q_{1-q}(\\Delta d)$\n$\\tilde e=\\hat e\\,Q_q(\\rho)$"),
             (64, 22, 17, 14, "Peak-state MPC (LP)\nenergy cost + demand\ncharge on increment\nabove month peak"),
             (43, 2, 38, 14, "Event-driven replay, 5-min steps\ntrue departure and energy revealed online\noptional top-up below month peak"),
             (85, 12, 14, 14, "Bill, peak,\nenergy served,\nunmet kWh")]
    for x, y, w, h, t in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4", fc="#f2f2f2", ec="0.3", lw=0.8))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=6.8)
    arr = dict(arrowstyle="->", color="0.25", lw=0.9)
    for (x0, y0), (x1, y1) in [((18.5, 29), (21.5, 29)), ((39.5, 29), (42.5, 29)), ((60.5, 29), (63.5, 29)),
                               ((72.5, 21.5), (70, 16.5)), ((62, 16.5), (51.5, 21.5)), ((81.5, 9), (84.5, 16))]:
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=arr)
    ax.text(58, 19, "rates", fontsize=6.5, color="0.3"); ax.text(52, 19.2, "state", fontsize=6.5, color="0.3")
    save(fig, "Fig1_framework")


# ---------------------------------------------------------------- data figures
def load_sessions(site):
    df = pd.read_pickle(f"data/processed/{site}_sessions.pkl")
    df["loc"] = df["arrival"].dt.tz_convert(TZ)
    return df


def fig_data():
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.6))
    for k, site in enumerate(SITES):
        df = load_sessions(site)
        d = df.groupby(df["loc"].dt.date).agg(n=("kwh", "size"), kwh=("kwh", "sum"))
        ax = axs[0, k]; ax.plot(pd.to_datetime(d.index), d["kwh"], lw=0.7, color="0.2")
        ax.set_title(f"{SN[site]}: daily energy delivered", fontsize=9); ax.set_ylabel("kWh/day")
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        ax = axs[1, k]
        wd = df[df["weekday"]]
        ax.hist(wd["arrival_local_hour"], bins=np.arange(0, 24.5, 0.5), color="#1f77b4", alpha=0.7, label="arrivals")
        dep_h = [t.hour + t.minute / 60 for t in wd["departure"].dt.tz_convert(TZ)]
        ax.hist(dep_h, bins=np.arange(0, 24.5, 0.5), color="#d62728", alpha=0.5, label="departures")
        ax.set_xlabel("Local hour (weekdays)"); ax.set_ylabel("Sessions"); ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); save(fig, "Fig2_data_overview")


def fig_errors():
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.4))
    for k, site in enumerate(SITES):
        df = load_sessions(site); cl = df[df["claimed"]]
        dd = (cl["departure"] - cl["req_departure"]).dt.total_seconds() / 3600
        rho = cl["kwh"] / cl["req_kwh"]
        ax = axs[0, k]; ax.hist(dd.clip(-10, 10), bins=80, color="0.35"); ax.axvline(0, color="k", lw=0.8)
        ax.axvline(dd.median(), color="#d62728", lw=1, ls="--")
        ax.set_title(f"{SN[site]} (claimed n={len(cl):,})", fontsize=9)
        ax.set_xlabel("Actual $-$ declared departure (h), clipped at $\\pm$10"); ax.set_ylabel("Sessions")
        ax = axs[1, k]; ax.hist(rho.clip(0, 2.5), bins=75, color="0.35"); ax.axvline(1, color="k", lw=0.8)
        ax.axvline(rho.median(), color="#d62728", lw=1, ls="--")
        ax.set_xlabel("Delivered / declared energy (-), clipped at 2.5"); ax.set_ylabel("Sessions")
    fig.tight_layout(); save(fig, "Fig3_declaration_errors")


# ---------------------------------------------------------------- results figures
def metrics():
    parts = [pd.read_csv(f"{R}/{s}_metrics.csv") for s in SITES]
    parts += [pd.read_csv(f"{R}/{s}_metrics_topup.csv") for s in SITES if os.path.exists(f"{R}/{s}_metrics_topup.csv")]
    m = pd.concat(parts, ignore_index=True)
    order = {"uncontrolled": 0, "edf": 1, "llf": 2, "mpc_nominal": 3, "mpc_robust": 4, "mpc_topup": 5, "offline": 6}
    m["_o"] = m.policy.map(order); m = m.sort_values(["site", "_o", "q", "month"], na_position="first").drop(columns="_o")
    m["label"] = m.apply(label, axis=1)
    off = m[m.policy == "offline"].set_index(["site", "month"])["bill"]
    m["gap_to_offline_pct"] = [100 * (b / off.get((s, mo), np.nan) - 1) for s, mo, b in zip(m.site, m.month, m.bill)]
    return m


def fig_frontier(m):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for k, site in enumerate(SITES):
        ax = axs[k]; g = m[m.site == site].groupby(["policy", "q"], dropna=False)[["bill", "served_ratio"]].mean().reset_index()
        rob = g[g.policy == "mpc_robust"].sort_values("q")
        ax.plot(100 * rob.served_ratio, rob.bill, "-o", color=COL["mpc_robust"], ms=4, label="Robust MPC")
        for _, r in rob.iterrows():
            ax.annotate(f"q={r.q:.1f}", (100 * r.served_ratio, r.bill), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
        top = g[g.policy == "mpc_topup"]
        if len(top):
            ax.plot(100 * top.served_ratio, top.bill, "P", color=COL["mpc_topup"], ms=6, ls="none", label="MPC + top-up")
            for _, r in top.iterrows():
                ax.annotate("nom." if pd.isna(r.q) else f"q={r.q:.1f}", (100 * r.served_ratio, r.bill), fontsize=6.5, xytext=(3, -8), textcoords="offset points", color=COL["mpc_topup"])
        for p, mk in [("uncontrolled", "s"), ("edf", "^"), ("llf", "v"), ("mpc_nominal", "D"), ("offline", "*")]:
            r = g[g.policy == p]
            ax.plot(100 * r.served_ratio, r.bill, mk, color=COL[p], ms=7 if p == "offline" else 5, label=PN[p], ls="none")
        ax.set_title(SN[site], fontsize=9); ax.set_xlabel("Energy served (% of demanded)"); ax.set_ylabel("Mean monthly bill (USD)")
    axs[1].legend(frameon=False, fontsize=6.5, loc="best"); fig.tight_layout(); save(fig, "Fig5_price_of_robustness")


def fig_bill(m):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=False)
    for k, site in enumerate(SITES):
        ax = axs[k]; g = m[m.site == site].groupby("label", sort=False)[["energy_cost", "frd_cost"]].mean()
        order = [l for l in m[m.site == site]["label"].unique()]
        g = g.loc[order]
        ax.bar(range(len(g)), g.energy_cost, color="#9ecae1", label="TOU energy")
        ax.bar(range(len(g)), g.frd_cost, bottom=g.energy_cost, color="#08519c", label="Demand charge")
        ax.set_xticks(range(len(g))); ax.set_xticklabels(g.index, rotation=40, ha="right", fontsize=7)
        ax.set_title(SN[site], fontsize=9); ax.set_ylabel("Mean monthly cost (USD)")
    axs[0].legend(frameon=False, fontsize=7); fig.tight_layout(); save(fig, "Fig6_bill_decomposition")


def fig_peaks(m):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for k, site in enumerate(SITES):
        ax = axs[k]; d = m[m.site == site]
        for p in ["uncontrolled", "llf", "mpc_nominal", "mpc_robust", "mpc_topup", "offline"]:
            if p == "mpc_robust":
                dd = d[(d.policy == p) & (d.q == 0.9)]; lab = PN[p] + " (q=0.9)"
            elif p == "mpc_topup":
                dd = d[(d.policy == p) & d.q.isna()]; lab = PN[p] + " (nominal)"
            else:
                dd = d[d.policy == p]; lab = PN[p]
            if len(dd):
                ax.plot(dd.month, dd.peak_all_kw, "-o", ms=3, color=COL[p], label=lab)
        ax.axhline(cfg["physics"]["site_capacity_kw"][site], color="k", ls=":", lw=0.8)
        ax.set_title(SN[site], fontsize=9); ax.set_ylabel("Monthly peak (kW)"); ax.tick_params(axis="x", labelrotation=30, labelsize=7)
    axs[1].legend(frameon=False, fontsize=6.5); fig.tight_layout(); save(fig, "Fig7_monthly_peaks")


def fig_profiles():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=False)
    for k, site in enumerate(SITES):
        ax = axs[k]; month = "2019-08"
        for p, tag in [("uncontrolled", "uncontrolled"), ("mpc_nominal", "mpc_nominal"), ("mpc_topup", "mpc_topup"), ("offline", "offline")]:
            fn = f"{R}/loads/{site}_{month}_{tag}.npz"
            if not os.path.exists(fn):
                continue
            z = np.load(fn); load = z["load"]; t0 = pd.Timestamp(str(z["t0"]))
            idx = pd.date_range(t0, periods=len(load), freq="5min").tz_convert(TZ)
            s = pd.Series(load, index=idx); s = s[(s.index.month == 8) & (s.index.weekday < 5)]
            prof = s.groupby(s.index.hour + s.index.minute / 60).mean()
            ax.plot(prof.index, prof.values, color=COL[p], lw=1.2, label=PN[p] + (" (nominal)" if p == "mpc_topup" else ""))
        ax.axvspan(12, 18, color="#fdd0a2", alpha=0.4, lw=0, label="Summer on-peak (12-18 h)")
        ax.set_title(f"{SN[site]}, weekdays, August 2019", fontsize=9); ax.set_xlabel("Local hour"); ax.set_ylabel("Mean load (kW)")
    axs[1].legend(frameon=False, fontsize=6.5); fig.tight_layout(); save(fig, "Fig4_load_profiles")


def fig_calibration():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for site, ls in [("jpl", "-"), ("caltech", "--")]:
        fn = f"{R}/{site}_calibration.csv"
        if not os.path.exists(fn):
            continue
        c = pd.read_csv(fn)
        for q, col in zip(sorted(c.q.unique()), ["#fdae6b", "#e6550d", "#a63603"]):
            d = c[c.q == q]
            axs[0].plot(d.month, d.claimed_shift_h, ls, color=col, marker="o", ms=3, label=f"{SN[site]} q={q}")
            axs[1].plot(d.month, d.claimed_energy_mult, ls, color=col, marker="o", ms=3, label=f"{SN[site]} q={q}")
    axs[0].set_ylabel("Deadline shift $Q_{1-q}(\\Delta d)$ (h)"); axs[1].set_ylabel("Energy multiplier $Q_q(\\rho)$ (-)")
    for a in axs:
        a.tick_params(axis="x", labelrotation=30, labelsize=7); a.axhline(0 if a is axs[0] else 1, color="k", lw=0.6)
    axs[1].legend(frameon=False, fontsize=6, ncol=2); fig.tight_layout(); save(fig, "Fig8_calibrated_sets")


def fig_sensitivity():
    fn = f"{R}/sensitivity_metrics.csv"
    if not os.path.exists(fn):
        return None
    s = pd.read_csv(fn)
    s["label"] = s.apply(label, axis=1)
    g = s.groupby(["scenario", "site", "label"])[["bill", "served_ratio"]].mean().reset_index()
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for k, site in enumerate(SITES):
        d = g[g.site == site]; scen = list(dict.fromkeys(d.scenario)); labs = list(dict.fromkeys(d.label))
        w = 0.8 / len(labs)
        for j, l in enumerate(labs):
            v = [d[(d.scenario == sc) & (d.label == l)]["bill"].mean() for sc in scen]
            axs[k].bar(np.arange(len(scen)) + j * w, v, w, label=l)
        axs[k].set_xticks(np.arange(len(scen)) + 0.4 - w / 2); axs[k].set_xticklabels(scen, rotation=35, ha="right", fontsize=7)
        axs[k].set_title(SN[site] + " (Aug & Nov 2019)", fontsize=9); axs[k].set_ylabel("Mean monthly bill (USD)")
    axs[1].legend(frameon=False, fontsize=6.5); fig.tight_layout(); save(fig, "Fig9_sensitivity")
    return s


def fig_unmet():
    fn = f"{R}/unmet_decomposition.csv"
    if not os.path.exists(fn):
        return
    u = pd.read_csv(fn)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for k, site in enumerate(SITES):
        d = u[u.site == site]; x = np.arange(len(d))
        axs[k].bar(x, d.kwh_decl_too_low, color="#fdae6b", label="Declared energy below actual (target met)")
        axs[k].bar(x, d.kwh_left_early, bottom=d.kwh_decl_too_low, color="#e6550d", label="Departed before target delivered")
        axs[k].set_xticks(x); axs[k].set_xticklabels(d.month, rotation=30, fontsize=7)
        axs[k].set_title(SN[site] + ": nominal MPC", fontsize=9); axs[k].set_ylabel("Undelivered energy (kWh/month)")
    axs[0].legend(frameon=False, fontsize=6.5); fig.tight_layout(); save(fig, "Fig10_unmet_decomposition")


# ---------------------------------------------------------------- tables
def table_render(df, name, title):
    """Render a table as SVG and PDF (the LaTeX version is written separately)."""
    fig, ax = plt.subplots(figsize=(min(7.2, 1.1 + 1.05 * len(df.columns)), 0.45 + 0.24 * (len(df) + 1)))
    ax.axis("off")
    tb = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(7); tb.scale(1, 1.25)
    for (r, c), cell in tb.get_celld().items():
        cell.set_linewidth(0.3)
        if r == 0:
            cell.set_facecolor("#e6e6e6"); cell.set_text_props(weight="bold")
    ax.set_title(title, fontsize=8)
    save(fig, name)


def _esc(x):
    x = str(x)
    return x if "\\%" in x else x.replace("%", "\\%")


def tex_table(df, name, caption, label_, colfmt=None, notes=None):
    df = df.copy(); df.columns = [_esc(c) for c in df.columns]; df = df.apply(lambda col: col.map(_esc))
    colfmt = colfmt or ("l" + "r" * (len(df.columns) - 1))
    lines = [r"\begin{table}[!htbp]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label_}}}",
             r"\small", (r"\resizebox{\textwidth}{!}{%" if len(df.columns) >= 5 else ""), rf"\begin{{tabular}}{{{colfmt}}}", r"\hline",
             " & ".join(df.columns) + r" \\", r"\hline"]
    for _, r in df.iterrows():
        lines.append(" & ".join(str(v) for v in r.values) + r" \\")
    lines += [r"\hline", r"\end{tabular}" + ("}" if len(df.columns) >= 5 else "")]
    if notes:
        lines.append(rf"\par\smallskip\footnotesize\raggedright {notes}")
    lines.append(r"\end{table}")
    open(os.path.join(T, f"{name}.tex"), "w").write("\n".join(lines) + "\n")


def tables(m, sens):
    # T2 dataset summary
    rows = []
    for site in SITES:
        df = load_sessions(site)
        fl = pd.read_csv(f"{R}/{site}_filter_flow.csv")
        rows.append([SN[site], f"{int(fl.n_remaining.iloc[0]):,}", f"{len(df):,}", f"{df.kwh.sum()/1000:,.1f}",
                     f"{100*df.claimed.mean():.1f}", f"{df.duration_h.median():.2f}", f"{df.kwh.median():.2f}",
                     f"{100*df.weekday.mean():.1f}"])
    t2 = pd.DataFrame(rows, columns=["Site", "Raw sessions", "Retained", "Energy (MWh)", "Claimed (%)",
                                     "Median stay (h)", "Median kWh", "Weekday (%)"])
    tex_table(t2, "Table2_dataset", "ACN-Data sessions used in this study (1 May--31 December 2019).", "tab:data")
    table_render(t2, "Table2_dataset", "Table 2. Dataset summary")
    # T3 tariff
    t3 = pd.DataFrame([["Summer (Jun--Sep) weekday", "0.05623", "0.09250", "0.26668", "0.05623"],
                       ["Winter (Oct--May) weekday", "0.06087", "0.07492", "0.08690", "0.06087"]],
                      columns=["Season / day", "00--08 h", "08--12, 18--23 h", "12--18 h", "23--24 h; weekends"])
    tex_table(t3, "Table3_tariff", r"Energy prices (USD/kWh) of SCE TOU-EV-4 effective 1 March 2019 \cite{SCETOUEV4}; demand charge 15.51 USD/kW-month on the monthly maximum demand.", "tab:tariff", "lrrrr")
    table_render(t3, "Table3_tariff", "Table 3. SCE TOU-EV-4 energy prices (USD/kWh); demand charge 15.51 USD/kW-month")
    # T4 main results
    g = m.groupby(["site", "label"], sort=False).agg(bill=("bill", "mean"), energy=("energy_cost", "mean"),
         dc=("frd_cost", "mean"), peak=("peak_all_kw", "mean"), served=("served_ratio", "mean"),
         full=("share_fully_served", "mean"), gap=("gap_to_offline_pct", "mean"), solve=("solve_s_mean", "mean"), pen=("penalised_cost", "mean")).reset_index()
    g = g.sort_values("site", key=lambda x: x.map({"jpl": 0, "caltech": 1}), kind="stable")
    t4 = pd.DataFrame({"Site": g.site.map(SN), "Policy": g.label, "Bill (USD)": g.bill.map("{:,.0f}".format),
                       "Energy (USD)": g.energy.map("{:,.0f}".format), "Demand (USD)": g.dc.map("{:,.0f}".format),
                       "Peak (kW)": g.peak.map("{:.1f}".format), "Served (%)": (100 * g.served).map("{:.2f}".format),
                       "EVs full (%)": (100 * g.full).map("{:.1f}".format), "Gap to opt. (%)": g.gap.map("{:+.1f}".format),
                       "Penalised (USD)": g.pen.map("{:,.0f}".format)})
    tex_table(t4, "Table4_main_results", "Mean monthly outcomes over the six test months (July--December 2019). Gap is the bill relative to the hindsight optimum; Served is delivered over demanded energy; Penalised adds 5 USD per undelivered kWh.", "tab:main", "llrrrrrrrr")
    table_render(t4, "Table4_main_results", "Table 4. Mean monthly outcomes, July-December 2019")
    # T5 sensitivity
    if sens is not None:
        g = sens.groupby(["scenario", "site", "label"]).agg(bill=("bill", "mean"), served=("served_ratio", "mean"),
                                                           peak=("peak_all_kw", "mean")).reset_index()
        t5 = pd.DataFrame({"Scenario": g.scenario.str.replace("_", r"\_"), "Site": g.site.map(SN), "Policy": g.label,
                           "Bill (USD)": g.bill.map("{:,.0f}".format), "Peak (kW)": g.peak.map("{:.1f}".format),
                           "Served (%)": (100 * g.served).map("{:.2f}".format)})
        tex_table(t5, "Table5_sensitivity", "Sensitivity analysis (means over August and November 2019).", "tab:sens", "lllrrr")
        t5r = t5.copy(); t5r["Scenario"] = t5r["Scenario"].str.replace(r"\_", "_", regex=False)
        table_render(t5r, "Table5_sensitivity", "Table 5. Sensitivity analysis (August and November 2019)")
    # T6 filter flow
    rows = []
    for site in SITES:
        fl = pd.read_csv(f"{R}/{site}_filter_flow.csv")
        for _, r in fl.iterrows():
            rows.append([SN[site], r.step.replace(">=", r"$\geq$").replace("<=", r"$\leq$"), int(r.n_remaining)])
    t6 = pd.DataFrame(rows, columns=["Site", "Step", "Sessions"])
    tex_table(t6, "Table6_filter_flow", "Session filtering flow.", "tab:flow", "lp{10cm}r")
    t6r = t6.copy(); t6r["Step"] = t6r["Step"].str.replace(r"$\geq$", ">=", regex=False).str.replace(r"$\leq$", "<=", regex=False)
    table_render(t6r, "Table6_filter_flow", "Table 6. Session filtering flow")
    return t4


if __name__ == "__main__":
    os.makedirs(F, exist_ok=True); os.makedirs(T, exist_ok=True)
    fig_framework(); fig_data(); fig_errors()
    m = metrics(); m.to_csv(f"{R}/all_metrics_labelled.csv", index=False)
    fig_frontier(m); fig_bill(m); fig_peaks(m); fig_profiles(); fig_calibration()
    sens = fig_sensitivity(); fig_unmet()
    tables(m, sens)
    print(sorted(os.listdir(F))); print(sorted(os.listdir(T)))
