"""Experiment runner.

  python -m acndc.run pull      --config config.json
  python -m acndc.run clean     --config config.json
  python -m acndc.run run       --config config.json [--site jpl] [--months 2019-01,2019-02]
  python -m acndc.run figures   --config config.json
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from .calibrate import Calibration
from .clean import clean, load_raw
from .simulate import MPC, Greedy, MonthContext, Offline, Uncontrolled, simulate
from .tariff import Tariff

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def metrics(res, ctx, policy_name, q=None):
    s, load, dt = ctx.s, res["load"], ctx.dt
    imp = np.maximum(load - ctx.pv, 0.0)       # billing is on metered grid import (equals load without PV)
    energy_cost = float(np.sum(ctx.price * imp * dt))
    p_all = float(imp.max()) if len(imp) else 0.0
    p_on = float(imp[ctx.trd].max()) if ctx.trd.any() else 0.0
    unmet = np.maximum(s["kwh"].to_numpy() - res["delivered"], 0.0)
    st = res["solve_times"]
    return {
        "policy": policy_name, "q": q, "n_sessions": ctx.n,
        "energy_kwh_true": float(s["kwh"].sum()), "energy_kwh_delivered": float(res["delivered"].sum()),
        "served_ratio": float(res["delivered"].sum() / max(s["kwh"].sum(), 1e-9)),
        "share_fully_served": float(np.mean(unmet <= 0.05)) if ctx.n else np.nan,
        "unmet_kwh": float(unmet.sum()),
        "energy_cost": energy_cost, "peak_all_kw": p_all, "peak_on_kw": p_on,
        "frd_cost": ctx.lam_f * p_all, "trd_cost": ctx.lam_t * p_on,
        "bill": energy_cost + ctx.lam_f * p_all + ctx.lam_t * p_on,
        "penalised_cost": energy_cost + ctx.lam_f * p_all + ctx.lam_t * p_on + ctx.M * float(unmet.sum()),
        "cap_violation_periods": res["cap_violations"],
        "pv_kwh": float(ctx.pv.sum() * dt), "pv_used_kwh": float(np.minimum(load, ctx.pv).sum() * dt),
        "import_kwh": float(imp.sum() * dt),
        "n_solves": len(st), "solve_s_mean": float(np.mean(st)) if st else np.nan,
        "solve_s_p95": float(np.quantile(st, .95)) if st else np.nan,
    }


def _save_load(load_dir, site, month, name, q, res, ctx):
    if load_dir:
        os.makedirs(load_dir, exist_ok=True)
        tag = f"{site}_{month}_{name}" + (f"_q{q}" if q is not None else "")
        np.savez_compressed(os.path.join(load_dir, tag + ".npz"), load=res["load"],
                            t0=str(ctx.index[0]), dt=ctx.dt, delivered=res["delivered"], kwh=ctx.s["kwh"].to_numpy())


def run_month(sessions, site, month, cfg, tariff, verbose=True, load_dir=None):
    tz = sessions["tz"].iloc[0]
    m0 = pd.Timestamp(month + "-01", tz=tz).tz_convert("UTC")
    train = sessions[(sessions["arrival"] < m0) &
                     (sessions["arrival"] >= m0 - pd.Timedelta(days=cfg["model"]["train_days"]))]
    if len(train) == 0:
        raise ValueError(f"No training sessions before {month}")
    cal = Calibration(train, cfg["model"]["arrival_hour_bucket"])
    test = sessions.copy()
    test["dec_departure"], test["dec_kwh"] = cal.declarations(test)
    ctx = MonthContext(month, tz, cfg["physics"]["period_minutes"], test, tariff,
                       cfg["physics"]["site_capacity_kw"][site], cfg)
    rows, calrows = [], []
    for pol in cfg["model"]["policies"]:
        if pol == "mpc_robust":
            for q in cfg["model"]["robust_quantiles"]:
                sh = cal.shifts(q)
                calrows.append({"site": site, "month": month, "q": q,
                                "claimed_shift_h": sh["claimed"][0], "claimed_energy_mult": sh["claimed"][1],
                                "unclaimed_shift_h": sh["unclaimed"][0], "unclaimed_energy_mult": sh["unclaimed"][1]})
                res = simulate(MPC("mpc_robust", sh), ctx)
                rows.append(metrics(res, ctx, "mpc_robust", q))
                _save_load(load_dir, site, month, "mpc_robust", q, res, ctx)
                if verbose:
                    print(f"  {site} {month} mpc_robust q={q}: done")
            continue
        if pol == "mpc_topup":
            for q in cfg["model"].get("topup_quantiles", [None]):
                res = simulate(MPC("mpc_topup", cal.shifts(q), topup=True), ctx)
                rows.append(metrics(res, ctx, "mpc_topup", q))
                _save_load(load_dir, site, month, "mpc_topup", q, res, ctx)
            continue
        p = {"uncontrolled": Uncontrolled(), "edf": Greedy("edf"), "llf": Greedy("llf"),
             "mpc_nominal": MPC("mpc_nominal", cal.shifts(None)), "offline": Offline()}[pol]
        res = simulate(p, ctx)
        rows.append(metrics(res, ctx, pol))
        _save_load(load_dir, site, month, pol, None, res, ctx)
        if verbose:
            print(f"  {site} {month} {pol}: done")
    out = pd.DataFrame(rows)
    out.insert(0, "month", month)
    out.insert(0, "site", site)
    summ = cal.summary()
    summ.update({"site": site, "month": month})
    return out, pd.DataFrame(calrows), summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pull", "clean", "run", "figures"])
    ap.add_argument("--config", default=os.path.join(ROOT, "config.json"))
    ap.add_argument("--site")
    ap.add_argument("--months")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    raw_dir, proc_dir, res_dir = (os.path.join(ROOT, "data", "raw"), os.path.join(ROOT, "data", "processed"),
                                  os.path.join(ROOT, "results"))
    sites = [a.site] if a.site else cfg["pull"]["sites"]

    if a.cmd == "pull":
        from .pull import pull_site
        for site in sites:
            _, meta = pull_site(site, cfg["pull"]["start_local"], cfg["pull"]["end_local"], cfg, raw_dir)
            print(meta)
    elif a.cmd == "clean":
        os.makedirs(proc_dir, exist_ok=True)
        for site in sites:
            paths = sorted(glob.glob(os.path.join(raw_dir, f"{site}_*.jsonl")))
            df, flow = clean(load_raw(paths), cfg)
            df.to_pickle(os.path.join(proc_dir, f"{site}_sessions.pkl"))
            flow.to_csv(os.path.join(res_dir, f"{site}_filter_flow.csv"), index=False)
            monthly = df.groupby(df["arrival"].dt.tz_convert(df["tz"].iloc[0]).dt.strftime("%Y-%m")).agg(
                sessions=("sessionID", "size"), kwh=("kwh", "sum"), claimed_share=("claimed", "mean"))
            monthly.to_csv(os.path.join(res_dir, f"{site}_monthly_counts.csv"))
            print(flow.to_string(index=False))
    elif a.cmd == "run":
        tariff = Tariff(cfg["tariff"])
        os.makedirs(res_dir, exist_ok=True)
        for site in sites:
            df = pd.read_pickle(os.path.join(proc_dir, f"{site}_sessions.pkl"))
            months = a.months.split(",") if a.months else cfg["experiments"]["test_months"][site]
            allm, allc, alls = [], [], []
            for m in months:
                mt, cal, summ = run_month(df, site, m, cfg, tariff)
                allm.append(mt); allc.append(cal); alls.append(summ)
                pd.concat(allm).to_csv(os.path.join(res_dir, f"{site}_metrics.csv"), index=False)
                pd.concat(allc).to_csv(os.path.join(res_dir, f"{site}_calibration.csv"), index=False)
                json.dump(alls, open(os.path.join(res_dir, f"{site}_error_summary.json"), "w"), indent=2)
    elif a.cmd == "figures":
        from .figures import make_all
        make_all(ROOT, sites)


if __name__ == "__main__":
    main()
