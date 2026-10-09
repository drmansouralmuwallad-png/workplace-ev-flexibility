"""Revision extensions: (1) PV-aware control, (2) transformer hosting limits, (3) tariff archetypes."""
import copy, json, sys, time
import numpy as np, pandas as pd, pvlib
from acndc.calibrate import Calibration
from acndc.simulate import MonthContext, MPC, Greedy, Offline, Uncontrolled, simulate
from acndc.run import metrics
from acndc.tariff import Tariff

TZ = "America/Los_Angeles"; R = "results"
SITE_LL = {"jpl": (34.2013, -118.1714), "caltech": (34.1377, -118.1253)}
cfg = json.load(open("config.json"))
log = open(f"{R}/run_log.txt", "a")
def say(*a):
    print(*a, flush=True); log.write(" ".join(map(str, a)) + "\n"); log.flush()


def make_ctx(df, site, month, c, tariff, cap=None):
    m0 = pd.Timestamp(month + "-01", tz=TZ).tz_convert("UTC")
    tr = df[(df.arrival < m0) & (df.arrival >= m0 - pd.Timedelta(days=c["model"]["train_days"]))]
    cal = Calibration(tr, c["model"]["arrival_hour_bucket"])
    t = df.copy(); t["dec_departure"], t["dec_kwh"] = cal.declarations(t)
    ctx = MonthContext(month, TZ, c["physics"]["period_minutes"], t, tariff, cap or c["physics"]["site_capacity_kw"][site], c)
    return ctx, cal


def pv_profile(ctx, site, scale=1.0):
    lat, lon = SITE_LL[site]
    ghi = pvlib.location.Location(lat, lon, tz="UTC", altitude=250).get_clearsky(ctx.index, model="ineichen")["ghi"].clip(lower=0).to_numpy()
    loc = ctx.index.tz_convert(TZ); inm = loc.strftime("%Y-%m") == ctx.index.tz_convert(TZ)[0].strftime("%Y-%m")
    E = ctx.s.kwh.sum()
    return scale * ghi * E / (ghi[inm].sum() * ctx.dt)   # energy-neutral over the calendar month


def pol(name, cal, q=None):
    return {"uncontrolled": lambda: Uncontrolled(), "llf": lambda: Greedy("llf"), "offline": lambda: Offline(),
            "mpc_nominal": lambda: MPC("mpc_nominal", cal.shifts(None)),
            "mpc_topup": lambda: MPC("mpc_topup", cal.shifts(q), topup=True)}[name]()


def overload(load, rating, dt):
    return {"rating_kw": rating, "max_loading_pu": load.max() / rating, "hours_over": float((load > rating + 1e-6).sum() * dt),
            "energy_over_kwh": float(np.maximum(load - rating, 0).sum() * dt)}


mode = sys.argv[1]
sites = ["jpl", "caltech"]
out = []
if mode == "pv":
    tariff = Tariff(cfg["tariff"])
    for site in sites:
        df = pd.read_pickle(f"data/processed/{site}_sessions.pkl")
        for month in cfg["experiments"]["test_months"][site]:
            t0 = time.time()
            for name, q, aware in [("uncontrolled", None, False), ("mpc_topup", None, False), ("mpc_topup", None, True),
                                   ("mpc_topup", 0.7, True), ("offline", None, True)]:
                ctx, cal = make_ctx(df, site, month, cfg, tariff)
                ctx.pv = pv_profile(ctx, site); ctx.pv_aware = aware
                res = simulate(pol(name, cal, q), ctx)
                row = metrics(res, ctx, name, q); row.update({"site": site, "month": month, "pv_aware": aware})
                out.append(row)
            pd.DataFrame(out).to_csv(f"{R}/pv_aware_metrics.csv", index=False)
            say("pv", site, month, f"{time.time()-t0:.0f}s")
elif mode == "xfmr":
    tariff = Tariff(cfg["tariff"])
    base_peak = {"jpl": 295.9, "caltech": 104.9}   # six-month mean uncontrolled peaks (Table 4b)
    for site in sites:
        df = pd.read_pickle(f"data/processed/{site}_sessions.pkl")
        for frac in [0.5, 0.35]:
            rating = round(frac * base_peak[site], 1)
            for month in ["2019-08", "2019-11"]:
                t0 = time.time()
                for name, q in [("uncontrolled", None), ("llf", None), ("mpc_topup", None), ("mpc_topup", 0.7), ("offline", None)]:
                    cap = None if name == "uncontrolled" else rating
                    ctx, cal = make_ctx(df, site, month, cfg, tariff, cap=cap)
                    res = simulate(pol(name, cal, q), ctx)
                    row = metrics(res, ctx, name, q); row.update({"site": site, "month": month, "rating_frac": frac})
                    row.update(overload(res["load"], rating, ctx.dt)); out.append(row)
                pd.DataFrame(out).to_csv(f"{R}/transformer_metrics.csv", index=False)
                say("xfmr", site, frac, month, f"{time.time()-t0:.0f}s")
elif mode == "tariff2":
    base = cfg["tariff"]
    def evening(b):
        t = copy.deepcopy(b); rules = []
        for season, months, off, mid, on in [("summer", [6, 7, 8, 9], 0.05623, 0.0925, 0.26668), ("winter", [1, 2, 3, 4, 5, 10, 11, 12], 0.06087, 0.07492, 0.0869)]:
            for a, z, p, trd in [(0, 8, off, False), (8, 16, mid, False), (16, 21, on, True), (21, 23, mid, False), (23, 24, off, False)]:
                rules.append({"name": f"{season}_wd_{a}", "months": months, "days": "weekday", "start_h": a, "end_h": z, "price_per_kwh": p, "trd": trd})
            rules.append({"name": f"{season}_we", "months": months, "days": "weekend", "start_h": 0, "end_h": 24, "price_per_kwh": off, "trd": False})
        t["rules"] = rules; t["name"] = "Stylised evening-peak TOU (TOU-EV-4 price levels, on-peak 16-21 h) + demand charge"
        return t
    for site in sites:
        df = pd.read_pickle(f"data/processed/{site}_sessions.pkl")
        for month in ["2019-08", "2019-11"]:
            # flat price = time-weighted mean of TOU-EV-4 energy prices over the month
            ctx0, _ = make_ctx(df, site, month, cfg, Tariff(base)); flat = float(ctx0.price.mean())
            fl = copy.deepcopy(base); fl["rules"] = [{"name": "flat", "months": list(range(1, 13)), "days": "all", "start_h": 0, "end_h": 24, "price_per_kwh": flat, "trd": False}]
            fl["name"] = f"Flat energy price {flat:.4f} USD/kWh + demand charge"
            for arch, tcfg in [("evening_tou_dc", evening(base)), ("flat_dc", fl)]:
                t0 = time.time()
                for name in ["uncontrolled", "mpc_topup", "offline"]:
                    ctx, cal = make_ctx(df, site, month, cfg, Tariff(tcfg))
                    res = simulate(pol(name, cal), ctx)
                    lt = ctx.index.tz_convert(TZ); hr = lt.hour
                    L = res["load"]; mid = (hr >= 8) & (hr < 16); eve = (hr >= 16) & (hr < 21)
                    row = metrics(res, ctx, name); row.update({"site": site, "month": month, "archetype": arch, "flat_price": flat,
                                                               "midday_share": float(L[mid].sum() / L.sum()), "evening_share": float(L[eve].sum() / L.sum())})
                    out.append(row)
                pd.DataFrame(out).to_csv(f"{R}/tariff_archetypes.csv", index=False)
                say("tariff2", arch, site, month, f"{time.time()-t0:.0f}s")
say("DONE", mode)
