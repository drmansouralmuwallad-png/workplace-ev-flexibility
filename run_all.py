"""Main experiments + sensitivity analysis on the real ACN-Data extracts (2019)."""
import copy, json, os, sys, time
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from acndc.run import run_month
from acndc.tariff import Tariff

R = "results"
cfg = json.load(open("config.json"))
log = open(os.path.join(R, "run_log.txt"), "a")
def say(*a):
    print(*a, flush=True); log.write(" ".join(map(str, a)) + "\n"); log.flush()

mode = sys.argv[1]
if mode == "main":
    for site in ["jpl", "caltech"]:
        df = pd.read_pickle(f"data/processed/{site}_sessions.pkl")
        M, C, S = [], [], []
        for m in cfg["experiments"]["test_months"][site]:
            t = time.time()
            mt, cal, summ = run_month(df, site, m, cfg, Tariff(cfg["tariff"]), verbose=False, load_dir=f"{R}/loads")
            M.append(mt); C.append(cal); S.append(summ)
            pd.concat(M).to_csv(f"{R}/{site}_metrics.csv", index=False)
            pd.concat(C).to_csv(f"{R}/{site}_calibration.csv", index=False)
            json.dump(S, open(f"{R}/{site}_error_summary.json", "w"), indent=1)
            say("main", site, m, f"{time.time()-t:.0f}s")
elif mode == "topup":
    c = json.loads(json.dumps(cfg)); c["model"]["policies"] = ["mpc_topup"]; c["model"]["topup_quantiles"] = [None, 0.7, 0.9]
    for site in ["jpl", "caltech"]:
        df = pd.read_pickle(f"data/processed/{site}_sessions.pkl"); M = []
        for m in cfg["experiments"]["test_months"][site]:
            t = time.time()
            mt, _, _ = run_month(df, site, m, c, Tariff(c["tariff"]), verbose=False, load_dir=f"{R}/loads")
            M.append(mt); pd.concat(M).to_csv(f"{R}/{site}_metrics_topup.csv", index=False)
            say("topup", site, m, f"{time.time()-t:.0f}s")
elif mode == "tariff":
    out = []
    for lam in [0.0, 5.0, 10.0, 20.0, 31.02]:
        c = json.loads(json.dumps(cfg)); c["tariff"]["frd_per_kw_month"] = lam
        c["model"]["policies"] = ["uncontrolled", "mpc_topup", "offline"]; c["model"]["topup_quantiles"] = [None]
        for site in ["jpl", "caltech"]:
            df = pd.read_pickle(f"data/processed/{site}_sessions.pkl")
            for m in ["2019-08", "2019-11"]:
                t = time.time()
                mt, _, _ = run_month(df, site, m, c, Tariff(c["tariff"]), verbose=False)
                mt.insert(0, "demand_charge", lam); out.append(mt)
                pd.concat(out).to_csv(f"{R}/tariff_sweep.csv", index=False)
                say("tariff", lam, site, m, f"{time.time()-t:.0f}s")
elif mode == "sens":
    scen = {
        "demand_x0":  lambda c: c["tariff"].__setitem__("frd_per_kw_month", 0.0),
        "demand_x2":  lambda c: c["tariff"].__setitem__("frd_per_kw_month", 2 * 15.51),
        "train_30d":  lambda c: c["model"].__setitem__("train_days", 30),
        "train_90d":  lambda c: c["model"].__setitem__("train_days", 90),
        "penalty_20": lambda c: c["model"].__setitem__("shortfall_penalty_per_kwh", 20.0),
    }
    out = []
    for name, f in scen.items():
        c = copy.deepcopy(cfg); f(c)
        c["model"]["policies"] = ["mpc_nominal", "mpc_robust"] + (["offline"] if name.startswith(("demand", "penalty")) else [])
        c["model"]["robust_quantiles"] = [0.9]
        for site in ["jpl", "caltech"]:
            df = pd.read_pickle(f"data/processed/{site}_sessions.pkl")
            for m in ["2019-08", "2019-11"]:
                t = time.time()
                mt, _, _ = run_month(df, site, m, c, Tariff(c["tariff"]), verbose=False)
                mt.insert(0, "scenario", name); out.append(mt)
                pd.concat(out).to_csv(f"{R}/sensitivity_metrics.csv", index=False)
                say("sens", name, site, m, f"{time.time()-t:.0f}s")
say("DONE", mode)

if mode == "heldout":
    # Out-of-sample check (2021 extracts, never used for design): calibrate on the 60 days before
    # each test month (May-June 2021) and test July and August 2021 with every policy.
    c = json.loads(json.dumps(cfg))
    c["model"]["policies"] = ["uncontrolled", "edf", "llf", "mpc_nominal", "mpc_robust", "mpc_topup", "offline"]
    c["model"]["robust_quantiles"] = [0.7, 0.9]; c["model"]["topup_quantiles"] = [None, 0.7, 0.9]
    for site in ["jpl", "caltech"]:
        df = pd.read_pickle(f"data/processed/{site}_sessions_2021.pkl"); M = []
        for m in ["2021-07", "2021-08"]:
            t = time.time()
            mt, cal, summ = run_month(df, site, m, c, Tariff(c["tariff"]), verbose=False, load_dir=f"{R}/loads_2021")
            M.append(mt); pd.concat(M).to_csv(f"{R}/{site}_metrics_heldout2021.csv", index=False)
            json.dump(summ, open(f"{R}/{site}_{m}_error_summary_heldout2021.json", "w"), indent=1)
            say("heldout", site, m, f"{time.time()-t:.0f}s")
    say("DONE heldout")
