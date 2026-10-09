"""Decompose undelivered energy of the nominal MPC into its two causes."""
import json, numpy as np, pandas as pd
from acndc.calibrate import Calibration
cfg = json.load(open("config.json")); rows = []
for site in ["jpl", "caltech"]:
    df = pd.read_pickle(f"data/processed/{site}_sessions.pkl")
    for m in cfg["experiments"]["test_months"][site]:
        m0 = pd.Timestamp(m + "-01", tz="America/Los_Angeles").tz_convert("UTC")
        m1 = (pd.Timestamp(m + "-01", tz="America/Los_Angeles") + pd.offsets.MonthBegin(1)).tz_convert("UTC")
        tr = df[(df.arrival < m0) & (df.arrival >= m0 - pd.Timedelta(days=60))]
        cal = Calibration(tr, 3); t = df[(df.arrival >= m0) & (df.arrival < m1)].copy()
        t["dec_dep"], t["dec_kwh"] = cal.declarations(t)
        z = np.load(f"results/loads/{site}_{m}_mpc_nominal.npz"); d = z["delivered"]; e = z["kwh"]
        assert len(d) == len(t)
        unmet = np.maximum(e - d, 0); bad = unmet > 0.05
        cap_lim = bad & (d >= np.minimum(t["dec_kwh"].to_numpy(), e) - 0.05)
        early = bad & ~cap_lim
        left_before_decl = (t["departure"] < t["dec_dep"]).to_numpy()
        rows.append({"site": site, "month": m, "n": len(t), "n_unmet": int(bad.sum()),
                     "unmet_kwh": unmet.sum(), "kwh_decl_too_low": unmet[cap_lim].sum(), "kwh_left_early": unmet[early].sum(),
                     "n_decl_too_low": int(cap_lim.sum()), "n_left_early": int(early.sum()),
                     "share_claimed_decl_below_actual": float(((t.dec_kwh < t.kwh) & t.claimed).sum() / max(t.claimed.sum(), 1)),
                     "share_left_before_declared": float(left_before_decl.mean())})
r = pd.DataFrame(rows); r.to_csv("results/unmet_decomposition.csv", index=False)
print(r.round(3).to_string())
print(r.groupby("site")[["unmet_kwh", "kwh_decl_too_low", "kwh_left_early", "n_unmet", "n_decl_too_low", "n_left_early"]].sum())
