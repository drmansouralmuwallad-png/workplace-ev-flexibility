"""SYNTHETIC sessions for software testing ONLY. Never report outputs from this file."""
import json
import numpy as np
import pandas as pd

FMT = "%a, %d %b %Y %H:%M:%S GMT"


def make_raw(path, start="2019-01-01", days=75, per_weekday=25, seed=0, tz="America/Los_Angeles"):
    rng = np.random.default_rng(seed)
    rows, sid = [], 0
    for d in pd.date_range(start, periods=days, freq="D", tz=tz):
        n = per_weekday if d.weekday() < 5 else 2
        for _ in range(n):
            arr = d + pd.Timedelta(hours=float(np.clip(rng.normal(8.5, 1.2), 5, 14)))
            dur = float(np.clip(rng.normal(8.5, 1.5), 1, 13))
            dep = arr + pd.Timedelta(hours=dur)
            kwh = float(np.clip(rng.gamma(3, 3.5), 1, 6.6 * dur * 0.9))
            done = arr + pd.Timedelta(hours=kwh / 6.6 * 1.05)
            u = lambda t: t.tz_convert("UTC").strftime(FMT)
            doc = {"sessionID": f"S{sid}", "siteID": "0000", "stationID": f"ST{sid % 50}", "timezone": tz,
                   "connectionTime": u(arr), "doneChargingTime": u(done), "disconnectTime": u(dep), "kWhDelivered": kwh,
                   "userID": None, "userInputs": None}
            if rng.random() < 0.6:
                rdep = dep + pd.Timedelta(hours=float(rng.normal(0.5, 1.5)))
                rdep = max(rdep, arr + pd.Timedelta(minutes=30))
                doc["userInputs"] = [{"userID": f"U{sid % 80}", "requestedDeparture": u(rdep),
                                      "kWhRequested": float(kwh * rng.lognormal(0.2, 0.4)),
                                      "modifiedAt": u(arr + pd.Timedelta(minutes=1))}]
            rows.append(doc); sid += 1
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


TEST_TARIFF = {  # arbitrary TEST_ONLY numbers, not a real tariff
    "name": "TEST_ONLY", "schedule_id": "TEST_ONLY", "effective_date": "TEST_ONLY", "source_url": "TEST_ONLY",
    "frd_per_kw_month": 10.0, "trd_per_kw_month": 5.0,
    "rules": [
        {"name": "on", "months": list(range(1, 13)), "days": "weekday", "start_h": 16, "end_h": 21, "price_per_kwh": 0.30, "trd": True},
        {"name": "super_off", "months": list(range(1, 13)), "days": "all", "start_h": 8, "end_h": 16, "price_per_kwh": 0.08, "trd": False},
        {"name": "off", "months": list(range(1, 13)), "days": "all", "start_h": 0, "end_h": 24, "price_per_kwh": 0.15, "trd": False}]}
