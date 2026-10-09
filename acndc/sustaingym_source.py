"""Load the ACN-Data session extracts redistributed with SustainGym v0.1.7 (PyPI).

SustainGym's `fetch_real_events` pulled sessions from the official ACN-Data API and
kept the FIRST user input of claimed sessions (kWhRequested, requestedDeparture).
For unclaimed sessions it fills requested energy and estimated departure with the
*realised* values, which would leak the future into an online controller; we
therefore discard those two fields for unclaimed sessions and let the
calibrated population predictor supply their declarations instead.
"""
import json
import pandas as pd

FMT = "%a, %d %b %Y %H:%M:%S GMT"


def to_raw_jsonl(csv_paths, site, out_path, tz="America/Los_Angeles"):
    n = 0
    with open(out_path, "w") as f:
        for p in csv_paths:
            df = pd.read_csv(p, compression="gzip")
            for c in ["arrival", "departure", "estimated_departure"]:
                df[c] = pd.to_datetime(df[c], utc=True)
            for r in df.itertuples(index=False):
                u = lambda t: t.strftime(FMT)
                doc = {"sessionID": r.session_id, "siteID": site, "stationID": r.station_id, "timezone": tz,
                       "connectionTime": u(r.arrival), "disconnectTime": u(r.departure),
                       "doneChargingTime": None, "kWhDelivered": float(getattr(r, "_3")),
                       "userInputs": None}
                if bool(r.claimed):
                    doc["userInputs"] = [{"requestedDeparture": u(r.estimated_departure),
                                          "kWhRequested": float(getattr(r, "_2"))}]
                f.write(json.dumps(doc) + "\n"); n += 1
    return n
