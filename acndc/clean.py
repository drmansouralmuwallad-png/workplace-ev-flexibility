"""Parse raw ACN-Data sessions into an analysis table with a logged filter flow.

Declarations (claimed sessions): the FIRST user input (earliest modifiedAt) is used,
because only information available at plug-in time may enter an online controller.
Later edits are ignored and counted in the log.
"""
import json

import numpy as np
import pandas as pd

HTTP_FMT = "%a, %d %b %Y %H:%M:%S GMT"


def _ts(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return pd.NaT
    try:
        return pd.to_datetime(x, format=HTTP_FMT, utc=True)
    except (ValueError, TypeError):
        return pd.to_datetime(x, utc=True, errors="coerce")


def _first_user_input(doc):
    """Return (userID, requestedDeparture, kWhRequested, n_inputs) at plug-in."""
    ui = doc.get("userInputs")
    if isinstance(ui, list) and ui:
        ui_sorted = sorted(ui, key=lambda u: _ts(u.get("modifiedAt")) if u.get("modifiedAt") else pd.Timestamp.max.tz_localize("UTC"))
        u = ui_sorted[0]
        return (u.get("userID", doc.get("userID")), _ts(u.get("requestedDeparture")),
                u.get("kWhRequested"), len(ui))
    # flat layout (as listed in Table 1 of Lee, Li & Low 2019)
    if doc.get("requestedDeparture") is not None or doc.get("kWhRequested") is not None:
        return doc.get("userID"), _ts(doc.get("requestedDeparture")), doc.get("kWhRequested"), 1
    return doc.get("userID"), pd.NaT, None, 0


def load_raw(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                d = json.loads(line)
                uid, rdep, rkwh, n_in = _first_user_input(d)
                rows.append({
                    "sessionID": d.get("sessionID"), "siteID": d.get("siteID"),
                    "stationID": d.get("stationID"), "tz": d.get("timezone", "America/Los_Angeles"),
                    "arrival": _ts(d.get("connectionTime")), "done": _ts(d.get("doneChargingTime")),
                    "departure": _ts(d.get("disconnectTime")), "kwh": d.get("kWhDelivered"),
                    "userID": uid, "req_departure": rdep,
                    "req_kwh": float(rkwh) if rkwh is not None else np.nan, "n_user_inputs": n_in,
                })
    return pd.DataFrame(rows)


def clean(df, cfg):
    c = cfg["cleaning"]
    log = [("raw", len(df))]

    def step(name, mask):
        nonlocal df
        df = df[mask].copy()
        log.append((name, len(df)))

    step("drop duplicate sessionID", ~df.duplicated("sessionID"))
    step("has arrival, departure, kWh", df["arrival"].notna() & df["departure"].notna() & df["kwh"].notna())
    df["kwh"] = df["kwh"].astype(float)
    df["done"] = pd.to_datetime(df["done"], utc=True).fillna(df["departure"])
    df["done"] = df["done"].where(df["done"] <= df["departure"], df["departure"])
    df["duration_h"] = (df["departure"] - df["arrival"]).dt.total_seconds() / 3600
    df["charging_h"] = (df["done"] - df["arrival"]).dt.total_seconds() / 3600
    step(f"kWh >= {c['min_kwh']}", df["kwh"] >= c["min_kwh"])
    step(f"{c['min_duration_h']} h <= duration <= {c['max_duration_h']} h",
         (df["duration_h"] >= c["min_duration_h"]) & (df["duration_h"] <= c["max_duration_h"]))
    step("done within [arrival, departure]", df["charging_h"] > 0)
    implied = df["kwh"] / df["charging_h"]
    step(f"implied avg power <= {c['max_implied_avg_kw']} kW (drops DC fast charging)", implied <= c["max_implied_avg_kw"])
    if c["exclude_station_ids"]:
        step("excluded stationIDs", ~df["stationID"].isin(c["exclude_station_ids"]))

    rbar0 = cfg["physics"]["default_rbar_kw"]
    df["rbar_kw"] = np.maximum(rbar0, df["kwh"] / df["charging_h"])
    # a declaration is usable only if it is complete and consistent at plug-in
    df["claimed"] = (df["req_departure"].notna() & df["req_kwh"].notna()
                     & (df["req_kwh"] > 0) & (df["req_departure"] > df["arrival"]))
    df["invalid_declaration"] = df["n_user_inputs"].gt(0) & ~df["claimed"]
    df["edited_after_plugin"] = df["n_user_inputs"] > 1
    local = df.apply(lambda r: r["arrival"].tz_convert(r["tz"]), axis=1)
    df["arrival_local_hour"] = [t.hour + t.minute / 60 for t in local]
    df["weekday"] = [t.weekday() < 5 for t in local]
    flow = pd.DataFrame(log, columns=["step", "n_remaining"])
    flow["n_removed"] = -flow["n_remaining"].diff().fillna(0).astype(int)
    extra = pd.DataFrame([
        ("claimed (usable declaration)", int(df["claimed"].sum()), None),
        ("declaration present but invalid -> treated as unclaimed", int(df["invalid_declaration"].sum()), None),
        ("declaration edited after plug-in (first input used)", int(df["edited_after_plugin"].sum()), None),
    ], columns=["step", "n_remaining", "n_removed"])
    return df.sort_values("arrival").reset_index(drop=True), pd.concat([flow, extra], ignore_index=True)
