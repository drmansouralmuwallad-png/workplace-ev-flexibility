"""Time-of-use energy prices plus two demand charges:
FRD (facilities-related, $/kW on the monthly maximum over all hours) and
TRD (time-related, $/kW on the monthly maximum within TRD-flagged periods).
Rules are matched in order; the first match wins. Holidays are not modelled.
"""
import numpy as np
import pandas as pd


class Tariff:
    def __init__(self, tcfg, allow_null=False):
        self.cfg = tcfg
        if not allow_null:
            missing = [k for k in ("name", "effective_date", "source_url", "frd_per_kw_month", "trd_per_kw_month")
                       if tcfg.get(k) is None]
            missing += [r["name"] for r in tcfg["rules"] if r.get("price_per_kwh") is None]
            if missing:
                raise ValueError("Tariff incomplete; fill from the live tariff sheet: " + ", ".join(missing))
        self.frd = float(tcfg["frd_per_kw_month"] or 0.0)
        self.trd = float(tcfg["trd_per_kw_month"] or 0.0)

    def arrays(self, index_utc: pd.DatetimeIndex, tz: str):
        loc = index_utc.tz_convert(tz)
        hour = loc.hour + loc.minute / 60
        price = np.full(len(loc), np.nan)
        trd = np.zeros(len(loc), dtype=bool)
        for r in self.cfg["rules"]:
            m = np.isin(loc.month, r["months"]) & (hour >= r["start_h"]) & (hour < r["end_h"])
            if r["days"] == "weekday":
                m &= loc.weekday < 5
            elif r["days"] == "weekend":
                m &= loc.weekday >= 5
            m &= np.isnan(price)
            price[m] = r["price_per_kwh"]
            trd[m] = bool(r.get("trd", False))
        if np.isnan(price).any():
            raise ValueError("Tariff rules do not cover every period; add a catch-all rule.")
        return price, trd
