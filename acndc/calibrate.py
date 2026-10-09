"""Data-driven uncertainty sets for declared departure time and energy.

Errors are measured on a training window strictly before the test month:
  dd  = actual departure - declared departure  (hours; negative = left early)
  rho = actual kWh / declared kWh
Claimed sessions use the driver's declaration; unclaimed sessions use a
population predictor (median duration and kWh by arrival-hour bucket and
weekday/weekend) fitted on the same training window.

For a protection level q, the controller plans with
  deadline  d~ = d_declared + Q_{1-q}(dd)     (earlier as q grows)
  energy    e~ = e_declared * Q_q(rho)
Because each EV's delivery constraint is monotone in (d, e), this is the exact
robust counterpart of the per-EV constraint under the box set
[Q_{1-q}(dd), Q_q(dd)] x [Q_{1-q}(rho), Q_q(rho)].
"""
import numpy as np
import pandas as pd


class Calibration:
    def __init__(self, train: pd.DataFrame, bucket_h: int = 3):
        self.bucket_h = bucket_h
        tr = train.copy()
        tr["bucket"] = (tr["arrival_local_hour"] // bucket_h).astype(int)
        # the predictor stands in for a declaration only for unclaimed sessions, so it is
        # fitted on unclaimed training sessions (all sessions if fewer than 30 exist)
        base = tr[~tr["claimed"]] if (~tr["claimed"]).sum() >= 30 else tr
        self.predictor_fitted_on = "unclaimed" if base is not tr else "all"
        g = base.groupby(["weekday", "bucket"])
        self.med_dur = g["duration_h"].median().to_dict()
        self.med_kwh = g["kwh"].median().to_dict()
        self.glob_dur = float(base["duration_h"].median())
        self.glob_kwh = float(base["kwh"].median())
        self.n_train = len(tr)

        cl = tr[tr["claimed"]]
        self.dd_claimed = ((cl["departure"] - cl["req_departure"]).dt.total_seconds() / 3600).to_numpy()
        self.rho_claimed = (cl["kwh"] / cl["req_kwh"]).to_numpy()
        un = tr[~tr["claimed"]]
        pd_dur, pd_kwh = self._predict(un)
        self.dd_unclaimed = (un["duration_h"].to_numpy() - pd_dur)
        self.rho_unclaimed = un["kwh"].to_numpy() / pd_kwh

    def _predict(self, df):
        b = (df["arrival_local_hour"] // self.bucket_h).astype(int)
        dur = np.array([self.med_dur.get((w, k), self.glob_dur) for w, k in zip(df["weekday"], b)])
        kwh = np.array([self.med_kwh.get((w, k), self.glob_kwh) for w, k in zip(df["weekday"], b)])
        return dur, kwh

    def declarations(self, test: pd.DataFrame):
        """Declared departure (UTC) and energy available to the controller at plug-in."""
        dur, kwh = self._predict(test)
        pred_dep = test["arrival"] + pd.to_timedelta(dur, unit="h")
        dec_dep = np.where(test["claimed"], test["req_departure"], pred_dep)
        dec_kwh = np.where(test["claimed"], test["req_kwh"], kwh)
        return pd.to_datetime(dec_dep, utc=True), dec_kwh.astype(float)

    @staticmethod
    def _q(x, q):
        x = x[np.isfinite(x)]
        return float(np.quantile(x, q)) if len(x) else 0.0

    def shifts(self, q):
        """Return {'claimed': (shift_h, mult), 'unclaimed': (shift_h, mult)} for level q.
        q=None gives the nominal controller (declarations taken at face value)."""
        if q is None:
            return {"claimed": (0.0, 1.0), "unclaimed": (0.0, 1.0)}
        return {
            "claimed": (self._q(self.dd_claimed, 1 - q), self._q(self.rho_claimed, q)),
            "unclaimed": (self._q(self.dd_unclaimed, 1 - q), self._q(self.rho_unclaimed, q)),
        }

    def summary(self):
        def s(x):
            x = x[np.isfinite(x)]
            if not len(x):
                return {}
            return {"n": len(x), "mean": float(x.mean()), "p10": float(np.quantile(x, .1)),
                    "p50": float(np.median(x)), "p90": float(np.quantile(x, .9))}
        return {"dd_claimed_h": s(self.dd_claimed), "rho_claimed": s(self.rho_claimed),
                "dd_unclaimed_h": s(self.dd_unclaimed), "rho_unclaimed": s(self.rho_unclaimed),
                "n_train": self.n_train, "predictor_fitted_on": self.predictor_fitted_on}
