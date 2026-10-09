"""Replay real sessions period by period under a charging policy.

Ground truth per session: arrival, disconnect time and delivered kWh (the energy
the vehicle will accept). A controller sees an EV only after plug-in, sees its
declaration (or the population prediction for unclaimed sessions), and learns
that an EV is full when its current drops to zero. Actual departure is revealed
only when the EV leaves.
"""
import time

import numpy as np
import pandas as pd

from .lp import solve_schedule


class MonthContext:
    def __init__(self, month, tz, dt_min, sessions, tariff, cap, cfg):
        start = pd.Timestamp(month + "-01", tz=tz)
        end = start + pd.offsets.MonthBegin(1)
        self.dt = dt_min / 60.0
        freq = f"{dt_min}min"
        s = sessions[(sessions["arrival"] >= start.tz_convert("UTC")) & (sessions["arrival"] < end.tz_convert("UTC"))].copy()
        # sessions are assigned to the month they arrive in; the horizon extends to the last departure
        horizon_end = max(end.tz_convert("UTC"), s["departure"].max() if len(s) else end.tz_convert("UTC"))
        self.index = pd.date_range(start.tz_convert("UTC"), horizon_end.ceil(freq) + pd.Timedelta(freq), freq=freq)
        self.T = len(self.index)
        self.price, self.trd = tariff.arrays(self.index, tz)
        self.lam_f, self.lam_t = tariff.frd, tariff.trd
        self.cap = cap
        self.M = cfg["model"]["shortfall_penalty_per_kwh"]
        self.eps = cfg["model"]["tiebreak_eps"]
        self.K = cfg["model"]["resolve_every_periods"]
        t0 = self.index[0]
        idx = lambda ts: np.floor((ts - t0).dt.total_seconds().to_numpy() / (dt_min * 60)).astype(int)
        s["arr"] = idx(s["arrival"])
        s["dep"] = np.maximum(idx(s["departure"]), s["arr"] + 1)
        s["dec_dep_i"] = np.maximum(idx(s["dec_departure"]), s["arr"] + 1)
        # effective max rate guarantees the observed energy is deliverable on the discrete grid
        s["rbar_eff"] = np.maximum(s["rbar_kw"], s["kwh"] / ((s["dep"] - s["arr"]) * self.dt))
        self.s = s.reset_index(drop=True)
        self.n = len(self.s)
        # numpy views for fast per-EV access inside the simulation loop
        self.a = {c: self.s[c].to_numpy() for c in ["arr", "dep", "dec_dep_i", "dec_kwh", "kwh", "rbar_eff", "claimed"]}
        # on-site PV (kW) behind the meter; zero unless a PV scenario sets it. pv_aware tells the MPC to use it.
        self.pv = np.zeros(self.T)
        self.pv_aware = False


# ----------------------------------------------------------------------------- policies
class Uncontrolled:
    name = "uncontrolled"
    enforce_cap = False

    def rates(self, t, active, st, ctx):
        return {i: ctx.a["rbar_eff"][i] for i in active}


class Greedy:
    """EDF or LLF on declared information, capacity-constrained."""
    enforce_cap = True

    def __init__(self, rule):
        self.name = rule

    def rates(self, t, active, st, ctx):
        s = ctx.s
        if self.name == "edf":
            key = lambda i: ctx.a["dec_dep_i"][i]
        else:
            key = lambda i: (ctx.a["dec_dep_i"][i] - t) * ctx.dt - max(ctx.a["dec_kwh"][i] - st["delivered"][i], 0) / ctx.a["rbar_eff"][i]
        out, room = {}, ctx.cap
        for i in sorted(active, key=key):
            r = min(ctx.a["rbar_eff"][i], room)
            out[i] = r
            room -= r
            if room <= 1e-9:
                break
        return out


class MPC:
    """Peak-state receding-horizon LP; nominal when shifts are (0, 1)."""
    enforce_cap = True

    def __init__(self, name, shifts, topup=False):
        self.name = name
        self.shifts = shifts
        self.topup = topup
        self.plan, self.plan_t0, self.last = {}, 0, -10**9
        self.solve_times = []

    def rates(self, t, active, st, ctx):
        if st["event"] or t - self.last >= ctx.K or not self.plan:
            self._solve(t, active, st, ctx)
        out = {i: (self.plan[i][t - self.plan_t0] if i in self.plan and t - self.plan_t0 < len(self.plan[i]) else 0.0)
               for i in active}
        if self.topup:
            # Peak-state top-up: capacity already billed this month (the realised month-to-date
            # peak) carries no further demand charge, so fill it with extra energy for vehicles
            # that are still accepting current, earliest declared deadline first.
            ceil = st["peak_all"]
            if ctx.trd[t] and ctx.lam_t > 0:
                ceil = min(ceil, st["peak_on"])
            ceil = min(ceil + ctx.pv[t], ctx.cap)   # import stays at or below the billed peak
            room = ceil - sum(out.values())
            for i in sorted(active, key=lambda i: ctx.a["dec_dep_i"][i]):
                if room <= 1e-9:
                    break
                extra = min(ctx.a["rbar_eff"][i] - out[i], room)
                if extra > 0:
                    out[i] += extra
                    room -= extra
        return out

    def _solve(self, t, active, st, ctx):
        s, evs, ids = ctx.s, [], []
        for i in active:
            sh_h, mult = self.shifts["claimed" if ctx.a["claimed"][i] else "unclaimed"]
            dec = int(ctx.a["dec_dep_i"][i])
            d_t = dec + int(round(sh_h / ctx.dt))
            if t >= dec:                      # overstaying its declaration: deliver any remaining target now
                we, te = t + 1, t + 1
            else:
                we = max(dec, d_t, t + 1)
                te = min(max(d_t, t + 1), we)
            tgt = max(ctx.a["dec_kwh"][i] * mult - st["delivered"][i], 0.0)
            evs.append({"ws": t, "we": min(we, ctx.T), "te": min(te, ctx.T), "target": tgt, "rbar": ctx.a["rbar_eff"][i]})
            ids.append(i)
        T = max([e["we"] for e in evs], default=t + 1)
        tic = time.perf_counter()
        sol, status, _ = solve_schedule(evs, t, T, ctx.price, ctx.trd, ctx.dt, ctx.cap, ctx.lam_f, ctx.lam_t,
                                        st["peak_all"], st["peak_on"], ctx.M, ctx.eps,
                                        pv=ctx.pv if ctx.pv_aware else None)
        self.solve_times.append(time.perf_counter() - tic)
        if status != 0:
            raise RuntimeError(f"MPC LP failed at t={t} (status {status})")
        self.plan = {ids[k]: v for k, v in sol.items()}
        self.plan_t0, self.last = t, t


class Offline:
    """Hindsight optimum: true arrival, departure and energy of every session in the month."""
    name = "offline"
    enforce_cap = True

    def prepare(self, ctx):
        s = ctx.s
        evs = [{"ws": int(a), "we": int(d), "te": int(d), "target": float(e), "rbar": float(r)}
               for a, d, e, r in zip(s["arr"], s["dep"], s["kwh"], s["rbar_eff"])]
        tic = time.perf_counter()
        sol, status, obj = solve_schedule(evs, 0, ctx.T, ctx.price, ctx.trd, ctx.dt, ctx.cap, ctx.lam_f, ctx.lam_t,
                                          0.0, 0.0, ctx.M, ctx.eps, pv=ctx.pv if ctx.pv_aware else None)
        self.solve_times = [time.perf_counter() - tic]
        if status != 0:
            raise RuntimeError(f"Offline LP failed (status {status})")
        self.sched = sol

    def rates(self, t, active, st, ctx):
        return {i: self.sched[i][t] for i in active}


# ----------------------------------------------------------------------------- simulator
def simulate(policy, ctx):
    s = ctx.s
    if isinstance(policy, Offline):
        policy.prepare(ctx)
    delivered = np.zeros(ctx.n)
    full = np.zeros(ctx.n, dtype=bool)
    load = np.zeros(ctx.T)
    st = {"delivered": delivered, "peak_all": 0.0, "peak_on": 0.0, "event": True}
    arr_at = {}
    for i, a in enumerate(s["arr"]):
        arr_at.setdefault(int(a), []).append(i)
    dep_at = {}
    for i, d in enumerate(s["dep"]):
        dep_at.setdefault(int(d), []).append(i)
    present = set()
    cap_violations = 0
    for t in range(ctx.T):
        ev_flag = False
        for i in dep_at.get(t, []):
            present.discard(i); ev_flag = True
        for i in arr_at.get(t, []):
            present.add(i); ev_flag = True
        active = [i for i in present if not full[i]]
        st["event"] = st["event"] or ev_flag
        req = policy.rates(t, active, st, ctx) if active else {}
        st["event"] = False
        tot = 0.0
        applied = {}
        for i, r in req.items():
            rem = ctx.a["kwh"][i] - delivered[i]
            a = max(0.0, min(r, ctx.a["rbar_eff"][i], rem / ctx.dt))
            applied[i] = a
            tot += a
        if tot > ctx.cap + 1e-6:
            if policy.enforce_cap:
                f = ctx.cap / tot
                applied = {i: a * f for i, a in applied.items()}
                tot = ctx.cap
            else:
                cap_violations += 1
        for i, a in applied.items():
            delivered[i] += a * ctx.dt
            if ctx.a["kwh"][i] - delivered[i] <= 1e-6:
                full[i] = True
                st["event"] = True
        load[t] = tot
        imp = max(tot - ctx.pv[t], 0.0)          # metered grid import
        st["peak_all"] = max(st["peak_all"], imp)
        if ctx.trd[t]:
            st["peak_on"] = max(st["peak_on"], imp)
    return {"load": load, "delivered": delivered, "cap_violations": cap_violations,
            "solve_times": getattr(policy, "solve_times", [])}
