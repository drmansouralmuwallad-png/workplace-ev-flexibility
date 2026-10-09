"""Demand-charge-aware charging LP, solved with HiGHS through scipy.

min  sum_t sum_i (c_t*dt + eps*(t-t0)) r_it + M sum_i s_i + lam_F P_all + lam_T P_on
s.t. dt * sum_{t < te_i} r_it + s_i >= target_i          (delivery by planning deadline)
     dt * sum_{t in window_i} r_it <= target_i            (do not plan beyond the target)
     sum_i r_it <= C                                      (site capacity)
     sum_i r_it <= P_all          for all t
     sum_i r_it <= P_on           for TRD periods
     0 <= r_it <= rbar_i,  0 <= s_i <= target_i,  P_all >= Phat_all,  P_on >= Phat_on
Phat_* are month-to-date realised peaks: capacity already paid for in the
demand charge is free, and only increments above it are priced.
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

try:  # direct HiGHS interface avoids per-call option validation overhead of scipy.linprog
    import highspy
    _HAS_HIGHSPY = True
except ImportError:  # pragma: no cover
    _HAS_HIGHSPY = False


def _solve_highs(c, A, b, lb, ub):
    """min c'x s.t. A x <= b, lb <= x <= ub, with HiGHS (same solver scipy uses)."""
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    lp = highspy.HighsLp()
    n, m = len(c), A.shape[0]
    lp.num_col_, lp.num_row_ = n, m
    lp.col_cost_ = np.asarray(c, dtype=float)
    lp.col_lower_ = np.asarray(lb, dtype=float)
    lp.col_upper_ = np.where(np.isinf(ub), highspy.kHighsInf, ub).astype(float)
    lp.row_lower_ = np.full(m, -highspy.kHighsInf)
    lp.row_upper_ = np.asarray(b, dtype=float)
    Ac = A.tocsc()
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = Ac.indptr.astype(np.int32)
    lp.a_matrix_.index_ = Ac.indices.astype(np.int32)
    lp.a_matrix_.value_ = Ac.data.astype(float)
    h.passModel(lp)
    h.run()
    if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        return None, np.nan
    return np.array(h.getSolution().col_value), h.getInfo().objective_function_value


def solve_schedule(evs, t0, T, price, trd_mask, dt, cap, lam_f, lam_t, phat_all, phat_on, M, eps, pv=None):
    """evs: list of dicts with keys ws, we (plannable window, absolute period
    indices), te (target deadline), target (kWh), rbar (kW).
    Returns (dict idx->np.array of rates over [t0, T)), status, objective)."""
    H = T - t0
    if H <= 0 or not evs:
        return {}, 0, 0.0
    cols_r, row_t, var_ev, costs, ub = [], [], [], [], []
    ev_slices = []
    v = 0
    for k, e in enumerate(evs):
        a, b = max(e["ws"], t0), min(e["we"], T)
        n = max(0, b - a)
        ts = np.arange(a, b)
        ev_slices.append((v, n, a))
        row_t.append(ts - t0)
        costs.append((0.0 if pv is not None else price[ts] * dt) + eps * (ts - t0))
        ub.append(np.full(n, e["rbar"]))
        v += n
    nr = v
    ns = len(evs)
    i_s = nr
    i_pa = nr + ns
    i_po = nr + ns + 1
    nv = nr + ns + 2
    # PV-aware variant: grid import g_t >= sum_i r_it - PV_t, energy cost and peaks are on import
    ng = H if pv is not None else 0
    i_g = nv
    nv += ng
    c = np.concatenate(costs + [np.full(ns, M), [lam_f, lam_t]] + ([price[t0:T] * dt] if ng else []))
    lb = np.zeros(nv)
    ubv = np.concatenate(ub + [np.array([max(e["target"], 0.0) for e in evs]), [np.inf, np.inf]] + ([np.full(ng, np.inf)] if ng else []))
    lb[i_pa], lb[i_po] = phat_all, phat_on

    rows, cols, vals, rhs = [], [], [], []
    r = 0
    for k, e in enumerate(evs):
        s0, n, a = ev_slices[k]
        tgt = max(e["target"], 0.0)
        m = n if e["te"] >= a + n else max(0, e["te"] - a)
        # delivery by deadline: -dt*sum r - s <= -target
        rows += [r] * (m + 1); cols += list(range(s0, s0 + m)) + [i_s + k]; vals += [-dt] * m + [-1.0]; rhs.append(-tgt); r += 1
        if n:
            rows += [r] * n; cols += list(range(s0, s0 + n)); vals += [dt] * n; rhs.append(tgt); r += 1
    tt = np.concatenate(row_t) if nr else np.array([], dtype=int)
    ridx = np.arange(nr)
    # capacity rows
    rows += list(r + tt); cols += list(ridx); vals += [1.0] * nr; rhs += [cap] * H; r += H
    if ng:
        # import rows: sum_i r_it - g_t <= PV_t
        rows += list(r + tt) + list(r + np.arange(H)); cols += list(ridx) + list(i_g + np.arange(H))
        vals += [1.0] * nr + [-1.0] * H; rhs += list(pv[t0:T]); r += H
        # peak rows on import: g_t - P_all <= 0
        rows += list(r + np.arange(H)) * 2; cols += list(i_g + np.arange(H)) + [i_pa] * H
        vals += [1.0] * H + [-1.0] * H; rhs += [0.0] * H; r += H
    else:
        # peak-all rows
        rows += list(r + tt) + list(r + np.arange(H)); cols += list(ridx) + [i_pa] * H
        vals += [1.0] * nr + [-1.0] * H; rhs += [0.0] * H; r += H
    # peak-on rows (TRD periods only)
    on_t = np.where(trd_mask[t0:T])[0]
    if len(on_t) and lam_t > 0 and ng:
        rows += [r + j for j in range(len(on_t))] * 2; cols += [i_g + t for t in on_t] + [i_po] * len(on_t)
        vals += [1.0] * len(on_t) + [-1.0] * len(on_t); rhs += [0.0] * len(on_t); r += len(on_t)
    elif len(on_t) and lam_t > 0:
        pos = {t: j for j, t in enumerate(on_t)}
        sel = np.isin(tt, on_t)
        rows += [r + pos[t] for t in tt[sel]] + [r + j for j in range(len(on_t))]
        cols += list(ridx[sel]) + [i_po] * len(on_t)
        vals += [1.0] * int(sel.sum()) + [-1.0] * len(on_t); rhs += [0.0] * len(on_t); r += len(on_t)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(r, nv))
    if _HAS_HIGHSPY:
        x, fun = _solve_highs(c, A, np.array(rhs), lb, ubv)
        if x is None:
            return {}, 2, np.nan
    else:
        res = linprog(c, A_ub=A, b_ub=np.array(rhs), bounds=np.column_stack([lb, ubv]), method="highs")
        if res.status != 0:
            return {}, res.status, np.nan
        x, fun = res.x, res.fun
    out = {}
    for k in range(len(evs)):
        s0, n, a = ev_slices[k]
        full = np.zeros(H)
        full[a - t0:a - t0 + n] = x[s0:s0 + n]
        out[k] = full
    return out, 0, float(fun)
