"""Pipeline integrity tests on SYNTHETIC data (software checks only, no research results)."""
import copy, json, os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthetic import make_raw, TEST_TARIFF
from acndc.clean import load_raw, clean
from acndc.tariff import Tariff
from acndc.run import run_month

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    cfg = json.load(open(os.path.join(ROOT, "config.json")))
    null_tariff = copy.deepcopy(cfg["tariff"]); null_tariff["frd_per_kw_month"] = None
    try:
        Tariff(null_tariff); raise AssertionError("null tariff accepted")
    except ValueError:
        print("PASS null tariff rejected")
    cfg = copy.deepcopy(cfg)
    cfg["tariff"] = TEST_TARIFF
    cfg["physics"]["site_capacity_kw"]["test"] = 80.0
    cfg["model"]["robust_quantiles"] = [0.5, 0.9]
    with tempfile.TemporaryDirectory() as d:
        p = make_raw(os.path.join(d, "test.jsonl"))
        df, flow = clean(load_raw([p]), cfg)
    assert flow["n_remaining"].iloc[0] >= len(df)
    print("PASS cleaning:", len(df), "sessions,", int(df["claimed"].sum()), "claimed")
    m, cal, summ = run_month(df, "test", "2019-03", cfg, Tariff(cfg["tariff"]), verbose=False)
    assert (m["energy_kwh_delivered"] <= m["energy_kwh_true"] + 1e-6).all()
    print("PASS delivered <= demanded for every policy")
    ctrl = m[m["policy"] != "uncontrolled"]
    assert (ctrl["peak_all_kw"] <= 80.0 + 1e-6).all()
    print("PASS capacity respected by controlled policies")
    off = float(m.loc[m["policy"] == "offline", "penalised_cost"].iloc[0])
    worse = ctrl[ctrl["penalised_cost"] < off - 1e-6]
    assert worse.empty, worse[["policy", "q", "penalised_cost"]]
    print("PASS offline penalised cost <= every controlled online policy")
    assert m.loc[m["policy"] == "offline", "served_ratio"].iloc[0] > 0.999
    print("PASS offline serves all demanded energy")
    print(m[["policy","q","n_solves","solve_s_mean"]].to_string(index=False))
    print("ALL TESTS PASSED (synthetic data; numbers are not results)")


if __name__ == "__main__":
    main()
