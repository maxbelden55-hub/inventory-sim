"""Tests for data cleaning, forecasting features, calibration and policy math."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import numpy as np
import pandas as pd
from data_prep import net_cancellations
from forecaster import FeatureBuilder, QUANTILES
from adaptive_policy import (fit_calibration, apply_calibration, optimal_reorder_point,
                             quantile_at, mean_from_quantiles)
from simulator import simulate, Scenarios, SSPolicy, Costs, LeadTimeModel


def test_cancellation_netting():
    t = pd.to_datetime
    sales = pd.DataFrame({"CustomerID": [1, 1, 2, 1], "StockCode": ["A", "A", "A", "B"],
                          "Quantity": [10, 10, 10, 5],
                          "InvoiceDate": t(["2010-01-01", "2010-01-03", "2010-01-02", "2010-01-01"])})
    cancels = pd.DataFrame({"CustomerID": [1, 3], "StockCode": ["A", "A"], "Quantity": [-10, -10],
                            "InvoiceDate": t(["2010-01-04", "2010-01-04"])})
    kept, removed = net_cancellations(sales, cancels)
    assert removed == 1                                   # customer 3 has nothing to cancel
    # the LATEST earlier matching line (Jan 3) is the one removed
    assert not ((kept.CustomerID == 1) & (kept.InvoiceDate == t("2010-01-03"))).any()
    assert len(kept) == 3


def test_cancellation_never_removes_a_later_order():
    t = pd.to_datetime
    sales = pd.DataFrame({"CustomerID": [1], "StockCode": ["A"], "Quantity": [10],
                          "InvoiceDate": t(["2010-02-01"])})
    cancels = pd.DataFrame({"CustomerID": [1], "StockCode": ["A"], "Quantity": [-10],
                            "InvoiceDate": t(["2010-01-01"])})
    kept, removed = net_cancellations(sales, cancels)
    assert removed == 0 and len(kept) == 1


def _toy_builder(lead=None, T=500, n=30, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.poisson(3, (n, T)).astype(float)
    days = pd.date_range("2009-12-01", periods=T)
    return FeatureBuilder(M, days, np.ones(n), lead=lead, seed=1), M, days


def test_features_do_not_look_ahead():
    fb, M, days = _toy_builder()
    t = 400
    X1 = fb.at(t)
    M2 = M.copy()
    M2[:, t + 1:] = 999                                    # change only the future
    fb2 = FeatureBuilder(M2, days, np.ones(M.shape[0]))
    X2 = fb2.at(t)
    pd.testing.assert_frame_equal(X1, X2)


def test_lead_time_targets_are_deterministic_and_bounded():
    lead = LeadTimeModel(5, 2.0)
    fb, M, _ = _toy_builder(lead)
    L1, L2 = fb.horizons(300), fb.horizons(300)
    assert (L1 == L2).all() and L1.min() >= 5
    y = fb.target(300)
    manual = np.array([M[i, 301:301 + L1[i]].sum() for i in range(M.shape[0])])
    assert np.allclose(y, manual)


def test_calibration_restores_coverage():
    rng = np.random.default_rng(3)
    y = rng.gamma(2.0, 50.0, 20000)
    true_q = np.quantile(y, QUANTILES)
    P = np.tile(true_q * 1.6, (len(y), 1))                 # forecasts 60% too high
    shifts = fit_calibration(P, y)
    Pc = apply_calibration(P, shifts)
    cov = (y[:, None] <= Pc).mean(0)
    assert np.allclose(cov, QUANTILES, atol=0.01)


def test_optimal_reorder_point():
    P = np.array([[10, 20, 30, 40, 50, 60, 80.]])
    # tiny order size -> reorder point = the target quantile itself
    assert optimal_reorder_point(P, np.array([1.0]), 0.9)[0] == 50
    assert optimal_reorder_point(P, np.array([1.0]), 0.99)[0] == 80
    # bigger order size -> lower reorder point; higher target -> higher reorder point
    s_small, s_big = (optimal_reorder_point(P, np.array([q]), 0.95)[0] for q in (5.0, 200.0))
    assert s_big < s_small
    assert optimal_reorder_point(P, np.array([50.0]), 0.99)[0] >= \
        optimal_reorder_point(P, np.array([50.0]), 0.9)[0]


def test_quantile_helpers():
    P = np.array([[10, 20, 30, 40, 50, 60, 80.]])
    assert quantile_at(P, 0.5)[0] == 30
    assert 30 < quantile_at(P, 0.6)[0] < 40
    m = mean_from_quantiles(P)[0]
    assert 20 < m < 50


def test_per_world_costs_match_scalar_runs():
    scen = Scenarios.generate(40, days=120, seed=4)
    c1, c2 = Costs(holding_per_unit_day=0.1), Costs(holding_per_unit_day=0.3)
    r1 = simulate(SSPolicy(100, 250), scen, c1)
    r2 = simulate(SSPolicy(100, 250), scen, c2)
    mixed = Costs(holding_per_unit_day=np.r_[np.full(20, 0.1), np.full(20, 0.3)])
    rm = simulate(SSPolicy(100, 250), scen, mixed)
    assert np.allclose(rm.holding_cost[:20], r1.holding_cost[:20])
    assert np.allclose(rm.holding_cost[20:], r2.holding_cost[20:])



def test_terminal_cost():
    from simulator import terminal_cost
    scen = Scenarios(np.zeros((2, 10), dtype=np.int64), np.full((2, 10), 3))
    c = Costs(holding_per_unit_day=0.1, backorder_per_unit_day=1.0)
    from simulator import NaivePolicy
    r = simulate(NaivePolicy(qty=0), scen, c, start_inventory=np.array([100, -5]))  # never orders
    tc = terminal_cost(r, c, daily_rate=np.array([10.0, 10.0]), lead_mean=7)
    # 100 units sold off at 10/day: 0.1 x 100 x 10 days / 2 = 50; 5 waiting x 1.0 x 7 = 35
    assert np.allclose(tc, [50.0, 35.0])
    # no demand at all -> capped at 365 days of holding
    tc0 = terminal_cost(r, c, daily_rate=np.array([0.0, 0.0]), lead_mean=7)
    assert np.isclose(tc0[0], 0.1 * 100 * 365 / 2)


def test_one_off_backorder_charge():
    from simulator import NaivePolicy
    dem = np.array([[0, 30, 0, 20, 0]], dtype=np.int64)
    scen = Scenarios(dem, np.full((1, 5), 3))
    c = Costs(holding_per_unit_day=0.0, backorder_per_unit_day=0.0, backorder_per_unit=2.0)
    r = simulate(NaivePolicy(qty=0), scen, c, start_inventory=np.array([10]))
    # 20 units short on day 1, 20 more on day 3 -> 40 x 2.0 = 80, charged once each
    assert np.isclose(r.backorder_cost[0], 80.0)


def test_online_calibration_works_with_long_lead_times():
    from adaptive_policy import online_calibration
    lead = LeadTimeModel(40, 15.0)                 # windows of ~40-70 days, longer than 8 weeks
    fb, M, _ = _toy_builder(lead, T=600, n=200)
    origins = list(range(100, 520, 7))
    P_raw = [np.tile(np.quantile(fb.target(t), QUANTILES) * 3.0, (M.shape[0], 1)) for t in origins]
    Pc, log = online_calibration(P_raw, origins, fb, by_tier=False)
    assert np.abs(log[-1]).sum() > 0, "recalibration never ran"
    late = [k for k in range(len(origins)) if k > 20]
    cov = np.mean([(fb.target(origins[k]) <= Pc[k][:, 4]).mean() for k in late])
    assert 0.8 < cov < 0.97                         # q90 coverage restored (raw forecasts were 3x too high)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)
