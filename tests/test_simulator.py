"""Check the vectorized simulator against a slow, obviously-correct loop."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import numpy as np
from simulator import simulate, Scenarios, SSPolicy, NaivePolicy, Costs


def reference(policy, demand, lead, costs, start=200):
    """One world, plain Python, no tricks."""
    net, pipeline = start, []          # pipeline: list of [arrival_day, qty]
    hold = fixed = back = 0.0
    shipped = so = n_orders = 0
    for t in range(len(demand)):
        arrived = sum(q for d, q in pipeline if d == t)
        pipeline = [p for p in pipeline if p[0] != t]
        net += arrived
        avail = max(net, 0)
        shipped += min(demand[t], avail)
        so += demand[t] > avail
        net -= demand[t]
        hold += costs.holding_per_unit_day * max(net, 0)
        back += costs.backorder_per_unit_day * max(-net, 0)
        on_order = sum(q for _, q in pipeline)
        q = int(policy.order(t, np.array([net]), np.array([on_order]))[0])
        if q > 0:
            fixed += costs.order_fixed
            pipeline.append([t + lead[n_orders], q])
            n_orders += 1
    return hold, fixed, back, shipped, so, n_orders


def test_matches_reference():
    costs = Costs()
    scen = Scenarios.generate(25, days=200, seed=7)
    for pol in (SSPolicy(120, 260), NaivePolicy(150, 7)):
        res = simulate(pol, scen, costs)
        for i in range(scen.n):
            h, f, b, sh, so, no = reference(pol, scen.demand[i], scen.lead[i], costs)
            assert np.isclose(res.holding_cost[i], h)
            assert np.isclose(res.ordering_cost[i], f)
            assert np.isclose(res.backorder_cost[i], b)
            assert res.shipped_from_stock[i] == sh
            assert res.stockout_days[i] == so
            assert res.n_orders[i] == no


def test_replay_mode():
    hist = np.array([10, 0, 30, 5, 50, 0, 0] * 20)
    scen = Scenarios.replay(hist, n_lead_draws=50, seed=1)
    assert scen.demand.shape == (50, len(hist))
    assert (scen.demand == hist).all()
    res = simulate(SSPolicy(60, 150), scen)
    assert (res.demand == hist.sum()).all()


if __name__ == "__main__":
    test_matches_reference(); test_replay_mode(); print("all tests passed")
