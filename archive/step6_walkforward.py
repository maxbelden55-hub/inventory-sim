"""
Step 6: the headline test. Every established product, real year-2 orders.

Policies, all facing the SAME real orders and the SAME supplier lead-time draws:
  1. Static (paper):      per product, fit hurdle-lognormal on year 1, grid-search (s,S)
                          on simulated years, freeze.
  2. Adaptive, no retrain: year-1 ML model, weekly forecasts, online recalibration,
                          optimal reorder point + EOQ (settings tuned in Step 5).
  3. Adaptive, retrained:  same, but the model is retrained every 4 weeks on all data
                          known at that time (gains 'same time last year' features).
  4. Hindsight static:    best fixed (s,S) per product had you known year 2 (reference).
"""
import sys, pathlib, time, json
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import os
import numpy as np
import pandas as pd
import static_baseline
K = float(os.environ.get("ORDER_COST", 50))
static_baseline.ORDER_FIXED = K
NO_RETRAIN = os.environ.get("NO_RETRAIN") == "1"
from forecaster import load_matrix, FeatureBuilder, build_panel, QuantileForecaster
from simulator import simulate, Scenarios, SSPolicy, DynamicSSPolicy, Costs
from static_baseline import costs_for, LEAD
from adaptive_policy import online_calibration, weekly_levels_optimal, eoq

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT, DATA = ROOT / "outputs", ROOT / "data"
R = 5                      # lead-time draws per product (same for every policy)
RETRAIN_EVERY = 4          # weeks
ROUNDS = 300
log = lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

t0 = time.time()
M, skus, days, prices = load_matrix()
fb = FeatureBuilder(M, days, prices, lead=LEAD, seed=1)
ix = lambda d: days.get_loc(pd.Timestamp(d))
y1_end, last = ix("2010-11-30"), len(days) - 1
T2 = last - y1_end
settings = pd.read_pickle(DATA / "step5_settings.pkl")
TARGET, KAPPA = settings["TARGET"], settings["KAPPA"]

est = np.where(fb.first_sale <= y1_end - 90)[0]          # established products
Np = len(est)
log(f"{Np:,} established products, {T2} test days, R={R} lead draws, order cost £{K:g}")

# shared year-2 scenarios: product i occupies rows i*R .. i*R+R-1
rep = lambda a: np.repeat(a, R, axis=0)
Y2 = M[est, y1_end + 1:].astype(np.int64)
scen2 = Scenarios.from_matrix(rep(Y2), LEAD, seed=2026)
costs_p = costs_for(prices[est])
costs_w = costs_for(rep(prices[est]))
open_w = rep(M[est, y1_end - 27:y1_end + 1].sum(1).astype(np.int64))


def per_product(x):
    return np.asarray(x).reshape(Np, R).mean(1)


def summarize(res):
    return dict(cost=per_product(res.operating_cost), hold=per_product(res.holding_cost),
                order=per_product(res.ordering_cost), back=per_product(res.backorder_cost),
                demand=per_product(res.demand), shipped=per_product(res.shipped_from_stock),
                stockout_days=per_product(res.stockout_days), orders=per_product(res.n_orders))


# =============================================================== 1. static (paper)
Y1 = M[est, :y1_end + 1]
wd1 = days[:y1_end + 1].dayofweek.to_numpy()
open1 = Y1[:, wd1 != 5]
p_ = (open1 > 0).mean(1)
logs = np.where(open1 > 0, np.log(np.maximum(open1, 1)), np.nan)
mu_ = np.nanmean(logs, 1)
cnt = (open1 > 0).sum(1)
sig_ = np.where(cnt >= 3, np.nanstd(logs, 1, ddof=1), 0.75)
mu_ = np.nan_to_num(mu_, nan=0.0)
mean_open = p_ * np.exp(mu_ + sig_ ** 2 / 2)
mean_day = mean_open * 6 / 7
ltd = mean_day * (LEAD.min_days + LEAD.mean_extra)
eoq_static = np.maximum(eoq(mean_day * 365, costs_p), 1)

A = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 9]          # s = a x expected lead-time demand
B = [0.25, 0.5, 1, 1.5, 2, 3, 5]             # Q = b x EOQ
NS = 20                                      # simulated years per product
wd2 = (days[y1_end + 1] .dayofweek + np.arange(T2)) % 7
rng = np.random.default_rng(11)
best_cost = np.full(Np, np.inf)
best_s = np.zeros(Np, dtype=np.int64)
best_S = np.zeros(Np, dtype=np.int64)
for lo in range(0, Np, 700):
    sl = slice(lo, min(lo + 700, Np))
    n = sl.stop - sl.start
    pw, mw, sw = (np.repeat(v[sl], NS) for v in (p_, mu_, sig_))
    happens = rng.random((n * NS, T2)) < pw[:, None]
    size = np.maximum(np.rint(rng.lognormal(mw[:, None], sw[:, None], (n * NS, T2))), 1)
    sim_dem = np.where(happens & (wd2 != 5)[None, :], size, 0).astype(np.int64)
    del happens, size
    sc = Scenarios(sim_dem, LEAD.sample(n * NS, T2, rng))
    cw = costs_for(np.repeat(prices[est][sl], NS))
    for a in A:
        for b in B:
            s = np.rint(a * ltd[sl]).astype(np.int64)
            S = s + np.maximum(np.rint(b * eoq_static[sl]), 1).astype(np.int64)
            r = simulate(SSPolicy(np.repeat(s, NS), np.repeat(S, NS)), sc, cw,
                         start_inventory=np.repeat(S, NS))
            c = r.operating_cost.reshape(n, NS).mean(1)
            better = c < best_cost[sl]
            best_cost[sl] = np.where(better, c, best_cost[sl])
            best_s[sl] = np.where(better, s, best_s[sl])
            best_S[sl] = np.where(better, S, best_S[sl])
    log(f"static: optimized products {sl.start}-{sl.stop}")
static = summarize(simulate(SSPolicy(rep(best_s), rep(best_S)), scen2, costs_w,
                            start_inventory=open_w))
log(f"static done ({time.time() - t0:.0f}s)")

# =============================================================== 4. hindsight static
h_cost = np.full(Np, np.inf)
hind = None
ltd2 = Y2.mean(1) * (LEAD.min_days + LEAD.mean_extra)
eoq2 = np.maximum(eoq(Y2.mean(1) * 365, costs_p), 1)
for grid_ltd, grid_eoq in ((ltd, eoq_static), (ltd2, eoq2)):     # search around both
    for a in A:
        for b in B:
            s = np.rint(a * grid_ltd).astype(np.int64)
            S = s + np.maximum(np.rint(b * grid_eoq), 1).astype(np.int64)
            r = summarize(simulate(SSPolicy(rep(s), rep(S)), scen2, costs_w,
                                   start_inventory=open_w))
            better = r["cost"] < h_cost
            h_cost = np.where(better, r["cost"], h_cost)
            hind = r if hind is None else {k: np.where(better, r[k], hind[k]) for k in r}
log(f"hindsight done ({time.time() - t0:.0f}s)")

# =============================================================== 2+3. adaptive
test_origins = list(range(y1_end, last - 7 + 1, 7))
all_origins = list(range(28, last - 7 + 1, 7))


def train_until(t_cut):
    """Train on every weekly origin whose outcome window ended by t_cut."""
    origins = [o for o in all_origins if o + fb.horizons(o).max() <= t_cut]
    X, y, _ = build_panel(fb, origins, skus)
    return QuantileForecaster(rounds=ROUNDS).fit(X, y), len(y)


def forecast(model, t):
    P = model.predict(fb.at(t))
    P[fb.first_sale > t] = 0
    return P


P_static_model, P_retrain = [], []
model0 = None
for k, t in enumerate(test_origins):
    if (k % RETRAIN_EVERY == 0 and not NO_RETRAIN) or model0 is None:
        model, nrows = train_until(t)
        if model0 is None:
            model0 = model
        log(f"retrained at {days[t].date()} on {nrows:,} rows")
    P_retrain.append(forecast(model, t))
    P_static_model.append(forecast(model0, t))

results = {}
variants = [("adaptive_no_retrain", P_static_model)]
if not NO_RETRAIN:
    variants.append(("adaptive_retrained", P_retrain))
for name, Plist in variants:
    Pcal, _ = online_calibration(Plist, test_origins, fb)
    Pe = [rep(P[est]) for P in Pcal]
    s, S = weekly_levels_optimal(Pe, costs_w, TARGET, KAPPA, T2)
    results[name] = summarize(simulate(DynamicSSPolicy(s, S), scen2, costs_w,
                                       start_inventory=open_w))
    log(f"{name} done")

results["static"] = static
results["hindsight"] = hind
tier0 = fb.tier(y1_end)[est]
df = pd.DataFrame({"sku": skus[est], "price": prices[est], "tier_at_start": tier0,
                   "y1_units": Y1.sum(1), "y2_units": Y2.sum(1)})
for name, r in results.items():
    for k, v in r.items():
        df[f"{name}__{k}"] = v
suffix = "" if K == 50 and not NO_RETRAIN else f"_K{K:g}"
df.to_parquet(DATA / f"step6_results{suffix}.parquet")
log(f"saved per-product results; total {time.time() - t0:.0f}s")
