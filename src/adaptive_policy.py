"""
Adaptive ML policy: every week, for every product, turn the probabilistic
forecast into an (s, S) ordering rule.

  reorder point  s = the tau-quantile of next-7-day demand (lead-time demand)
                     -> newsvendor / critical-ratio logic: tau trades off the
                        cost of running short vs. the cost of holding stock
  order size     Q = kappa x EOQ = kappa x sqrt(2 K D / h)
                     -> balances the fixed cost per order (K) against holding (h)
  order-up-to    S = s + Q

tau and kappa are the only two knobs, tuned by Monte Carlo grid search on a
validation period. Before use, the forecast quantiles are recalibrated so that
"q90" really means demand stays below it 90% of the time.
"""
import numpy as np
from forecaster import QUANTILES


def critical_ratio(costs):
    """Textbook optimal service quantile: b / (b + h)."""
    b, h = costs.backorder_per_unit_day, costs.holding_per_unit_day
    return b / (b + h)


# ------------------------------------------------------------ recalibration

def fit_calibration(P, y, quantiles=QUANTILES):
    """For each quantile, the shift (in log1p units) that makes coverage match.
    Fit on weeks the model did NOT train on."""
    resid = np.log1p(y)[:, None] - np.log1p(P)
    return np.array([np.quantile(resid[:, i], q) for i, q in enumerate(quantiles)])


def online_calibration(P_by_origin, origins, fb, window=8, quantiles=QUANTILES,
                       by_tier=True, min_rows=300):
    """Recalibrate each week using only the model's OWN past errors.
    At origin k, use the `window` most recent forecasts whose outcome is already
    known (outcome window fully in the past), however far back that is. Corrections are fitted
    separately per volume tier (intermittent / low / mid / high), falling back
    to the pooled correction when a tier has too little data."""
    out, log = [], []
    tiers_at = {}
    horizon_max = [int(fb.horizons(o).max()) for o in origins]
    for k, (P, t) in enumerate(zip(P_by_origin, origins)):
        # the `window` most recent forecasts whose outcome window has fully ended.
        # (Searching only the last `window` weeks found nothing when lead times
        # exceed `window` weeks, which silently disabled recalibration.)
        known = [i for i in range(k) if origins[i] + horizon_max[i] <= t]
        past = known[-window:]
        sh_rows = np.zeros((P.shape[0], len(quantiles)))
        if past:
            Pp = np.vstack([P_by_origin[i] for i in past])
            yp = np.concatenate([fb.target(origins[i]) for i in past])
            live = np.concatenate([fb.first_sale <= origins[i] for i in past])
            tp = np.concatenate([tiers_at.setdefault(origins[i], fb.tier(origins[i]))
                                 for i in past])
            pooled = fit_calibration(Pp[live], yp[live], quantiles)
            sh_rows[:] = pooled
            if by_tier:
                now = tiers_at.setdefault(t, fb.tier(t))
                for g in np.unique(now):
                    m = live & (tp == g)
                    if m.sum() >= min_rows:
                        sh_rows[now == g] = fit_calibration(Pp[m], yp[m], quantiles)
            log.append(pooled)
        else:
            log.append(np.zeros(len(quantiles)))
        out.append(np.sort(np.maximum(np.expm1(np.log1p(P) + sh_rows), 0), axis=1))
    return out, np.array(log)


def apply_calibration(P, shifts):
    return np.sort(np.maximum(np.expm1(np.log1p(P) + shifts[None, :]), 0), axis=1)


def quantile_at(P, tau, quantiles=QUANTILES):
    """Interpolate the forecast at any tau between q10 and q99 (in log1p space)."""
    tau = float(np.clip(tau, quantiles[0], quantiles[-1]))
    L = np.log1p(P)
    j = np.searchsorted(quantiles, tau)
    if quantiles[min(j, len(quantiles) - 1)] == tau:
        return P[:, j]
    w = (tau - quantiles[j - 1]) / (quantiles[j] - quantiles[j - 1])
    return np.expm1((1 - w) * L[:, j - 1] + w * L[:, j])


def mean_from_quantiles(P, quantiles=QUANTILES):
    """Approximate the forecast mean = area under the quantile curve
    (trapezoid, with Q(0)=0 and Q(1)=q99)."""
    taus = np.r_[0.0, quantiles, 1.0]
    Q = np.column_stack([np.zeros(len(P)), P, P[:, -1]])
    return (np.diff(taus)[None, :] * (Q[:, 1:] + Q[:, :-1]) / 2).sum(1)


# ------------------------------------------------------------ policy levels

def eoq(annual_demand, costs):
    h_year = np.asarray(costs.holding_per_unit_day) * 365
    return np.sqrt(2 * costs.order_fixed * np.maximum(annual_demand, 0) / h_year)


def _knots(P, quantiles=QUANTILES):
    """Full quantile function per product: Q(0)=0, the forecast quantiles, and
    an exponential tail fitted to q95->q99 out to the 99.9th percentile."""
    beta = np.maximum(P[:, -1] - P[:, -2], 0) / np.log(5)
    q999 = P[:, -1] + beta * np.log(10)
    X = np.column_stack([np.zeros(len(P)), P, q999])             # demand values
    T = np.r_[0.0, quantiles, 0.999]                             # probabilities
    return np.maximum.accumulate(X, axis=1), T


def _cdf(y, X, T):
    """P(demand <= y) for each product (row-wise piecewise-linear CDF)."""
    y = y[..., None]                                             # (n, m, 1)
    lo, hi = X[:, None, :-1], X[:, None, 1:]
    width = hi - lo
    frac = np.where(width > 0, np.clip((y - lo) / np.where(width > 0, width, 1), 0, 1),
                    (y >= hi).astype(float))
    return (frac * np.diff(T)[None, None, :]).sum(-1)


def optimal_reorder_point(P, Q, target, m=24, iters=32):
    """Pick s so that the AVERAGE no-stockout probability across the band
    s..s+Q equals the target (b/(b+h)). With (s, Q) policies, inventory
    position spends time evenly across that band, so this is the right
    condition, not F(s) = target as in the single-period newsvendor."""
    X, T = _knots(P)
    lo = np.full(len(P), -1.0) * Q            # allow s < 0 when Q alone covers it
    hi = X[:, -1] + 1.0
    grid = (np.arange(m) + 0.5) / m
    for _ in range(iters):
        mid = (lo + hi) / 2
        avg = _cdf(mid[:, None] + grid[None, :] * Q[:, None], X, T).mean(1)
        too_low = avg < target
        lo = np.where(too_low, mid, lo)
        hi = np.where(too_low, hi, mid)
    return np.ceil(hi)


def weekly_levels_optimal(P_by_origin, costs, target, kappa, days):
    """Like weekly_levels, but the reorder point comes from
    optimal_reorder_point (per product, per week, full distribution)."""
    n = P_by_origin[0].shape[0]
    s = np.zeros((n, days), dtype=np.int64)
    S = np.zeros((n, days), dtype=np.int64)
    hi = 0
    for k, P in enumerate(P_by_origin):
        lo, hi = 7 * k, min(7 * k + 7, days)
        if lo >= days:
            break
        Q = np.maximum(np.rint(kappa * eoq(mean_from_quantiles(P) * 52, costs)), 1)
        rp = optimal_reorder_point(P, Q.astype(float), target)
        s[:, lo:hi] = rp[:, None]
        S[:, lo:hi] = (rp + Q)[:, None]
    if hi < days:
        s[:, hi:] = s[:, hi - 1:hi]
        S[:, hi:] = S[:, hi - 1:hi]
    return s, S


def weekly_levels(P_by_origin, costs, tau, kappa, days):
    """Build per-product, per-day s and S arrays.
    P_by_origin: list of (n_products x n_quantiles) calibrated forecasts of
    next-7-day demand, one per weekly origin; origin k's rule applies to
    simulated days 7k .. 7k+6. Order size uses the FORECAST demand rate
    (forward-looking), so it shrinks before a season ends, not after."""
    n = P_by_origin[0].shape[0]
    s = np.zeros((n, days), dtype=np.int64)
    S = np.zeros((n, days), dtype=np.int64)
    for k, P in enumerate(P_by_origin):
        lo, hi = 7 * k, min(7 * k + 7, days)
        if lo >= days:
            break
        rp = np.ceil(quantile_at(P, tau))
        Q = np.maximum(np.rint(kappa * eoq(mean_from_quantiles(P) * 52, costs)), 1)
        s[:, lo:hi] = rp[:, None]
        S[:, lo:hi] = (rp + Q)[:, None]
    if hi < days:                          # pad the final partial week
        s[:, hi:] = s[:, hi - 1:hi]
        S[:, hi:] = S[:, hi - 1:hi]
    return s, S
