"""
Global probabilistic demand forecaster.

Question it answers, every week, for every product:
    "How many units will customers order over the NEXT 14 DAYS, and how
     uncertain is that?"  -> quantiles q10, q25, q50, q75, q90, q95, q99

14 days ~ supplier lead time (5-7+ days) + one week until the next review,
which is exactly the window an ordering decision has to cover.

One model is trained on ALL products at once (a "global" model), so patterns
learned from thousands of items (momentum, decay after a bulk order, the
autumn ramp-up) transfer to each individual item, including new ones.

Every feature is computed only from data available at the forecast date
(no peeking at the future).
"""
import pathlib
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = pathlib.Path(__file__).resolve().parent.parent
H = 14                                         # forecast horizon (days)
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
WINDOWS = [7, 14, 28, 56, 91, 182]


def load_matrix():
    """Dense demand matrix: products x calendar days (closed days = 0)."""
    d = pd.read_parquet(ROOT / "data" / "daily_demand.parquet")
    prods = pd.read_parquet(ROOT / "data" / "products.parquet").set_index("StockCode")
    days = pd.date_range(d.date.min(), d.date.max())
    skus = np.sort(d.StockCode.unique())
    M = (d.pivot_table(index="StockCode", columns="date", values="qty", aggfunc="sum")
           .reindex(index=skus, columns=days).fillna(0).to_numpy(np.float64))
    return M, skus, days, prods.loc[skus, "price"].to_numpy()


class FeatureBuilder:
    """horizon: fixed forecast window in days, OR pass `lead` (a LeadTimeModel)
    to make each target the demand over a RANDOM lead time drawn per product
    and origin. The model then learns lead-time demand directly, including the
    risk that a delivery runs late. Draws are seeded by origin, so the same
    origin always gets the same windows (training, calibration and scoring agree)."""

    def __init__(self, M, days, prices, horizon=H, lead=None, seed=0):
        self.M, self.days, self.prices, self.h = M, days, prices, horizon
        self.lead, self.seed = lead, seed
        if lead is not None:
            self.h = int(round(lead.min_days + lead.mean_extra))
        self.cs = np.concatenate([np.zeros((M.shape[0], 1)), M.cumsum(1)], 1)
        self.cs_pos = np.concatenate([np.zeros((M.shape[0], 1)), (M > 0).cumsum(1)], 1)
        nz = M > 0
        self.first_sale = np.where(nz.any(1), nz.argmax(1), M.shape[1])

    def wsum(self, t, w, cs=None):
        """Sum over the w days ending at day index t (inclusive)."""
        cs = self.cs if cs is None else cs
        lo = max(t + 1 - w, 0)
        return cs[:, t + 1] - cs[:, lo]

    def at(self, t):
        """Feature table for all products at forecast origin t (day index)."""
        M, n = self.M, self.M.shape[0]
        f = {}
        for w in WINDOWS:
            f[f"sum_{w}"] = self.wsum(t, w)
            f[f"days_sold_{w}"] = self.wsum(t, w, self.cs_pos)
        lo = max(t - 90, 0)
        recent = M[:, lo:t + 1]
        f["max_day_91"] = recent.max(1)
        f["std_day_91"] = recent.std(1)
        # share of last 91 days' volume that came from the single biggest day (lumpiness)
        f["top_day_share_91"] = f["max_day_91"] / np.maximum(f["sum_91"], 1)
        last_sale = np.where((M[:, :t + 1] > 0).any(1),
                             t - (M[:, :t + 1] > 0)[:, ::-1].argmax(1), 999)
        f["days_since_sale"] = last_sale
        f["age_days"] = t - self.first_sale
        f["momentum_7_28"] = f["sum_7"] / np.maximum(f["sum_28"] / 4, 1)
        f["momentum_28_91"] = f["sum_28"] / np.maximum(f["sum_91"] * 28 / 91, 1)
        f["price"] = self.prices
        # same time last year (NaN until a year of history exists)
        if t - 364 - 27 >= 0:
            f["ly_next_28"] = np.log1p(self.cs[:, t - 364 + 29] - self.cs[:, t - 364 + 1])
            f["ly_last_28"] = np.log1p(self.wsum(t - 364, 28))
            f["ly_growth"] = np.log1p(self.wsum(t, 28)) - f["ly_last_28"]
        else:
            f["ly_next_28"] = f["ly_last_28"] = f["ly_growth"] = np.full(n, np.nan)
        date = self.days[t]
        target_mid = date + pd.Timedelta(days=self.h // 2)
        f["week_of_year"] = np.full(n, target_mid.isocalendar().week)
        f["days_to_xmas"] = np.full(n, (pd.Timestamp(date.year, 12, 25) - date).days % 365)
        X = pd.DataFrame(f)
        for c in [c for c in X if c.startswith("sum_") or c in ("max_day_91", "std_day_91")]:
            X[c] = np.log1p(X[c])
        return X

    def horizons(self, t):
        """Window length per product for origin t."""
        n = self.M.shape[0]
        if self.lead is None:
            return np.full(n, self.h)
        rng = np.random.default_rng(self.seed * 100_003 + t)
        L = self.lead.min_days + rng.poisson(self.lead.mean_extra, n)
        return np.minimum(L, self.M.shape[1] - 1 - t)

    def target(self, t):
        """Demand over days t+1 .. t+window."""
        L = self.horizons(t)
        return self.cs[np.arange(len(L)), t + 1 + L] - self.cs[:, t + 1]

    def tier(self, t):
        """Volume tier at origin t: 0 intermittent (sold on < 1/4 of recent
        open days), then 1/2/3 = low/mid/high recent volume."""
        ds, vol = self.wsum(t, 91, self.cs_pos), self.wsum(t, 91)
        tier = np.where(vol < 300, 1, np.where(vol < 2000, 2, 3))
        return np.where(ds < 20, 0, tier)


def build_panel(fb, origins, skus):
    """Stack features for many weekly origins; keep products already launched."""
    Xs, ys, meta = [], [], []
    for t in origins:
        X = fb.at(t)
        live = fb.first_sale <= t
        Xs.append(X[live])
        ys.append(fb.target(t)[live])
        meta.append(pd.DataFrame({"t": t, "sku": skus[live]}))
    return (pd.concat(Xs, ignore_index=True), np.concatenate(ys),
            pd.concat(meta, ignore_index=True))


PARAMS = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=1.0, verbose=-1, num_threads=2)


class QuantileForecaster:
    """One LightGBM model per quantile, trained on log1p(demand).
    Quantiles survive monotone transforms, so expm1(q_tau of log1p y) is
    exactly q_tau of y."""

    def __init__(self, quantiles=QUANTILES, rounds=400):
        self.quantiles, self.rounds, self.models = quantiles, rounds, {}

    def fit(self, X, y):
        ly = np.log1p(y)
        for q in self.quantiles:
            ds = lgb.Dataset(X, ly)
            self.models[q] = lgb.train({**PARAMS, "objective": "quantile", "alpha": q},
                                       ds, self.rounds)
        return self

    def predict(self, X):
        P = np.column_stack([self.models[q].predict(X) for q in self.quantiles])
        P = np.sort(P, axis=1)                    # no crossing quantiles
        return np.maximum(np.expm1(P), 0)


# ------------------------------------------------------------------ baselines

def empirical_quantiles(windows_sums, quantiles=QUANTILES):
    return np.quantile(windows_sums, quantiles, axis=1).T


def static_baseline(fb, t_fit_end, quantiles=QUANTILES):
    """Paper-style: one fixed distribution of 14-day demand per product,
    estimated from every 14-day window in year 1, never updated."""
    sums = np.stack([fb.target(t) for t in range(0, t_fit_end - H)], 1)
    return empirical_quantiles(sums, quantiles)


def moving_window_baseline(fb, t, lookback=91, quantiles=QUANTILES):
    """Simple adaptive rule: quantiles of 14-day demand over the last 91 days."""
    lo = max(t - lookback, 0)
    sums = np.stack([fb.cs[:, s + H] - fb.cs[:, s] for s in range(lo, t - H + 2)], 1)
    return empirical_quantiles(sums, quantiles)


def pinball(y, P, quantiles=QUANTILES):
    """Quantile (pinball) loss per row and quantile."""
    q = np.asarray(quantiles)[None, :]
    d = y[:, None] - P
    return np.maximum(q * d, (q - 1) * d)
