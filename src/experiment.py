"""
Experiment engine: evaluate static vs adaptive policies on real year-2 demand
for any cost / lead-time configuration.

    prepare_forecasts(lead)      train forecasters once and cache their raw forecasts
    evaluate(cfg)                tune on validation, test on year 2, save per-product results

Timeline (no look-ahead anywhere):
    Fold A: forecaster trained Dec 2009 - Mar 2010, validated Apr - Jun 2010 (flat demand)
    Fold B: forecaster trained Dec 2009 - Jun 2010, validated Jul - Nov 2010 (demand +119%)
            adaptive settings are tuned per volume tier on the SUM of both folds; tuning on
            fold B alone picked 2x order sizes that compensated for under-forecasting during
            the ramp-up and overstocked in year 2
    Dec 2010 - Dec 2011   test: year-1 forecaster (optionally retrained every 4 weeks),
                          online recalibration from its own past errors, real orders replayed

Policies compared on the SAME real orders and the SAME supplier lead-time draws:
    static            paper method: per product, fit hurdle-lognormal on year 1,
                      grid-search (s, S) on simulated years, freeze
    adaptive_textbook weekly (s, S) from forecasts, textbook settings (target = b/(b+h), 1x EOQ)
    adaptive_global   weekly (s, S), one setting tuned on validation for all products
    adaptive_tiered   weekly (s, S), settings tuned per volume tier on validation
    adaptive_tiered_retrained  as above, forecaster retrained every 4 weeks (if cached)
    hindsight         best fixed (s, S) per product had year 2 been known (reference only)
"""
import json, pathlib, time
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from forecaster import load_matrix, FeatureBuilder, build_panel, QuantileForecaster
from simulator import simulate, Scenarios, SSPolicy, DynamicSSPolicy, Costs, LeadTimeModel, terminal_cost
from adaptive_policy import online_calibration, weekly_levels_optimal, weekly_levels, eoq

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "outputs"
RES = DATA / "results"
RES.mkdir(parents=True, exist_ok=True)
TIERS = {0: "Intermittent", 1: "Low volume", 2: "Mid volume", 3: "High volume"}
log = lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


@dataclass
class Config:
    name: str = "base"
    order_cost: float = 5.0          # £ per product order line
    purchase_share: float = 0.60     # purchase cost as share of selling price
    holding_rate: float = 0.25       # per year, on purchase cost
    backorder_rate: float = 0.05     # share of price per unit per day a customer waits
    backorder_once: float = 0.0      # share of price charged once per backordered unit
    lead_min: int = 5
    lead_extra: float = 2.0          # Poisson mean of extra days
    retrain: bool = False

    @property
    def lead(self):
        return LeadTimeModel(self.lead_min, self.lead_extra)

    def costs(self, price):
        unit = self.purchase_share * np.asarray(price, float)
        return Costs(holding_per_unit_day=self.holding_rate * unit / 365,
                     order_fixed=self.order_cost, order_per_unit=unit,
                     backorder_per_unit_day=self.backorder_rate * np.asarray(price, float),
                     backorder_per_unit=self.backorder_once * np.asarray(price, float))

    @property
    def opening_days(self):
        """Opening stock = this many days of prior demand: 28, or lead time + 14
        when lead times are long (otherwise every policy starts out of stock)."""
        return max(28, int(round(self.lead_min + self.lead_extra + 14)))

    def critical_ratio(self):
        b = self.backorder_rate
        h = self.holding_rate * self.purchase_share / 365
        return b / (b + h)


# --------------------------------------------------------------------- setup

class Setup:
    """Data, calendar and feature builder for one lead-time model."""

    def __init__(self, lead):
        self.M, self.skus, self.days, self.prices = load_matrix()
        self.fb = FeatureBuilder(self.M, self.days, self.prices, lead=lead, seed=1)
        ix = lambda d: self.days.get_loc(pd.Timestamp(d))
        self.val_start, self.y1_end = ix("2010-07-01"), ix("2010-11-30")
        self.last = len(self.days) - 1
        self.val_origins = list(range(self.val_start - 1, self.y1_end - 14 + 1, 7))
        self.valA_start = ix("2010-04-01")
        self.valA_origins = list(range(self.valA_start - 1, self.val_start - 14 + 1, 7))
        self.folds = [(self.valA_start, self.valA_origins, "PvA"),
                      (self.val_start, self.val_origins, "Pv")]
        self.test_origins = list(range(self.y1_end, self.last - 7 + 1, 7))
        self.all_origins = list(range(28, self.last - 7 + 1, 7))

    def train_until(self, t_cut, rounds=400):
        fb = self.fb
        origins = [o for o in self.all_origins if o + fb.horizons(o).max() <= t_cut]
        X, y, _ = build_panel(fb, origins, self.skus)
        return QuantileForecaster(rounds=rounds).fit(X, y), len(y)

    def forecast(self, model, t):
        P = model.predict(self.fb.at(t))
        P[self.fb.first_sale > t] = 0
        return P


def forecast_cache(lead, retrain=False):
    return DATA / f"forecasts_L{lead.min_days}_{lead.mean_extra:g}{'_retrain' if retrain else ''}.pkl"


def prepare_forecasts(lead, retrain=False, every=4):
    """Train and cache raw forecasts. Base: early model -> validation origins,
    year-1 model -> test origins. retrain=True: test forecasts from a model
    retrained every `every` weeks on all outcomes known at that time."""
    path = forecast_cache(lead, retrain)
    if path.exists():
        out = pd.read_pickle(path)
        if retrain or "PvA" in out:
            return out
    st = Setup(lead)
    t0 = time.time()
    if not retrain:
        earlyA, n_a = st.train_until(st.valA_start - 1)
        PvA = [st.forecast(earlyA, t) for t in st.valA_origins]
        early, n_e = st.train_until(st.val_start - 1)
        Pv = [st.forecast(early, t) for t in st.val_origins]
        full, n_f = st.train_until(st.y1_end)
        Pt = [st.forecast(full, t) for t in st.test_origins]
        imp = pd.Series(full.models[0.5].feature_importance("gain"),
                        index=full.models[0.5].feature_name())
        out = dict(PvA=PvA, Pv=Pv, Pt=Pt, n_foldA=n_a, n_early=n_e, n_full=n_f, importance=imp)
        log(f"forecasts ready: fold-A model {n_a:,} rows, fold-B model {n_e:,} rows, year-1 model {n_f:,} rows "
            f"({time.time() - t0:.0f}s)")
    else:
        Pt, model, rows = [], None, []
        for k, t in enumerate(st.test_origins):
            if k % every == 0:
                model, n = st.train_until(t, rounds=300)
                rows.append((str(st.days[t].date()), n))
                log(f"retrained at {st.days[t].date()} on {n:,} rows")
            Pt.append(st.forecast(model, t))
        out = dict(Pt=Pt, retrain_log=rows)
    pd.to_pickle(out, path)
    return out


# --------------------------------------------------------------------- static

def optimize_static(st, idx, cfg, costs_p, n_sims=20, seed=11):
    """Paper baseline for many products at once: fit a hurdle-lognormal to
    year 1, simulate year-2-shaped years, pick the cheapest (s, S) per product
    from a grid scaled to each product (s = a x lead-time demand, Q = b x EOQ)."""
    lead = cfg.lead
    Y1 = st.M[idx, :st.y1_end + 1]
    wd1 = st.days[:st.y1_end + 1].dayofweek.to_numpy()
    open1 = Y1[:, wd1 != 5]
    pos = open1 > 0
    cnt = pos.sum(1)
    p_ = pos.mean(1)
    logs = np.log(np.where(pos, np.maximum(open1, 1), 1.0))
    mu_ = np.where(cnt > 0, (logs * pos).sum(1) / np.maximum(cnt, 1), 0.0)
    var_ = ((logs - mu_[:, None]) ** 2 * pos).sum(1) / np.maximum(cnt - 1, 1)
    sig_ = np.where(cnt >= 3, np.sqrt(var_), 0.75)        # too few points -> default spread
    mean_day = p_ * np.exp(mu_ + sig_ ** 2 / 2) * 6 / 7
    ltd = mean_day * (lead.min_days + lead.mean_extra)
    q_eoq = np.maximum(eoq(mean_day * 365, costs_p), 1)
    A = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 9]
    B = [0.25, 0.5, 1, 1.5, 2, 3, 5]
    T2 = st.last - st.y1_end
    wd2 = (st.days[st.y1_end + 1].dayofweek + np.arange(T2)) % 7
    rng = np.random.default_rng(seed)
    n = len(idx)
    best_cost = np.full(n, np.inf)
    best_s = np.zeros(n, dtype=np.int64)
    best_S = np.zeros(n, dtype=np.int64)
    for lo in range(0, n, 700):
        sl = slice(lo, min(lo + 700, n))
        k = sl.stop - sl.start
        pw, mw, sw = (np.repeat(v[sl], n_sims) for v in (p_, mu_, sig_))
        happens = rng.random((k * n_sims, T2)) < pw[:, None]
        size = np.maximum(np.rint(rng.lognormal(mw[:, None], sw[:, None], (k * n_sims, T2))), 1)
        dem = np.where(happens & (wd2 != 5)[None, :], size, 0).astype(np.int64)
        del happens, size
        sc = Scenarios(dem, lead.sample(k * n_sims, T2, rng))
        cw = cfg.costs(np.repeat(st.prices[idx][sl], n_sims))
        for a in A:
            for b in B:
                s = np.rint(a * ltd[sl]).astype(np.int64)
                S = s + np.maximum(np.rint(b * q_eoq[sl]), 1).astype(np.int64)
                r = simulate(SSPolicy(np.repeat(s, n_sims), np.repeat(S, n_sims)), sc, cw,
                             start_inventory=np.repeat(S, n_sims))
                tc = terminal_cost(r, cw, np.repeat(mean_day[sl], n_sims), lead.min_days + lead.mean_extra)
                c = (r.operating_cost + tc).reshape(k, n_sims).mean(1)
                better = c < best_cost[sl]
                best_cost[sl] = np.where(better, c, best_cost[sl])
                best_s[sl] = np.where(better, s, best_s[sl])
                best_S[sl] = np.where(better, S, best_S[sl])
    return best_s, best_S, dict(ltd=ltd, eoq=q_eoq, A=A, B=B)


# --------------------------------------------------------------------- evaluate

def _summ(res, n, R, costs, rate, lead_mean):
    """Per-product means over R lead-time draws. cost = in-window operating cost
    + terminal cost of leftover stock (headline); in_window kept separately."""
    m = lambda x: np.asarray(x).reshape(n, R).mean(1)
    term = terminal_cost(res, costs, rate, lead_mean)
    return dict(cost=m(res.operating_cost + term), in_window=m(res.operating_cost), terminal=m(term),
                hold=m(res.holding_cost), order=m(res.ordering_cost),
                back=m(res.backorder_cost), demand=m(res.demand), shipped=m(res.shipped_from_stock),
                stockout_days=m(res.stockout_days), orders=m(res.n_orders))


def evaluate(cfg: Config, R=5, R_val=2):
    t0 = time.time()
    lead = cfg.lead
    lead_mean = lead.min_days + lead.mean_extra
    st = Setup(lead)
    fc = prepare_forecasts(lead)
    fb, M = st.fb, st.M
    rep = lambda a, r=R: np.repeat(a, r, axis=0)

    # ---------- validation: tune per tier on two folds
    # a setting = (target, kappa, family); family "band" = band-average optimal reorder
    # point, "quantile" = reorder point at the target quantile of lead-time demand.
    # The family is tuned like any other setting, per tier, on validation only.
    targets = [0.6, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998]
    kappas = [0.5, 0.75, 1.0, 1.5, 2.0]
    keys = ([(tg, kp, "band") for tg in targets for kp in kappas]
            + [(tg, kp, "quantile") for tg in [0.5, 0.75, 0.85, 0.9, 0.95, 0.98, 0.99] for kp in kappas])
    fold_tot = {}                                   # fold -> {setting: per-tier totals}
    for f_start, f_origins, f_key in st.folds:
        est_v = np.where(fb.first_sale <= f_start - 90)[0]
        vdays = len(f_origins) * 7
        Pv, _ = online_calibration(fc[f_key], f_origins, fb)
        Pv = [P[est_v] for P in Pv]
        cost_v = cfg.costs(st.prices[est_v])
        scen_v = Scenarios.from_matrix(rep(M[est_v, f_start:f_start + vdays], R_val), lead, seed=5)
        od = cfg.opening_days
        open_v = rep(M[est_v, f_start - od:f_start].sum(1).astype(np.int64), R_val)
        cost_vw = cfg.costs(rep(st.prices[est_v], R_val))
        tier_v = fb.tier(f_start)[est_v]
        rate_v = rep(M[est_v, f_start:f_start + vdays].mean(1), R_val)
        fold_tot[f_key] = {}
        for tg, kp, fam in keys:
            s, S = LEVELS[fam](Pv, cost_v, tg, kp, vdays)
            r = simulate(DynamicSSPolicy(rep(s, R_val), rep(S, R_val)), scen_v, cost_vw,
                         start_inventory=open_v)
            c = (r.operating_cost + terminal_cost(r, cost_vw, rate_v, lead_mean)).reshape(len(est_v), R_val).mean(1)
            fold_tot[f_key][(tg, kp, fam)] = np.array([c[tier_v == g].sum() for g in TIERS])
    combined = {k: sum(fold_tot[f][k] for f in fold_tot) for k in keys}
    band_keys = [k for k in keys if k[2] == "band"]
    global_best = min(band_keys, key=lambda k: combined[k].sum())
    tier_best_band = {g: min(band_keys, key=lambda k: combined[k][g]) for g in TIERS}
    tier_best = {g: min(keys, key=lambda k: combined[k][g]) for g in TIERS}
    foldB_only = {g: min(keys, key=lambda k: fold_tot["Pv"][k][g]) for g in TIERS}
    log(f"[{cfg.name}] tuned (both folds): global {global_best}, per tier "
        + ", ".join(f"{TIERS[g]} {tier_best[g]}" for g in TIERS)
        + f" | fold B alone would pick " + ", ".join(f"{foldB_only[g]}" for g in TIERS)
        + f" ({time.time() - t0:.0f}s)")

    # ---------- test setup
    est = np.where(fb.first_sale <= st.y1_end - 90)[0]
    n = len(est)
    T2 = st.last - st.y1_end
    Y2 = M[est, st.y1_end + 1:].astype(np.int64)
    scen = Scenarios.from_matrix(rep(Y2), lead, seed=2026)
    cost_p = cfg.costs(st.prices[est])
    cost_w = cfg.costs(rep(st.prices[est]))
    od = cfg.opening_days
    open_w = rep(M[est, st.y1_end + 1 - od:st.y1_end + 1].sum(1).astype(np.int64))
    tier_t = fb.tier(st.y1_end)[est]
    rate_w = rep(Y2.mean(1))
    summ = lambda res: _summ(res, n, R, cost_w, rate_w, lead_mean)
    results = {}

    # static
    s_st, S_st, info = optimize_static(st, est, cfg, cost_p)
    results["static"] = summ(simulate(SSPolicy(rep(s_st), rep(S_st)), scen, cost_w,
                                      start_inventory=open_w))
    log(f"[{cfg.name}] static done ({time.time() - t0:.0f}s)")

    # hindsight: same grid family, searched on the real year-2 orders
    Y2m = Y2.mean(1)
    grids = [(info["ltd"], info["eoq"]),
             (Y2m * (lead.min_days + lead.mean_extra), np.maximum(eoq(Y2m * 365, cost_p), 1))]
    hind, hc = None, np.full(n, np.inf)
    for g_ltd, g_eoq in grids:
        for a in info["A"]:
            for b in info["B"]:
                s = np.rint(a * g_ltd).astype(np.int64)
                S = s + np.maximum(np.rint(b * g_eoq), 1).astype(np.int64)
                r = summ(simulate(SSPolicy(rep(s), rep(S)), scen, cost_w, start_inventory=open_w))
                better = r["cost"] < hc
                hc = np.where(better, r["cost"], hc)
                hind = r if hind is None else {k: np.where(better, r[k], hind[k]) for k in r}
    results["hindsight"] = hind
    log(f"[{cfg.name}] hindsight done ({time.time() - t0:.0f}s)")

    # adaptive variants
    def run_adaptive(P_raw, setting_for_tier, keep_levels=False):
        Pc, _ = online_calibration(P_raw, st.test_origins, fb)
        Pe = [P[est] for P in Pc]
        s_all = np.zeros((n, T2), dtype=np.int64)
        S_all = np.zeros((n, T2), dtype=np.int64)
        for setting in set(setting_for_tier.values()):
            rows = np.isin(tier_t, [g for g, v in setting_for_tier.items() if v == setting])
            if not rows.any():
                continue
            s, S = LEVELS[setting[2]]([P[rows] for P in Pe], cfg.costs(st.prices[est][rows]),
                                      setting[0], setting[1], T2)
            s_all[rows], S_all[rows] = s, S
        if keep_levels:
            LAST_LEVELS["levels"] = (s_all, S_all)
        return summ(simulate(DynamicSSPolicy(rep(s_all), rep(S_all)), scen, cost_w,
                             start_inventory=open_w))

    cr = min(cfg.critical_ratio(), 0.999)
    results["adaptive_textbook"] = run_adaptive(fc["Pt"], {g: (cr, 1.0, "band") for g in TIERS})
    results["adaptive_global"] = run_adaptive(fc["Pt"], {g: global_best for g in TIERS})
    results["adaptive_tiered"] = run_adaptive(fc["Pt"], tier_best, keep_levels=True)
    # dead-product rule: no orders after RULE_WEEKS weeks without a sale
    cs = np.concatenate([np.zeros((n, 1)), M[est].cumsum(1)], 1)
    days_idx = st.y1_end + 1 + np.arange(T2)
    alive = (cs[:, days_idx + 1] - cs[:, np.maximum(days_idx + 1 - 7 * RULE_WEEKS, 0)]) > 0
    s_full = np.repeat(s_st[:, None], T2, 1)
    S_full = np.repeat(S_st[:, None], T2, 1)
    results["static_rule"] = summ(simulate(DynamicSSPolicy(rep(np.where(alive, s_full, -10**12)), rep(S_full)),
                                           scen, cost_w, start_inventory=open_w))
    sa, Sa = LAST_LEVELS["levels"]
    results["adaptive_tiered_rule"] = summ(simulate(DynamicSSPolicy(rep(np.where(alive, sa, -10**12)), rep(Sa)),
                                                    scen, cost_w, start_inventory=open_w))
    results["adaptive_tiered_band_only"] = run_adaptive(fc["Pt"], tier_best_band)
    results["adaptive_tiered_foldB_only"] = run_adaptive(fc["Pt"], foldB_only)
    if cfg.retrain:
        fr = prepare_forecasts(lead, retrain=True)
        results["adaptive_tiered_retrained"] = run_adaptive(fr["Pt"], tier_best)
    log(f"[{cfg.name}] adaptive done ({time.time() - t0:.0f}s)")

    # ---------- save
    df = pd.DataFrame({"sku": st.skus[est], "price": st.prices[est], "tier": tier_t,
                       "y1_units": M[est, :st.y1_end + 1].sum(1), "y2_units": Y2.sum(1),
                       "static_s": s_st, "static_S": S_st})
    df = pd.concat([df, pd.DataFrame({f"{name}__{k}": v for name, r in results.items()
                                      for k, v in r.items()})], axis=1)
    df.to_parquet(RES / f"{cfg.name}.parquet")
    summary = dict(config=asdict(cfg), critical_ratio=cfg.critical_ratio(),
                   tuned_global=list(global_best),
                   tuned_tiers={TIERS[g]: list(v) for g, v in tier_best.items()},
                   tuned_tiers_foldB_only={TIERS[g]: list(v) for g, v in foldB_only.items()},
                   tuned_tiers_band_only={TIERS[g]: list(v) for g, v in tier_best_band.items()},
                   n_products=int(n), seconds=round(time.time() - t0),
                   totals={p: dict(cost=float(df[f"{p}__cost"].sum()),
                                   in_window=float(df[f"{p}__in_window"].sum()),
                                   terminal=float(df[f"{p}__terminal"].sum()),
                                   fill=float(df[f"{p}__shipped"].sum() / df[f"{p}__demand"].sum()),
                                   hold=float(df[f"{p}__hold"].sum()),
                                   order=float(df[f"{p}__order"].sum()),
                                   back=float(df[f"{p}__back"].sum()))
                           for p in results})
    (RES / f"{cfg.name}.json").write_text(json.dumps(summary, indent=2))
    base = summary["totals"]["static"]["cost"]
    log(f"[{cfg.name}] " + " | ".join(f"{p} £{v['cost']:,.0f} ({v['cost'] / base - 1:+.1%}, fill {v['fill']:.1%})"
                                      for p, v in summary["totals"].items()))
    return df, summary


LEVELS = {"band": weekly_levels_optimal, "quantile": weekly_levels}
RULE_WEEKS = 1          # dead-product rule window, best in the N = 1..8 sweep (dead_rule.py)
LAST_LEVELS = {}

CONFIGS = {
    "base": Config("base", retrain=True),
    "order_cost_2": Config("order_cost_2", order_cost=2),
    "order_cost_10": Config("order_cost_10", order_cost=10),
    "order_cost_20": Config("order_cost_20", order_cost=20),
    "order_cost_40": Config("order_cost_40", order_cost=40),
    "purchase_40pct": Config("purchase_40pct", purchase_share=0.40),
    "purchase_85pct": Config("purchase_85pct", purchase_share=0.85),
    "backorder_1pct": Config("backorder_1pct", backorder_rate=0.01),
    "backorder_20pct": Config("backorder_20pct", backorder_rate=0.20),
    "long_lead": Config("long_lead", lead_min=10, lead_extra=4.0),
    # researched values (see Project doc inventory-sim/data-sources-for-assumptions.md):
    # purchase 70% of price (Census wholesale margins), ~£3.50 per order line (APQC $42/PO
    # over 7.8 lines), holding 20%/yr, backorder one-off 35% of price + 0.5%/day
    "calibrated": Config("calibrated", order_cost=3.5, purchase_share=0.70, holding_rate=0.20,
                         backorder_rate=0.005, backorder_once=0.35),
    # same, with China-sourced lead times (~45-70 days)
    "calibrated_china": Config("calibrated_china", order_cost=3.5, purchase_share=0.70,
                               holding_rate=0.20, backorder_rate=0.005, backorder_once=0.35,
                               lead_min=40, lead_extra=15.0),
}

if __name__ == "__main__":
    import sys
    names = sys.argv[1:] or list(CONFIGS)
    for nm in names:
        cfg = CONFIGS[nm]
        if (RES / f"{nm}.json").exists():
            log(f"[{nm}] already done, skipping")
            continue
        evaluate(cfg)
