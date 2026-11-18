"""
Demand-shock tests: feed the trained ML new, changed data and see how it responds.

The year-1 forecaster is trained once on real year-1 data and then frozen. In year 2,
from Monday 4 April 2011, demand for a random sample of 400 selling products is altered:
    surge      demand doubles overnight and stays doubled
    collapse   demand falls to 30% overnight
    trend      demand grows steadily to 2.5x by December
    spike      one surprise bulk order of 4 weeks' typical demand on a random day
The forecaster only sees the altered sales history (features are rebuilt from it); its
weights never change. Online recalibration, the weekly (s, S) policy and the dead-product
rule run exactly as in the base case. The static policy keeps its frozen year-1 settings.

Outputs: outputs/shock_test.csv, outputs/shock_test.png, outputs/shock_test_summary.json
"""
import json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiment import Setup, Config, TIERS, LEVELS, _summ, RES
from forecaster import FeatureBuilder
from adaptive_policy import online_calibration, mean_from_quantiles
from simulator import simulate, Scenarios, DynamicSSPolicy

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
R, N_SHOCK, SEED = 5, 400, 7
SHOCK_DATE = "2011-04-04"
OFF = -10 ** 12


def apply_shock(M, rows, d0, kind, rng):
    M2 = M.copy()
    T = M.shape[1]
    if kind == "surge":
        M2[rows, d0:] = M[rows, d0:] * 2
    elif kind == "collapse":
        M2[rows, d0:] = rng.binomial(M[rows, d0:].astype(np.int64), 0.3)
    elif kind == "trend":
        mult = np.linspace(1.0, 2.5, T - d0)
        M2[rows, d0:] = np.rint(M[rows, d0:] * mult[None, :])
    elif kind == "spike":
        base = np.maximum(M[rows, d0 - 28:d0].sum(1) * 4, 50)          # ~4 weeks of demand
        day = rng.integers(d0, T - 30, len(rows))
        M2[rows, day] = M[rows, day] + base
    return M2


def main():
    t0 = time.time()
    cfg = Config()
    summary = json.loads((RES / "base.json").read_text())
    base = pd.read_parquet(RES / "base.parquet")
    st = Setup(cfg.lead)
    M, days, prices = st.M, st.days, st.prices
    lead_mean = cfg.lead.min_days + cfg.lead.mean_extra
    rng = np.random.default_rng(SEED)

    est = np.where(st.fb.first_sale <= st.y1_end - 90)[0]
    tier_t = st.fb.tier(st.y1_end)[est]
    y2 = M[est, st.y1_end + 1:].sum(1)
    pool = np.where((y2 > 0) & (tier_t >= 1))[0]                     # selling, not intermittent
    pick = np.sort(rng.choice(pool, N_SHOCK, replace=False))        # positions within est
    rows_M = est[pick]                                              # rows of M
    d0 = days.get_loc(pd.Timestamp(SHOCK_DATE))
    print(f"{len(pool):,} eligible products, {N_SHOCK} shocked from {SHOCK_DATE}")

    model, nrows = st.train_until(st.y1_end)                        # year-1 model, then frozen
    print(f"year-1 model trained on {nrows:,} rows ({time.time() - t0:.0f}s)")
    tuned = {g: tuple(summary["tuned_tiers"][TIERS[g]]) for g in TIERS}
    s_static = base.static_s.to_numpy()[pick]
    S_static = base.static_S.to_numpy()[pick]
    tiers = tier_t[pick]
    T2 = st.last - st.y1_end
    rep = lambda a: np.repeat(a, R, axis=0)

    rows_out, traces = [], {}
    for kind in ["none", "surge", "collapse", "trend", "spike"]:
        M2 = M if kind == "none" else apply_shock(M, rows_M, d0, kind, rng)
        fb = FeatureBuilder(M2, days, prices, lead=cfg.lead, seed=1)
        P_raw = []
        for t in st.test_origins:
            P = model.predict(fb.at(t))
            P[fb.first_sale > t] = 0
            P_raw.append(P)
        Pc, _ = online_calibration(P_raw, st.test_origins, fb)
        Pe = [P[rows_M] for P in Pc]

        s_ad = np.zeros((N_SHOCK, T2), dtype=np.int64)
        S_ad = np.zeros((N_SHOCK, T2), dtype=np.int64)
        for setting in set(tuned.values()):
            r = np.isin(tiers, [g for g, v in tuned.items() if v == setting])
            if r.any():
                s, S = LEVELS[setting[2]]([P[r] for P in Pe], cfg.costs(prices[rows_M][r]),
                                          setting[0], setting[1], T2)
                s_ad[r], S_ad[r] = s, S
        s_st = np.repeat(s_static[:, None], T2, 1)
        S_st = np.repeat(S_static[:, None], T2, 1)
        cs = np.concatenate([np.zeros((N_SHOCK, 1)), M2[rows_M].cumsum(1)], 1)
        di = st.y1_end + 1 + np.arange(T2)
        alive = (cs[:, di + 1] - cs[:, np.maximum(di + 1 - 7, 0)]) > 0

        Y2 = M2[rows_M, st.y1_end + 1:].astype(np.int64)
        scen = Scenarios.from_matrix(rep(Y2), cfg.lead, seed=2026)
        cost_w = cfg.costs(rep(prices[rows_M]))
        open_w = rep(M2[rows_M, st.y1_end + 1 - cfg.opening_days:st.y1_end + 1].sum(1).astype(np.int64))
        summ = lambda res: _summ(res, N_SHOCK, R, cost_w, rep(Y2.mean(1)), lead_mean)
        run = lambda s, S: summ(simulate(DynamicSSPolicy(rep(s), rep(S)), scen, cost_w, start_inventory=open_w))
        res = {"static": run(s_st, S_st), "static_rule": run(np.where(alive, s_st, OFF), S_st),
               "adaptive": run(s_ad, S_ad), "adaptive_rule": run(np.where(alive, s_ad, OFF), S_ad)}
        for pol, r in res.items():
            rows_out.append(dict(shock=kind, policy=pol, cost=r["cost"].sum(),
                                 fill=r["shipped"].sum() / r["demand"].sum(),
                                 backorder=r["back"].sum(), holding=r["hold"].sum(),
                                 stockout_days=r["stockout_days"].mean()))

        # response trace: forecast mean vs what actually happened, weekly, summed over shocked products
        fmean = np.array([mean_from_quantiles(P).sum() for P in Pe])
        actual = np.array([fb.target(t)[rows_M].sum() for t in st.test_origins])
        traces[kind] = (fmean, actual)
        print(f"{kind:8} static £{res['static']['cost'].sum():,.0f} | adaptive £{res['adaptive']['cost'].sum():,.0f} "
              f"| adaptive+rule £{res['adaptive_rule']['cost'].sum():,.0f} ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows_out)
    sc = df[df.policy == "static"].set_index("shock").cost
    df["vs_static"] = df.cost / df.shock.map(sc) - 1
    df.to_csv(OUT / "shock_test.csv", index=False)

    # adaptation speed: weeks after the shock until the forecast is within 20% of reality
    dates = days[st.test_origins] + pd.Timedelta(days=1)
    k0 = int(np.searchsorted(dates, pd.Timestamp(SHOCK_DATE)))
    speed = {}
    for kind in ["surge", "collapse", "trend"]:
        f, a = traces[kind]
        ratio = f[k0:] / np.maximum(a[k0:], 1)
        ok = np.where(np.abs(ratio - 1) <= 0.2)[0]
        # the first week from which it stays within 20% for 3 weeks running
        weeks = next((int(w) for w in ok if all(abs(r - 1) <= 0.2 for r in ratio[w:w + 3])), None)
        speed[kind] = weeks
    pre = slice(max(k0 - 8, 0), k0)
    none_ratio = float(traces["none"][0][pre].sum() / traces["none"][1][pre].sum())
    out = dict(shock_date=SHOCK_DATE, products=N_SHOCK, weeks_to_track_within_20pct=speed,
               pre_shock_forecast_to_actual=none_ratio)
    (OUT / "shock_test_summary.json").write_text(json.dumps(out, indent=1))
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, formatters={"cost": "£{:,.0f}".format, "fill": "{:.1%}".format,
          "backorder": "£{:,.0f}".format, "holding": "£{:,.0f}".format, "vs_static": "{:+.1%}".format,
          "stockout_days": "{:.1f}".format}))
    print(json.dumps(out, indent=1))

    # figure: forecast vs reality around the shock, one panel per shock
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), sharex=True)
    names = {"surge": "Demand doubles", "collapse": "Demand falls to 30%",
             "trend": "Demand grows to 2.5x", "spike": "One surprise bulk order"}
    for ax, kind in zip(axes.flat, ["surge", "collapse", "trend", "spike"]):
        f, a = traces[kind]
        fn, an = traces["none"]
        norm = an[max(k0 - 8, 0):k0].mean()
        ax.plot(dates, an / norm, color="#c9ced6", lw=1, label="Actual, no shock")
        ax.plot(dates, a / norm, color="black", lw=1.3, label="Actual demand (next lead time)")
        ax.plot(dates, f / norm, color="#2a6f97", lw=1.6, label="ML forecast (mean)")
        ax.axvline(pd.Timestamp(SHOCK_DATE), color="#c1121f", lw=0.8, ls="--")
        d = df[df.shock == kind].set_index("policy")
        title = (f"{names[kind]}: adaptive {d.loc['adaptive', 'vs_static']:+.0%}, "
                 f"adaptive + rule {d.loc['adaptive_rule', 'vs_static']:+.0%} vs static")
        if speed.get(kind) is not None:
            title += f"; tracks within {speed[kind]} wk"
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.set_ylabel("Demand index (pre-shock = 1)", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %y"))
        ax.tick_params(labelsize=8)
    axes[0, 0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle(f"How the frozen year-1 ML responds to new demand: {N_SHOCK} products shocked from 4 Apr 2011",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "shock_test.png", dpi=130)
    print(f"saved outputs/shock_test.png ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
