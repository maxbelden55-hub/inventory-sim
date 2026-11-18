"""
Dead-product rule: does a simple "no sales in N weeks -> stop reordering" rule
capture the adaptive policy's saving without any ML?

For N = 1..8 weeks, on the SAME year-2 orders, lead-time draws, opening stock
and static (s, S) settings as the base case (data/results/base.parquet):
  static + rule    the paper's static policy, but no order is placed while a
                   product has had zero sales in the trailing N weeks
  adaptive + rule  the base-case adaptive policy with the same gate
Picking the best N from this sweep uses year-2 outcomes (hindsight), which
favours the baseline; the ML policy never saw year 2.
"""
import json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiment import Setup, Config, prepare_forecasts, TIERS, LEVELS, _summ, RES
from simulator import simulate, Scenarios, DynamicSSPolicy
from adaptive_policy import online_calibration

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
NS = list(range(1, 9))
R = 5
OFF = -10 ** 12          # reorder point that can never trigger an order


def main():
    t0 = time.time()
    cfg = Config()
    base = pd.read_parquet(RES / "base.parquet")
    summary = json.loads((RES / "base.json").read_text())
    st = Setup(cfg.lead)
    fb, M = st.fb, st.M
    lead_mean = cfg.lead.min_days + cfg.lead.mean_extra
    rep = lambda a: np.repeat(a, R, axis=0)

    est = np.where(fb.first_sale <= st.y1_end - 90)[0]
    assert (st.skus[est] == base.sku.to_numpy()).all(), "product order differs from base run"
    n, T2 = len(est), st.last - st.y1_end
    Y2 = M[est, st.y1_end + 1:].astype(np.int64)
    scen = Scenarios.from_matrix(rep(Y2), cfg.lead, seed=2026)
    cost_w = cfg.costs(rep(st.prices[est]))
    open_w = rep(M[est, st.y1_end - 27:st.y1_end + 1].sum(1).astype(np.int64))
    rate_w = rep(Y2.mean(1))
    summ = lambda r: _summ(r, n, R, cost_w, rate_w, lead_mean)

    # static levels from the base run (constant over the year)
    s_st = np.repeat(base.static_s.to_numpy()[:, None], T2, 1)
    S_st = np.repeat(base.static_S.to_numpy()[:, None], T2, 1)

    # adaptive levels, rebuilt exactly as in the base run
    fc = prepare_forecasts(cfg.lead)
    Pc, _ = online_calibration(fc["Pt"], st.test_origins, fb)
    Pe = [P[est] for P in Pc]
    tier_t = fb.tier(st.y1_end)[est]
    tuned = {g: tuple(summary["tuned_tiers"][TIERS[g]]) for g in TIERS}
    s_ad = np.zeros((n, T2), dtype=np.int64)
    S_ad = np.zeros((n, T2), dtype=np.int64)
    for setting in set(tuned.values()):
        rows = np.isin(tier_t, [g for g, v in tuned.items() if v == setting])
        s, S = LEVELS[setting[2]]([P[rows] for P in Pe], cfg.costs(st.prices[est][rows]),
                                  setting[0], setting[1], T2)
        s_ad[rows], S_ad[rows] = s, S

    def run(s, S):
        return summ(simulate(DynamicSSPolicy(rep(s), rep(S)), scen, cost_w, start_inventory=open_w))

    # sanity: no gate reproduces the base run
    ref_static, ref_adapt = run(s_st, S_st), run(s_ad, S_ad)
    for name, r in (("static", ref_static), ("adaptive_tiered", ref_adapt)):
        assert np.isclose(r["cost"].sum(), summary["totals"][name]["cost"], rtol=1e-9), name
    print(f"reproduced base run: static £{ref_static['cost'].sum():,.0f}, "
          f"adaptive £{ref_adapt['cost'].sum():,.0f} ({time.time() - t0:.0f}s)")

    # trailing sales over the last 7N days, known at the end of each test day
    cs = np.concatenate([np.zeros((n, 1)), M[est].cumsum(1)], 1)
    days_idx = st.y1_end + 1 + np.arange(T2)                 # calendar index of each test day
    rows_out, per_n = [], {}
    dead = (Y2.sum(1) == 0)
    for N in NS:
        w = 7 * N
        trailing = cs[:, days_idx + 1] - cs[:, np.maximum(days_idx + 1 - w, 0)]
        alive = trailing > 0
        r_st = run(np.where(alive, s_st, OFF), S_st)
        r_ad = run(np.where(alive, s_ad, OFF), S_ad)
        per_n[N] = (r_st, r_ad)
        for pol, r in (("static + rule", r_st), ("adaptive + rule", r_ad)):
            rows_out.append(dict(N_weeks=N, policy=pol, cost=r["cost"].sum(),
                                 fill=r["shipped"].sum() / r["demand"].sum(),
                                 cost_dead_products=r["cost"][dead].sum(),
                                 cost_active_products=r["cost"][~dead].sum()))
        print(f"N={N}: static+rule £{r_st['cost'].sum():,.0f} | adaptive+rule £{r_ad['cost'].sum():,.0f}")

    df = pd.DataFrame(rows_out)
    base_static, base_adapt = ref_static["cost"].sum(), ref_adapt["cost"].sum()
    df["vs_static"] = df.cost / base_static - 1
    df["vs_adaptive"] = df.cost / base_adapt - 1
    df.to_csv(OUT / "dead_rule_sweep.csv", index=False)

    sr = df[df.policy == "static + rule"].set_index("N_weeks")
    ar = df[df.policy == "adaptive + rule"].set_index("N_weeks")
    best_n = int(sr.cost.idxmin())
    extra = dict(base_static=base_static, base_adaptive=base_adapt,
                 base_static_fill=ref_static["shipped"].sum() / ref_static["demand"].sum(),
                 base_adaptive_fill=ref_adapt["shipped"].sum() / ref_adapt["demand"].sum(),
                 base_static_dead=ref_static["cost"][dead].sum(), base_adaptive_dead=ref_adapt["cost"][dead].sum(),
                 base_static_active=ref_static["cost"][~dead].sum(), base_adaptive_active=ref_adapt["cost"][~dead].sum(),
                 best_N_static_rule=best_n, best_static_rule_cost=float(sr.cost.min()),
                 share_of_ml_saving_captured=float((base_static - sr.cost.min()) / (base_static - base_adapt)),
                 best_N_adaptive_rule=int(ar.cost.idxmin()), best_adaptive_rule_cost=float(ar.cost.min()),
                 dead_products=int(dead.sum()))
    (OUT / "dead_rule_summary.json").write_text(json.dumps(extra, indent=1))
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, formatters={"cost": "£{:,.0f}".format, "fill": "{:.1%}".format,
          "cost_dead_products": "£{:,.0f}".format, "cost_active_products": "£{:,.0f}".format,
          "vs_static": "{:+.1%}".format, "vs_adaptive": "{:+.1%}".format}))
    print(json.dumps(extra, indent=1))

    # figure
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axhline(base_static / 1e3, color="#8d99ae", lw=1, ls="--")
    ax.text(0.7, base_static / 1e3 + 1.2, "Static policy, no rule", va="bottom", fontsize=8, color="#5c677d")
    ax.axhline(base_adapt / 1e3, color="#2a6f97", lw=1, ls="--")
    ax.text(0.7, base_adapt / 1e3 + 1.2, "Adaptive ML, no rule", va="bottom", fontsize=8, color="#2a6f97")
    ax.plot(NS, sr.cost / 1e3, marker="o", color="#8d99ae", label="Static + rule")
    ax.plot(NS, ar.cost / 1e3, marker="o", color="#2a6f97", label="Adaptive ML + rule")
    for N in NS:
        ax.annotate(f"£{sr.cost[N] / 1e3:,.0f}k", (N, sr.cost[N] / 1e3), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7.5, color="#5c677d")
    ax.set_xticks(NS)
    ax.set_xlim(0.6, 8.4)
    ax.set_xlabel("Stop reordering after N weeks with no sales", fontsize=9)
    ax.set_ylabel("Year-2 cost, £ thousands", fontsize=9)
    ax.set_title("Dead-product rule sweep: 3,799 products, real year-2 orders", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 0.82))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "dead_rule_sweep.png", dpi=130)
    print(f"saved outputs/dead_rule_sweep.png ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
