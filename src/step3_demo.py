"""Step 3: static baseline on 4 real products, then replay real year-2 demand."""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from simulator import simulate, Scenarios, SSPolicy
from monte_carlo import estimate
from static_baseline import (HurdleLognormalDemand, costs_for, LEAD, optimize_ss,
                             opening_stock, total_cost)

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
BLUE, GREY, RED, GOLD = "#2a6f97", "#8d99ae", "#c1121f", "#e09f3e"

PRODUCTS = {  # sku: role
    "85099B": "Steady seller",
    "85123A": "Demand fell in year 2",
    "22633": "Seasonal (winter)",
    "21985": "Lumpy bulk orders",
}

daily = pd.read_parquet(ROOT / "data" / "daily_demand.parquet")
prods = pd.read_parquet(ROOT / "data" / "products.parquet").set_index("StockCode")
Y1 = pd.date_range("2009-12-01", "2010-11-30")
Y2 = pd.date_range("2010-12-01", "2011-12-09")


def series(sku, dates):
    s = daily[daily.StockCode == sku].set_index("date").qty
    return s.reindex(dates, fill_value=0).to_numpy().astype(np.int64)


rows, figdata = [], []
t0 = time.time()
for k, (sku, role) in enumerate(PRODUCTS.items()):
    y1, y2 = series(sku, Y1), series(sku, Y2)
    price = prods.loc[sku, "price"]
    costs = costs_for(price)

    # 1. fit on year 1 only
    model = HurdleLognormalDemand.fit(y1, Y1.dayofweek.to_numpy())
    model.start_weekday = Y2[0].dayofweek       # simulate a year shaped like year 2

    # 2. optimize (s, S) on 1,000 simulated years
    search = Scenarios.generate(1000, len(Y2), model, LEAD, seed=100 + k)
    policy, (sg, qg, cmat) = optimize_ss(search, costs, y1.mean())

    # evaluate on 5,000 FRESH simulated years (avoid optimizer's lucky draw)
    pred_scen = Scenarios.generate(5000, len(Y2), model, LEAD, seed=900 + k)
    pred = simulate(policy, pred_scen, costs, start_inventory=policy.S)

    # 3. reality: replay the actual year-2 orders (500 lead-time draws)
    open2 = opening_stock(y1)                    # same opening stock for every year-2 run
    scen2 = Scenarios.replay(y2, LEAD, 500, seed=7)
    rep2 = simulate(policy, scen2, costs, start_inventory=open2, keep_trace=True)
    # in-sample check: replay year 1 itself
    scen1 = Scenarios.replay(y1, LEAD, 500, seed=7)
    rep1 = simulate(policy, scen1, costs, start_inventory=policy.S)
    # hindsight: best static (s,S) if you had known year 2 in advance
    hind_scen = Scenarios.replay(y2, LEAD, 200, seed=8)
    hind_policy, _ = optimize_ss(hind_scen, costs, y2.mean(), start=open2)
    hind = simulate(hind_policy, scen2, costs, start_inventory=open2)
    if total_cost(hind, costs, scen2).mean() > total_cost(rep2, costs, scen2).mean():
        hind_policy, hind = policy, rep2                 # search can miss; never worse than static

    def row(label, r, pol, sc):
        tc = total_cost(r, costs, sc)
        return dict(sku=sku, role=role, scenario=label, s=pol.s, S=pol.S,
                    cost=tc.mean(), cost_se=estimate(tc).se,
                    fill=r.fill_rate.mean(), stockout_days=r.stockout_days.mean(),
                    backorder=r.backorder_cost.mean(), holding=r.holding_cost.mean())
    for lab, r, pol, sc in [("Model's prediction", pred, policy, pred_scen),
                            ("Replay year 1 (in-sample)", rep1, policy, scen1),
                            ("Replay year 2 (reality)", rep2, policy, scen2),
                            ("Hindsight best for year 2", hind, hind_policy, scen2)]:
        rows.append(row(lab, r, pol, sc))
    print(f"{sku} {prods.loc[sku, 'description'][:30]:30} price £{price:.2f}  "
          f"fit p={model.p:.2f} mu={model.mu:.2f} sigma={model.sigma:.2f}  "
          f"policy (s,S)=({policy.s},{policy.S})  hindsight ({hind_policy.s},{hind_policy.S})")
    figdata.append((sku, role, y1, y2, model, policy, rep2, price))

print(f"\nfinished in {time.time() - t0:.0f}s\n")
res = pd.DataFrame(rows)
res.to_csv(OUT / "step3_results.csv", index=False)
pd.set_option("display.width", 200)
show = res.copy()
show["cost"] = show.cost.map("£{:,.0f}".format)
show["fill"] = show.fill.map("{:.1%}".format)
show["stockout_days"] = show.stockout_days.round(1)
show["holding"] = show.holding.map("£{:,.0f}".format)
show["backorder"] = show.backorder.map("£{:,.0f}".format)
print(show[["sku", "role", "scenario", "cost", "holding", "backorder", "fill", "stockout_days"]].to_string(index=False))

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(len(PRODUCTS), 3, figsize=(15, 3.3 * len(PRODUCTS)),
                         gridspec_kw=dict(width_ratios=[1.6, 1, 1.1]))
for (sku, role, y1, y2, model, policy, rep2, price), ax in zip(figdata, axes):
    name = prods.loc[sku, "description"].title()
    # (a) weekly demand, both years
    wk = pd.Series(np.r_[y1, y2], index=Y1.append(Y2)).resample("W").sum()
    ax[0].plot(wk.index, wk.values, color=GREY, lw=1)
    ax[0].axvline(Y2[0], color="black", lw=0.8, ls="--")
    mean_wk = model.mean_per_open_day * 6
    ax[0].hlines(mean_wk, Y1[0], Y2[-1], color=BLUE, lw=1.5,
                 label="what the static model assumes, forever")
    ax[0].set_title(f"{name}  ·  {role}", loc="left", fontsize=10.5)
    ax[0].set_ylabel("Units / week")
    ax[0].text(0.01, 0.93, "Year 1 (fit)", transform=ax[0].transAxes, fontsize=8)
    ax[0].text(0.51, 0.93, "Year 2 (unseen)", transform=ax[0].transAxes, fontsize=8)
    ax[0].legend(frameon=False, fontsize=7.5, loc="upper right", bbox_to_anchor=(1, 0.88))
    ax[0].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %y"))

    # (b) order-size distribution: data vs fitted lognormal
    pos = np.log10(y1[y1 > 0])
    ax[1].hist(pos, bins=30, density=True, color=GREY, alpha=0.8, label="Year 1 days with demand")
    xs = np.linspace(0, pos.max() * 1.05, 200)
    # lognormal in natural log -> normal in log10 units
    ax[1].plot(xs, stats.norm.pdf(xs, model.mu / np.log(10), model.sigma / np.log(10)),
               color=BLUE, lw=1.5, label="Fitted lognormal")
    ax[1].set_xticks(range(int(np.ceil(xs.max())) + 1),
                     [f"{10**i:,}" for i in range(int(np.ceil(xs.max())) + 1)])
    ax[1].set_xlabel("Units that day (log scale)", fontsize=8)
    ax[1].set_title(f"Daily volume when it sells (p={model.p:.0%} of open days)",
                    loc="left", fontsize=9.5)
    ax[1].legend(frameon=False, fontsize=7.5)
    ax[1].set_yticks([])

    # (c) predicted vs real cost
    r = res[res.sku == sku]
    colors = [BLUE, GREY, RED, GOLD]
    ax[2].barh(range(4)[::-1], r.cost, xerr=1.96 * r.cost_se, color=colors, alpha=0.85)
    for yy, (c, f) in zip(range(4)[::-1], zip(r.cost, r.fill)):
        ax[2].text(c, yy, f"  £{c:,.0f} · fill {f:.0%}", va="center", fontsize=8)
    ax[2].set_yticks(range(4)[::-1], r.scenario, fontsize=8)
    ax[2].set_xlim(0, r.cost.max() * 1.6)
    ax[2].set_title(f"Operating cost, policy (s,S)=({policy.s:,},{policy.S:,})",
                    loc="left", fontsize=9.5)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
axes[-1][2].set_xlabel("£ per year")
fig.tight_layout()
fig.savefig(OUT / "step3_static_baseline.png", dpi=130)
print("\nsaved", OUT / "step3_static_baseline.png")
