"""
Step 5: forecast -> optimized weekly ordering rules.

Timeline (no peeking):
  Dec 2009 - Jun 2010  "early" model trains here
  Jul 2010 - Nov 2010  validation: recalibrate quantiles + tune tau, kappa
  Dec 2010 - Dec 2011  test: full year-1 model + chosen settings, replay real orders
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from forecaster import load_matrix, FeatureBuilder, build_panel, QuantileForecaster, QUANTILES
from simulator import simulate, Scenarios, DynamicSSPolicy, terminal_cost
from static_baseline import costs_for, LEAD, opening_stock
from adaptive_policy import (fit_calibration, apply_calibration, online_calibration, weekly_levels,
                             weekly_levels_optimal, critical_ratio)

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
BLUE, GREY, RED, GOLD = "#2a6f97", "#8d99ae", "#c1121f", "#e09f3e"
H7 = 7
SHOW = {"85099B": "Steady seller", "85123A": "Demand fell in year 2",
        "22633": "Seasonal (winter)", "21985": "Lumpy bulk orders"}

t0 = time.time()
M, skus, days, prices = load_matrix()
fb = FeatureBuilder(M, days, prices, lead=LEAD, seed=1)   # targets = demand over a random lead time
ix = lambda d: days.get_loc(pd.Timestamp(d))
val_start, y1_end, last = ix("2010-07-01"), ix("2010-11-30"), len(days) - 1


def forecasts(model, origins):
    """Forecast for EVERY product at each origin (unlaunched products get 0)."""
    out = []
    for t in origins:
        X = fb.at(t)
        P = model.predict(X)
        P[fb.first_sale > t] = 0
        out.append(P)
    return out


# ---------------------------------------------------------------- 1. early model
early_origins = list(range(28, val_start - 14, 7))
val_origins = list(range(val_start - 1, y1_end - 14 + 1, 7))
Xe, ye, _ = build_panel(fb, early_origins, skus)
early = QuantileForecaster().fit(Xe, ye)
Pv_raw = forecasts(early, val_origins)
print(f"early model trained on {len(ye):,} rows ({time.time() - t0:.0f}s)")

# ---------------------------------------------------------------- 2. recalibrate
yv = np.concatenate([fb.target(t) for t in val_origins])
Pv_flat = np.vstack(Pv_raw)
live = np.concatenate([fb.first_sale <= t for t in val_origins])
shifts = fit_calibration(Pv_flat[live], yv[live])
cov = lambda P, y: 0.5 * ((y[:, None] < P).mean(0) + (y[:, None] <= P).mean(0))
print("coverage before:", np.round(cov(Pv_flat[live], yv[live]), 2))
print("coverage after: ", np.round(cov(apply_calibration(Pv_flat[live], shifts), yv[live]), 2),
      " (target", QUANTILES, ")")
Pv, _ = online_calibration(Pv_raw, val_origins, fb)
Pv_cal_flat = np.vstack(Pv)
print("coverage, online-recalibrated:", np.round(cov(Pv_cal_flat[live], yv[live]), 2))

# ---------------------------------------------------------------- 3. tune tau, kappa
est = fb.first_sale <= val_start - 90                      # established products
vdays = len(val_origins) * 7
v_dem = M[est, val_start:val_start + vdays].astype(np.int64)
v_costs = costs_for(prices[est])
v_scen = Scenarios.from_matrix(v_dem, LEAD, seed=5)
print(f"tuning on {est.sum():,} products x {vdays} validation days")


v_open = M[est, val_start - 28:val_start].sum(1).astype(np.int64)


def run(Plist, costs, knob, kappa, scen, mask=None, optimal=True):
    Pl = [P[mask] for P in Plist] if mask is not None else Plist
    f = weekly_levels_optimal if optimal else weekly_levels
    s, S = f(Pl, costs, knob, kappa, scen.days)
    r = simulate(DynamicSSPolicy(s, S), scen, costs, start_inventory=v_open)
    r.total = r.operating_cost + terminal_cost(r, costs, scen.demand.mean(1), 7.0)   # + leftover stock
    return r


cr = float(np.median(critical_ratio(v_costs)))
kappas = [0.5, 1.0, 1.5]
# Family A: reorder point = a fixed forecast quantile tau (simple rule)
taus = [0.5, 0.65, 0.75, 0.85, 0.95]
gridA = np.array([[run(Pv, v_costs, t, k, v_scen, est, optimal=False).total.sum()
                   for k in kappas] for t in taus])
# Family B: per-product optimal reorder point for a service target
targets = [0.90, 0.95, 0.98, cr, 0.997]
grid = np.array([[run(Pv, v_costs, t, k, v_scen, est).total.sum()
                  for k in kappas] for t in targets])
ia, ja = np.unravel_index(gridA.argmin(), gridA.shape)
i, j = np.unravel_index(grid.argmin(), grid.shape)
TARGET, KAPPA = targets[i], kappas[j]
textbookA = run(Pv, v_costs, cr, 1.0, v_scen, est, optimal=False).total.sum()
textbookB = grid[targets.index(cr), kappas.index(1.0)]
raw = run(Pv_raw, v_costs, TARGET, KAPPA, v_scen, est).total.sum()
print(f"\nvalidation total operating cost, {est.sum():,} products (critical ratio = {cr:.3f}):")
print(f"  A. simple rule, textbook (s = q{cr*100:.1f}, EOQ):        £{textbookA:,.0f}")
print(f"  A. simple rule, tuned (s = q{taus[ia]*100:g}, {kappas[ja]}x EOQ):      £{gridA.min():,.0f}")
print(f"  B. optimal reorder point, textbook (target {cr:.3f}, EOQ): £{textbookB:,.0f}")
print(f"  B. optimal reorder point, tuned (target {TARGET:.3f}, {KAPPA}x EOQ): £{grid.min():,.0f}")
print(f"  B tuned, but WITHOUT recalibration:                     £{raw:,.0f}")

# ---------------------------------------------------------------- 4. test on year 2
y1_origins = list(range(28, y1_end - 14 + 1, 7))
X1, y1, _ = build_panel(fb, y1_origins, skus)
full = QuantileForecaster().fit(X1, y1)
test_origins = list(range(y1_end, last - 7 + 1, 7))
Pt_raw = forecasts(full, test_origins)
Pt, test_shifts = online_calibration(Pt_raw, test_origins, fb)
yt = np.concatenate([fb.target(t) for t in test_origins])
lt = np.concatenate([fb.first_sale <= t for t in test_origins])
print("test coverage raw:   ", np.round(cov(np.vstack(Pt_raw)[lt], yt[lt]), 2))
print("test coverage online:", np.round(cov(np.vstack(Pt)[lt], yt[lt]), 2))
T2 = last - y1_end
step3 = pd.read_csv(OUT / "step3_results.csv")

rows, traces = [], {}
sku_i = {s: k for k, s in enumerate(skus)}
for sku, role in SHOW.items():
    k = sku_i[sku]
    dem = np.repeat(M[k:k + 1, y1_end + 1:], 500, 0).astype(np.int64)
    scen = Scenarios.from_matrix(dem, LEAD, seed=7)
    c = costs_for(prices[k])
    Pk = [np.repeat(P[k:k + 1], 500, 0) for P in Pt]
    s, S = weekly_levels_optimal(Pk, c, TARGET, KAPPA, T2)
    r = simulate(DynamicSSPolicy(s, S), scen, c, start_inventory=opening_stock(M[k, :y1_end + 1]),
                 keep_trace=True)
    r.total = r.operating_cost + terminal_cost(r, c, scen.demand.mean(1), 7.0)
    traces[sku] = (s[0], S[0], r.trace_net[0], M[k, y1_end + 1:])
    st = step3[step3.sku == sku].set_index("scenario")
    rows.append(dict(sku=sku, role=role,
                     static=st.loc["Replay year 2 (reality)", "cost"],
                     static_fill=st.loc["Replay year 2 (reality)", "fill"],
                     hindsight=st.loc["Hindsight best for year 2", "cost"],
                     adaptive=r.total.mean(), adaptive_fill=r.fill_rate.mean(),
                     a_hold=r.holding_cost.mean(), a_order=r.ordering_cost.mean(),
                     a_back=r.backorder_cost.mean(), a_orders=r.n_orders.mean(),
                     adaptive_stockout_days=r.stockout_days.mean()))
res = pd.DataFrame(rows)
res["saving_vs_static"] = 1 - res.adaptive / res.static
res.to_csv(OUT / "step5_results.csv", index=False)
pd.set_option("display.width", 200)
print("\nYEAR 2 replay (real orders, 500 lead-time draws each):")
print(res.round(3).to_string(index=False))
print(f"\ntotal {time.time() - t0:.0f}s")

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], hspace=0.6, wspace=0.35)
ax = fig.add_subplot(gs[0, 0:2])
im = ax.imshow(grid / 1e3, cmap="Blues_r", aspect="auto")
ax.set_xticks(range(len(kappas)), [f"{k}x" for k in kappas])
ax.set_yticks(range(len(targets)), [f"{t:.3f}" + (" (textbook)" if t == cr else "") for t in targets])
ax.set_xlabel("Order size, as a multiple of EOQ (kappa)", fontsize=8)
ax.set_ylabel("Service target b/(b+h)", fontsize=8)
for a in range(len(targets)):
    for b in range(len(kappas)):
        ax.text(b, a, f"{grid[a, b] / 1e3:,.0f}", ha="center", va="center", fontsize=7,
                color="white" if grid[a, b] < np.median(grid) else "black",
                fontweight="bold" if (a, b) == (i, j) else None)
ax.set_title("Tuning, optimal reorder point (validation Jul–Nov 2010): total £k",
             loc="left", fontsize=10)

ax = fig.add_subplot(gs[0, 2:4])
x = np.arange(len(res)); w = 0.27
ax.bar(x - w, res.static, w, color=GREY, label="Static (paper)")
ax.bar(x, res.adaptive, w, color=BLUE, label="Adaptive ML")
ax.bar(x + w, res.hindsight, w, color=GOLD, label="Best fixed policy in hindsight")
for xx, (a, sv) in enumerate(zip(res.adaptive, res.saving_vs_static)):
    ax.text(xx, a, f"{sv:+.0%}", ha="center", va="bottom", fontsize=8, color=BLUE)
ax.set_xticks(x, res.role, fontsize=8)
ax.set_ylabel("£ operating cost, year 2", fontsize=8)
ax.set_title("Year 2 replay on real orders (labels = saving vs static)", loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=7.5)

tdates = days[y1_end + 1:]
for n, (sku, role) in enumerate(SHOW.items()):
    s_, S_, net, dem = traces[sku]
    ax = fig.add_subplot(gs[1 + n // 2, (n % 2) * 2:(n % 2) * 2 + 2])
    ax.fill_between(tdates, s_, S_, step="post", color=BLUE, alpha=0.15, lw=0,
                    label="Band between reorder point s and order-up-to S")
    ax.step(tdates, s_, where="post", color=BLUE, lw=1, label="Reorder point s (moves weekly)")
    ax.plot(tdates, net, color="black", lw=0.8, label="Inventory (one sample run)")
    ax.axhline(0, color=RED, lw=0.6)
    ax.set_title(f"{role}: adaptive policy, year 2", loc="left", fontsize=10)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %y"))
    ax.tick_params(labelsize=8)
    if n == 0:
        ax.legend(frameon=False, fontsize=7, loc="upper left")
for a in fig.axes:
    a.spines[["top", "right"]].set_visible(False)
fig.savefig(OUT / "step5_adaptive_policy.png", dpi=130, bbox_inches="tight")
pd.to_pickle(dict(shifts=shifts, TARGET=TARGET, KAPPA=KAPPA, grid=grid, gridA=gridA,
                    targets=targets, taus=taus, kappas=kappas, cr=cr),
             ROOT / "data" / "step5_settings.pkl")
print("saved", OUT / "step5_adaptive_policy.png")
