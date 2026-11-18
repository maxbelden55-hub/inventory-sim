"""Step 4: train the global forecaster on year 1, test on year 2."""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from forecaster import (load_matrix, FeatureBuilder, build_panel, QuantileForecaster,
                        static_baseline, moving_window_baseline, pinball, QUANTILES, H)

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
BLUE, GREY, RED, GOLD = "#2a6f97", "#8d99ae", "#c1121f", "#e09f3e"
SHOW = {"85099B": "Steady seller", "85123A": "Demand fell in year 2",
        "22633": "Seasonal (winter)", "21985": "Lumpy bulk orders"}

t0 = time.time()
M, skus, days, prices = load_matrix()
fb = FeatureBuilder(M, days, prices)
y1_end = days.get_loc(pd.Timestamp("2010-11-30"))
last = len(days) - 1
train_origins = list(range(28, y1_end - H + 1, 7))
test_origins = list(range(y1_end, last - H + 1, 7))

Xtr, ytr, mtr = build_panel(fb, train_origins, skus)
Xte, yte, mte = build_panel(fb, test_origins, skus)
print(f"train rows {len(ytr):,} ({len(train_origins)} weekly origins, year 1 only)")
print(f"test rows  {len(yte):,} ({len(test_origins)} weekly origins, year 2)")

qf = QuantileForecaster().fit(Xtr, ytr)
P_ml = qf.predict(Xte)
print(f"trained 7 quantile models in {time.time() - t0:.0f}s")

# baselines, aligned to test rows
sku_idx = pd.Series(np.arange(len(skus)), index=skus)
rows = sku_idx[mte.sku].to_numpy()
P_static = static_baseline(fb, y1_end + 1)[rows]
mw = {t: moving_window_baseline(fb, t) for t in test_origins}
P_mw = np.vstack([mw[t][sku_idx[g.sku].to_numpy()] for t, g in mte.groupby("t", sort=False)])
# (groupby keeps row order because build_panel stacked origins in order)

established = (fb.first_sale[rows] <= y1_end - 90)       # had a year-1 history
methods = {"Static (paper)": P_static, "Moving window (last 91 days)": P_mw,
           "ML global model": P_ml}


def report(mask, label):
    print(f"\n{label}: {mask.sum():,} product-weeks")
    base = pinball(yte[mask], P_static[mask]).sum()
    out = {}
    for name, P in methods.items():
        L = pinball(yte[mask], P[mask])
        # tie-aware coverage: many product-weeks are exactly 0 = forecast 0, so count
        # a tie as half below / half above (otherwise low quantiles look way off)
        yy, PP = yte[mask][:, None], P[mask]
        cov = 0.5 * ((yy < PP).mean(0) + (yy <= PP).mean(0))
        med_err = np.abs(yte[mask] - P[mask][:, 2]).sum() / yte[mask].sum()
        out[name] = (L.sum(0), cov)
        print(f"  {name:30} quantile loss {L.sum():>12,.0f}  skill vs static {1 - L.sum() / base:+.0%}"
              f"  median error {med_err:.0%}  q90 coverage {cov[4]:.0%}")
    return out


res_est = report(established, "Established products (sold in year 1)")
new = fb.first_sale[rows] > y1_end
res_new = report(new, "Brand-new products launched in year 2 (static has nothing)")

imp = pd.Series(qf.models[0.5].feature_importance("gain"), index=Xtr.columns)
print("\nTop features (median model):",
      ", ".join(f"{k}" for k in (imp / imp.sum()).sort_values(ascending=False).index[:8]))

# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(15, 9.5))
gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.05], hspace=0.55, wspace=0.3)

# fan charts for the 4 showcase products
dates = days[test_origins] + pd.Timedelta(days=1)
for k, (sku, role) in enumerate(SHOW.items()):
    ax = fig.add_subplot(gs[k // 2, (k % 2) * 2:(k % 2) * 2 + 2])
    m = (mte.sku == sku).to_numpy()
    y = yte[m]
    ax.fill_between(dates, P_static[m][:, 0], P_static[m][:, 4], color=GREY, alpha=0.25,
                    lw=0, label="Static 10–90% range")
    ax.fill_between(dates, P_ml[m][:, 0], P_ml[m][:, 4], color=BLUE, alpha=0.3, lw=0,
                    label="ML 10–90% range")
    ax.plot(dates, P_ml[m][:, 2], color=BLUE, lw=1.2, label="ML median")
    ax.plot(dates, y, color="black", lw=1, marker="o", ms=2.5, label="What actually happened")
    ax.set_title(f"{role}: next-14-day demand forecast, year 2", loc="left", fontsize=10)
    ax.set_ylabel("Units in next 14 days", fontsize=8)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %y"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    if k == 0:
        ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")

# loss by quantile (established)
ax = fig.add_subplot(gs[2, 0:2])
x = np.arange(len(QUANTILES)); w = 0.27
for i, (name, (L, cov)) in enumerate(res_est.items()):
    ax.bar(x + (i - 1) * w, L / 1e3, w, color=[GREY, GOLD, BLUE][i], label=name)
ax.set_xticks(x, [f"q{int(q * 100)}" for q in QUANTILES])
ax.set_ylabel("Quantile loss (thousands; lower = better)", fontsize=8)
ax.set_title("Forecast error by quantile, established products", loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=7.5)
ax.spines[["top", "right"]].set_visible(False)

# calibration
ax = fig.add_subplot(gs[2, 2:4])
ax.plot([0, 1], [0, 1], color="black", lw=0.7, ls=":", label="Perfectly calibrated")
for i, (name, (L, cov)) in enumerate(res_est.items()):
    ax.plot(QUANTILES, cov, marker="o", color=[GREY, GOLD, BLUE][i], label=name)
ax.set_xlabel("Forecast quantile (e.g. 0.9 = 'demand should be below this 90% of the time')",
              fontsize=8)
ax.set_ylabel("How often demand was actually below it", fontsize=8)
ax.set_title("Calibration: are the uncertainty ranges honest?", loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=7.5)
ax.spines[["top", "right"]].set_visible(False)

fig.savefig(OUT / "step4_forecaster.png", dpi=130, bbox_inches="tight")
pd.to_pickle(dict(P_ml=P_ml, P_static=P_static, P_mw=P_mw, yte=yte, mte=mte),
             ROOT / "data" / "step4_forecasts.pkl")
print("\nsaved", OUT / "step4_forecaster.png", f"  total {time.time() - t0:.0f}s")
