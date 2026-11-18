"""Step 6 report: headline numbers + figure from the walk-forward results."""
import pathlib, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
BLUE, LBLUE, GREY, GOLD, RED = "#2a6f97", "#89b6d1", "#8d99ae", "#e09f3e", "#c1121f"
TAG = sys.argv[1] if len(sys.argv) > 1 else ""
df = pd.read_parquet(ROOT / "data" / f"step6_results{TAG}.parquet")
POL = {"static": "Static (paper)", "adaptive_no_retrain": "Adaptive ML, year-1 model",
       "adaptive_retrained": "Adaptive ML, retrained monthly",
       "hindsight": "Best fixed policy in hindsight"}
COL = {"static": GREY, "adaptive_no_retrain": LBLUE, "adaptive_retrained": BLUE,
       "hindsight": GOLD}
if "adaptive_retrained__cost" not in df:
    del POL["adaptive_retrained"]
ADAPT = "adaptive_retrained" if "adaptive_retrained" in POL else "adaptive_no_retrain"
TIERS = {0: "Intermittent", 1: "Low volume", 2: "Mid volume", 3: "High volume"}

rows = []
for p, name in POL.items():
    g = lambda k: df[f"{p}__{k}"].sum()
    rows.append(dict(policy=name, total_cost=g("cost"), holding=g("hold"), ordering=g("order"),
                     backorder=g("back"), fill_rate=g("shipped") / g("demand"),
                     stockout_days_per_product=df[f"{p}__stockout_days"].mean(),
                     orders_per_product=df[f"{p}__orders"].mean()))
tot = pd.DataFrame(rows)
base = tot.total_cost.iloc[0]
tot["vs_static"] = tot.total_cost / base - 1
pd.set_option("display.width", 220)
print(f"{len(df):,} products, real year-2 orders\n")
print(tot.to_string(index=False, formatters={
    "total_cost": "£{:,.0f}".format, "holding": "£{:,.0f}".format,
    "ordering": "£{:,.0f}".format, "backorder": "£{:,.0f}".format,
    "fill_rate": "{:.1%}".format, "vs_static": "{:+.1%}".format,
    "stockout_days_per_product": "{:.1f}".format, "orders_per_product": "{:.1f}".format}))

a, s = df[f"{ADAPT}__cost"], df["static__cost"]
active = (df.y2_units > 0)
print(f"\nadaptive (retrained) cheaper than static on {(a < s - 0.01)[active].mean():.0%} "
      f"of products that sold in year 2")
gap_closed = (s.sum() - a.sum()) / (s.sum() - df["hindsight__cost"].sum())
print(f"share of the static-vs-hindsight gap closed: {gap_closed:.0%}")
if ADAPT == "adaptive_retrained":
    print(f"retraining effect: {df['adaptive_retrained__cost'].sum() / df['adaptive_no_retrain__cost'].sum() - 1:+.1%}")
z = df.y2_units == 0
print(f"products with zero year-2 demand: {z.sum():,}; adaptive vs static on the rest: "
      f"{df.loc[~z, f'{ADAPT}__cost'].sum() / df.loc[~z, 'static__cost'].sum() - 1:+.1%}")

bt = []
for t, name in TIERS.items():
    m = df.tier_at_start == t
    bt.append(dict(tier=name, products=m.sum(),
                   static=df.loc[m, "static__cost"].sum(),
                   adaptive=df.loc[m, f"{ADAPT}__cost"].sum(),
                   hindsight=df.loc[m, "hindsight__cost"].sum(),
                   static_fill=df.loc[m, "static__shipped"].sum() / df.loc[m, "static__demand"].sum(),
                   adaptive_fill=df.loc[m, f"{ADAPT}__shipped"].sum()
                   / df.loc[m, f"{ADAPT}__demand"].sum()))
bt = pd.DataFrame(bt)
bt["saving"] = 1 - bt.adaptive / bt.static
print("\nby volume tier (at start of year 2):")
print(bt.round(3).to_string(index=False))
tot.to_csv(OUT / f"step6_totals{TAG}.csv", index=False)
bt.to_csv(OUT / f"step6_by_tier{TAG}.csv", index=False)

# ------------------------------------------------------------------ figure
fig, ax = plt.subplots(2, 2, figsize=(14, 9))
fig.subplots_adjust(hspace=0.45, wspace=0.3)

# (a) total cost, stacked
x = np.arange(len(tot))
bottom = np.zeros(len(tot))
for comp, c, alpha in (("holding", None, 1.0), ("ordering", None, 0.65), ("backorder", None, 0.35)):
    vals = tot[comp].to_numpy() / 1e3
    ax[0, 0].bar(x, vals, bottom=bottom, color=[COL[p] for p in POL], alpha=alpha,
                 edgecolor="white", label=comp.capitalize())
    bottom += vals
for xx, (v, d, f) in enumerate(zip(tot.total_cost, tot.vs_static, tot.fill_rate)):
    ax[0, 0].text(xx, v / 1e3, f"£{v / 1e3:,.0f}k\n{d:+.0%} · fill {f:.1%}",
                  ha="center", va="bottom", fontsize=8)
ax[0, 0].set_xticks(x, [n.replace(", ", ",\n").replace(" in hindsight", "\nin hindsight")
                        for n in POL.values()], fontsize=8)
ax[0, 0].set_ylabel("£ thousands, year 2", fontsize=8)
ax[0, 0].set_ylim(0, bottom.max() * 1.25)
ax[0, 0].set_title(f"Total operating cost, {len(df):,} real products (shade = holding / ordering / backorder)",
                   loc="left", fontsize=10)

# (b) per-product saving distribution
sv = (1 - a / s.where(s > 0))[active].clip(-1, 1)
ax[0, 1].hist(sv, bins=np.linspace(-1, 1, 81), color=BLUE, alpha=0.85)
ax[0, 1].axvline(0, color="black", lw=0.8)
ax[0, 1].axvline(sv.median(), color=RED, lw=1.2, label=f"median {sv.median():+.0%}")
ax[0, 1].set_xlabel("Saving vs static, per product (clipped to ±100%)", fontsize=8)
ax[0, 1].set_ylabel("Products", fontsize=8)
ax[0, 1].set_title(f"Adaptive cheaper on {(a < s - 0.01)[active].mean():.0%} of products",
                   loc="left", fontsize=10)
ax[0, 1].legend(frameon=False, fontsize=8)

# (c) by tier
xt = np.arange(len(bt)); w = 0.27
ax[1, 0].bar(xt - w, bt.static / 1e3, w, color=GREY, label="Static")
ax[1, 0].bar(xt, bt.adaptive / 1e3, w, color=BLUE, label=POL[ADAPT].replace("Adaptive ML, ", "Adaptive, "))
ax[1, 0].bar(xt + w, bt.hindsight / 1e3, w, color=GOLD, label="Hindsight")
for xx, (v, sv_) in enumerate(zip(bt.adaptive, bt.saving)):
    ax[1, 0].text(xx, v / 1e3, f"{sv_:+.0%}", ha="center", va="bottom", fontsize=8, color=BLUE)
ax[1, 0].set_xticks(xt, [f"{t}\n({n:,})" for t, n in zip(bt.tier, bt.products)], fontsize=8)
ax[1, 0].set_ylabel("£ thousands", fontsize=8)
ax[1, 0].set_title("Cost by product type (labels = adaptive saving vs static)", loc="left", fontsize=10)
ax[1, 0].legend(frameon=False, fontsize=8)

# (d) cost vs service, per tier
for t, name in TIERS.items():
    r = bt[bt.tier == name].iloc[0]
    ax[1, 1].annotate("", xy=(r.adaptive_fill, r.adaptive / r.products),
                      xytext=(r.static_fill, r.static / r.products),
                      arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
    ax[1, 1].scatter([r.static_fill], [r.static / r.products], color=GREY, zorder=3)
    ax[1, 1].scatter([r.adaptive_fill], [r.adaptive / r.products], color=BLUE, zorder=3)
    ax[1, 1].text(r.adaptive_fill, r.adaptive / r.products, f"  {name}", fontsize=8, va="center")
ax[1, 1].set_yscale("log")
ax[1, 1].set_xlabel("Fill rate (share of units shipped same day)", fontsize=8)
ax[1, 1].set_ylabel("£ cost per product (log)", fontsize=8)
ax[1, 1].set_title("Static (grey) → adaptive (blue): cost vs service", loc="left", fontsize=10)

for a_ in ax.flat:
    a_.spines[["top", "right"]].set_visible(False)
    a_.tick_params(labelsize=8)
fig.savefig(OUT / f"step6_walkforward{TAG}.png", dpi=130, bbox_inches="tight")
print("\nsaved", OUT / "step6_walkforward.png")
