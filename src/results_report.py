"""Turn experiment results (data/results/*.json|parquet) into tables and figures."""
import json, pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES, OUT = ROOT / "data" / "results", ROOT / "outputs"
TIERS = {0: "Intermittent", 1: "Low volume", 2: "Mid volume", 3: "High volume"}
LABEL = {"static": "Static (paper)", "adaptive_textbook": "Adaptive, textbook settings",
         "adaptive_global": "Adaptive, one tuned setting", "adaptive_tiered": "Adaptive, tuned per tier",
         "adaptive_tiered_band_only": "Adaptive per tier, band rule only",
         "adaptive_tiered_foldB_only": "Adaptive per tier, tuned on Jul-Nov fold only",
         "static_rule": "Static + dead-product rule",
         "adaptive_tiered_rule": "Adaptive per tier + dead-product rule",
         "adaptive_tiered_retrained": "Adaptive per tier, retrained",
         "hindsight": "Best fixed policy in hindsight"}
GREY, BLUE, LBLUE, GOLD, RED = "#8d99ae", "#2a6f97", "#89b6d1", "#e09f3e", "#c1121f"
COL = {"static": GREY, "adaptive_textbook": "#b9d3e3", "adaptive_global": LBLUE,
       "adaptive_tiered_foldB_only": "#d0dde6", "adaptive_tiered_band_only": "#6a9fc0",
       "static_rule": "#b0b7c3", "adaptive_tiered_rule": "#14375a",
       "adaptive_tiered": BLUE, "adaptive_tiered_retrained": "#1b4965", "hindsight": GOLD}


def load(name):
    return (pd.read_parquet(RES / f"{name}.parquet"),
            json.loads((RES / f"{name}.json").read_text()))


def totals(df, policies):
    rows = []
    for p in policies:
        g = lambda k: df[f"{p}__{k}"].sum()
        rows.append(dict(policy=LABEL[p], key=p, cost=g("cost"), holding=g("hold"), ordering=g("order"),
                         backorder=g("back"), leftover=g("terminal"), fill=g("shipped") / g("demand"),
                         stockout_days=df[f"{p}__stockout_days"].mean(), orders=df[f"{p}__orders"].mean()))
    t = pd.DataFrame(rows)
    t["vs_static"] = t.cost / t.cost.iloc[0] - 1
    return t


def by_tier(df, policy="adaptive_tiered"):
    rows = []
    for g, name in TIERS.items():
        m = df.tier == g
        rows.append(dict(tier=name, products=int(m.sum()),
                         static=df.loc[m, "static__cost"].sum(), adaptive=df.loc[m, f"{policy}__cost"].sum(),
                         hindsight=df.loc[m, "hindsight__cost"].sum(),
                         static_fill=df.loc[m, "static__shipped"].sum() / df.loc[m, "static__demand"].sum(),
                         adaptive_fill=df.loc[m, f"{policy}__shipped"].sum() / df.loc[m, f"{policy}__demand"].sum()))
    t = pd.DataFrame(rows)
    t["saving"] = 1 - t.adaptive / t.static
    return t


def base_report():
    df, sm = load("base")
    pols = [p for p in LABEL if f"{p}__cost" in df]
    tot = totals(df, pols)
    bt = by_tier(df)
    rev = pd.read_parquet(ROOT / "data" / "daily_demand.parquet")
    rev = rev[rev.date > "2010-11-30"].groupby("StockCode").revenue.sum()
    df["y2_revenue"] = df.sku.map(rev).fillna(0)
    bt["revenue_share"] = [df.loc[df.tier == g, "y2_revenue"].sum() / df.y2_revenue.sum() for g in TIERS]
    pd.set_option("display.width", 220)
    print(f"BASE CASE: {len(df):,} products, order cost £{sm['config']['order_cost']:g}/line, "
          f"critical ratio {sm['critical_ratio']:.4f}")
    print("tuned:", sm["tuned_global"], sm["tuned_tiers"])
    print(tot.drop(columns="key").to_string(index=False, formatters={
        "cost": "£{:,.0f}".format, "holding": "£{:,.0f}".format, "ordering": "£{:,.0f}".format,
        "backorder": "£{:,.0f}".format, "leftover": "£{:,.0f}".format,
        "fill": "{:.1%}".format, "vs_static": "{:+.1%}".format,
        "stockout_days": "{:.2f}".format, "orders": "{:.2f}".format}))
    print("\nby tier (adaptive tuned per tier):")
    print(bt.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    a, s = df["adaptive_tiered__cost"], df["static__cost"]
    act = df.y2_units > 0
    z = df.y2_units == 0
    extra = dict(
        win_rate_active=float((a < s - 0.01)[act].mean()),
        dead_products=int(z.sum()),
        dead_static=float(s[z].sum()), dead_adaptive=float(a[z].sum()),
        active_change=float(a[act].sum() / s[act].sum() - 1),
        gap_closed=float((s.sum() - a.sum()) / (s.sum() - df["hindsight__cost"].sum())))
    print("\n", json.dumps(extra, indent=1))
    tot.to_csv(OUT / "base_totals.csv", index=False)
    bt.to_csv(OUT / "base_by_tier.csv", index=False)
    (OUT / "base_extra.json").write_text(json.dumps(extra, indent=1))

    # figure (the Jul-Nov-fold-only variant is identical in the base case, so it is left out)
    tf = tot[~tot.key.isin(["adaptive_tiered_foldB_only", "adaptive_tiered_band_only", "adaptive_textbook"])].reset_index(drop=True)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw=dict(width_ratios=[1.35, 1]))
    x = np.arange(len(tf))
    bottom = np.zeros(len(tf))
    for comp, alpha in (("holding", 1.0), ("ordering", 0.65), ("backorder", 0.4), ("leftover", 0.2)):
        v = tf[comp].to_numpy() / 1e3
        ax[0].bar(x, v, bottom=bottom, color=[COL[k] for k in tf.key], alpha=alpha, edgecolor="white")
        bottom += v
    for xx, (c, d, f) in enumerate(zip(tf.cost, tf.vs_static, tf.fill)):
        ax[0].text(xx, c / 1e3 + 8, f"£{c / 1e3:,.0f}k\n{d:+.1%}\nfill {f:.1%}", ha="center", va="bottom", fontsize=8)
    ax[0].set_xticks(x, [l.replace(", ", ",\n").replace(" in hindsight", "\nin hindsight")
                         for l in tf.policy], fontsize=7.5)
    ax[0].set_ylim(0, bottom.max() * 1.28)
    ax[0].set_ylabel("£ thousands, year 2\n(shades, bottom to top: holding / ordering / backorder / leftover stock)", fontsize=8)
    ax[0].set_title(f"Year-2 operating cost, {len(df):,} products, £{sm['config']['order_cost']:g} per order line",
                    loc="left", fontsize=10)
    xt = np.arange(len(bt)); w = 0.27
    ax[1].bar(xt - w, bt.static / 1e3, w, color=GREY, label="Static (paper)")
    ax[1].bar(xt, bt.adaptive / 1e3, w, color=BLUE, label="Adaptive, tuned per tier")
    ax[1].bar(xt + w, bt.hindsight / 1e3, w, color=GOLD, label="Hindsight")
    for xx, (v, sv) in enumerate(zip(bt.adaptive, bt.saving)):
        ax[1].text(xx, v / 1e3, f"{sv:.0%}\ncheaper", ha="center", va="bottom", fontsize=8, color=BLUE)
    ax[1].set_xticks(xt, [f"{t}\n{n:,} products\n{r:.0%} of revenue"
                          for t, n, r in zip(bt.tier, bt.products, bt.revenue_share)], fontsize=7.5)
    ax[1].set_ylabel("£ thousands", fontsize=8)
    ax[1].set_title("By product tier: adaptive cost vs static", loc="left", fontsize=10)
    ax[1].legend(frameon=False, fontsize=8)
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "final_base_case.png", dpi=130)
    return tot, bt, extra


def sweep_report():
    rows = []
    for f in sorted(RES.glob("*.json")):
        sm = json.loads(f.read_text())
        df = pd.read_parquet(RES / f"{f.stem}.parquet")
        t = sm["totals"]
        mh = df.tier >= 2
        rows.append(dict(config=f.stem, order_cost=sm["config"]["order_cost"],
                         purchase_share=sm["config"]["purchase_share"],
                         backorder_rate=sm["config"]["backorder_rate"],
                         lead=f"{sm['config']['lead_min']}+P({sm['config']['lead_extra']:g})",
                         static=t["static"]["cost"], adaptive_tiered=t["adaptive_tiered"]["cost"],
                         adaptive_global=t["adaptive_global"]["cost"],
                         hindsight=t["hindsight"]["cost"],
                         saving_all=1 - t["adaptive_tiered"]["cost"] / t["static"]["cost"],
                         saving_mid_high=1 - df.loc[mh, "adaptive_tiered__cost"].sum() / df.loc[mh, "static__cost"].sum(),
                         saving_low_int=1 - df.loc[~mh, "adaptive_tiered__cost"].sum() / df.loc[~mh, "static__cost"].sum(),
                         static_fill=t["static"]["fill"], adaptive_fill=t["adaptive_tiered"]["fill"],
                         tuned=json.dumps(sm["tuned_tiers"])))
    sw = pd.DataFrame(rows)
    sw.to_csv(OUT / "sweep_results.csv", index=False)
    pd.set_option("display.width", 250)
    print(sw.drop(columns="tuned").to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    oc = sw[sw.config.isin(["base"]) | sw.config.str.startswith("order_cost")].sort_values("order_cost")
    if len(oc) >= 2:
        fig, ax = plt.subplots(figsize=(8.5, 4.6))
        ax.axhline(0, color="black", lw=0.7)
        ax.plot(oc.order_cost, oc.saving_mid_high * 100, marker="o", color=BLUE,
                label="Mid + high volume products")
        ax.plot(oc.order_cost, oc.saving_all * 100, marker="o", color=GREY, label="All products")
        ax.plot(oc.order_cost, oc.saving_low_int * 100, marker="o", color=GREY, ls=":",
                label="Low volume + intermittent")
        for _, r in oc.iterrows():
            ax.annotate(f"{r.saving_mid_high:+.1%}", (r.order_cost, r.saving_mid_high * 100),
                        textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color=BLUE)
        ax.set_xscale("log")
        ax.set_xticks(oc.order_cost, [f"£{v:g}" for v in oc.order_cost])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("Assumed cost per product order line (log scale)", fontsize=9)
        ax.set_ylabel("Adaptive saving vs static (%)", fontsize=9)
        ax.set_title("Order-cost sweep: adaptive (tuned per tier) vs static, year 2", loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT / "final_order_cost_sweep.png", dpi=130)
    return sw


if __name__ == "__main__":
    base_report()
    print()
    sweep_report()
