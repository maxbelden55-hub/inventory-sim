"""Step 2 demo: Monte Carlo over thousands of simulated years."""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simulator import simulate, Scenarios, SSPolicy, NaivePolicy
from monte_carlo import (summarize, running_stats, batch_means, samples_needed,
                         compare, estimate)

OUT = pathlib.Path(__file__).resolve().parent.parent / "outputs"
N = 10_000
BLUE, GREY, RED = "#2a6f97", "#8d99ae", "#c1121f"

# ---- 1. Run 10,000 years for both policies on the SAME worlds --------------
t0 = time.time()
scen = Scenarios.generate(N, seed=2026)
res = {"Naive (140 every Monday)": simulate(NaivePolicy(), scen),
       "(s, S) = (150, 300)": simulate(SSPolicy(), scen)}
print(f"Simulated {N:,} years x 2 policies in {time.time() - t0:.1f}s\n")

for name, r in res.items():
    s = summarize(r)
    print(name)
    print(f"  operating cost/yr  £{s['operating_cost']}")
    print(f"    holding £{s['holding'].mean:,.0f} | ordering £{s['ordering'].mean:,.0f}"
          f" | backorder £{s['backorder'].mean:,.0f}")
    print(f"  fill rate {s['fill_rate'].mean:.1%}   stockout days {s['stockout_days'].mean:.1f}"
          f"   orders {s['orders'].mean:.0f}   avg on hand {s['avg_on_hand'].mean:.0f}")
    oc = r.operating_cost
    print(f"  worst 5% of years cost more than £{np.percentile(oc, 95):,.0f}\n")

ss_cost = res["(s, S) = (150, 300)"].operating_cost
print(f"Years needed for +/- £100 precision on (s,S): {samples_needed(ss_cost, 100):,}")

# ---- 2. Common random numbers: is (150,300) really better than (130,300)? --
small = Scenarios.generate(2000, seed=11)
a, b = SSPolicy(80, 250), SSPolicy(70, 250)
ra, rb, diff, paired = compare(a, b, small)
# Unpaired version: B on a different set of worlds
rb_ind = simulate(b, Scenarios.generate(2000, seed=99))
se_unpaired = np.sqrt(ra.operating_cost.var(ddof=1) / 2000 + rb_ind.operating_cost.var(ddof=1) / 2000)
print(f"\n(70,250) minus (80,250), 2000 years each:")
print(f"  paired (same worlds):      {paired.mean:+,.0f} +/- {1.96 * paired.se:,.0f}")
print(f"  unpaired (different luck): {rb_ind.operating_cost.mean() - ra.operating_cost.mean():+,.0f}"
      f" +/- {1.96 * se_unpaired:,.0f}")
print(f"  -> paired CI is {se_unpaired / paired.se:.1f}x tighter, i.e. {(se_unpaired / paired.se)**2:.1f}x fewer simulations for the same precision")

# ---- 3. Figures ------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))

# (a) cost distribution
edges = np.linspace(8000, 30000, 111)    # clip the naive policy's long tail for readability
for (name, r), c in zip(res.items(), (GREY, BLUE)):
    ax[0].hist(np.clip(r.operating_cost, None, 30000), bins=edges, alpha=0.8, color=c, label=name)
ax[0].set_yscale("log")
ax[0].set_title("Operating cost across 10,000 simulated years", loc="left", fontsize=11)
ax[0].text(0.98, 0.55, "naive tail runs to £100K+\n(lumped at £30K)", transform=ax[0].transAxes,
           ha="right", fontsize=7, color=GREY)
ax[0].set_xlabel("£ per year"); ax[0].set_ylabel("Number of years (log scale)")
ax[0].legend(frameon=False, fontsize=8)

# (b) convergence
n, m, se = running_stats(ss_cost)
ax[1].fill_between(n, m - 1.96 * se, m + 1.96 * se, color=BLUE, alpha=0.2, lw=0)
ax[1].plot(n, m, color=BLUE, lw=1.3)
ax[1].set_xscale("log")
ax[1].axhline(ss_cost.mean(), color="black", lw=0.6, ls=":")
ax[1].set_title("Convergence: running mean ± 95% CI, (s, S)", loc="left", fontsize=11)
ax[1].set_xlabel("Simulated years (log scale)"); ax[1].set_ylabel("£ per year")
bm = batch_means(ss_cost, 20)
ins = ax[1].inset_axes([0.55, 0.62, 0.42, 0.33])
ins.plot(bm, "o", ms=3, color=BLUE); ins.axhline(ss_cost.mean(), color="black", lw=0.6)
ins.set_title("20 batch means", fontsize=7); ins.tick_params(labelsize=6)

# (c) paired vs unpaired
ax[2].hist(diff, bins=50, color=BLUE, alpha=0.8)
ax[2].axvline(0, color="black", lw=0.8)
ax[2].axvline(diff.mean(), color=RED, lw=1.2, label=f"mean {diff.mean():+,.0f}")
ax[2].set_title("Same worlds: (70,250) minus (80,250)", loc="left", fontsize=11)
ax[2].set_xlabel("£ difference per year"); ax[2].set_ylabel("Number of years")
ax[2].legend(frameon=False, fontsize=8)

for a_ in ax:
    a_.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "step2_monte_carlo.png", dpi=140)
print("\nsaved", OUT / "step2_monte_carlo.png")
