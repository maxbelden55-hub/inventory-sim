"""Step 1 demo: simulate one year under two policies and chart it.
(Uses the vectorized simulator with a single world; synthetic demand.)"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simulator import simulate, Scenarios, NaivePolicy, SSPolicy

OUT = pathlib.Path(__file__).resolve().parent.parent / "outputs"
OUT.mkdir(exist_ok=True)
policies = {"Naive: 140 units every Monday": NaivePolicy(),
            "(s, S) = (150, 300)": SSPolicy()}
scen = Scenarios.generate(1, days=365, seed=42)           # one simulated year, same for both

fig, axes = plt.subplots(len(policies), 1, figsize=(11, 6.5), sharex=True)
for ax, (name, pol) in zip(axes, policies.items()):
    r = simulate(pol, scen, keep_trace=True)
    net = r.trace_net[0]
    print(f"{name}\n  operating cost £{r.operating_cost[0]:,.0f}  "
          f"(holding £{r.holding_cost[0]:,.0f}, ordering £{r.ordering_cost[0]:,.0f}, "
          f"backorder £{r.backorder_cost[0]:,.0f})\n"
          f"  fill rate {r.fill_rate[0]:.1%}, stockout days {r.stockout_days[0]}, "
          f"orders placed {r.n_orders[0]}\n")
    ax.plot(net, color="#2a6f97", lw=1.2, label="Net inventory (negative = backordered)")
    ax.axhline(0, color="#c1121f", lw=0.6)
    ax.set_title(f"{name}  |  operating cost £{r.operating_cost[0]:,.0f}, fill rate {r.fill_rate[0]:.1%}",
                 fontsize=11, loc="left")
    ax.set_ylabel("Units")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
axes[-1].set_xlabel("Day of year")
fig.tight_layout()
fig.savefig(OUT / "step1_one_year.png", dpi=140)
print("saved", OUT / "step1_one_year.png")
