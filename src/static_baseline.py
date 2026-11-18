"""
Static baseline = the Maitra (2024) approach, applied to real products:

  1. Fit ONE demand distribution to a product's history (year 1):
       - on each open day, an order happens with probability p
       - if it happens, its size is lognormal(mu, sigma)
       - Saturdays are closed (the real warehouse almost never shipped on Saturday)
  2. Simulate many years from that distribution and grid-search the (s, S)
     policy with the lowest expected operating cost (common random numbers).
  3. Freeze the policy. It never updates, whatever happens next.
"""
from dataclasses import dataclass
import numpy as np
from simulator import simulate, Scenarios, SSPolicy, Costs, LeadTimeModel, terminal_cost
from monte_carlo import estimate

SATURDAY = 5


@dataclass
class HurdleLognormalDemand:
    p: float                 # chance of any demand on an open day
    mu: float                # lognormal params of order size, given demand > 0
    sigma: float
    start_weekday: int = 0   # weekday of simulated day 0 (Mon=0)

    @classmethod
    def fit(cls, daily, weekdays, start_weekday=0):
        """daily: demand per calendar day; weekdays: weekday number per day."""
        open_days = daily[weekdays != SATURDAY]
        pos = open_days[open_days > 0]
        logs = np.log(pos)
        return cls(p=len(pos) / len(open_days), mu=logs.mean(),
                   sigma=logs.std(ddof=1), start_weekday=start_weekday)

    def sample(self, n, days, rng):
        wd = (self.start_weekday + np.arange(days)) % 7
        happens = rng.random((n, days)) < self.p
        size = np.maximum(np.rint(rng.lognormal(self.mu, self.sigma, (n, days))), 1)
        return np.where(happens & (wd != SATURDAY), size, 0).astype(np.int64)

    @property
    def mean_per_open_day(self):
        return self.p * np.exp(self.mu + self.sigma ** 2 / 2)


ORDER_FIXED = 5.0      # £ per product order line (base case; see experiment.py sweeps)


def costs_for(price):
    """Costs scaled to the product's real selling price (assumptions, in GBP):
       buy at 60% of price; holding = 25%/yr of purchase cost; ORDER_FIXED per
       order line; a backordered unit costs 5% of its price per day it waits.
       Same base case as experiment.Config."""
    unit_cost = 0.6 * price
    return Costs(holding_per_unit_day=0.25 * unit_cost / 365,
                 order_fixed=ORDER_FIXED,
                 order_per_unit=unit_cost,
                 backorder_per_unit_day=0.05 * price)


LEAD = LeadTimeModel(min_days=5, mean_extra=2.0)   # assumption: domestic supplier


def total_cost(r, costs, scen):
    """Operating cost + cost of stock left at the end of the window."""
    rate = scen.demand.mean(1)
    return r.operating_cost + terminal_cost(r, costs, rate, LEAD.min_days + LEAD.mean_extra)


def grid_search(scen, costs, s_grid, Q_grid, start=None):
    """Evaluate every (s, Q) with S = s + Q on the same scenarios.
    Returns cost matrix [len(s_grid), len(Q_grid)]."""
    cost = np.empty((len(s_grid), len(Q_grid)))
    for i, s in enumerate(s_grid):
        for j, Q in enumerate(Q_grid):
            r = simulate(SSPolicy(int(s), int(s + Q)), scen, costs,
                         start_inventory=int(s + Q) if start is None else start)
            cost[i, j] = total_cost(r, costs, scen).mean()
    return cost


def optimize_ss(scen, costs, mean_daily, coarse=22, fine=15, start=None):
    """Coarse grid, then a finer grid around the best point."""
    ltd = mean_daily * (LEAD.min_days + LEAD.mean_extra)
    s_grid = np.unique(np.linspace(0, 6 * ltd + 10, coarse).astype(int))
    Q_grid = np.unique(np.geomspace(max(mean_daily, 1), 60 * mean_daily + 10, coarse).astype(int))
    c = grid_search(scen, costs, s_grid, Q_grid, start)
    i, j = np.unravel_index(c.argmin(), c.shape)
    s_lo, s_hi = s_grid[max(i - 1, 0)], s_grid[min(i + 1, len(s_grid) - 1)]
    q_lo, q_hi = Q_grid[max(j - 1, 0)], Q_grid[min(j + 1, len(Q_grid) - 1)]
    s2 = np.unique(np.linspace(s_lo, s_hi, fine).astype(int))
    q2 = np.unique(np.linspace(max(q_lo, 1), q_hi, fine).astype(int))
    c2 = grid_search(scen, costs, s2, q2, start)
    i2, j2 = np.unravel_index(c2.argmin(), c2.shape)
    best = SSPolicy(int(s2[i2]), int(s2[i2] + q2[j2]))
    return best, (s_grid, Q_grid, c)


def opening_stock(history_before):
    """Same opening inventory for every policy in a head-to-head:
    the last 28 days of demand before the test period starts."""
    return int(np.asarray(history_before)[-28:].sum())
