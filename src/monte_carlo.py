"""
Monte Carlo engine: summaries with confidence intervals, convergence checks,
and paired policy comparisons using common random numbers.
"""

from dataclasses import dataclass
import numpy as np
from simulator import simulate, Scenarios, Costs

Z95 = 1.96


@dataclass
class Estimate:
    mean: float
    se: float          # standard error of the mean

    @property
    def ci(self):
        return self.mean - Z95 * self.se, self.mean + Z95 * self.se

    def __str__(self):
        lo, hi = self.ci
        return f"{self.mean:,.0f}  (95% CI {lo:,.0f} to {hi:,.0f})"


def estimate(x) -> Estimate:
    x = np.asarray(x, float)
    return Estimate(x.mean(), x.std(ddof=1) / np.sqrt(len(x)))


def summarize(res):
    """Headline numbers for one policy across all simulated years."""
    return {
        "operating_cost": estimate(res.operating_cost),
        "holding": estimate(res.holding_cost),
        "ordering": estimate(res.ordering_cost),
        "backorder": estimate(res.backorder_cost),
        "fill_rate": estimate(res.fill_rate),
        "stockout_days": estimate(res.stockout_days),
        "orders": estimate(res.n_orders),
        "avg_on_hand": estimate(res.avg_on_hand),
    }


def running_stats(x):
    """Running mean and standard error after 1..N samples (convergence check,
    as in Maitra 2024)."""
    x = np.asarray(x, float)
    n = np.arange(1, len(x) + 1)
    mean = np.cumsum(x) / n
    var = (np.cumsum(x ** 2) - n * mean ** 2) / np.maximum(n - 1, 1)
    se = np.sqrt(np.maximum(var, 0) / n)
    return n, mean, se


def batch_means(x, n_batches=20):
    """Mean of each consecutive batch. If batches wander a lot, run more."""
    return np.array([b.mean() for b in np.array_split(np.asarray(x, float), n_batches)])


def samples_needed(x, target_halfwidth):
    """How many simulated years for the 95% CI to be +/- target_halfwidth."""
    sd = np.std(x, ddof=1)
    return int(np.ceil((Z95 * sd / target_halfwidth) ** 2))


def compare(policy_a, policy_b, scen: Scenarios, costs=None):
    """Paired comparison on identical worlds (common random numbers).
    Returns per-world cost difference B - A and its estimate."""
    ra, rb = simulate(policy_a, scen, costs), simulate(policy_b, scen, costs)
    diff = rb.operating_cost - ra.operating_cost
    return ra, rb, diff, estimate(diff)
