"""
Warehouse simulator, vectorized: runs N independent simulated years at once.

Each day, for every one of the N parallel worlds:
  1. Orders whose lead time has elapsed arrive.
  2. Customer demand happens. We ship from stock; anything we can't ship is
     BACKORDERED (the wholesale customer waits, and we pay a penalty per unit
     per day they wait). Backorders are filled first when stock arrives.
  3. We pay holding cost on stock left on the shelf, backorder cost on units
     customers are still waiting for.
  4. The policy looks at inventory position (net stock + on order) and decides
     how much to order. Orders get a random lead time.

Randomness is generated UP FRONT as a `Scenarios` object (a demand matrix and a
lead-time matrix). That gives two things:
  * Replay mode: pass in real historical demand instead of simulated demand.
  * Common random numbers: every policy can be tested against the exact same
    N worlds, so differences between policies are due to the policy, not luck.
    Lead times are indexed by order number (1st order, 2nd order, ...) rather
    than by calendar day, so supplier luck stays aligned across policies.
"""

from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass
class DemandModel:
    """Daily demand ~ Negative Binomial (overdispersed) with a weekly pattern."""
    mean: float = 20.0
    dispersion: float = 5.0                  # NB shape k: smaller = wilder days
    weekly: tuple = (0.9, 0.85, 0.85, 0.9, 1.1, 1.3, 1.1)   # Mon..Sun

    def sample(self, n, days, rng):
        mu = self.mean * np.resize(np.asarray(self.weekly), days)   # (days,)
        k = self.dispersion
        p = k / (k + mu)
        return rng.negative_binomial(k, p, size=(n, days))


@dataclass
class LeadTimeModel:
    """Lead time = min_days + Poisson(mean_extra) random delay."""
    min_days: int = 3
    mean_extra: float = 2.0

    def sample(self, n, days, rng):
        return self.min_days + rng.poisson(self.mean_extra, size=(n, days))


@dataclass
class Costs:
    """Each field can be a number or an array with one value per world
    (e.g. per product, when costs are scaled to each product's price)."""
    holding_per_unit_day: float = 0.10       # storage + tied-up cash
    order_fixed: float = 100.0               # truck, admin, receiving per order
    order_per_unit: float = 5.0              # purchase price (reported separately)
    backorder_per_unit_day: float = 1.00     # penalty per unit per day a customer waits
    backorder_per_unit: float = 0.0          # one-off penalty when a unit goes on backorder


@dataclass
class Scenarios:
    """N pre-generated worlds: what customers will want, and how long each
    order placed on each day would take to arrive."""
    demand: np.ndarray        # (N, days) ints
    lead: np.ndarray          # (N, days) ints: lead time of the 1st, 2nd, 3rd... order placed

    @property
    def n(self):
        return self.demand.shape[0]

    @property
    def days(self):
        return self.demand.shape[1]

    @classmethod
    def generate(cls, n, days=365, demand=None, lead=None, seed=None):
        rng = np.random.default_rng(seed)
        demand = demand or DemandModel()
        lead = lead or LeadTimeModel()
        return cls(demand.sample(n, days, rng), lead.sample(n, days, rng))

    @classmethod
    def from_matrix(cls, demand_matrix, lead=None, seed=None):
        """Replay many real series at once (e.g. one row per product)."""
        d = np.asarray(demand_matrix, dtype=np.int64)
        lead = lead or LeadTimeModel()
        return cls(d, lead.sample(d.shape[0], d.shape[1], np.random.default_rng(seed)))

    @classmethod
    def replay(cls, demand_history, lead=None, n_lead_draws=1, seed=None):
        """Replay mode: real demand history (1-D array), optionally with many
        random lead-time draws so we see the range of outcomes."""
        d = np.asarray(demand_history, dtype=int)[None, :].repeat(n_lead_draws, 0)
        lead = lead or LeadTimeModel()
        rng = np.random.default_rng(seed)
        return cls(d, lead.sample(n_lead_draws, d.shape[1], rng))


# ---------------------------------------------------------------------------
# Policies: each gets (day, net_inventory[N], on_order[N]) and returns qty[N]
# ---------------------------------------------------------------------------

@dataclass
class NaivePolicy:
    """Order a fixed amount every `every` days, regardless of stock."""
    qty: int = 140
    every: int = 7

    def order(self, t, net, on_order):
        return np.full_like(net, self.qty if t % self.every == 0 else 0)


@dataclass
class SSPolicy:
    """(s, S): if inventory position <= s, order up to S."""
    s: int = 150
    S: int = 300

    def order(self, t, net, on_order):
        ip = net + on_order
        return np.where(ip <= self.s, self.S - ip, 0)


@dataclass
class DynamicSSPolicy:
    """(s, S) whose levels can differ by world (product) and by day.
    s, S: arrays of shape (N, days). Used by the adaptive ML policy, whose
    levels are recomputed every week from the latest forecast."""
    s: np.ndarray
    S: np.ndarray

    def order(self, t, net, on_order):
        ip = net + on_order
        return np.where(ip <= self.s[:, t], self.S[:, t] - ip, 0)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    """Per-world results; every field is an array of length N."""
    holding_cost: np.ndarray
    ordering_cost: np.ndarray        # fixed cost per order (truck/admin)
    backorder_cost: np.ndarray
    purchase_cost: np.ndarray        # units bought x price (reported separately)
    demand: np.ndarray
    shipped_from_stock: np.ndarray   # units shipped the same day they were ordered
    stockout_days: np.ndarray
    n_orders: np.ndarray
    avg_on_hand: np.ndarray
    trace_net: np.ndarray = None     # (N, days) net inventory, if requested
    end_net: np.ndarray = None       # net inventory after the last day
    end_on_order: np.ndarray = None  # units ordered but not yet arrived after the last day

    @property
    def operating_cost(self):
        """The cost the policy actually controls: holding + ordering + backorders.
        Purchase cost is excluded because with backorders, every unit demanded
        is eventually bought no matter which policy you use."""
        return self.holding_cost + self.ordering_cost + self.backorder_cost

    @property
    def fill_rate(self):
        return self.shipped_from_stock / np.maximum(self.demand, 1)


def simulate(policy, scen: Scenarios, costs=None, start_inventory=200,
             keep_trace=False) -> SimResult:
    costs = costs or Costs()
    N, T = scen.n, scen.days
    max_lead = int(scen.lead.max()) + 1
    arrivals = np.zeros((N, T + max_lead), dtype=np.int64)  # scheduled deliveries
    net = np.full(N, start_inventory, dtype=np.int64)        # on hand minus backorders
    on_order = np.zeros(N, dtype=np.int64)
    rows = np.arange(N)

    hold = np.zeros(N); fixed = np.zeros(N); back = np.zeros(N); buy = np.zeros(N)
    shipped = np.zeros(N, dtype=np.int64); so_days = np.zeros(N, dtype=np.int64)
    n_orders = np.zeros(N, dtype=np.int64); on_hand_sum = np.zeros(N)
    trace = np.zeros((N, T), dtype=np.int64) if keep_trace else None

    for t in range(T):
        # 1. arrivals
        arr = arrivals[:, t]
        net += arr
        on_order -= arr

        # 2. demand: ship what's on the shelf, backorder the rest
        d = scen.demand[:, t]
        available = np.maximum(net, 0)
        shipped += np.minimum(d, available)
        so_days += (d > available)
        back += costs.backorder_per_unit * np.maximum(d - available, 0)   # one-off charge
        net -= d

        # 3. daily costs
        on_hand = np.maximum(net, 0)
        hold += costs.holding_per_unit_day * on_hand
        back += costs.backorder_per_unit_day * np.maximum(-net, 0)
        on_hand_sum += on_hand

        # 4. policy
        q = np.maximum(policy.order(t, net, on_order), 0).astype(np.int64)
        placed = q > 0
        fixed += costs.order_fixed * placed
        buy += costs.order_per_unit * q
        n_orders += placed
        on_order += q
        # k-th order in a world always gets that world's k-th lead time, so two
        # policies see the same supplier luck even if they order on different days
        arrive_day = t + scen.lead[rows, np.minimum(n_orders - placed, T - 1)]
        np.add.at(arrivals, (rows[placed], arrive_day[placed]), q[placed])

        if keep_trace:
            trace[:, t] = net

    return SimResult(hold, fixed, back, buy, scen.demand.sum(1), shipped,
                     so_days, n_orders, on_hand_sum / T, trace, net.copy(), on_order.copy())


def terminal_cost(res, costs, daily_rate, lead_mean, max_days=365):
    """Cost of what is left when the simulated window ends, so that stock bought
    near the end is not free. Leftover units (on hand + on order) are held while
    they are sold off at the product's average daily demand: a linear draw-down
    costs h x I^2 / (2 d), capped at max_days of holding. Customers still waiting
    are charged backorder cost for about one more lead time."""
    inv = np.maximum(res.end_net, 0) + res.end_on_order
    d = np.maximum(np.asarray(daily_rate, float), 1e-9)
    days = np.minimum(inv / d, max_days)
    hold = np.asarray(costs.holding_per_unit_day) * inv * days / 2
    back = np.asarray(costs.backorder_per_unit_day) * np.maximum(-res.end_net, 0) * lead_mean
    return hold + back
