# Inventory under stochastic demand: Monte Carlo + ML for a wholesale distribution center

A weekly ordering system for ~3,800 products of a real UK giftware wholesaler. It combines a
stochastic demand model, a Monte Carlo warehouse simulator and a global probabilistic
forecaster (LightGBM quantile regression), and it is tested walk-forward on a full year of
real orders the models never saw.

**Headline (base case, £5 per order line):** the adaptive ML policy, with every setting tuned
on year-1 data only, cut year-2 inventory cost by **13.8%** versus a published static
simulation-optimization method (Maitra, 2024) re-implemented per product, and by **17.1%** with a
simple dead-product rule added (fill rate 92.9% vs 92.2%). The rule alone gets static to **-12.0%**;
on products that kept selling, ML forecasting adds a further **5.0%**. With researched cost values
the figures are -16.2% / -20.0%. Two limits: at China-style lead times (~55 days) the adaptive policy
**loses** to static (+10.3%), and a one-off bulk order cannot be anticipated. In demand-shock tests the
frozen model cut cost 20-29% versus static when demand surged, collapsed or trended.

## Data

UCI Machine Learning Repository, *Online Retail II* (Chen, 2012),
<https://archive.ics.uci.edu/dataset/502/online+retail+ii>: every invoice line of a UK
non-store giftware wholesaler, 1 Dec 2009 to 9 Dec 2011 (1,067,371 rows). Place
`online_retail_II.xlsx` in `data/raw/uci_original/`. The data has quantities, dates and
selling prices only; ordering cost, purchase cost, holding cost, backorder penalty and lead
times are **assumptions** (see below).

Cleaning (`src/data_prep.py`): product lines only, positive prices, and each cancellation
removes the order line it reverses (6,255 lines; without this a cancelled 80,995-unit order
looks like demand). Result: 1,008,690 order lines, 4,700 products.

## How it works

| Stage | File | What it does |
|---|---|---|
| Simulator | `src/simulator.py` | Vectorized daily warehouse model; backorders; random lead times (5 + Poisson(2) days, per order); replay of real demand; common random numbers; end-of-window cost of leftover stock |
| Monte Carlo | `src/monte_carlo.py` | Confidence intervals, convergence checks, paired comparisons |
| Static baseline | `src/static_baseline.py`, `src/experiment.py` | Paper method: hurdle-lognormal demand fit on year 1, (s, S) grid search on simulated years, frozen |
| Forecaster | `src/forecaster.py` | One LightGBM model per quantile (q10..q99) across all products; target = demand over a random lead time; no look-ahead (tested) |
| Adaptive policy | `src/adaptive_policy.py` | Weekly: order size = kappa x EOQ from forecast mean; reorder point from the exact (s, Q) band-average optimality condition; online per-tier recalibration of forecast quantiles |
| Experiments | `src/experiment.py` | Per-tier tuning on two year-1 validation folds; year-2 walk-forward for all products; sweeps and stress tests |
| Report | `src/results_report.py` | Tables and final figures |

Demo scripts `src/step1_demo.py` ... `src/step5_demo.py` reproduce the step-by-step build.

## Results

`outputs/final_base_case.png`, `outputs/final_order_cost_sweep.png`, `outputs/dead_rule_sweep.png`, `outputs/shock_test.png`, `outputs/sweep_results.csv`.

**Base case (£5 per order line), 3,799 products, year 2 (Dec 2010 - Dec 2011), real orders:**

| Policy | Cost incl. leftover stock | vs static | Fill rate |
|---|---|---|---|
| Static (paper) | £422,378 | +0.0% | 92.2% |
| Adaptive, textbook settings | £580,542 | +37.4% | 96.8% |
| Adaptive, one tuned setting | £389,331 | -7.8% | 88.6% |
| Adaptive, tuned per tier | £363,997 | -13.8% | 93.0% |
| Adaptive per tier, band rule only | £360,928 | -14.5% | 92.6% |
| Adaptive per tier, tuned on Jul-Nov fold only | £373,400 | -11.6% | 94.7% |
| Static + dead-product rule | £371,642 | -12.0% | 92.1% |
| Adaptive per tier + dead-product rule | £350,007 | -17.1% | 92.9% |
| Adaptive per tier, retrained | £362,056 | -14.3% | 92.9% |
| Best fixed policy in hindsight | £204,244 | -51.6% | 98.2% |

**By tier (adaptive, tuned per tier):**

| Tier | Products | Share of revenue | Static | Adaptive | Change |
|---|---|---|---|---|---|
| Intermittent | 2,432 | 9% | £137,525 | £95,170 | -30.8% |
| Low volume | 437 | 10% | £60,845 | £55,547 | -8.7% |
| Mid volume | 721 | 38% | £150,029 | £140,539 | -6.3% |
| High volume | 209 | 43% | £73,980 | £72,741 | -1.7% |

**Dead-product rule** (`src/dead_rule.py`, `outputs/dead_rule_sweep.png`): stop reordering after N weeks with no sales, N swept 1-8; N = 1 was best for both policies. The rule alone captures 87% of the ML saving, almost all of it on the 908 products that stopped selling. On the 2,891 products still selling, adaptive + rule costs £348,666 vs £370,062 for static + rule (-5.8%): that gap is what forecasting adds. Picking N from the year-2 sweep favours the baseline.

**Sensitivity** (each row changes one assumption from the base case; `calibrated` uses the researched values below):

| Configuration | Static | Adaptive (per tier) | Change | Mid+high volume | Low+intermittent | Fill static → adaptive |
|---|---|---|---|---|---|---|
| Order cost £2 | £378,888 | £322,979 | -14.8% | -7.3% | -23.7% | 91.5% → 93.9% |
| Order cost £5 (base) | £422,378 | £363,997 | -13.8% | -4.8% | -24.0% | 92.2% → 93.0% |
| Order cost £10 | £484,730 | £410,175 | -15.4% | -5.8% | -25.5% | 92.7% → 93.2% |
| Order cost £20 | £582,508 | £499,338 | -14.3% | -3.7% | -24.3% | 93.3% → 93.3% |
| Order cost £40 | £762,603 | £652,552 | -14.4% | -4.2% | -22.9% | 93.6% → 92.5% |
| Purchase cost 40% of price | £349,170 | £297,650 | -14.8% | -7.0% | -23.1% | 93.5% → 95.1% |
| Purchase cost 85% of price | £504,441 | £428,897 | -15.0% | -7.8% | -23.5% | 90.6% → 92.6% |
| Backorder 1% of price/day | £257,761 | £215,603 | -16.4% | -5.4% | -28.9% | 85.1% → 84.4% |
| Backorder 20% of price/day | £779,975 | £616,328 | -21.0% | -15.1% | -27.4% | 94.9% → 97.0% |
| Lead time 10 + Poisson(4) days | £621,546 | £523,575 | -15.8% | -4.6% | -27.9% | 90.7% → 94.9% |
| **Calibrated costs**, lead 5 + Poisson(2) | £440,824 | £369,567 | -16.2% | -7.0% | -26.4% | 93.2% → 95.6% |
| **Calibrated costs, China lead time 40 + Poisson(15)** | £996,817 | £1,099,162 | +10.3% | +11.7% | +8.2% | 84.8% → 91.5% |

With the dead-product rule: calibrated, static + rule -11.0%, adaptive + rule -20.0%; China lead time, static + rule -6.3%, adaptive + rule +7.9%. **At ~55-day lead times the adaptive policy loses to static**: its forecasts of two-month demand still run 13-40% high by tier after recalibration, so it overstocks (holding roughly doubles). The sweep rows other than base, calibrated and China were run before the recalibration look-back fix (#30 in CHANGELOG), which only matters for lead times over 8 weeks; the base case moved from -14.1% to -13.8%.

**Demand-shock test** (`src/shock_test.py`, `outputs/shock_test.png`): the year-1 model is frozen; from 4 Apr 2011 demand for 400 selling products is altered and the model sees only the new history.

| Shock | Static | Adaptive | Change | Fill static → adaptive | Forecast within 20% after |
|---|---|---|---|---|---|
| None (control) | £97,242 | £92,266 | -5.1% | 93.4% → 94.1% | — |
| Demand doubles | £217,309 | £155,913 | -28.3% | 83.6% → 92.1% | 7 weeks |
| Demand falls to 30% | £66,323 | £52,851 | -20.3% | 97.4% → 96.2% | 15 weeks |
| Steady growth to 2.5x | £210,080 | £149,437 | -28.9% | 83.4% → 92.2% | 11 weeks |
| One surprise bulk order | £256,592 | £260,074 | +1.4% | 79.1% → 79.5% | — |

## Assumptions (not in the data)

| Assumption | Base value | Grounding | Tested range |
|---|---|---|---|
| Cost per product order line | £5 (calibrated: £3.50) | APQC: $14-$54.50 per purchase order, median 7.8 lines per invoice | £2 to £40 |
| Purchase cost | 60% (calibrated: 70%) | US Census wholesale gross margins 2009-11 (home furnishings 28-32%, misc. durables 19-20%); data ceiling ~85% | 40%, 85% |
| Holding cost | 25%/yr (calibrated: 20%) | Carrying-cost studies, 12-35% | (fixed) |
| Backorder penalty | 5% of price/day (calibrated: 35% once + 0.5%/day) | Anderson, Fitzsimons & Simester (2006) stockout costs, consumer catalog | 1%, 20%/day |
| Lead time | 5 days + Poisson(2) | UK/EU stockists; China-made giftware ~45-70 days | 10 + Poisson(4); 40 + Poisson(15) |

## Reproduce

```bash
pip install -r requirements.txt
./run_all.sh          # ~75 min on 2 cores; tests, data, steps 1-5, all experiments, report
```

Tests: `python3 tests/test_simulator.py && python3 tests/test_pipeline.py`
(13 tests: simulator vs a plain-Python reference, cancellation netting, no look-ahead in features,
lead-time targets, calibration incl. long lead times, reorder-point math, per-product costs,
end-of-window cost, one-off backorder charge).

Researched values and their sources: Project doc `inventory-sim/data-sources-for-assumptions.md`.

`CHANGELOG.md` lists every defect found during development, its root cause and how the fix
was verified. `archive/` keeps superseded scripts and pre-fix results for traceability.
