# Monte Carlo — Planet Labs (PL)

> **Not financial advice.** See [LEGAL.md](../../../../LEGAL.md).

---

## V1 — Baseline (C+)

**Script:** [`monte_carlo_v1.py`](../../../../scripts/monte_carlo_v1.py)

Basic Geometric Brownian Motion (GBM) with constant drift and volatility. 10,000 paths over 2 years.

| Detail | Value |
|--------|-------|
| Model | Constant-vol GBM |
| 2yr Median | $31.02 |
| P(Profit) | 46.8% |
| Stress Tests | Made-up multipliers |
| Validation | None |

**What went wrong:**
- Constant volatility — PL's real vol swings between 46% and 134%
- No fat tails or sudden jumps
- Stress scenarios were arbitrary, not grounded in data

---

## V2 — Upgraded (A-)

**Script:** [`monte_carlo_v2.py`](../../../../scripts/monte_carlo_v2.py)

Three new approaches on top of the V1 baseline.

### Heston (Stochastic Vol)
Volatility is now random and mean-reverting. When vol spikes, it tends to come back down. Captures how real markets work.

### Merton (Jump Diffusion)
Adds sudden jumps (crashes/spikes) on top of normal movement. Models the fat tails that GBM misses. Found 12 jumps in PL's history.

### HMM (Regime Switching)
Hidden Markov Model detects 2 regimes from PL's actual data:
- **Calm** — 85% of days, ~46% annualized vol
- **Volatile** — 15% of days, ~134% annualized vol

Stress tests now use these real regimes instead of made-up multipliers.

### Results (2yr)

| Model | Median | P(Profit) | P(Double) | VaR 95% |
|-------|-------:|----------:|----------:|--------:|
| V1 GBM | $31.02 | 46.8% | 23.6% | $5.32 |
| V2 Heston | $23.89 | 39.3% | 21.2% | $2.76 |
| V2 Jump Diff | $24.84 | 40.4% | 22.0% | $3.22 |

V2 is more pessimistic because it's more realistic about tail risk.

### Stress Tests

| Scenario | 2yr Median | P(Profit) |
|----------|----------:|----------:|
| Market Crash | $0.01 | 1.5% |
| Bear Market | $0.96 | 8.6% |
| Base Case | $20.24 | 43.4% |
| Calm Bull | $29.39 | 60.7% |
| Volatile Bull | $1.59 | 15.9% |

---

## Takeaways

- Constant vol is wrong — real stocks have volatility clusters
- Jumps matter for tail risk — GBM underestimates crashes
- HMM regime detection works — the calm/volatile split matches what you see on a chart
- More realistic = more pessimistic. That's the point.

---

## Charts

See the animated demo in the [main README](../../../README.md).

## Files

| File | What |
|------|------|
| `results.json` | Full V1 vs V2 comparison |
| `paths.csv` | Heston percentile paths |
| `stress_paths.csv` | Stress scenario paths |
| `sensitivity.csv` | Parameter sensitivity grid |

*Seed: 42 for reproducibility*

*Data: [Yahoo Finance](https://finance.yahoo.com/quote/PL/) via yfinance*
