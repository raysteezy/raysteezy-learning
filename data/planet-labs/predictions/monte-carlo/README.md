# Monte Carlo Simulation — Planet Labs (PL) — v2

> **Disclaimer:** This simulation is for educational and academic purposes only. It does not constitute financial or investment advice. Past performance does not guarantee future results. See [DISCLAIMER.md](../../../../DISCLAIMER.md) for full details.

## What Changed from V1

The original Monte Carlo (v1) used constant drift and volatility (basic GBM). It worked as a starting point, but real stock prices have:
- **Volatility clustering** — calm periods followed by wild periods
- **Sudden jumps** — earnings surprises, crashes, news events
- **Regime changes** — the market behaves differently in bull vs bear modes

V2 adds three new model types to address all of this.

---

## The Models

### V1 Baseline: Standard GBM (Geometric Brownian Motion)
- `S(t+dt) = S(t) × exp((μ − σ²/2)·dt + σ·√dt·Z)`
- Constant drift μ and volatility σ
- 10,000 paths, 504 trading days (~2 years)
- Good for learning, but the constant-vol assumption is wrong for real markets

### V2: Heston Stochastic Volatility
- Volatility is no longer fixed — it's a random process that mean-reverts
- When volatility is high, it tends to come back down (and vice versa)
- Captures the "leverage effect" — price drops often come with volatility spikes
- Parameters (κ, θ, ξ, ρ) are estimated from PL's actual historical data
- This is more realistic because PL's annualized vol swings between ~45% and ~131%

### V2: Merton Jump Diffusion
- Adds a Poisson jump process on top of regular GBM
- Detects jumps from PL's history (returns > 3 standard deviations)
- Each jump has a random size drawn from a normal distribution
- The drift is adjusted with a "compensator" so jumps don't bias the expected return
- This models the fat tails that GBM misses

### V2: HMM Regime-Switching Stress Tests
- Uses a Hidden Markov Model to automatically detect market regimes from PL's history
- Found 2 regimes: **Calm** (~85% of days, ~46% vol) and **Volatile** (~15% of days, ~131% vol)
- Stress-test scenarios are now based on these real regimes instead of made-up multipliers
- Much more empirically grounded than v1's static multiplier approach

---

## Model Parameters

| Parameter | V1 GBM | V2 Heston | V2 Jump Diffusion |
|-----------|---------|-----------|-------------------|
| Daily μ | 0.000742 | 0.000742 | 0.000742 |
| Daily σ | 0.047398 | varies (stochastic) | 0.047398 |
| Volatility of vol (ξ) | N/A | ~0.067 | N/A |
| Mean reversion (κ) | N/A | 3.0 | N/A |
| Jump intensity (λ) | N/A | N/A | ~2.5/year |
| Jump mean | N/A | N/A | ~0.07 |
| Jump std | N/A | N/A | ~0.27 |
| Price-vol correlation (ρ) | N/A | ~-0.10 | N/A |

---

## 2-Year Price Forecasts (Model Comparison)

| Metric | V1 GBM | V2 Heston | V2 Jump Diffusion |
|--------|--------|-----------|-------------------|
| Median | ~$20 | ~$16 | ~$16 |
| P5 (Downside) | ~$4 | — | — |
| P95 (Upside) | ~$117 | — | — |
| P(Profit) | ~42% | ~36% | ~37% |
| P(Double) | ~20% | — | — |
| VaR 95% | ~$4 | — | — |

The v2 models generally give more conservative medians because stochastic volatility and jumps both increase the variance-drain effect. The P(Profit) drops because the models are more realistic about downside risk.

---

## Stress Test Scenarios (HMM-Based)

Instead of using arbitrary multipliers like v1 did, v2 uses a Hidden Markov Model to detect the actual market regimes in PL's price history, then builds stress tests from those real parameters.

| Scenario | Source | Description |
|----------|--------|-------------|
| Market Crash | HMM volatile regime × amplified | Extreme negative drift + very high vol |
| Bear Market | HMM volatile regime | Negative drift + elevated vol |
| Base Case | Full historical data | All parameters as-is |
| Calm Bull | HMM calm regime | Positive drift + low vol |
| Volatile Bull | Calm drift × 3 + volatile vol | Strong growth but wild swings |

---

## Robustness Analysis

### Walk-Forward Validation (V1 GBM vs V2 Heston)

| Lookback | GBM RMSE | Heston RMSE | Tests |
|----------|----------|-------------|-------|
| 63 days | ~0.23 | ~0.23 | 19 |
| 126 days | ~0.19 | ~0.19 | 18 |
| 252 days | ~0.19 | ~0.19 | 16 |

The Heston model doesn't dramatically beat GBM in 21-day-ahead walk-forward tests. This is expected — the benefits of stochastic volatility show up more in the tails and over longer horizons, not in short-term median prediction.

### Bootstrap Confidence Intervals (5,000 resamples)
- μ 95% CI: wide range — includes negative drift
- σ 95% CI: more stable — volatility estimate is more reliable

### Parameter Sensitivity
Same grid as v1 — higher volatility consistently reduces median outcomes due to the variance drain effect (−σ²/2 term).

---

## What I Learned Building V2

1. **Constant volatility is wrong** — PL's vol ranges from 46% to 131% depending on regime. Treating it as fixed at 75% misses this completely.
2. **Jumps matter for tail risk** — GBM produces smooth, bell-curve-ish terminal distributions. Real stocks have fat tails from sudden jumps. The Merton model captures this.
3. **HMM regime detection actually works** — it found that PL spends ~85% of its time in a calm regime and ~15% in a volatile one. This matches what you'd see on a chart.
4. **More realistic models are more pessimistic** — both stochastic vol and jump diffusion give lower median prices and lower P(Profit) than basic GBM. This is because they're better at modeling downside risk.
5. **The models still have limitations** — parameters are estimated from ~5 years of data, which isn't much. The Heston parameters are rough estimates, not formally calibrated to an options surface. These are learning models, not trading models.

---

## Files

| File | What It Is |
|------|-----------|
| `pl_monte_carlo_simulation.py` | Full v2 simulation script (all three models) |
| `monte_carlo_summary.json` | Complete results including v1 vs v2 comparison |
| `mc_percentile_paths.csv` | Percentile paths from the Heston model |
| `stress_test_paths.csv` | Median paths for each HMM-based stress scenario |
| `sensitivity_analysis.csv` | Parameter sensitivity grid |
| `mc_fan_chart.png` | Fan chart comparing v1 GBM vs v2 Heston + Jump Diffusion |
| `stress_test_chart.png` | HMM-based stress test scenarios |
| `robustness_dashboard.png` | 4-panel dashboard with distributions, bootstrap, sensitivity, comparison |

---

## How to Reproduce

```bash
pip install numpy pandas yfinance matplotlib scipy hmmlearn
python pl_monte_carlo_simulation.py
```

Seed is set to 42 for reproducibility.

---

*Generated on 2026-03-17 | Data source: Yahoo Finance via yfinance*
