# Monte Carlo Simulation — Planet Labs (PL)

> **Disclaimer:** This simulation is for educational and academic purposes only. It does not constitute financial or investment advice. Past performance does not guarantee future results. See [DISCLAIMER.md](../../../../DISCLAIMER.md) for full details.

## What This Is

After building the basic regression models, I wanted to try something that accounts for randomness. Stock prices don't follow a neat curve — they bounce around a lot. A Monte Carlo simulation lets you run thousands of random "what if" scenarios and see what range of outcomes is plausible.

I used Geometric Brownian Motion (GBM), which is a standard model in quantitative finance. It's not perfect (no model is), but it's a solid starting point for learning how stochastic processes work.

---

## How the Model Works

| Setting | Value |
|---------|-------|
| **Method** | Geometric Brownian Motion (GBM) |
| **The Math** | `S(t+dt) = S(t) × exp((μ − σ²/2)·dt + σ·√dt·Z)`, where Z ~ N(0,1) |
| **Number of Simulations** | 10,000 random paths |
| **Forecast Horizon** | 504 trading days (about 2 years) |
| **Starting Price** | $24.60 |
| **Daily Drift (μ)** | 0.000742 (18.69% annualized) |
| **Daily Volatility (σ)** | 0.047398 (75.24% annualized) |
| **Historical Data Used** | 2021-04-26 to 2026-03-16 (1,228 trading days) |

In plain English: the model takes PL's historical average return (drift) and how much the price bounces around (volatility), then simulates 10,000 possible futures using random noise. Each path is different because of the random element, but they all start from the same price and use the same drift/volatility.

---

## Price Forecasts

| Horizon | Median | P5 (Downside) | P95 (Upside) |
|---------|--------|---------------|--------------|
| **6 Months** | $23.38 | $9.65 | $54.65 |
| **1 Year** | $22.29 | $6.48 | $76.08 |
| **2 Years** | $20.10 | $3.52 | $117.12 |

The median goes down slightly over time, which surprised me at first. But it makes sense — the "variance drain" effect (the −σ²/2 term in the equation) pulls the median down when volatility is high. PL has really high volatility (75% annualized), so this effect is strong.

### Probability Breakdown (2-Year)

| What | Chance |
|------|--------|
| Price ends above $24.60 (profit) | 42.3% |
| Price doubles to $49.20+ | 20.1% |
| Price drops below $10 | 25.2% |
| Price goes above $50 | 19.6% |

---

## Risk Metrics

| Metric | 2-Year Value |
|--------|-------------|
| **VaR 95%** | $3.52 — 95% chance the price stays above this |
| **VaR 99%** | $1.70 |
| **CVaR 95%** | $2.39 — average price in the worst 5% of outcomes |
| **CVaR 99%** | $1.24 |

VaR (Value at Risk) tells you the worst-case scenario at a given confidence level. CVaR (Conditional VaR) is even more conservative — it averages all the outcomes that are worse than VaR.

---

## Stress Tests

I wanted to see what happens under different market conditions, so I ran the simulation 5 times with adjusted drift and volatility to model different scenarios.

| Scenario | What It Simulates | Median 2yr | P(Profit) | Max Drawdown |
|----------|-------------------|-----------|-----------|-------------|
| 🔴 **Market Crash** | 2008-style severe downturn | $0.15 | 2.5% | −99.7% |
| 🟠 **Bear Market** | Prolonged negative sentiment | $3.28 | 10.1% | −91.5% |
| 🟡 **Base Case** | Things keep going like they have been | $20.14 | 42.0% | −55.9% |
| 🟢 **Bull Market** | Major contracts, good news | $49.26 | 79.3% | −18.6% |
| 🟣 **Extreme Bull** | AI/Space sector goes crazy | $31.44 | 55.3% | −63.5% |

The Extreme Bull scenario having a lower median than the Bull scenario confused me at first, but it's because I cranked the volatility way up for that scenario, and high volatility drags the median down even when drift is very positive. Another example of variance drain in action.

---

## Robustness Checks

I didn't want to just trust the model blindly, so I ran a few tests to see how reliable it is.

### Walk-Forward Validation

Instead of just testing the model on the same data I trained it on, I tested it on data it hadn't seen. I used rolling windows: fit the model on X days of data, predict 21 days ahead, then compare to what actually happened.

| Lookback Window | Mean Error | MAE | RMSE | Tests |
|----------------|------------|-----|------|-------|
| 63 days (3 months) | −0.072 | 0.446 | 0.742 | 18 |
| 126 days (6 months) | +0.035 | 0.330 | 0.451 | 17 |
| 252 days (1 year) | +0.110 | 0.302 | 0.387 | 15 |

Longer lookback windows give more stable predictions. Makes sense — more data means better estimates of drift and volatility.

### Bootstrap Confidence Intervals (5,000 resamples)

I used bootstrapping (randomly resampling the historical returns with replacement) to see how uncertain the drift and volatility estimates are.

| Parameter | 95% Confidence Interval |
|-----------|------------------------|
| Annualized μ (drift) | [−0.1974%, +0.3458%] daily — pretty wide, includes negative drift |
| Annualized σ (volatility) | [4.23%, 5.28%] daily — more stable than drift |

The drift estimate is very uncertain — we can't even be sure it's positive. The volatility estimate is more reliable. This tells me the model's forecasts should be taken with a grain of salt, especially the direction.

### Parameter Sensitivity

I varied drift and volatility to see how much the 2-year median price changes:

| σ \ μ | −50% | −25% | +0% | +25% | +50% |
|-------|------|------|-----|------|------|
| **−30%** | $22 | $24 | $26 | $30 | $32 |
| **+0%** | $17 | $18 | $20 | $22 | $25 |
| **+30%** | $12 | $13 | $13 | $15 | $16 |

Higher volatility hurts the median in every case. This is the variance drain effect again — the −σ²/2 term in GBM means more volatility = lower expected median, even if drift stays the same.

---

## Files in This Folder

| File | What It Is |
|------|-----------|
| `pl_monte_carlo_simulation.py` | The full simulation script (you can reproduce everything) |
| `monte_carlo_summary.json` | All the model results in one JSON file |
| `mc_percentile_paths.csv` | P5/P10/P25/P50/P75/P90/P95 price paths over time |
| `stress_test_paths.csv` | Median paths for each stress test scenario |
| `sensitivity_analysis.csv` | The parameter sensitivity grid results |
| `mc_fan_chart.png` | Fan chart showing confidence bands around the forecast |
| `stress_test_chart.png` | All 5 stress test scenarios side by side |
| `robustness_dashboard.png` | 4-panel dashboard with distribution, bootstrap, sensitivity, and summary |

---

## How to Reproduce

```bash
pip install numpy pandas yfinance matplotlib scipy
python pl_monte_carlo_simulation.py
```

Everything gets regenerated in the current directory. The random seed is set to 42, so you'll get the same results as long as the price history hasn't changed.

---

*Generated on 2026-03-17 | Data source: Yahoo Finance via yfinance*
