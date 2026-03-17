# Monte Carlo Simulation — Planet Labs (PL)

> **Geometric Brownian Motion (GBM) with 10,000 simulation paths, 5 stress-test scenarios, and full robustness analysis.**

⚠️ **Disclaimer:** Educational simulation only — not financial advice. Past volatility and drift do not guarantee future results.

---

## Model Overview

| Parameter | Value |
|-----------|-------|
| **Method** | Geometric Brownian Motion (GBM) |
| **Equation** | `S(t+dt) = S(t) × exp((μ − σ²/2)·dt + σ·√dt·Z)`, Z ~ N(0,1) |
| **Simulations** | 10,000 paths |
| **Forecast Horizon** | 504 trading days (~2 years) |
| **Current Price** | $24.60 |
| **Daily Drift (μ)** | 0.000742 (18.69% annualized) |
| **Daily Volatility (σ)** | 0.047398 (75.24% annualized) |
| **Historical Window** | 2021-04-26 to 2026-03-16 (1,228 trading days) |

---

## Price Forecasts

| Horizon | Median | P5 (Downside) | P95 (Upside) |
|---------|--------|---------------|--------------|
| **6 Months** | $23.38 | $9.65 | $54.65 |
| **1 Year** | $22.29 | $6.48 | $76.08 |
| **2 Years** | $20.10 | $3.52 | $117.12 |

### Key Probabilities (2-Year)

| Metric | Value |
|--------|-------|
| P(Profit) — above $24.60 | 42.3% |
| P(Double) — above $49.20 | 20.1% |
| P(Below $10) | 25.2% |
| P(Above $50) | 19.6% |

---

## Risk Metrics

| Metric | 2-Year Value |
|--------|-------------|
| **VaR 95%** | $3.52 (95% chance price stays above this) |
| **VaR 99%** | $1.70 |
| **CVaR 95%** | $2.39 (avg price in worst 5% of outcomes) |
| **CVaR 99%** | $1.24 |

---

## Stress Test Scenarios

| Scenario | Description | Median 2yr | P(Profit) | Max Drawdown |
|----------|-------------|-----------|-----------|--------------|
| 🔴 **Market Crash** | 2008-style severe downturn | $0.15 | 2.5% | −99.7% |
| 🟠 **Bear Market** | Prolonged negative sentiment | $3.28 | 10.1% | −91.5% |
| 🟡 **Base Case** | Historical parameters continue | $20.14 | 42.0% | −55.9% |
| 🟢 **Bull Market** | Major contracts, positive sentiment | $49.26 | 79.3% | −18.6% |
| 🟣 **Extreme Bull** | AI/Space sector boom | $31.44 | 55.3% | −63.5% |

---

## Robustness Analysis

### Walk-Forward Validation

The model was tested using rolling windows to predict 21-day-ahead returns:

| Lookback Window | Mean Error | MAE | RMSE | Tests |
|----------------|------------|-----|------|-------|
| 63 days (3 mo) | −0.072 | 0.446 | 0.742 | 18 |
| 126 days (6 mo) | +0.035 | 0.330 | 0.451 | 17 |
| 252 days (1 yr) | +0.110 | 0.302 | 0.387 | 15 |

Longer lookback windows produce more stable predictions (lower RMSE).

### Bootstrap Confidence Intervals (5,000 samples)

| Parameter | 95% CI |
|-----------|--------|
| Annualized μ | [−0.1974%, +0.3458%] daily → wide range includes negative drift |
| Annualized σ | [4.23%, 5.28%] daily → volatility estimate is more stable |

### Parameter Sensitivity

2-year median price under different drift (μ) and volatility (σ) shifts:

| σ \ μ | −50% | −25% | +0% | +25% | +50% |
|-------|------|------|-----|------|------|
| **−30%** | $22 | $24 | $26 | $30 | $32 |
| **+0%** | $17 | $18 | $20 | $22 | $25 |
| **+30%** | $12 | $13 | $13 | $15 | $16 |

Higher volatility consistently reduces median outcomes due to the variance drain effect (−σ²/2 term in GBM drift).

---

## Files

| File | Description |
|------|-------------|
| `pl_monte_carlo_simulation.py` | Full simulation script (reproducible) |
| `monte_carlo_summary.json` | Complete model results in JSON |
| `mc_percentile_paths.csv` | P5/P10/P25/P50/P75/P90/P95 price paths |
| `stress_test_paths.csv` | Median paths for all 5 stress scenarios |
| `sensitivity_analysis.csv` | Parameter sensitivity grid results |
| `mc_fan_chart.png` | Monte Carlo fan chart with confidence bands |
| `stress_test_chart.png` | 5 stress-test scenario comparison |
| `robustness_dashboard.png` | 4-panel robustness & risk dashboard |

---

## How to Reproduce

```bash
pip install numpy pandas yfinance matplotlib scipy
python pl_monte_carlo_simulation.py
```

All outputs will be regenerated in the current directory. The random seed (42) ensures reproducible results for the same price history.

---

*Generated on 2026-03-17 | Data source: Yahoo Finance via yfinance*
