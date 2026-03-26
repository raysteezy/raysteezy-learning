# Price Predictions — Planet Labs (PL)

> **Not financial advice.** See [LEGAL.md](../../../LEGAL.md).

---

## V1 — Baseline (C+)

**Script:** [`prediction_v1.py`](../../../scripts/prediction_v1.py)

My first try. I used linear and polynomial regression with just the date as input. No validation, no intervals.

| Model | R² (Train) | Issue |
|-------|----------:|-------|
| Linear | 0.036 | Barely explains anything |
| Poly (deg-3) | 0.853 | Overfits — predicts $115+ in 2 years |

**What went wrong:**
- R² computed on training data only (no held-out test set)
- Only feature was the date — no returns, volume, or momentum
- No confidence intervals
- Polynomial curves shoot to infinity past the data

---

## V2 — Upgraded (A-)

**Script:** [`prediction_v2.py`](../../../scripts/prediction_v2.py)

Rebuilt everything. Added proper time-series models, walk-forward validation, and multiple features.

### ARIMA
- Handles autocorrelation (today's price depends on yesterday's)
- Auto-selected parameters via AIC
- Validated on 63 unseen trading days
- Includes 90% and 50% prediction intervals

### Ridge Regression
- 10+ features: lagged returns, SMA crossovers, volatility, volume, momentum
- Regularization to prevent overfitting
- Same 63-day walk-forward test

| Model | R² (OOS) | MAE | RMSE |
|-------|--------:|----:|-----:|
| ARIMA | 0.703 | $1.14 | $1.57 |
| Ridge | 0.737 | $1.11 | $1.47 |

**6-month forecast (ARIMA):** $36.35 — 90% CI [$34.22, $39.79]

---

## Takeaways

- V1's R² of 0.85 was on training data (cheating). V2's R² of 0.74 is on unseen data (honest).
- More features help — Ridge with returns and volume beats ARIMA with price alone.
- Predicting direction (~50%) is way harder than predicting level.
- Always use confidence intervals — single-number predictions are misleading.

---

## Charts

See the animated demo in the [main README](../../../README.md).

## Files

| File | What |
|------|------|
| `results.json` | All model metrics |
| `prices.csv` | V2 forecasts with CIs |

*Data: [Yahoo Finance](https://finance.yahoo.com/quote/PL/) via yfinance*
