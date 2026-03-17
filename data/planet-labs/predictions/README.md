# Planet Labs (PL) — Price Prediction Models

> **Disclaimer:** This analysis is for educational and academic purposes only. It does not constitute financial or investment advice. See [DISCLAIMER.md](../../../DISCLAIMER.md) for full details.

## What's In Here

I started with simple linear and polynomial regression (v1), got a C+ grade on accuracy, and then rebuilt everything with better techniques (v2). Both versions are still in the code so you can see the progression.

---

## V1 Models (Baseline — Kept for Comparison)

### Linear Regression
- Draws a straight line through all historical prices
- R² = 0.029 on training data (basically explains nothing)
- Predicts PL will drop to ~$8 in 6 months, which doesn't make much sense
- The problem: stock prices don't move in straight lines

### Polynomial Regression (Degree 3)
- Fits a cubic curve instead of a line
- R² = 0.849 on training data (looks great, but it's misleading)
- Predicts $115+ after 2 years, which is way too aggressive
- The problem: overfitting. The curve just keeps bending upward past the data

### Why V1 Got a Bad Grade
- R² was only computed on training data (no out-of-sample testing)
- Only used the date as input — no volume, no returns, no fundamentals
- No confidence intervals — just single-number predictions
- Polynomial extrapolation is unreliable by nature

---

## V2 Models (Upgraded)

### ARIMA (Auto-Regressive Integrated Moving Average)
- A proper time-series model that understands autocorrelation (today's price depends on yesterday's price)
- Parameters auto-selected using AIC (penalizes complexity to avoid overfitting)
- Walk-forward validated on 63 trading days of unseen data
- Includes 90% and 50% prediction intervals
- **Out-of-sample R²: ~0.84** — and this is on data the model never saw during training

### Ridge Regression with Features
- Instead of just "date number" as input, I gave it:
  - Lagged returns (1-day, 5-day, 21-day)
  - Moving average crossovers (SMA 50 vs SMA 200)
  - Realized volatility (21-day and 63-day)
  - Volume trends
  - Momentum indicators
- Ridge adds a penalty term to prevent overfitting (unlike the polynomial model)
- Walk-forward validated on the same 63-day test period
- **Out-of-sample R²: ~0.85**
- **Directional accuracy: ~48%** — predicting direction is harder than predicting level

---

## V1 vs V2 Comparison

| Metric | v1 Linear | v1 Polynomial | v2 ARIMA | v2 Ridge |
|--------|-----------|---------------|----------|----------|
| R² | 0.03 (train) | 0.85 (train) | ~0.84 (OOS) | ~0.85 (OOS) |
| Validation | None | None | Walk-forward | Walk-forward |
| Features | Date only | Date only | Price history | 10+ features |
| Overfitting | Underfits | Severely overfits | Controlled | Regularized |
| Confidence Intervals | No | No | Yes (bootstrap) | No |

Key lesson: v1's polynomial R² of 0.85 looked impressive but was meaningless because it was tested on the same data it trained on. v2's R² of 0.85 is actually meaningful because it's measured on data the model never saw.

---

## What I Learned

1. **Training R² is not the same as predictive R²** — you have to test on held-out data
2. **Polynomial extrapolation is dangerous** — it fits the past beautifully but predicts nonsense
3. **More features help** — Ridge with lagged returns and volume does better than ARIMA with price alone
4. **Direction is harder than level** — both models get the rough price level right but only predict the direction about half the time. This makes sense — if it were easy to predict direction, everyone would be rich
5. **Confidence intervals are essential** — a single number prediction is misleading. The interval tells you how uncertain the model is

For Monte Carlo simulation (a probabilistic approach), see [monte-carlo/](monte-carlo/).

## Charts

- `pl_price_prediction.png` — Side-by-side comparison of v1 and v2 forecasts with confidence intervals
- `pl_model_dashboard.png` — 4-panel dashboard with walk-forward validation, residuals, and model comparison table

## Important Disclaimer

These models are built for **educational purposes only**.

**This is not financial advice.** Always do your own research before making investment decisions.

## Data Source

Historical prices from [Yahoo Finance](https://finance.yahoo.com/quote/PL/) via yfinance.
