# Planet Labs (PL) — Price Prediction Models

> **Disclaimer:** This analysis is for educational and academic purposes only. It does not constitute financial or investment advice. See [DISCLAIMER.md](../../../DISCLAIMER.md) for full details.


## Models

### Linear Regression
- Fits a straight trend line through all historical prices
- R² = 0.029 (very low — PL's price movement is highly non-linear)
- Predicts a conservative, slow-growth trajectory
- Useful as a baseline / lower-bound estimate

### Polynomial Regression (Degree 3)
- Fits a cubic curve that captures the recent upward acceleration
- R² = 0.849 (strong fit to historical data)
- Predicts aggressive growth based on recent momentum
- Caution: polynomial extrapolation can overshoot significantly

## Predictions (from $24.60 current price)

| Timeframe | Linear | Polynomial |
|-----------|--------|------------|
| 6 months  | $8.06  | $37.17     |
| 1 year    | $8.36  | $57.15     |
| 2 years   | $8.97  | $115.49    |

## Visualizations

- `pl_price_prediction.png` — Main chart with historical data and 2-year forecast
- `pl_model_dashboard.png` — 4-panel analysis dashboard with model fits, residuals, forecasts, and summary table

## Important Disclaimer

These are simple trend-extrapolation models built for **educational purposes only**. They do not account for:

- Earnings surprises or fundamental changes
- Macroeconomic conditions (interest rates, inflation)
- Industry competition or regulatory changes
- Market sentiment, momentum shifts, or black swan events
- Planet Labs' specific business metrics (ARR growth, contract wins, satellite launches)

**This is not financial advice.** Always do your own research before making investment decisions.

## Data Source

Historical prices sourced from [Yahoo Finance](https://finance.yahoo.com/quote/PL/) via yfinance.
