# Planet Labs (PL) — Price Prediction Models

> **Disclaimer:** This analysis is for educational and academic purposes only. It does not constitute financial or investment advice. See [DISCLAIMER.md](../../../DISCLAIMER.md) for full details.


## What I Did

I tried two regression models on PL's historical stock prices to see if I could forecast the price 6 months, 1 year, and 2 years out. This was my first real attempt at machine learning on financial data, so I kept it simple.

### Linear Regression
- Draws a straight line through all the historical prices
- R² = 0.029 (pretty bad — the price doesn't move in a straight line)
- Gives a very conservative prediction
- I used this mostly as a baseline to compare against

### Polynomial Regression (Degree 3)
- Fits a curve (cubic polynomial) instead of a straight line
- R² = 0.849 (much better fit on the training data)
- Captures the recent upward trend in the price
- But I learned that polynomials can predict crazy numbers once you go past the training data — they just keep curving up or down forever

## Prediction Results (starting from $24.60)

| Timeframe | Linear | Polynomial |
|-----------|--------|------------|
| 6 months  | $8.06  | $37.17     |
| 1 year    | $8.36  | $57.15     |
| 2 years   | $8.97  | $115.49    |

You can see the problem — the linear model says the price is going down, and the polynomial says it's going to the moon. The truth is probably somewhere in between, and neither model really "knows" what will happen.

## Charts

- `pl_price_prediction.png` — Main chart showing historical data and the 2-year forecasts from both models
- `pl_model_dashboard.png` — 4-panel dashboard with model fits, residuals, forecasts, and a summary table

## What These Models Don't Know About

These are simple curve-fitting models. They have no idea about:

- Earnings surprises or changes in the business
- Interest rates, inflation, or the economy
- Competitors or new regulations
- Whether people are feeling bullish or bearish
- Planet Labs' actual business metrics (contracts, satellite launches, revenue growth)

I built these to learn how regression works, not to actually predict the stock price. For a more advanced approach, check out the [Monte Carlo simulation](monte-carlo/).

## Important Disclaimer

These are simple trend-extrapolation models built for **educational purposes only**.

**This is not financial advice.** Always do your own research before making investment decisions.

## Data Source

Historical prices from [Yahoo Finance](https://finance.yahoo.com/quote/PL/) via yfinance.
