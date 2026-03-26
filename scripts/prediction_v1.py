"""
Planet Labs (PL) — Price Prediction V1 (Baseline)
===================================================
My first attempt at predicting stock prices. I used basic regression
models and trained/tested on the same data (which I later learned
is a bad idea — it makes the models look way better than they are).

Models:
  - Linear regression (R² ~ 0.03 — basically useless)
  - Polynomial regression deg-3 (R² ~ 0.85 — overfits badly)

These earned a C+ because of no out-of-sample testing, only date
as input, no confidence intervals, and polynomial goes to infinity.

See prediction_v2.py for the upgraded version.

Outputs: v1_model_summary.json, v1_predicted_prices.csv
Disclaimer: Educational project only — not financial advice.
"""

import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

TICKER = "PL"
OUTPUT_DIR = os.path.join("data", "planet-labs", "predictions")
FORECAST_DAYS = 126
RANDOM_SEED = 42


def load_data():
    try:
        import yfinance as yf
    except ImportError:
        os.system("pip install yfinance")
        import yfinance as yf
    return yf.Ticker(TICKER).history(period="max").reset_index()


def build_v1_models(hist):
    """Linear + polynomial-3 regression using only date ordinals."""
    df = hist.copy()
    df["date_ordinal"] = df["Date"].apply(lambda x: x.toordinal())
    X = df["date_ordinal"].values.reshape(-1, 1)
    y = df["Close"].values

    linear = LinearRegression().fit(X, y)
    poly_model = np.poly1d(np.polyfit(df["date_ordinal"].values, y, 3))

    return {
        "linear": linear, "poly_model": poly_model,
        "r2_linear": r2_score(y, linear.predict(X)),
        "r2_poly": r2_score(y, poly_model(df["date_ordinal"].values)),
        "mae_linear": mean_absolute_error(y, linear.predict(X)),
        "mae_poly": mean_absolute_error(y, poly_model(df["date_ordinal"].values)),
    }


def save_summary(hist, models, current_price):
    summary = {
        "ticker": TICKER, "company": "Planet Labs PBC",
        "version": "v1 — baseline (linear + polynomial)", "grade": "C+",
        "current_price": round(current_price, 2),
        "run_date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "training_days": len(hist),
        "models": {
            "linear_regression": {
                "r2_train": round(models["r2_linear"], 4),
                "mae_train": round(models["mae_linear"], 2),
                "grade": "Poor — barely explains any price movement",
            },
            "polynomial_deg3": {
                "r2_train": round(models["r2_poly"], 4),
                "mae_train": round(models["mae_poly"], 2),
                "grade": "Overfits — high training R² but unreliable forecasts",
            },
        },
        "problems": [
            "R² computed on training data (no out-of-sample testing)",
            "Only used date as input — no volume, returns, or fundamentals",
            "No confidence intervals — just single-number predictions",
            "Polynomial extrapolation goes to infinity past training data",
        ],
        "disclaimer": "Educational model only. Not financial advice.",
    }
    filepath = os.path.join(OUTPUT_DIR, "v1_model_summary.json")
    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {filepath}")


def save_predictions_csv(models, last_date):
    future = pd.bdate_range(start=last_date + timedelta(days=1), periods=FORECAST_DAYS)
    ords = np.array([d.toordinal() for d in future])

    pd.DataFrame({
        "Date": future.strftime("%Y-%m-%d"),
        "v1_Linear": np.round(models["linear"].predict(ords.reshape(-1, 1)), 2),
        "v1_Polynomial": np.round(np.clip(models["poly_model"](ords), 0, None), 2),
    }).to_csv(os.path.join(OUTPUT_DIR, "v1_predicted_prices.csv"), index=False)
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'v1_predicted_prices.csv')}")


def main():
    np.random.seed(RANDOM_SEED)
    print(f"{'=' * 60}\n  Planet Labs (PL) — Price Prediction V1 (Baseline)\n{'=' * 60}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[1/3] Loading price data...")
    hist = load_data()
    current_price = float(hist["Close"].iloc[-1])
    last_date = hist["Date"].iloc[-1]
    print(f"  {len(hist)} trading days, current price: ${current_price:.2f}")

    print("\n[2/3] Fitting V1 baseline models...")
    models = build_v1_models(hist)
    print(f"  Linear R²={models['r2_linear']:.4f}  Poly R²={models['r2_poly']:.4f}  (training data only)")

    print("\n[3/3] Saving results...")
    save_summary(hist, models, current_price)
    save_predictions_csv(models, last_date)

    end_ord = last_date.toordinal() + FORECAST_DAYS
    print(f"\n{'=' * 60}")
    print(f"  Linear 6mo: ${models['linear'].predict([[end_ord]])[0]:.2f}")
    print(f"  Poly   6mo: ${models['poly_model'](end_ord):.2f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
