"""
Planet Labs (PL) — Price Prediction V1 (Baseline)

My first attempt at predicting stock prices. I used basic regression
models and trained/tested on the same data (which I later learned
is a bad idea — it makes the models look way better than they are).

Models:
  - Linear regression (R² ~ 0.03 — basically useless)
  - Polynomial regression deg-3 (R² ~ 0.85 — overfits badly)

These models earned a C+ grade because:
  - No out-of-sample testing
  - Only used the date as input (no volume, returns, etc.)
  - No confidence intervals
  - Polynomial extrapolation goes to infinity

I'm keeping this file as a baseline so you can compare it to V2.
See prediction_v2.py for the upgraded version.

Outputs (saved to data/planet-labs/predictions/):
  - v1_model_summary.json    Results for linear + polynomial
  - v1_predicted_prices.csv  Forecasts from each model
  - v1_price_prediction.png  Chart showing both fits

Disclaimer: Educational project only — not financial advice.
"""

import json
import os
from datetime import datetime, timedelta, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

TICKER = "PL"
OUTPUT_DIR = os.path.join("data", "planet-labs", "predictions")
FORECAST_DAYS = 126
RANDOM_SEED = 42

COLORS = {
    "bg": "#0d1117", "panel": "#161b22", "text": "#e6edf3",
    "muted": "#8b949e", "accent": "#58a6ff", "green": "#3fb950",
    "orange": "#d29922", "purple": "#bc8cff", "red": "#f85149",
    "teal": "#39d2c0", "grid": "#21262d",
}


def load_data():
    """Fetch PL's full price history from Yahoo Finance."""
    stock = yf.Ticker(TICKER)
    return stock.history(period="max").reset_index()


def build_v1_models(hist):
    """Fit linear and polynomial-3 regression on date ordinals."""
    df = hist.copy()
    df["date_ordinal"] = df["Date"].apply(lambda x: x.toordinal())
    X = df["date_ordinal"].values.reshape(-1, 1)
    y = df["Close"].values

    # fit the linear model
    linear = LinearRegression().fit(X, y)
    y_pred_lin = linear.predict(X)

    # fit a degree-3 polynomial
    poly_coeffs = np.polyfit(df["date_ordinal"].values, y, 3)
    poly_model = np.poly1d(poly_coeffs)
    y_pred_poly = poly_model(df["date_ordinal"].values)

    return {
        "linear": linear,
        "poly_model": poly_model,
        "y_pred_linear": y_pred_lin,
        "y_pred_poly": y_pred_poly,
        "r2_linear": r2_score(y, y_pred_lin),
        "r2_poly": r2_score(y, y_pred_poly),
        "mae_linear": mean_absolute_error(y, y_pred_lin),
        "mae_poly": mean_absolute_error(y, y_pred_poly),
    }


def plot_v1_chart(hist, v1_models, current_price, last_date):
    """Chart showing both V1 model fits and forecasts."""
    plt.style.use("dark_background")
    hist_dates = pd.to_datetime(hist["Date"])
    future_trading = pd.bdate_range(
        start=last_date + timedelta(days=1), periods=FORECAST_DAYS
    )

    fig, ax = plt.subplots(figsize=(16, 9), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["panel"])

    ax.plot(hist_dates, hist["Close"], color=COLORS["accent"],
            linewidth=1.2, alpha=0.9, label="Historical Price")

    # shade the forecast region
    ax.axvspan(future_trading[0], future_trading[-1],
               alpha=0.06, color=COLORS["accent"])
    ax.axvline(x=last_date, color=COLORS["muted"], linestyle=":", alpha=0.5)

    # linear forecast
    v1_ordinals = np.array([d.toordinal() for d in future_trading])
    v1_linear = v1_models["linear"].predict(v1_ordinals.reshape(-1, 1))
    ax.plot(future_trading, v1_linear, color=COLORS["orange"],
            linewidth=2, linestyle="--",
            label=f"Linear (R²={v1_models['r2_linear']:.3f})")

    # polynomial forecast (clip to avoid negative prices)
    v1_poly = np.clip(v1_models["poly_model"](v1_ordinals), 0, None)
    ax.plot(future_trading, v1_poly, color=COLORS["purple"],
            linewidth=2, linestyle=":",
            label=f"Polynomial deg-3 (R²={v1_models['r2_poly']:.3f})")

    # show the historical fits too (faded)
    hist_ordinals = np.array([d.toordinal() for d in hist_dates])
    ax.plot(hist_dates,
            v1_models["linear"].predict(hist_ordinals.reshape(-1, 1)),
            color=COLORS["orange"], linewidth=0.8, alpha=0.4)
    ax.plot(hist_dates, v1_models["poly_model"](hist_ordinals),
            color=COLORS["purple"], linewidth=0.8, alpha=0.4)

    ax.scatter(last_date, current_price, color=COLORS["orange"],
               s=100, zorder=5, edgecolors="white", linewidth=1)
    ax.annotate(f"Current: ${current_price:.2f}",
                (last_date, current_price),
                textcoords="offset points", xytext=(-90, 25),
                fontsize=11, color=COLORS["orange"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=COLORS["orange"],
                                lw=1.5))

    ax.set_title("Planet Labs (PL) — V1 Price Prediction (Baseline)",
                 fontsize=20, color=COLORS["text"], fontweight="bold", pad=20)
    ax.set_xlabel("Date", fontsize=12, color=COLORS["muted"])
    ax.set_ylabel("Stock Price (USD)", fontsize=12, color=COLORS["muted"])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.xticks(rotation=45, fontsize=10, color=COLORS["muted"])
    plt.yticks(fontsize=10, color=COLORS["muted"])
    ax.grid(True, alpha=0.12, color=COLORS["grid"])
    ax.legend(loc="upper left", fontsize=9, framealpha=0.3,
              facecolor=COLORS["panel"], edgecolor=COLORS["grid"])

    fig.text(0.5, 0.01,
             "V1 baseline — no out-of-sample validation, R² on training "
             "data only. Not financial advice.",
             ha="center", fontsize=8, color=COLORS["muted"], style="italic")
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    filepath = os.path.join(OUTPUT_DIR, "v1_price_prediction.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.close()
    print(f"  Saved: {filepath}")


def save_summary(hist, v1_models, current_price):
    """Save V1 model results to JSON."""
    summary = {
        "ticker": TICKER,
        "company": "Planet Labs PBC",
        "version": "v1 — baseline (linear + polynomial)",
        "grade": "C+",
        "current_price": round(current_price, 2),
        "run_date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "training_days": len(hist),
        "models": {
            "linear_regression": {
                "r2_train": round(v1_models["r2_linear"], 4),
                "mae_train": round(v1_models["mae_linear"], 2),
                "grade": "Poor — barely explains any price movement",
            },
            "polynomial_deg3": {
                "r2_train": round(v1_models["r2_poly"], 4),
                "mae_train": round(v1_models["mae_poly"], 2),
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
    return summary


def save_predictions_csv(v1_models, last_date):
    """Save V1 forecast data to CSV."""
    future_trading = pd.bdate_range(
        start=last_date + timedelta(days=1), periods=FORECAST_DAYS
    )
    v1_ordinals = np.array([d.toordinal() for d in future_trading])
    v1_lin = v1_models["linear"].predict(v1_ordinals.reshape(-1, 1))
    v1_poly = np.clip(v1_models["poly_model"](v1_ordinals), 0, None)

    df = pd.DataFrame({
        "Date": future_trading.strftime("%Y-%m-%d"),
        "v1_Linear": np.round(v1_lin, 2),
        "v1_Polynomial": np.round(v1_poly, 2),
    })

    filepath = os.path.join(OUTPUT_DIR, "v1_predicted_prices.csv")
    df.to_csv(filepath, index=False)
    print(f"  Saved: {filepath}")


def main():
    np.random.seed(RANDOM_SEED)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"{'=' * 60}")
    print(f"  Planet Labs (PL) — Price Prediction V1 (Baseline)")
    print(f"  {timestamp}")
    print(f"{'=' * 60}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[1/4] Loading price data...")
    hist = load_data()
    current_price = float(hist["Close"].iloc[-1])
    last_date = hist["Date"].iloc[-1]
    print(f"  {len(hist)} trading days loaded")
    print(f"  Current price: ${current_price:.2f}")

    print("\n[2/4] Fitting V1 baseline models...")
    v1_models = build_v1_models(hist)
    print(f"  Linear R² (train):     {v1_models['r2_linear']:.4f}")
    print(f"  Polynomial R² (train): {v1_models['r2_poly']:.4f}")
    print(f"  (These are on training data — no out-of-sample validation)")

    print("\n[3/4] Saving results...")
    save_summary(hist, v1_models, current_price)
    save_predictions_csv(v1_models, last_date)

    print("\n[4/4] Making chart...")
    plot_v1_chart(hist, v1_models, current_price, last_date)

    end_ordinal = last_date.toordinal() + FORECAST_DAYS
    lin_6mo = v1_models["linear"].predict([[end_ordinal]])[0]
    poly_6mo = v1_models["poly_model"](end_ordinal)

    print(f"\n{'=' * 60}")
    print(f"  V1 Linear  6-month forecast:     ${lin_6mo:.2f}")
    print(f"  V1 Polynomial 6-month forecast:  ${poly_6mo:.2f}")
    print(f"  (No confidence intervals — that's one of the problems)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
