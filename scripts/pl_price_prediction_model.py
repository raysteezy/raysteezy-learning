"""
Planet Labs (PL) — Price Prediction Model
==========================================
Fits a linear regression and a polynomial regression (degree 3)
on PL's historical stock prices, then forecasts 2 years ahead.

Outputs:
  - data/planet-labs/predictions/model_summary.json
  - data/planet-labs/predictions/predicted_prices.csv
  - data/planet-labs/predictions/pl_price_prediction.png
  - data/planet-labs/predictions/pl_model_dashboard.png

This was my first attempt at using ML on real stock data.
The main takeaway: simple regression isn't great at predicting
stock prices, but it taught me a lot about how models work.

Disclaimer: Educational project only — not financial advice.
"""

import json
import os
from datetime import datetime, timedelta, timezone

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend (needed for servers/CI)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# ── Settings ─────────────────────────────────────────────────────────
TICKER = "PL"
OUTPUT_DIR = os.path.join("data", "planet-labs", "predictions")

# Dark theme colors for the charts
COLORS = {
    "bg": "#0d1117",
    "panel": "#161b22",
    "text": "#e6edf3",
    "muted": "#8b949e",
    "accent": "#58a6ff",
    "green": "#3fb950",
    "orange": "#d29922",
    "purple": "#bc8cff",
    "grid": "#21262d",
}


def load_price_data():
    """
    Load all available historical prices for PL from Yahoo Finance.

    We use yfinance with period="max" to get every trading day since
    PL went public. This gives us as much data as possible to train on.
    """
    try:
        import yfinance as yf
        stock = yf.Ticker(TICKER)
        hist = stock.history(period="max").reset_index()
    except ImportError:
        os.system("pip install yfinance")
        import yfinance as yf
        stock = yf.Ticker(TICKER)
        hist = stock.history(period="max").reset_index()
    return hist


def build_models(hist):
    """
    Fit both regression models on the historical data.

    How it works:
    1. Convert each date to a number (ordinal) so the models can use it
    2. Fit a linear regression — just a straight line through the prices
    3. Fit a degree-3 polynomial — a curve that can bend and follow trends
    4. Calculate R² and MAE to see how good each model is

    Returns a dictionary with both fitted models and their metrics.
    """
    # Turn dates into numbers (days since year 1) so regression can use them
    hist["date_ordinal"] = hist["Date"].apply(lambda x: x.toordinal())
    X = hist["date_ordinal"].values.reshape(-1, 1)  # sklearn wants 2D input
    y = hist["Close"].values

    # Linear regression: y = mx + b
    linear = LinearRegression()
    linear.fit(X, y)
    y_pred_linear = linear.predict(X)

    # Polynomial regression (degree 3): y = ax³ + bx² + cx + d
    poly_coeffs = np.polyfit(hist["date_ordinal"].values, y, 3)
    poly_model = np.poly1d(poly_coeffs)
    y_pred_poly = poly_model(hist["date_ordinal"].values)

    # R² = how much of the price movement the model explains (1.0 = perfect)
    r2_lin = r2_score(y, y_pred_linear)
    r2_poly = r2_score(y, y_pred_poly)

    # MAE = on average, how many dollars off is the prediction
    mae_lin = mean_absolute_error(y, y_pred_linear)

    return {
        "linear": linear,
        "poly_model": poly_model,
        "y_pred_linear": y_pred_linear,
        "y_pred_poly": y_pred_poly,
        "r2_linear": r2_lin,
        "r2_poly": r2_poly,
        "mae": mae_lin,
    }


def generate_predictions(hist, models):
    """
    Use the fitted models to predict prices for the next 2 years.

    We generate one prediction per calendar day (730 days total).
    The polynomial predictions get clipped at 0 because a stock
    price can't go negative.
    """
    last_date = hist["Date"].iloc[-1]
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=365 * 2,
        freq="D",
    )
    future_ordinals = np.array([d.toordinal() for d in future_dates])

    pred_linear = models["linear"].predict(future_ordinals.reshape(-1, 1))
    pred_poly = np.clip(models["poly_model"](future_ordinals), 0, None)

    return future_dates, pred_linear, pred_poly


def save_model_summary(hist, models, pred_linear, pred_poly):
    """Save all model results to a JSON file for easy reference."""
    current_price = float(hist["Close"].iloc[-1])

    summary = {
        "ticker": TICKER,
        "company": "Planet Labs PBC",
        "training_period": (
            f"{hist['Date'].iloc[0].strftime('%Y-%m-%d')} to "
            f"{hist['Date'].iloc[-1].strftime('%Y-%m-%d')}"
        ),
        "training_days": int(len(hist)),
        "current_price": round(current_price, 2),
        "models": {
            "linear_regression": {
                "r_squared": round(models["r2_linear"], 4),
                "mean_absolute_error": round(models["mae"], 2),
                "slope_per_year": round(float(models["linear"].coef_[0]) * 365.25, 2),
                "predictions": {
                    "6_months": round(float(pred_linear[182]), 2),
                    "1_year": round(float(pred_linear[364]), 2),
                    "2_years": round(float(pred_linear[-1]), 2),
                },
            },
            "polynomial_regression_deg3": {
                "r_squared": round(models["r2_poly"], 4),
                "predictions": {
                    "6_months": round(float(pred_poly[182]), 2),
                    "1_year": round(float(pred_poly[364]), 2),
                    "2_years": round(float(pred_poly[-1]), 2),
                },
            },
        },
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "disclaimer": "Educational model only. Not financial advice.",
    }

    filepath = os.path.join(OUTPUT_DIR, "model_summary.json")
    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {filepath}")
    return summary


def save_predictions_csv(future_dates, pred_linear, pred_poly):
    """Save the daily predictions to a CSV so they're easy to look at."""
    df = pd.DataFrame({
        "Date": future_dates.strftime("%Y-%m-%d"),
        "Linear_Prediction": np.round(pred_linear, 2),
        "Polynomial_Prediction": np.round(pred_poly, 2),
    })
    filepath = os.path.join(OUTPUT_DIR, "predicted_prices.csv")
    df.to_csv(filepath, index=False)
    print(f"  Saved: {filepath}")


def plot_main_chart(hist, models, future_dates, pred_linear, pred_poly):
    """
    Create the main prediction chart.

    Shows historical prices on the left, then the forecast zone on
    the right with both model predictions. Dot annotations mark
    the 6-month, 1-year, and 2-year price targets.
    """
    plt.style.use("dark_background")
    hist_dates = pd.to_datetime(hist["Date"])
    current_price = hist["Close"].iloc[-1]
    last_date = hist_dates.iloc[-1]

    fig, ax = plt.subplots(figsize=(16, 9), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["panel"])

    # Historical prices
    ax.plot(
        hist_dates, hist["Close"],
        color=COLORS["accent"], linewidth=1.2, alpha=0.9,
        label="Historical Price",
    )

    # Faint linear fit line over historical data
    ax.plot(
        hist_dates, models["y_pred_linear"],
        color=COLORS["muted"], linewidth=1, linestyle="--", alpha=0.4,
    )

    # Shade the forecast zone
    ax.axvspan(future_dates[0], future_dates[-1], alpha=0.06, color=COLORS["accent"])
    ax.axvline(x=last_date, color=COLORS["muted"], linestyle=":", alpha=0.5)

    # Future predictions
    ax.plot(
        future_dates, pred_linear,
        color=COLORS["green"], linewidth=2.5, linestyle="--",
        label="Linear Regression", alpha=0.9,
    )
    ax.plot(
        future_dates, pred_poly,
        color=COLORS["purple"], linewidth=2.5,
        label="Polynomial (deg 3)", alpha=0.9,
    )

    # Confidence band around linear prediction (using MAE)
    ax.fill_between(
        future_dates,
        pred_linear - models["mae"],
        pred_linear + models["mae"],
        color=COLORS["green"], alpha=0.08,
    )

    # Annotate milestone predictions (6 months, 1 year, 2 years)
    for idx, label in [(182, "6 Mo"), (364, "1 Yr"), (-1, "2 Yr")]:
        # Linear model dot
        ax.scatter(
            future_dates[idx], pred_linear[idx],
            color=COLORS["green"], s=60, zorder=5,
            edgecolors="white", linewidth=0.5,
        )
        ax.annotate(
            f"${pred_linear[idx]:.2f}",
            (future_dates[idx], pred_linear[idx]),
            textcoords="offset points", xytext=(0, 18),
            fontsize=9, color=COLORS["green"], fontweight="bold", ha="center",
        )

        # Polynomial model dot
        ax.scatter(
            future_dates[idx], pred_poly[idx],
            color=COLORS["purple"], s=60, zorder=5,
            edgecolors="white", linewidth=0.5,
        )
        ax.annotate(
            f"${pred_poly[idx]:.2f}",
            (future_dates[idx], pred_poly[idx]),
            textcoords="offset points", xytext=(0, 18),
            fontsize=9, color=COLORS["purple"], fontweight="bold", ha="center",
        )

    # Mark the current price
    ax.scatter(
        last_date, current_price,
        color=COLORS["orange"], s=100, zorder=5,
        edgecolors="white", linewidth=1,
    )
    ax.annotate(
        f"Current: ${current_price:.2f}",
        (last_date, current_price),
        textcoords="offset points", xytext=(-90, 25),
        fontsize=11, color=COLORS["orange"], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=COLORS["orange"], lw=1.5),
    )

    # Labels and formatting
    ax.set_title(
        "Planet Labs (PL) — 2-Year Price Prediction",
        fontsize=20, color=COLORS["text"], fontweight="bold", pad=20,
    )
    ax.set_xlabel("Date", fontsize=12, color=COLORS["muted"])
    ax.set_ylabel("Stock Price (USD)", fontsize=12, color=COLORS["muted"])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45, fontsize=10, color=COLORS["muted"])
    plt.yticks(fontsize=10, color=COLORS["muted"])
    ax.grid(True, alpha=0.12, color=COLORS["grid"])
    ax.legend(
        loc="upper left", fontsize=11, framealpha=0.3,
        facecolor=COLORS["panel"], edgecolor=COLORS["grid"],
    )

    # Disclaimer at the bottom of the chart
    fig.text(
        0.5, 0.01,
        "Disclaimer: Educational model only. Not financial advice. "
        "Past performance does not predict future results.",
        ha="center", fontsize=8, color=COLORS["muted"], style="italic",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    filepath = os.path.join(OUTPUT_DIR, "pl_price_prediction.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"  Saved: {filepath}")


def main():
    """Run everything: load data, fit models, predict, save, plot."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"{'=' * 60}")
    print(f"  Planet Labs (PL) — Price Prediction Model")
    print(f"  {timestamp}")
    print(f"{'=' * 60}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[1/5] Loading price data...")
    hist = load_price_data()
    print(f"  {len(hist)} trading days loaded")

    print("\n[2/5] Building regression models...")
    models = build_models(hist)
    print(f"  Linear R²:     {models['r2_linear']:.4f}")
    print(f"  Polynomial R²: {models['r2_poly']:.4f}")

    print("\n[3/5] Generating 2-year predictions...")
    future_dates, pred_linear, pred_poly = generate_predictions(hist, models)

    print("\n[4/5] Saving model data...")
    summary = save_model_summary(hist, models, pred_linear, pred_poly)
    save_predictions_csv(future_dates, pred_linear, pred_poly)

    print("\n[5/5] Generating charts...")
    plot_main_chart(hist, models, future_dates, pred_linear, pred_poly)

    # Final summary
    lin_2yr = summary["models"]["linear_regression"]["predictions"]["2_years"]
    poly_2yr = summary["models"]["polynomial_regression_deg3"]["predictions"]["2_years"]
    print(f"\n{'=' * 60}")
    print(f"  Current Price:  ${summary['current_price']}")
    print(f"  Linear 2yr:     ${lin_2yr}")
    print(f"  Polynomial 2yr: ${poly_2yr}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
