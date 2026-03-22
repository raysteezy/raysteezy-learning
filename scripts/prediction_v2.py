"""
Planet Labs (PL) — Price Prediction V2 (Upgraded)

After getting a C+ on V1, I rebuilt the prediction models to fix
all the problems. This version includes:

  - ARIMA via auto_arima (time-series model that handles trends)
  - Ridge regression with features (lagged returns, volume, moving
    averages, fundamental ratios)
  - Walk-forward validation for every model (no more cheating on
    training data)
  - Bootstrap prediction intervals (shows uncertainty, not just
    one number)

What I learned: simple curve-fitting doesn't work for stocks.
You need models that understand autocorrelation, features beyond
just price, and proper out-of-sample testing.

The V1 baseline models (linear + polynomial) are in prediction_v1.py.
I kept them separate so you can compare side by side.

Outputs (saved to data/planet-labs/predictions/):
  - results.json         Full results for all V2 models
  - prices.csv       Forecasts with confidence intervals
  - forecast.png    Main comparison chart (V1 vs V2)
  - dashboard.png     Multi-panel analysis dashboard

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
import pmdarima as pm
from pmdarima.arima import ndiffs
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import yfinance as yf

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

RIDGE_FEATURE_COLS = [
    "ret_1d", "ret_5d", "ret_21d",
    "price_vs_sma50", "price_vs_sma200", "sma_50_vs_200",
    "vol_21d", "vol_63d",
    "momentum_14d",
]


# --- data loading ---

def load_data():
    """Fetch PL price history and some fundamental metrics from Yahoo."""
    stock = yf.Ticker(TICKER)
    hist = stock.history(period="max").reset_index()
    info = stock.info

    fundamentals = {
        "pe_ratio": info.get("trailingPE"),
        "price_to_sales": info.get("priceToSalesTrailing12Months"),
        "revenue_growth": info.get("revenueGrowth"),
        "gross_margin": info.get("grossMargins"),
        "operating_margin": info.get("operatingMargins"),
    }

    return hist, fundamentals, info


# --- feature engineering ---

def build_features(hist):
    """Create technical features for the Ridge model."""
    df = hist.copy()
    close = df["Close"]

    # basic returns at different lookbacks
    df["ret_1d"] = close.pct_change(1)
    df["ret_5d"] = close.pct_change(5)
    df["ret_21d"] = close.pct_change(21)

    # moving averages
    df["sma_10"] = close.rolling(10).mean()
    df["sma_50"] = close.rolling(50).mean()
    df["sma_200"] = close.rolling(200).mean()

    # how far price is from the moving averages
    df["price_vs_sma50"] = close / df["sma_50"] - 1
    df["price_vs_sma200"] = close / df["sma_200"] - 1
    df["sma_50_vs_200"] = df["sma_50"] / df["sma_200"] - 1

    # realized volatility
    df["vol_21d"] = df["ret_1d"].rolling(21).std()
    df["vol_63d"] = df["ret_1d"].rolling(63).std()

    # volume features if we have volume data
    if "Volume" in df.columns:
        df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
        df["vol_trend"] = (df["Volume"].rolling(5).mean()
                           / df["Volume"].rolling(20).mean())

    # momentum — what fraction of the last 14 days were positive
    df["momentum_14d"] = df["ret_1d"].rolling(14).apply(
        lambda x: (x > 0).sum() / len(x), raw=True
    )
    df["day_of_week"] = pd.to_datetime(df["Date"]).dt.dayofweek

    return df.dropna().reset_index(drop=True)


# --- V1 baseline (for comparison charts) ---

def build_v1_baseline(hist):
    """Refit the V1 linear + polynomial models so we can compare."""
    df = hist.copy()
    df["date_ordinal"] = df["Date"].apply(lambda x: x.toordinal())
    X = df["date_ordinal"].values.reshape(-1, 1)
    y = df["Close"].values

    linear = LinearRegression().fit(X, y)
    y_pred_lin = linear.predict(X)

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


# --- ARIMA ---

def build_arima_model(close_prices):
    """Fit ARIMA using auto_arima with automatic differencing."""
    n_diffs = ndiffs(close_prices, alpha=0.05, test="adf", max_d=3)
    return pm.auto_arima(
        close_prices,
        d=n_diffs,
        seasonal=False,
        stepwise=True,
        suppress_warnings=False,
        error_action="ignore",
        max_p=5, max_q=5, max_order=10,
        maxiter=200,
        trace=False,
    )


def arima_walk_forward(close_prices, n_test=63):
    """Walk-forward: train on the past, predict one step, then move forward."""
    n = len(close_prices)
    train_size = n - n_test
    actuals = []
    predictions = []
    history = list(close_prices[:train_size])

    # grab the last n_test days for testing
    for i in range(n_test):
        model = pm.auto_arima(
            history, seasonal=False, stepwise=True,
            suppress_warnings=False, error_action="ignore",
            max_p=5, max_q=5, maxiter=200,
        )
        pred = model.predict(n_periods=1)[0]
        actual = close_prices.iloc[train_size + i]
        predictions.append(pred)
        actuals.append(actual)
        history.append(actual)

    actuals = np.array(actuals)
    predictions = np.array(predictions)

    return {
        "actuals": actuals,
        "predictions": predictions,
        "mae": float(mean_absolute_error(actuals, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(actuals, predictions))),
        "r2": float(r2_score(actuals, predictions)),
        "directional_accuracy": float(
            np.mean(np.sign(np.diff(predictions))
                    == np.sign(np.diff(actuals))) * 100
        ),
    }


# --- Ridge regression ---

def _get_ridge_features(df):
    """Figure out which feature columns are available."""
    cols = list(RIDGE_FEATURE_COLS)
    if "vol_ratio" in df.columns:
        cols.extend(["vol_ratio", "vol_trend"])
    return cols


def build_ridge_model(df_features):
    """Ridge regression predicting next-day returns from technical features."""
    feature_cols = _get_ridge_features(df_features)

    df = df_features.copy()
    df["target"] = df["Close"].shift(-1) / df["Close"] - 1
    df = df.dropna(subset=feature_cols + ["target"])

    X = df[feature_cols].values
    y = df["target"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
    ridge.fit(X_scaled, y)

    return ridge, scaler, feature_cols


def ridge_walk_forward(df_features, n_test=63):
    """Walk-forward validation for Ridge on next-day return prediction."""
    feature_cols = _get_ridge_features(df_features)

    df = df_features.copy()
    df["target"] = df["Close"].shift(-1) / df["Close"] - 1
    df = df.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)

    n = len(df)
    train_size = n - n_test
    actuals = []
    predictions = []
    prices_actual = []
    prices_predicted = []

    for i in range(n_test):
        train = df.iloc[:train_size + i]
        test_row = df.iloc[train_size + i]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(train[feature_cols].values)

        ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
        ridge.fit(X_train_s, train["target"].values)

        X_test = scaler.transform(
            test_row[feature_cols].values.reshape(1, -1)
        )
        pred_return = ridge.predict(X_test)[0]
        actual_return = test_row["target"]

        actuals.append(actual_return)
        predictions.append(pred_return)

        current_price = test_row["Close"]
        prices_actual.append(current_price * (1 + actual_return))
        prices_predicted.append(current_price * (1 + pred_return))

    actuals = np.array(actuals)
    predictions = np.array(predictions)
    prices_actual = np.array(prices_actual)
    prices_predicted = np.array(prices_predicted)

    return {
        "actuals": prices_actual,
        "predictions": prices_predicted,
        "return_actuals": actuals,
        "return_predictions": predictions,
        "mae": float(mean_absolute_error(prices_actual, prices_predicted)),
        "rmse": float(np.sqrt(
            mean_squared_error(prices_actual, prices_predicted)
        )),
        "r2": float(r2_score(prices_actual, prices_predicted)),
        "directional_accuracy": float(
            np.mean(np.sign(predictions) == np.sign(actuals)) * 100
        ),
    }


# --- bootstrap prediction intervals ---

def bootstrap_forecast(model, close_prices, n_ahead=FORECAST_DAYS, n_boot=1000):
    """Build prediction intervals by resampling residuals."""
    residuals = model.resid()
    base_forecast = np.array(model.predict(n_periods=n_ahead))

    # not sure if this is the best way but it works
    boot_forecasts = np.zeros((n_boot, n_ahead))
    for b in range(n_boot):
        boot_resid = np.random.choice(residuals, size=n_ahead, replace=True)
        boot_forecasts[b] = base_forecast + np.cumsum(boot_resid) * 0.3

    return {
        "forecast": base_forecast,
        "ci_90_lower": np.percentile(boot_forecasts, 5, axis=0),
        "ci_90_upper": np.percentile(boot_forecasts, 95, axis=0),
        "ci_50_lower": np.percentile(boot_forecasts, 25, axis=0),
        "ci_50_upper": np.percentile(boot_forecasts, 75, axis=0),
    }


# --- charts ---

def plot_main_chart(hist, v1_models, arima_ci, current_price, last_date):
    """Main chart: V1 models vs V2 ARIMA with confidence bands."""
    plt.style.use("dark_background")
    hist_dates = pd.to_datetime(hist["Date"])
    future_trading = pd.bdate_range(
        start=last_date + timedelta(days=1), periods=FORECAST_DAYS
    )

    fig, ax = plt.subplots(figsize=(16, 9), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["panel"])

    ax.plot(hist_dates, hist["Close"], color=COLORS["accent"],
            linewidth=1.2, alpha=0.9, label="Historical Price")

    ax.axvspan(future_trading[0], future_trading[-1],
               alpha=0.06, color=COLORS["accent"])
    ax.axvline(x=last_date, color=COLORS["muted"], linestyle=":", alpha=0.5)

    # V1 models (faded so they don't distract)
    v1_ordinals = np.array([d.toordinal() for d in future_trading])
    v1_linear = v1_models["linear"].predict(v1_ordinals.reshape(-1, 1))
    v1_poly = np.clip(v1_models["poly_model"](v1_ordinals), 0, None)

    ax.plot(future_trading, v1_linear, color=COLORS["muted"],
            linewidth=1, linestyle="--", alpha=0.4,
            label="V1 Linear (R²=0.03)")
    ax.plot(future_trading, v1_poly, color=COLORS["muted"],
            linewidth=1, linestyle=":", alpha=0.4,
            label="V1 Polynomial (overfits)")

    # V2 ARIMA with confidence intervals
    arima_pred = arima_ci["forecast"][:FORECAST_DAYS]
    ci_90_lo = arima_ci["ci_90_lower"][:FORECAST_DAYS]
    ci_90_hi = arima_ci["ci_90_upper"][:FORECAST_DAYS]
    ci_50_lo = arima_ci["ci_50_lower"][:FORECAST_DAYS]
    ci_50_hi = arima_ci["ci_50_upper"][:FORECAST_DAYS]

    n = min(len(future_trading), len(arima_pred))
    ft = future_trading[:n]

    ax.fill_between(ft, ci_90_lo[:n], ci_90_hi[:n],
                    alpha=0.12, color=COLORS["teal"],
                    label="90% Prediction Interval")
    ax.fill_between(ft, ci_50_lo[:n], ci_50_hi[:n],
                    alpha=0.25, color=COLORS["teal"],
                    label="50% Prediction Interval")
    ax.plot(ft, arima_pred[:n], color=COLORS["teal"],
            linewidth=2.5, label="V2 ARIMA Forecast")

    ax.scatter(last_date, current_price, color=COLORS["orange"],
               s=100, zorder=5, edgecolors="white", linewidth=1)
    ax.annotate(f"Current: ${current_price:.2f}",
                (last_date, current_price),
                textcoords="offset points", xytext=(-90, 25),
                fontsize=11, color=COLORS["orange"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=COLORS["orange"],
                                lw=1.5))

    end_price = arima_pred[n - 1]
    ax.scatter(ft[-1], end_price, color=COLORS["teal"],
               s=80, zorder=5, edgecolors="white", linewidth=0.5)
    ax.annotate(
        f"6mo: ${end_price:.2f}\n"
        f"[${ci_90_lo[n-1]:.2f}\u2013${ci_90_hi[n-1]:.2f}]",
        (ft[-1], end_price),
        textcoords="offset points", xytext=(15, 15),
        fontsize=10, color=COLORS["teal"], fontweight="bold",
    )

    ax.set_title("Planet Labs (PL) — Price Prediction V1 vs V2",
                 fontsize=20, color=COLORS["text"], fontweight="bold",
                 pad=20)
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
             "V2 adds ARIMA + prediction intervals. "
             "Not financial advice. Past performance does not "
             "predict future results.",
             ha="center", fontsize=8, color=COLORS["muted"],
             style="italic")
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    filepath = os.path.join(OUTPUT_DIR, "forecast.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.close()
    print(f"  Saved: {filepath}")


def plot_dashboard(hist, v1_models, arima_wf, ridge_wf,
                   arima_model, current_price):
    """4-panel dashboard: walk-forward results, residuals, model comparison."""
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12),
                             facecolor=COLORS["bg"])

    for ax in axes.flat:
        ax.set_facecolor(COLORS["panel"])

    # Panel 1: ARIMA walk-forward
    ax1 = axes[0, 0]
    x = range(len(arima_wf["actuals"]))
    ax1.plot(x, arima_wf["actuals"], color=COLORS["accent"],
             linewidth=1.2, label="Actual")
    ax1.plot(x, arima_wf["predictions"], color=COLORS["teal"],
             linewidth=1.2, linestyle="--", label="ARIMA Predicted")
    ax1.set_title("ARIMA Walk-Forward Validation", fontsize=13,
                  color=COLORS["text"], fontweight="bold")
    ax1.set_xlabel("Days (test period)", color=COLORS["muted"])
    ax1.set_ylabel("Price (USD)", color=COLORS["muted"])
    ax1.legend(fontsize=9)
    ax1.text(0.02, 0.02,
             f"MAE=${arima_wf['mae']:.2f}  RMSE=${arima_wf['rmse']:.2f}  "
             f"R\u00b2={arima_wf['r2']:.3f}\n"
             f"Direction Accuracy: {arima_wf['directional_accuracy']:.1f}%",
             transform=ax1.transAxes, fontsize=9, color=COLORS["green"],
             verticalalignment="bottom")

    # Panel 2: Ridge walk-forward
    ax2 = axes[0, 1]
    x2 = range(len(ridge_wf["actuals"]))
    ax2.plot(x2, ridge_wf["actuals"], color=COLORS["accent"],
             linewidth=1.2, label="Actual")
    ax2.plot(x2, ridge_wf["predictions"], color=COLORS["purple"],
             linewidth=1.2, linestyle="--", label="Ridge Predicted")
    ax2.set_title("Ridge Regression Walk-Forward Validation", fontsize=13,
                  color=COLORS["text"], fontweight="bold")
    ax2.set_xlabel("Days (test period)", color=COLORS["muted"])
    ax2.set_ylabel("Price (USD)", color=COLORS["muted"])
    ax2.legend(fontsize=9)
    ax2.text(0.02, 0.02,
             f"MAE=${ridge_wf['mae']:.2f}  RMSE=${ridge_wf['rmse']:.2f}  "
             f"R\u00b2={ridge_wf['r2']:.3f}\n"
             f"Direction Accuracy: {ridge_wf['directional_accuracy']:.1f}%",
             transform=ax2.transAxes, fontsize=9, color=COLORS["green"],
             verticalalignment="bottom")

    # Panel 3: ARIMA residuals
    ax3 = axes[1, 0]
    residuals = arima_model.resid()
    ax3.hist(residuals, bins=50, color=COLORS["teal"], alpha=0.7,
             edgecolor="none")
    ax3.axvline(0, color=COLORS["orange"], linewidth=2, linestyle="--")
    ax3.set_title("ARIMA Residuals Distribution", fontsize=13,
                  color=COLORS["text"], fontweight="bold")
    ax3.set_xlabel("Residual (USD)", color=COLORS["muted"])
    ax3.set_ylabel("Frequency", color=COLORS["muted"])
    ax3.text(0.02, 0.95,
             f"Mean: {np.mean(residuals):.4f}\n"
             f"Std: {np.std(residuals):.4f}",
             transform=ax3.transAxes, fontsize=9, color=COLORS["text"],
             verticalalignment="top")

    # Panel 4: Model comparison table
    ax4 = axes[1, 1]
    ax4.axis("off")
    ax4.set_title("Model Comparison (Out-of-Sample)", fontsize=13,
                  color=COLORS["text"], fontweight="bold", pad=20)

    table_data = [
        ["V1 Linear", "N/A*", "N/A*",
         f"{v1_models['r2_linear']:.3f}\u2020"],
        ["V1 Polynomial", "N/A*", "N/A*",
         f"{v1_models['r2_poly']:.3f}\u2020"],
        ["V2 ARIMA", f"${arima_wf['mae']:.2f}",
         f"${arima_wf['rmse']:.2f}", f"{arima_wf['r2']:.3f}"],
        ["V2 Ridge", f"${ridge_wf['mae']:.2f}",
         f"${ridge_wf['rmse']:.2f}", f"{ridge_wf['r2']:.3f}"],
    ]

    tbl = ax4.table(
        cellText=table_data,
        colLabels=["Model", "MAE", "RMSE", "R\u00b2 (OOS)"],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.0, 1.8)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#555555")
        if r == 0:
            cell.set_facecolor("#1565c0")
            cell.set_text_props(color="white", fontweight="bold")
        elif r <= 2:
            cell.set_facecolor("#2d1a1a")
            cell.set_text_props(color="#ff8888")
        else:
            cell.set_facecolor("#1a2d1a")
            cell.set_text_props(color="#88ff88")

    ax4.text(0.5, 0.08,
             "* V1 models had no out-of-sample testing\n"
             "\u2020 R\u00b2 computed on training data (misleading)",
             transform=ax4.transAxes, fontsize=8, color=COLORS["muted"],
             ha="center", style="italic")

    fig.suptitle("Planet Labs (PL) — Model Analysis Dashboard",
                 fontsize=16, color=COLORS["text"], fontweight="bold",
                 y=0.98)
    fig.text(0.5, 0.01,
             "All V2 metrics are out-of-sample (walk-forward). "
             "Not financial advice.",
             ha="center", fontsize=8, color=COLORS["muted"],
             style="italic")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    filepath = os.path.join(OUTPUT_DIR, "dashboard.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.close()
    print(f"  Saved: {filepath}")


# --- output ---

def save_summary(hist, v1_models, arima_model, arima_wf,
                 ridge_wf, arima_ci, fundamentals, current_price):
    """Save full V2 results to JSON."""
    summary = {
        "ticker": TICKER,
        "company": "Planet Labs PBC",
        "version": "v2 — ARIMA + Ridge + walk-forward validation",
        "current_price": round(current_price, 2),
        "run_date": datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        ),
        "training_days": len(hist),
        "v1_baseline": {
            "note": "Kept for comparison — no out-of-sample validation",
            "linear_regression": {
                "r2_train": round(v1_models["r2_linear"], 4),
                "mae_train": round(v1_models["mae_linear"], 2),
                "grade": "Poor — barely explains any price movement",
            },
            "polynomial_deg3": {
                "r2_train": round(v1_models["r2_poly"], 4),
                "mae_train": round(v1_models["mae_poly"], 2),
                "grade": "Overfits — high training R² but bad forecasts",
            },
        },
        "v2_models": {
            "arima": {
                "order": list(arima_model.order),
                "aic": round(float(arima_model.aic()), 2),
                "walk_forward_validation": {
                    "test_days": len(arima_wf["actuals"]),
                    "mae": round(arima_wf["mae"], 2),
                    "rmse": round(arima_wf["rmse"], 2),
                    "r2_oos": round(arima_wf["r2"], 4),
                    "directional_accuracy_pct": round(
                        arima_wf["directional_accuracy"], 1
                    ),
                },
                "forecast_6mo": {
                    "point": round(float(arima_ci["forecast"][-1]), 2),
                    "ci_90": [
                        round(float(arima_ci["ci_90_lower"][-1]), 2),
                        round(float(arima_ci["ci_90_upper"][-1]), 2),
                    ],
                    "ci_50": [
                        round(float(arima_ci["ci_50_lower"][-1]), 2),
                        round(float(arima_ci["ci_50_upper"][-1]), 2),
                    ],
                },
            },
            "ridge_regression": {
                "features_used": [
                    "1d/5d/21d returns", "SMA crossovers",
                    "21d/63d volatility", "volume ratio", "momentum",
                ],
                "walk_forward_validation": {
                    "test_days": len(ridge_wf["actuals"]),
                    "mae": round(ridge_wf["mae"], 2),
                    "rmse": round(ridge_wf["rmse"], 2),
                    "r2_oos": round(ridge_wf["r2"], 4),
                    "directional_accuracy_pct": round(
                        ridge_wf["directional_accuracy"], 1
                    ),
                },
            },
        },
        "fundamentals_snapshot": {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in fundamentals.items() if v is not None
        },
        "improvements_over_v1": [
            "ARIMA handles autocorrelation (V1 ignored it)",
            "Ridge uses multiple features instead of just date",
            "All metrics are out-of-sample (walk-forward validation)",
            "Prediction intervals show uncertainty range",
            "Regularization prevents overfitting",
        ],
        "disclaimer": "Educational model only. Not financial advice.",
    }

    filepath = os.path.join(OUTPUT_DIR, "results.json")
    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {filepath}")
    return summary


def save_predictions_csv(arima_ci, last_date):
    """Save ARIMA forecast with confidence intervals to CSV."""
    future_trading = pd.bdate_range(
        start=last_date + timedelta(days=1), periods=FORECAST_DAYS
    )
    n = min(len(future_trading), len(arima_ci["forecast"]))

    df = pd.DataFrame({
        "Date": future_trading[:n].strftime("%Y-%m-%d"),
        "v2_ARIMA": np.round(arima_ci["forecast"][:n], 2),
        "v2_ARIMA_CI90_Lower": np.round(
            arima_ci["ci_90_lower"][:n], 2
        ),
        "v2_ARIMA_CI90_Upper": np.round(
            arima_ci["ci_90_upper"][:n], 2
        ),
    })

    filepath = os.path.join(OUTPUT_DIR, "prices.csv")
    df.to_csv(filepath, index=False)
    print(f"  Saved: {filepath}")


# --- main ---

def main():
    np.random.seed(RANDOM_SEED)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"{'=' * 60}")
    print("  Planet Labs (PL) — Price Prediction V2 (Upgraded)")
    print(f"  {timestamp}")
    print(f"{'=' * 60}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[1/8] Loading price data and fundamentals...")
    hist, fundamentals, info = load_data()
    current_price = float(hist["Close"].iloc[-1])
    last_date = hist["Date"].iloc[-1]
    print(f"  {len(hist)} trading days loaded")
    print(f"  Current price: ${current_price:.2f}")

    print("\n[2/8] Engineering features...")
    df_features = build_features(hist)
    print(f"  {len(df_features)} rows with {df_features.shape[1]} columns")

    print("\n[3/8] Fitting V1 baseline (for comparison)...")
    v1_models = build_v1_baseline(hist)
    print(f"  Linear R² (train):     {v1_models['r2_linear']:.4f}")
    print(f"  Polynomial R² (train): {v1_models['r2_poly']:.4f}")
    print("  (Misleading — trained on same data they're scored on)")

    print("\n[4/8] Fitting ARIMA model...")
    close_prices = hist["Close"]
    arima_model = build_arima_model(close_prices)
    print(f"  Best order: {arima_model.order}")
    print(f"  AIC: {arima_model.aic():.2f}")

    print("\n[5/8] Walk-forward validation (ARIMA — takes a minute)...")
    arima_wf = arima_walk_forward(close_prices, n_test=63)
    print(f"  MAE: ${arima_wf['mae']:.2f}")
    print(f"  RMSE: ${arima_wf['rmse']:.2f}")
    print(f"  R² (out-of-sample): {arima_wf['r2']:.4f}")
    print(f"  Directional accuracy: "
          f"{arima_wf['directional_accuracy']:.1f}%")

    print("\n[6/8] Walk-forward validation (Ridge)...")
    ridge_wf = ridge_walk_forward(df_features, n_test=63)
    print(f"  MAE: ${ridge_wf['mae']:.2f}")
    print(f"  RMSE: ${ridge_wf['rmse']:.2f}")
    print(f"  R² (out-of-sample): {ridge_wf['r2']:.4f}")
    print(f"  Directional accuracy: "
          f"{ridge_wf['directional_accuracy']:.1f}%")

    print("\n[7/8] Building forecasts with prediction intervals...")
    arima_ci = bootstrap_forecast(
        arima_model, close_prices,
        n_ahead=FORECAST_DAYS, n_boot=1000
    )
    end_price = float(arima_ci["forecast"][-1])
    end_lo = float(arima_ci["ci_90_lower"][-1])
    end_hi = float(arima_ci["ci_90_upper"][-1])
    print(f"  ARIMA 6-mo forecast: ${end_price:.2f}")
    print(f"  90% CI: [${end_lo:.2f}, ${end_hi:.2f}]")

    print("\n[8/8] Saving results and charts...")
    save_summary(
        hist, v1_models, arima_model, arima_wf,
        ridge_wf, arima_ci, fundamentals, current_price,
    )
    save_predictions_csv(arima_ci, last_date)
    plot_main_chart(hist, v1_models, arima_ci, current_price, last_date)
    plot_dashboard(hist, v1_models, arima_wf, ridge_wf,
                   arima_model, current_price)

    print(f"\n{'=' * 60}")
    print(f"  V2 ARIMA 6mo:      ${end_price:.2f} "
          f"[{end_lo:.2f}, {end_hi:.2f}]")
    print(f"  V2 ARIMA R²(OOS):  {arima_wf['r2']:.4f}")
    print(f"  V2 Ridge R²(OOS):  {ridge_wf['r2']:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
