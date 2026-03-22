#!/usr/bin/env python3
"""
(PL) — Monte Carlo Simulation V1 (Baseline)
My first Monte Carlo simulation. Uses standard Geometric Brownian
Motion (GBM) with constant drift and volatility. The stress tests
are just made-up multipliers — not based on real market data.

This earned a C+ grade alongside V1 predictions because:
  - Constant volatility is unrealistic (real markets have vol clustering)
  - No jump modeling (can't simulate crashes or spikes)
  - Stress scenarios are arbitrary, not data-driven
  - No walk-forward validation

I'm keeping this as a baseline so you can compare it to V2.
See monte_carlo_v2.py for the upgraded version.

Outputs (saved to current directory):
  - v1_mc_fan_chart.png          Fan chart showing GBM paths
  - v1_mc_summary.json           Summary statistics
  - v1_mc_percentile_paths.csv   Percentile price paths

Disclaimer: Educational simulation only — not financial advice.
"""

import json
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

TICKER = "PL"
N_SIMULATIONS = 10_000
FORECAST_DAYS = 504
RANDOM_SEED = 42

SCENARIOS = {
    "market_crash": {"name": "Market Crash (2008-style)", "mu_mult": -3.0, "sig_mult": 2.5},
    "bear_market": {"name": "Prolonged Bear Market", "mu_mult": -1.5, "sig_mult": 1.5},
    "base_case": {"name": "Base Case (Historical)", "mu_mult": 1.0, "sig_mult": 1.0},
    "bull_market": {"name": "Strong Bull Market", "mu_mult": 3.0, "sig_mult": 0.8},
    "extreme_bull": {"name": "Extreme Bull (Sector Boom)", "mu_mult": 5.0, "sig_mult": 2.0},
}


def fetch_data():
    """Grab PL price history and compute log returns."""
    stock = yf.Ticker(TICKER)
    hist = stock.history(period="max")
    hist = hist[hist.index >= "2021-01-01"]
    prices = hist["Close"].dropna()
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return prices, log_returns


def run_gbm(current_price, mu, sigma, n_days=FORECAST_DAYS, n_sims=N_SIMULATIONS):
    """Run a basic GBM simulation and return the price paths."""
    Z = np.random.standard_normal((n_days, n_sims))
    drift = (mu - 0.5 * sigma ** 2)
    diffusion = sigma * Z
    log_paths = np.cumsum(drift + diffusion, axis=0)
    log_paths = np.vstack([np.zeros(n_sims), log_paths])
    return current_price * np.exp(log_paths)


def run_stress_tests(mu_daily, sigma_daily, current_price):
    """Run GBM under each stress scenario."""
    stress_results = {}
    stress_paths = {}

    for key, sc in SCENARIOS.items():
        sc_mu = mu_daily * sc["mu_mult"]
        sc_sigma = sigma_daily * sc["sig_mult"]
        sc_paths = run_gbm(current_price, sc_mu, sc_sigma)
        stress_paths[key] = sc_paths

        final_prices = sc_paths[FORECAST_DAYS]
        stress_results[key] = {
            "name": sc["name"],
            "median_2yr": round(float(np.median(final_prices)), 2),
            "p10_2yr": round(float(np.percentile(final_prices, 10)), 2),
            "p90_2yr": round(float(np.percentile(final_prices, 90)), 2),
            "prob_profit": round(
                float(np.mean(final_prices > current_price) * 100), 1
            ),
        }
        print(f"  {sc['name']:35s} -> "
              f"Median: ${stress_results[key]['median_2yr']:.2f}")

    return stress_results, stress_paths


def plot_fan_chart(prices, paths, current_price):
    """Fan chart showing GBM percentile bands."""
    plt.style.use("dark_background")
    forecast_start = prices.index[-1]
    forecast_dates = pd.bdate_range(
        start=forecast_start, periods=FORECAST_DAYS + 1
    )

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(prices.index, prices.values, color="#4fc3f7",
            linewidth=1.2, label="Historical Price")

    # calculate percentile bands for the fan
    pct = np.percentile(paths, [5, 25, 50, 75, 95], axis=1)
    ax.fill_between(forecast_dates, pct[0], pct[4],
                    alpha=0.15, color="#888888", label="90% CI")
    ax.fill_between(forecast_dates, pct[1], pct[3],
                    alpha=0.25, color="#888888", label="50% CI")
    ax.plot(forecast_dates, pct[2], "--", color="#FFA500",
            linewidth=2, label="GBM Median")

    ax.axhline(y=current_price, color="white", linestyle=":", alpha=0.3)
    ax.annotate(f"Current: ${current_price:.2f}",
                xy=(forecast_start, current_price),
                fontsize=10, color="#FFA500", fontweight="bold",
                xytext=(-120, 20), textcoords="offset points")

    # mark the 6mo, 1yr, 2yr points on the median line
    for label, day_idx in {"6 Mo": 126, "1 Yr": 252, "2 Yr": 504}.items():
        val = pct[2][day_idx]
        ax.plot(forecast_dates[day_idx], val, "o", color="white",
                markersize=6, zorder=5)
        ax.annotate(f"{label}: ${val:.2f}",
                    xy=(forecast_dates[day_idx], val),
                    fontsize=9, color="#FFA500",
                    xytext=(15, 15), textcoords="offset points")

    ax.set_title(f"Planet Labs ({TICKER}) — V1 Monte Carlo "
                 f"({N_SIMULATIONS:,} GBM paths)",
                 fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Stock Price (USD)", fontsize=12)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.7)
    fig.text(0.5, 0.01,
             "V1 baseline — constant volatility GBM. "
             "Not financial advice.",
             ha="center", fontsize=8, alpha=0.5)
    plt.tight_layout()
    fig.savefig("v1_mc_fan_chart.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print("  Saved: v1_mc_fan_chart.png")


def save_results(prices, paths, current_price, mu_daily,
                 sigma_daily, log_returns, stress_results):
    """Save summary JSON and percentile paths CSV."""
    terminal = paths[FORECAST_DAYS]
    var_95 = round(float(np.percentile(terminal, 5)), 2)
    var_99 = round(float(np.percentile(terminal, 1)), 2)

    stock = yf.Ticker(TICKER)
    summary = {
        "ticker": TICKER,
        "company": stock.info.get("longName", "Planet Labs PBC"),
        "version": "v1 — constant-volatility GBM",
        "grade": "C+",
        "current_price": round(float(current_price), 2),
        "run_date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "parameters": {
            "trading_days": len(log_returns),
            "daily_drift": round(float(mu_daily), 6),
            "daily_volatility": round(float(sigma_daily), 6),
            "annual_drift": round(float(mu_daily * 252), 4),
            "annual_volatility": round(float(sigma_daily * np.sqrt(252)), 4),
        },
        "results": {
            "6_months": {
                "median": round(float(np.median(paths[126])), 2),
                "p5": round(float(np.percentile(paths[126], 5)), 2),
                "p95": round(float(np.percentile(paths[126], 95)), 2),
            },
            "1_year": {
                "median": round(float(np.median(paths[252])), 2),
                "p5": round(float(np.percentile(paths[252], 5)), 2),
                "p95": round(float(np.percentile(paths[252], 95)), 2),
            },
            "2_years": {
                "median": round(float(np.median(terminal)), 2),
                "p5": round(float(np.percentile(terminal, 5)), 2),
                "p95": round(float(np.percentile(terminal, 95)), 2),
            },
        },
        "risk_metrics": {
            "var_95": var_95,
            "var_99": var_99,
            "cvar_95": round(float(np.mean(terminal[terminal <= var_95])), 2),
            "prob_profit": round(
                float(np.mean(terminal > current_price) * 100), 1
            ),
            "prob_double": round(
                float(np.mean(terminal > 2 * current_price) * 100), 1
            ),
        },
        "stress_tests": stress_results,
        "problems": [
            "Constant volatility assumption (real markets have vol clustering)",
            "No jump modeling (can't simulate sudden crashes or spikes)",
            "Stress scenarios use arbitrary multipliers, not real data",
            "No walk-forward validation",
        ],
        "disclaimer": (
            "Educational Monte Carlo simulation. Not financial advice. "
            "Past volatility does not guarantee future results."
        ),
    }

    with open("v1_mc_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("  Saved: v1_mc_summary.json")

    # save percentile paths to CSV
    forecast_dates = pd.bdate_range(
        start=prices.index[-1], periods=FORECAST_DAYS + 1
    )
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    pct_paths = np.percentile(paths, percentiles, axis=1)
    pct_df = pd.DataFrame(
        {f"P{p}": pct_paths[i] for i, p in enumerate(percentiles)},
        index=forecast_dates,
    )
    pct_df.index.name = "date"
    pct_df.to_csv("v1_mc_percentile_paths.csv")
    print("  Saved: v1_mc_percentile_paths.csv")


def main():
    np.random.seed(RANDOM_SEED)

    print(f"[1/4] Fetching {TICKER} price history ...")
    prices, log_returns = fetch_data()
    current_price = prices.iloc[-1]
    mu_daily = log_returns.mean()
    sigma_daily = log_returns.std()

    print(f"  Current price : ${current_price:.2f}")
    print(f"  Trading days  : {len(log_returns)}")
    print(f"  Daily mu/sigma: {mu_daily:.6f} / {sigma_daily:.6f}")
    print(f"  Annual mu/sigma: {mu_daily * 252:.2%} / "
          f"{sigma_daily * np.sqrt(252):.2%}")

    print(f"\n[2/4] Running constant-vol GBM ({N_SIMULATIONS:,} paths) ...")
    paths = run_gbm(current_price, mu_daily, sigma_daily)
    terminal = paths[FORECAST_DAYS]
    print(f"  2yr median: ${np.median(terminal):.2f}")
    print(f"  2yr mean:   ${np.mean(terminal):.2f}")
    print(f"  P(Profit):  {np.mean(terminal > current_price) * 100:.1f}%")

    print("\n[3/4] Running V1 stress-test scenarios (static multipliers) ...")
    stress_results, _ = run_stress_tests(mu_daily, sigma_daily, current_price)

    print("\n[4/4] Making chart and saving results ...")
    plot_fan_chart(prices, paths, current_price)
    save_results(prices, paths, current_price, mu_daily, sigma_daily,
                 log_returns, stress_results)

    print(f"\nV1 GBM 2yr median: ${np.median(terminal):.2f}")
    print(f"P(Profit): {np.mean(terminal > current_price) * 100:.1f}%")


if __name__ == "__main__":
    main()
