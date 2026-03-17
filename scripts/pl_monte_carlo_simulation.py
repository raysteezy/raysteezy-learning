#!/usr/bin/env python3
"""
Planet Labs (PL) — Monte Carlo Simulation
==========================================
Simulates 10,000 possible future price paths for PL stock using
Geometric Brownian Motion (GBM). Also includes stress testing
under 5 different market scenarios and robustness analysis.

The GBM equation:
    S(t+dt) = S(t) * exp((mu - sigma^2/2) * dt + sigma * sqrt(dt) * Z)
    where Z ~ N(0,1)

In plain English: each day, the stock price changes by its average
daily return (drift) plus a random amount (volatility * random noise).
We do this 10,000 times to see the range of possible outcomes.

Usage:
    python pl_monte_carlo_simulation.py

Requirements:
    pip install numpy pandas yfinance matplotlib scipy

Disclaimer: Educational simulation only — not financial advice.
"""

import json
import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

# Suppress some noisy warnings from yfinance/matplotlib
warnings.filterwarnings("ignore")


# ── Settings ─────────────────────────────────────────────────────────

TICKER = "PL"
N_SIMULATIONS = 10_000       # number of random paths to simulate
FORECAST_DAYS = 504          # ~2 trading years (252 days per year)
BOOTSTRAP_SAMPLES = 5_000    # for confidence interval estimation
RANDOM_SEED = 42             # makes results reproducible


# =====================================================================
# STEP 1: Get Historical Data
# =====================================================================

print(f"[1/6] Fetching {TICKER} price history ...")

stock = yf.Ticker(TICKER)
hist = stock.history(period="max")
hist = hist[hist.index >= "2021-01-01"]   # only use data from 2021 onward
close = hist["Close"].dropna()

# Calculate log returns (the standard way to measure daily price changes)
# log return = ln(today's price / yesterday's price)
log_returns = np.log(close / close.shift(1)).dropna()

# Current price and statistical parameters
current_price = close.iloc[-1]
mu_daily = log_returns.mean()             # average daily return (drift)
sigma_daily = log_returns.std()           # daily volatility (how much it bounces)
mu_annual = mu_daily * 252                # annualized drift
sigma_annual = sigma_daily * np.sqrt(252) # annualized volatility

print(f"  Current price : ${current_price:.2f}")
print(f"  Trading days  : {len(log_returns)}")
print(f"  Daily mu/sigma: {mu_daily:.6f} / {sigma_daily:.6f}")
print(f"  Annual mu/sigma: {mu_annual:.2%} / {sigma_annual:.2%}")
print(f"  Skewness      : {log_returns.skew():.4f}")
print(f"  Kurtosis      : {log_returns.kurtosis():.4f}")


# =====================================================================
# STEP 2: Run the Monte Carlo Simulation
# =====================================================================

print(f"\n[2/6] Running {N_SIMULATIONS:,} GBM simulations ({FORECAST_DAYS} days) ...")

np.random.seed(RANDOM_SEED)

# Time step (1 day)
dt = 1

# Generate all the random noise at once (much faster than looping)
# Z is a matrix: rows = days, columns = simulation paths
Z = np.random.standard_normal((FORECAST_DAYS, N_SIMULATIONS))

# GBM formula broken into two parts:
# 1. drift: the expected daily move (adjusted by -sigma^2/2 for the log-normal correction)
# 2. diffusion: the random daily move (volatility * random noise)
drift = (mu_daily - 0.5 * sigma_daily ** 2) * dt
diffusion = sigma_daily * np.sqrt(dt) * Z

# Each day's log-return is drift + diffusion
log_increments = drift + diffusion

# Cumulative sum of log-returns gives the total path
# We prepend a row of zeros so paths[0] = current_price
log_paths = np.cumsum(log_increments, axis=0)
log_paths = np.vstack([np.zeros(N_SIMULATIONS), log_paths])

# Convert from log-space back to actual prices
paths = current_price * np.exp(log_paths)

# Extract statistics at key time horizons
milestones = {"6_months": 126, "1_year": 252, "2_years": 504}
mc_stats = {}
for label, day in milestones.items():
    terminal = paths[day]
    mc_stats[label] = {
        "mean": round(float(np.mean(terminal)), 2),
        "median": round(float(np.median(terminal)), 2),
        "std": round(float(np.std(terminal)), 2),
        "min": round(float(np.min(terminal)), 2),
        "max": round(float(np.max(terminal)), 2),
        "p5": round(float(np.percentile(terminal, 5)), 2),
        "p10": round(float(np.percentile(terminal, 10)), 2),
        "p25": round(float(np.percentile(terminal, 25)), 2),
        "p50": round(float(np.percentile(terminal, 50)), 2),
        "p75": round(float(np.percentile(terminal, 75)), 2),
        "p90": round(float(np.percentile(terminal, 90)), 2),
        "p95": round(float(np.percentile(terminal, 95)), 2),
    }

# Probability calculations at the 2-year mark
terminal_2yr = paths[FORECAST_DAYS]
probabilities = {
    "above_current_2yr": round(float(np.mean(terminal_2yr > current_price) * 100), 1),
    "double_2yr": round(float(np.mean(terminal_2yr > 2 * current_price) * 100), 1),
    "below_10_2yr": round(float(np.mean(terminal_2yr < 10) * 100), 1),
    "above_50_2yr": round(float(np.mean(terminal_2yr > 50) * 100), 1),
}

print(f"  2yr Median: ${mc_stats['2_years']['median']}")
print(f"  P(profit 2yr): {probabilities['above_current_2yr']}%")


# =====================================================================
# STEP 3: Stress Testing
# =====================================================================

print("\n[3/6] Running stress-test scenarios ...")

# Each scenario adjusts drift (mu) and volatility (sigma) by a multiplier
# to simulate different market conditions
SCENARIOS = {
    "market_crash": {
        "name": "Market Crash (2008-style)",
        "description": "Severe market downturn — 60% drop over 6 months, slow recovery",
        "mu_mult": -3.0,   # strong negative drift
        "sigma_mult": 2.5,  # much higher volatility
    },
    "bear_market": {
        "name": "Prolonged Bear Market",
        "description": "Extended downturn — negative sentiment, rising rates",
        "mu_mult": -1.5,
        "sigma_mult": 1.5,
    },
    "base_case": {
        "name": "Base Case",
        "description": "Historical drift and volatility continue",
        "mu_mult": 1.0,    # no change
        "sigma_mult": 1.0,
    },
    "bull_market": {
        "name": "Strong Bull Market",
        "description": "Accelerating growth — major contracts, positive sentiment",
        "mu_mult": 3.0,
        "sigma_mult": 0.8,  # slightly lower volatility in a steady bull
    },
    "extreme_bull": {
        "name": "Extreme Bull (AI/Space Boom)",
        "description": "Massive sector rotation into space/AI — parabolic move",
        "mu_mult": 5.0,
        "sigma_mult": 2.0,  # high vol even in a boom (wild swings)
    },
}

stress_results = {}
stress_paths = {}

for key, scenario in SCENARIOS.items():
    # Adjust drift and volatility for this scenario
    sc_mu = mu_daily * scenario["mu_mult"]
    sc_sigma = sigma_daily * scenario["sigma_mult"]

    # Run the simulation with these adjusted parameters
    Z_sc = np.random.standard_normal((FORECAST_DAYS, N_SIMULATIONS))
    sc_drift = (sc_mu - 0.5 * sc_sigma ** 2) * dt
    sc_diff = sc_sigma * np.sqrt(dt) * Z_sc
    sc_log = np.cumsum(sc_drift + sc_diff, axis=0)
    sc_log = np.vstack([np.zeros(N_SIMULATIONS), sc_log])
    sc_paths = current_price * np.exp(sc_log)
    stress_paths[key] = sc_paths

    # Calculate statistics for this scenario
    sc_terminal = sc_paths[FORECAST_DAYS]
    sc_median_path = np.median(sc_paths, axis=1)

    # Max drawdown: the worst peak-to-trough decline in the median path
    sc_max_dd = float(
        np.min(sc_median_path / np.maximum.accumulate(sc_median_path) - 1) * 100
    )

    stress_results[key] = {
        "name": scenario["name"],
        "description": scenario["description"],
        "median_2yr": round(float(np.median(sc_terminal)), 2),
        "p10_2yr": round(float(np.percentile(sc_terminal, 10)), 2),
        "p90_2yr": round(float(np.percentile(sc_terminal, 90)), 2),
        "prob_profit": round(float(np.mean(sc_terminal > current_price) * 100), 1),
        "max_drawdown_median": round(sc_max_dd, 1),
    }
    print(f"  {scenario['name']:35s} -> Median 2yr: ${stress_results[key]['median_2yr']:.2f}")


# =====================================================================
# STEP 4: Robustness Analysis
# =====================================================================

print("\n[4/6] Walk-forward validation ...")

# Walk-forward: test the model on data it hasn't seen
# For each window, fit mu/sigma, predict 21 days ahead, compare to reality
lookbacks = [63, 126, 252]  # 3 months, 6 months, 1 year
wf_results = {}

for lb in lookbacks:
    n_tests = 0
    errors = []

    for start in range(lb, len(close) - 21, 63):
        window = log_returns.iloc[start - lb : start]
        if len(window) < lb:
            continue

        w_mu = window.mean()
        w_sigma = window.std()
        actual = close.iloc[min(start + 21, len(close) - 1)]

        # GBM median prediction for 21 days ahead
        predicted_median = float(
            close.iloc[start] * np.exp((w_mu - 0.5 * w_sigma ** 2) * 21)
        )
        errors.append(np.log(actual / predicted_median))
        n_tests += 1

    errors = np.array(errors)
    wf_results[f"{lb}_day_lookback"] = {
        "mean_error": round(float(np.mean(errors)), 4),
        "mae": round(float(np.mean(np.abs(errors))), 4),
        "rmse": round(float(np.sqrt(np.mean(errors ** 2))), 4),
        "n_tests": n_tests,
    }
    print(f"  {lb}-day lookback  RMSE={wf_results[f'{lb}_day_lookback']['rmse']:.4f}  (n={n_tests})")


print("\n[5/6] Bootstrap confidence intervals ...")

# Bootstrap: resample the historical returns many times to see how
# uncertain our estimates of mu and sigma are
boot_mus = []
boot_sigmas = []
lr_array = log_returns.values

for _ in range(BOOTSTRAP_SAMPLES):
    sample = np.random.choice(lr_array, size=len(lr_array), replace=True)
    boot_mus.append(sample.mean() * 252)        # annualize
    boot_sigmas.append(sample.std() * np.sqrt(252))

mu_ci = [
    round(float(np.percentile(boot_mus, 2.5)), 6),
    round(float(np.percentile(boot_mus, 97.5)), 6),
]
sigma_ci = [
    round(float(np.percentile(boot_sigmas, 2.5)), 6),
    round(float(np.percentile(boot_sigmas, 97.5)), 6),
]
print(f"  mu 95% CI: [{mu_ci[0]:.4f}, {mu_ci[1]:.4f}]")
print(f"  sigma 95% CI: [{sigma_ci[0]:.4f}, {sigma_ci[1]:.4f}]")


# VaR and CVaR — standard risk metrics
# VaR: the price threshold at a given confidence level
# CVaR: the average price in the worst outcomes below VaR
var_95 = round(float(np.percentile(terminal_2yr, 5)), 2)
var_99 = round(float(np.percentile(terminal_2yr, 1)), 2)
cvar_95 = round(float(np.mean(terminal_2yr[terminal_2yr <= var_95])), 2)
cvar_99 = round(float(np.mean(terminal_2yr[terminal_2yr <= var_99])), 2)


# Parameter sensitivity: what happens if mu or sigma are different
# than what we estimated?
print("\n  Parameter sensitivity grid ...")

mu_shifts = [-0.50, -0.25, 0.0, 0.25, 0.50]
sigma_shifts = [-0.30, 0.0, 0.30]
sensitivity = []

for s_shift in sigma_shifts:
    for m_shift in mu_shifts:
        adj_mu = mu_daily * (1 + m_shift)
        adj_sigma = sigma_daily * (1 + s_shift)

        # Smaller simulation (2000 paths) since this is just for the grid
        Z_sens = np.random.standard_normal((FORECAST_DAYS, 2000))
        d = (adj_mu - 0.5 * adj_sigma ** 2) * dt
        diff = adj_sigma * np.sqrt(dt) * Z_sens
        lp = np.cumsum(d + diff, axis=0)
        tp = current_price * np.exp(lp[-1])

        sensitivity.append({
            "mu_shift": f"{m_shift:+.0%}",
            "sigma_shift": f"{s_shift:+.0%}",
            "median_2yr": round(float(np.median(tp)), 2),
        })

sens_df = pd.DataFrame(sensitivity)


# =====================================================================
# STEP 5: Generate Charts
# =====================================================================

print("\n[6/6] Generating charts ...")
plt.style.use("dark_background")

# ── Chart 1: Monte Carlo Fan Chart ──────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 7))

hist_dates = close.index
forecast_start = hist_dates[-1]
forecast_dates = pd.bdate_range(start=forecast_start, periods=FORECAST_DAYS + 1)

# Calculate percentile paths for the confidence bands
percentiles = [5, 10, 25, 50, 75, 90, 95]
pct_paths = np.percentile(paths, percentiles, axis=1)
median_path = pct_paths[3]  # index 3 = P50

# Fan chart: wider bands = less confident
ax.fill_between(forecast_dates, pct_paths[0], pct_paths[6], alpha=0.15, color="#00d4aa", label="90% Confidence")
ax.fill_between(forecast_dates, pct_paths[1], pct_paths[5], alpha=0.20, color="#00d4aa")
ax.fill_between(forecast_dates, pct_paths[2], pct_paths[4], alpha=0.30, color="#00d4aa", label="50% Confidence")
ax.plot(forecast_dates, median_path, color="#FFA500", linewidth=2, label="Median (P50)")
ax.plot(forecast_dates, pct_paths[0], "--", color="#ff6b6b", linewidth=0.8, alpha=0.7, label="P5 / P95")
ax.plot(forecast_dates, pct_paths[6], "--", color="#ff6b6b", linewidth=0.8, alpha=0.7)

# Historical prices
ax.plot(hist_dates, close.values, color="#4fc3f7", linewidth=1.2, label="Historical Price")
ax.axhline(y=current_price, color="white", linestyle=":", alpha=0.3)

# Annotate current price and milestones
ax.annotate(
    f"Current: ${current_price:.2f}",
    xy=(forecast_start, current_price),
    fontsize=10, color="#FFA500", fontweight="bold",
    xytext=(-120, 20), textcoords="offset points",
)

for label_name, day_idx in milestones.items():
    nice = label_name.replace("_", " ").title().replace("Months", "Mo").replace("Years", "Yr").replace("Year", "Yr")
    val = median_path[day_idx]
    ax.plot(forecast_dates[day_idx], val, "o", color="white", markersize=6, zorder=5)
    ax.annotate(
        f"{nice}: ${val:.2f}",
        xy=(forecast_dates[day_idx], val),
        fontsize=9, color="#FFA500",
        xytext=(15, 15), textcoords="offset points",
    )

# VaR line
ax.axhline(y=var_95, color="#ff6b6b", linestyle="-.", alpha=0.4, linewidth=0.8)
ax.annotate(
    f"VaR 95%: ${var_95:.2f}",
    xy=(forecast_dates[FORECAST_DAYS // 2], var_95),
    fontsize=8, color="#ff6b6b", alpha=0.7,
)

ax.set_title(
    f"Planet Labs ({TICKER}) — Monte Carlo Simulation ({N_SIMULATIONS:,} paths)",
    fontsize=16, fontweight="bold", pad=15,
)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Stock Price (USD)", fontsize=12)
ax.legend(loc="upper left", fontsize=9, framealpha=0.7)
fig.text(
    0.5, 0.01,
    "GBM Model: S(t+dt) = S(t) * exp((mu - sigma^2/2)dt + sigma*sqrt(dt)*Z)  |  Not financial advice.",
    ha="center", fontsize=8, alpha=0.5,
)
plt.tight_layout()
fig.savefig("mc_fan_chart.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("  Done: mc_fan_chart.png")


# ── Chart 2: Stress Test Scenarios ──────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(14, 7))

colors_st = {
    "market_crash": "#ff6b6b",
    "bear_market": "#ff99cc",
    "base_case": "#FFA500",
    "bull_market": "#00e676",
    "extreme_bull": "#b388ff",
}
styles_st = {
    "market_crash": "--",
    "bear_market": "--",
    "base_case": "-",
    "bull_market": "-",
    "extreme_bull": "-",
}

for key in SCENARIOS:
    median_p = np.median(stress_paths[key], axis=1)
    ax2.plot(
        forecast_dates, median_p, styles_st[key],
        color=colors_st[key], linewidth=2,
        label=stress_results[key]["name"],
    )

    # Label the endpoint
    end_val = stress_results[key]["median_2yr"]
    ax2.plot(forecast_dates[-1], end_val, "o", color=colors_st[key], markersize=7, zorder=5)
    ax2.annotate(
        f"${end_val:.2f}",
        xy=(forecast_dates[-1], end_val),
        fontsize=10, fontweight="bold", color=colors_st[key],
        xytext=(10, 0), textcoords="offset points",
    )

ax2.axhline(y=current_price, color="white", linestyle=":", alpha=0.3)
ax2.set_yscale("log")  # log scale because the scenarios span a huge range
ax2.set_title(
    f"Planet Labs ({TICKER}) — Stress Test Scenarios",
    fontsize=16, fontweight="bold", pad=15,
)
ax2.set_xlabel("Date", fontsize=12)
ax2.set_ylabel("Stock Price (USD)", fontsize=12)
ax2.legend(loc="upper left", fontsize=9, framealpha=0.7)
fig2.text(
    0.5, 0.01,
    "Median paths under each scenario. Log scale. Not financial advice.",
    ha="center", fontsize=8, alpha=0.5,
)
plt.tight_layout()
fig2.savefig("stress_test_chart.png", dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
print("  Done: stress_test_chart.png")


# ── Chart 3: Robustness Dashboard (4 panels) ────────────────────────

fig3, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel 1: Distribution of final prices at 2 years
ax_dist = axes[0, 0]
ax_dist.hist(terminal_2yr, bins=200, range=(0, 150), color="#4fc3f7", alpha=0.7, edgecolor="none")
ax_dist.axvline(current_price, color="#FFA500", linewidth=2, label=f"Current ${current_price:.2f}")
ax_dist.axvline(
    mc_stats["2_years"]["median"], color="#00e676", linewidth=2, linestyle="--",
    label=f'Median ${mc_stats["2_years"]["median"]}',
)
ax_dist.axvline(var_95, color="#ff6b6b", linewidth=2, linestyle="-.", label=f"VaR 95% ${var_95}")
ax_dist.set_title("2-Year Terminal Price Distribution", fontsize=13, fontweight="bold")
ax_dist.set_xlabel("Price (USD)")
ax_dist.set_ylabel("Frequency")
ax_dist.legend(fontsize=9)

# Panel 2: Bootstrap distributions of mu and sigma
ax_boot = axes[0, 1]
ax_boot.hist(boot_mus, bins=80, color="#00e676", alpha=0.6, label="Annualized mu")
ax_boot2 = ax_boot.twinx()  # second y-axis for sigma
ax_boot2.hist(boot_sigmas, bins=80, color="#b388ff", alpha=0.5, label="Annualized sigma")
ax_boot.set_title("Bootstrap Parameter Distributions", fontsize=13, fontweight="bold")
ax_boot.set_xlabel("Value")
ax_boot.set_ylabel("mu Frequency", color="#00e676")
ax_boot2.set_ylabel("sigma Frequency", color="#b388ff")
ax_boot.legend(loc="upper left", fontsize=9)
ax_boot2.legend(loc="upper right", fontsize=9)

# Panel 3: Sensitivity heatmap
ax_heat = axes[1, 0]
pivot = sens_df.pivot(index="sigma_shift", columns="mu_shift", values="median_2yr")
pivot = pivot.reindex(index=["-30%", "+0%", "+30%"])
cmap = LinearSegmentedColormap.from_list("rg", ["#ff6b6b", "#ffeb3b", "#00e676"])
im = ax_heat.imshow(pivot.values, cmap=cmap, aspect="auto")
ax_heat.set_xticks(range(len(pivot.columns)))
ax_heat.set_xticklabels(pivot.columns, fontsize=10)
ax_heat.set_yticks(range(len(pivot.index)))
ax_heat.set_yticklabels(pivot.index, fontsize=10)
ax_heat.set_xlabel("Drift (mu) Shift")
ax_heat.set_ylabel("Volatility (sigma) Shift")
ax_heat.set_title("Parameter Sensitivity (2yr Median Price)", fontsize=13, fontweight="bold")

# Add price labels inside each cell
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        color = "black" if val > 20 else "white"
        ax_heat.text(j, i, f"${val:.0f}", ha="center", va="center", fontsize=12, fontweight="bold", color=color)

plt.colorbar(im, ax=ax_heat, label="Price ($)")

# Panel 4: Summary table
ax_tbl = axes[1, 1]
ax_tbl.axis("off")
ax_tbl.set_title("Monte Carlo Summary", fontsize=13, fontweight="bold", pad=20)

table_data = [
    ["Median", f'${mc_stats["6_months"]["median"]}', f'${mc_stats["1_year"]["median"]}', f'${mc_stats["2_years"]["median"]}'],
    ["P5 (Downside)", f'${mc_stats["6_months"]["p5"]}', f'${mc_stats["1_year"]["p5"]}', f'${mc_stats["2_years"]["p5"]}'],
    ["P95 (Upside)", f'${mc_stats["6_months"]["p95"]}', f'${mc_stats["1_year"]["p95"]}', f'${mc_stats["2_years"]["p95"]}'],
    ["VaR 95%", "—", "—", f"${var_95}"],
    ["CVaR 95%", "—", "—", f"${cvar_95}"],
    ["P(Profit)", "—", "—", f'{probabilities["above_current_2yr"]}%'],
    ["P(Double)", "—", "—", f'{probabilities["double_2yr"]}%'],
    ["P(<$10)", "—", "—", f'{probabilities["below_10_2yr"]}%'],
]

tbl = ax_tbl.table(
    cellText=table_data,
    colLabels=["Metric", "6 Month", "1 Year", "2 Year"],
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.0, 1.6)

# Style the table cells
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#555555")
    if r == 0:
        cell.set_facecolor("#1565c0")
        cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor("#1a1a2e" if r % 2 == 0 else "#16213e")
        cell.set_text_props(color="white")

fig3.suptitle(
    f"Planet Labs ({TICKER}) — Robustness & Risk Analysis",
    fontsize=16, fontweight="bold", y=0.98,
)
fig3.text(
    0.5, 0.01,
    f"{N_SIMULATIONS:,} GBM simulations  |  {BOOTSTRAP_SAMPLES:,} bootstrap samples  |  Not financial advice.",
    ha="center", fontsize=8, alpha=0.5,
)
plt.tight_layout(rect=[0, 0.02, 1, 0.96])
fig3.savefig("robustness_dashboard.png", dpi=150, bbox_inches="tight", facecolor=fig3.get_facecolor())
print("  Done: robustness_dashboard.png")


# =====================================================================
# STEP 6: Save All Results
# =====================================================================

# Main JSON summary with everything
summary = {
    "ticker": TICKER,
    "company": stock.info.get("longName", "Planet Labs PBC"),
    "current_price": round(float(current_price), 2),
    "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
    "historical_parameters": {
        "trading_days": len(log_returns),
        "date_range": (
            f"{log_returns.index[0].strftime('%Y-%m-%d')} to "
            f"{log_returns.index[-1].strftime('%Y-%m-%d')}"
        ),
        "daily_drift_mu": round(float(mu_daily), 6),
        "daily_volatility_sigma": round(float(sigma_daily), 6),
        "annualized_drift": round(float(mu_annual), 4),
        "annualized_volatility": round(float(sigma_annual), 4),
        "log_return_skewness": round(float(log_returns.skew()), 4),
        "log_return_kurtosis": round(float(log_returns.kurtosis()), 4),
    },
    "monte_carlo": {
        "simulations": N_SIMULATIONS,
        "forecast_days": FORECAST_DAYS,
        "method": "Geometric Brownian Motion (GBM)",
        "equation": "S(t+dt) = S(t) * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z), Z ~ N(0,1)",
        "statistics": mc_stats,
        "probabilities": probabilities,
    },
    "stress_testing": stress_results,
    "robustness": {
        "walk_forward_validation": wf_results,
        "bootstrap_confidence_intervals": {
            "mu_95_ci": mu_ci,
            "sigma_95_ci": sigma_ci,
        },
        "value_at_risk": {
            "var_95_2yr": var_95,
            "var_99_2yr": var_99,
            "cvar_95_2yr": cvar_95,
            "cvar_99_2yr": cvar_99,
        },
    },
    "disclaimer": (
        "Educational Monte Carlo simulation. Not financial advice. "
        "Past volatility and drift do not guarantee future results. "
        "The GBM model assumes log-normal returns and constant parameters, "
        "which may not hold in practice."
    ),
}

with open("monte_carlo_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# Percentile paths CSV — one column per percentile, one row per day
pct_df = pd.DataFrame(
    {f"P{p}": pct_paths[i] for i, p in enumerate(percentiles)},
    index=forecast_dates,
)
pct_df.index.name = "date"
pct_df.to_csv("mc_percentile_paths.csv")

# Stress test paths CSV — median path for each scenario
st_rows = []
for key, res in stress_results.items():
    median_p = np.median(stress_paths[key], axis=1)
    for i, d in enumerate(forecast_dates):
        st_rows.append({
            "date": d,
            "scenario": res["name"],
            "median_price": round(float(median_p[i]), 4),
        })
pd.DataFrame(st_rows).to_csv("stress_test_paths.csv", index=False)

# Sensitivity analysis CSV
sens_df.to_csv("sensitivity_analysis.csv", index=False)

# Final output
print("\nAll outputs saved:")
print("   mc_fan_chart.png")
print("   stress_test_chart.png")
print("   robustness_dashboard.png")
print("   monte_carlo_summary.json")
print("   mc_percentile_paths.csv")
print("   stress_test_paths.csv")
print("   sensitivity_analysis.csv")
print(f"\nModel: {N_SIMULATIONS:,} GBM paths x {FORECAST_DAYS} days")
print(f"   2yr Median: ${mc_stats['2_years']['median']}  |  P(Profit): {probabilities['above_current_2yr']}%  |  VaR 95%: ${var_95}")
