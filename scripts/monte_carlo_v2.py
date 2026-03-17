#!/usr/bin/env python3
"""
Planet Labs (PL) — Monte Carlo Simulation V2 (Upgraded)
========================================================
After V1 got a C+ for using constant volatility and made-up stress
scenarios, I rebuilt the simulation with proper stochastic models:

  - Stochastic volatility (Heston-inspired model where volatility
    itself is random and mean-reverts)
  - Jump diffusion (Merton model — accounts for sudden crashes or
    spikes that GBM can't produce)
  - Regime-switching stress tests (uses a Hidden Markov Model to
    identify real bull/bear regimes from historical data instead
    of made-up multipliers)
  - Walk-forward validation comparing GBM vs Heston
  - Bootstrap confidence intervals for parameters
  - Combined risk metrics across all three models

What I learned: constant volatility is a terrible assumption for
stocks. Real markets have volatility clustering (calm periods
followed by wild periods), jumps (earnings surprises, crashes),
and regime changes. Modeling these makes the simulation way more
realistic.

The V1 baseline (constant-vol GBM) is in monte_carlo_v1.py.
I kept it separate so you can compare side by side.

Usage:
    python monte_carlo_v2.py

Requirements:
    pip install numpy pandas yfinance matplotlib scipy hmmlearn

Disclaimer: Educational simulation only — not financial advice.
"""

import json
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

warnings.filterwarnings("ignore")


# ── Settings ─────────────────────────────────────────────────────────

TICKER = "PL"
N_SIMULATIONS = 10_000
FORECAST_DAYS = 504          # ~2 trading years
BOOTSTRAP_SAMPLES = 5_000
RANDOM_SEED = 42


# =====================================================================
# STEP 1: Get Historical Data and Compute Parameters
# =====================================================================

print(f"[1/7] Fetching {TICKER} price history ...")

stock = yf.Ticker(TICKER)
hist = stock.history(period="max")
hist = hist[hist.index >= "2021-01-01"]
close = hist["Close"].dropna()
log_returns = np.log(close / close.shift(1)).dropna()

current_price = close.iloc[-1]
mu_daily = log_returns.mean()
sigma_daily = log_returns.std()
mu_annual = mu_daily * 252
sigma_annual = sigma_daily * np.sqrt(252)

print(f"  Current price : ${current_price:.2f}")
print(f"  Trading days  : {len(log_returns)}")
print(f"  Daily mu/sigma: {mu_daily:.6f} / {sigma_daily:.6f}")
print(f"  Annual mu/sigma: {mu_annual:.2%} / {sigma_annual:.2%}")
print(f"  Skewness      : {log_returns.skew():.4f}")
print(f"  Kurtosis      : {log_returns.kurtosis():.4f}")


# =====================================================================
# STEP 2: V1 Baseline GBM (for comparison only)
# =====================================================================

print(f"\n[2/7] Running V1 baseline GBM ({N_SIMULATIONS:,} paths) ...")

np.random.seed(RANDOM_SEED)
dt = 1

Z_v1 = np.random.standard_normal((FORECAST_DAYS, N_SIMULATIONS))
drift_v1 = (mu_daily - 0.5 * sigma_daily ** 2) * dt
diffusion_v1 = sigma_daily * np.sqrt(dt) * Z_v1
log_paths_v1 = np.cumsum(drift_v1 + diffusion_v1, axis=0)
log_paths_v1 = np.vstack([np.zeros(N_SIMULATIONS), log_paths_v1])
paths_v1 = current_price * np.exp(log_paths_v1)

v1_terminal = paths_v1[FORECAST_DAYS]
print(f"  V1 GBM 2yr median: ${np.median(v1_terminal):.2f}")


# =====================================================================
# STEP 3: Stochastic Volatility (Heston-inspired)
# =====================================================================

print(f"\n[3/7] Running V2 stochastic volatility model ...")

# Heston model parameters calibrated from PL's data:
# - kappa: how fast volatility returns to its long-term average
# - theta: the long-term average variance
# - xi: volatility of volatility (how much vol itself bounces around)
# - rho: correlation between price and volatility (usually negative)

realized_var = log_returns.rolling(21).var().dropna()
theta = float(sigma_daily ** 2)                   # long-run variance
v0 = float(realized_var.iloc[-1])                 # current variance
kappa = 3.0                                        # mean reversion speed
xi = float(realized_var.std() * np.sqrt(252)) * 2  # vol of vol
rho = float(log_returns.corr(
    log_returns.rolling(21).std().pct_change()
))
rho = max(min(rho, -0.1), -0.9)  # keep rho negative and reasonable

print(f"  Long-run variance (theta): {theta:.6f}")
print(f"  Current variance (v0):     {v0:.6f}")
print(f"  Mean reversion (kappa):    {kappa:.2f}")
print(f"  Vol of vol (xi):           {xi:.4f}")
print(f"  Correlation (rho):         {rho:.4f}")

# Simulate the Heston model
# Two correlated Brownian motions: one for price, one for volatility
paths_heston = np.zeros((FORECAST_DAYS + 1, N_SIMULATIONS))
paths_heston[0] = current_price
variance = np.full(N_SIMULATIONS, v0)

for t in range(FORECAST_DAYS):
    z1 = np.random.standard_normal(N_SIMULATIONS)
    z2 = (rho * z1
          + np.sqrt(1 - rho ** 2)
          * np.random.standard_normal(N_SIMULATIONS))

    # Variance can't go negative — truncation scheme
    var_pos = np.maximum(variance, 0)
    vol = np.sqrt(var_pos)

    # Price update
    paths_heston[t + 1] = paths_heston[t] * np.exp(
        (mu_daily - 0.5 * var_pos) * dt + vol * np.sqrt(dt) * z1
    )

    # Variance update (mean-reverts toward theta)
    variance = (var_pos
                + kappa * (theta - var_pos) * dt
                + xi * vol * np.sqrt(dt) * z2)

heston_terminal = paths_heston[FORECAST_DAYS]
print(f"  Heston 2yr median: ${np.median(heston_terminal):.2f}")


# =====================================================================
# STEP 4: Jump Diffusion (Merton model)
# =====================================================================

print(f"\n[4/7] Running V2 jump diffusion model ...")

# Jump parameters estimated from PL's tail behavior:
# I detect jumps as returns more than 3 standard deviations from mean.
threshold = 3 * sigma_daily
jumps = log_returns[np.abs(log_returns - mu_daily) > threshold]
n_jumps = len(jumps)
n_years = len(log_returns) / 252

lam = n_jumps / n_years if n_years > 0 else 2.0    # jumps per year
jump_mu = float(jumps.mean()) if n_jumps > 0 else 0.0
jump_sigma = float(jumps.std()) if n_jumps > 1 else sigma_daily

print(f"  Detected {n_jumps} jumps in {n_years:.1f} years")
print(f"  Jump intensity (lambda): {lam:.2f}/year")
print(f"  Jump mean: {jump_mu:.4f}")
print(f"  Jump std:  {jump_sigma:.4f}")

# Simulate jump diffusion paths
paths_jd = np.zeros((FORECAST_DAYS + 1, N_SIMULATIONS))
paths_jd[0] = current_price

# Compensator: adjust drift so jumps don't bias the expected return
jump_compensator = (
    lam * (np.exp(jump_mu + 0.5 * jump_sigma ** 2) - 1) / 252
)

for t in range(FORECAST_DAYS):
    z = np.random.standard_normal(N_SIMULATIONS)

    # Poisson process: how many jumps happen today?
    n_jumps_today = np.random.poisson(lam / 252, N_SIMULATIONS)

    # Total jump size for today
    jump_sizes = np.zeros(N_SIMULATIONS)
    for i in range(N_SIMULATIONS):
        if n_jumps_today[i] > 0:
            jump_sizes[i] = np.sum(
                np.random.normal(jump_mu, jump_sigma, n_jumps_today[i])
            )

    # GBM part + jump part
    continuous = (
        (mu_daily - 0.5 * sigma_daily ** 2 - jump_compensator) * dt
    )
    diffuse = sigma_daily * np.sqrt(dt) * z
    paths_jd[t + 1] = paths_jd[t] * np.exp(
        continuous + diffuse + jump_sizes
    )

jd_terminal = paths_jd[FORECAST_DAYS]
print(f"  Jump diffusion 2yr median: ${np.median(jd_terminal):.2f}")


# =====================================================================
# STEP 5: Regime-Switching Stress Tests (HMM-based)
# =====================================================================

print(f"\n[5/7] Fitting Hidden Markov Model for regime detection ...")

# Instead of making up stress-test multipliers (V1 approach), I use
# a Hidden Markov Model to find real market regimes in PL's history.
try:
    from hmmlearn.hmm import GaussianHMM

    returns_2d = log_returns.values.reshape(-1, 1)
    hmm = GaussianHMM(
        n_components=2, covariance_type="full",
        n_iter=1000, random_state=RANDOM_SEED,
    )
    hmm.fit(returns_2d)

    regimes = hmm.predict(returns_2d)
    regime_means = hmm.means_.flatten()
    regime_stds = np.sqrt(hmm.covars_.flatten())

    calm_idx = np.argmin(regime_stds)
    vol_idx = np.argmax(regime_stds)

    hmm_regimes = {
        "calm": {
            "mu_daily": float(regime_means[calm_idx]),
            "sigma_daily": float(regime_stds[calm_idx]),
            "frequency": float(np.mean(regimes == calm_idx) * 100),
            "mu_annual": float(regime_means[calm_idx] * 252),
            "sigma_annual": float(regime_stds[calm_idx] * np.sqrt(252)),
        },
        "volatile": {
            "mu_daily": float(regime_means[vol_idx]),
            "sigma_daily": float(regime_stds[vol_idx]),
            "frequency": float(np.mean(regimes == vol_idx) * 100),
            "mu_annual": float(regime_means[vol_idx] * 252),
            "sigma_annual": float(regime_stds[vol_idx] * np.sqrt(252)),
        },
    }

    print(f"  Calm regime:     "
          f"mu={hmm_regimes['calm']['mu_annual']:.2%}, "
          f"vol={hmm_regimes['calm']['sigma_annual']:.2%}, "
          f"({hmm_regimes['calm']['frequency']:.0f}% of days)")
    print(f"  Volatile regime: "
          f"mu={hmm_regimes['volatile']['mu_annual']:.2%}, "
          f"vol={hmm_regimes['volatile']['sigma_annual']:.2%}, "
          f"({hmm_regimes['volatile']['frequency']:.0f}% of days)")

    hmm_available = True
except Exception as e:
    print(f"  HMM fitting failed ({e}), using V1 multiplier fallback")
    hmm_available = False
    hmm_regimes = None

# Build stress-test scenarios
if hmm_available:
    calm_mu = hmm_regimes["calm"]["mu_daily"]
    calm_sig = hmm_regimes["calm"]["sigma_daily"]
    vol_mu = hmm_regimes["volatile"]["mu_daily"]
    vol_sig = hmm_regimes["volatile"]["sigma_daily"]

    SCENARIOS = {
        "market_crash": {
            "name": "Market Crash (2008-style)",
            "description": "Extreme volatile regime with forced "
                           "negative drift",
            "mu": vol_mu * -3,
            "sigma": vol_sig * 2.0,
            "source": "HMM volatile regime, amplified",
        },
        "bear_market": {
            "name": "Prolonged Bear Market",
            "description": "Volatile regime continues with "
                           "negative drift",
            "mu": vol_mu if vol_mu < 0 else -abs(vol_mu),
            "sigma": vol_sig * 1.3,
            "source": "HMM volatile regime, slightly amplified",
        },
        "base_case": {
            "name": "Base Case (Historical)",
            "description": "Blend of both regimes weighted by "
                           "their frequency",
            "mu": mu_daily,
            "sigma": sigma_daily,
            "source": "Full historical parameters",
        },
        "calm_bull": {
            "name": "Calm Bull Market",
            "description": "Calm regime dominates with positive drift",
            "mu": calm_mu if calm_mu > 0 else abs(calm_mu),
            "sigma": calm_sig,
            "source": "HMM calm regime",
        },
        "volatile_bull": {
            "name": "Volatile Bull (Sector Boom)",
            "description": "Strong positive drift but wild swings",
            "mu": abs(calm_mu) * 3,
            "sigma": vol_sig * 1.5,
            "source": "HMM calm drift amplified, volatile vol amplified",
        },
    }
else:
    SCENARIOS = {
        "market_crash": {
            "name": "Market Crash (2008-style)",
            "description": "Severe downturn",
            "mu": mu_daily * -3.0,
            "sigma": sigma_daily * 2.5,
            "source": "Static multipliers (V1 fallback)",
        },
        "bear_market": {
            "name": "Prolonged Bear Market",
            "description": "Extended negative sentiment",
            "mu": mu_daily * -1.5,
            "sigma": sigma_daily * 1.5,
            "source": "Static multipliers (V1 fallback)",
        },
        "base_case": {
            "name": "Base Case",
            "description": "Historical parameters continue",
            "mu": mu_daily,
            "sigma": sigma_daily,
            "source": "Historical",
        },
        "calm_bull": {
            "name": "Strong Bull Market",
            "description": "Positive sentiment, lower vol",
            "mu": mu_daily * 3.0,
            "sigma": sigma_daily * 0.8,
            "source": "Static multipliers (V1 fallback)",
        },
        "volatile_bull": {
            "name": "Extreme Bull (Sector Boom)",
            "description": "Parabolic move with high vol",
            "mu": mu_daily * 5.0,
            "sigma": sigma_daily * 2.0,
            "source": "Static multipliers (V1 fallback)",
        },
    }

print("\n  Running stress-test scenarios ...")

stress_results = {}
stress_paths = {}

for key, sc in SCENARIOS.items():
    sc_mu = sc["mu"]
    sc_sigma = sc["sigma"]
    Z_sc = np.random.standard_normal((FORECAST_DAYS, N_SIMULATIONS))
    sc_drift = (sc_mu - 0.5 * sc_sigma ** 2) * dt
    sc_diff = sc_sigma * np.sqrt(dt) * Z_sc
    sc_log = np.cumsum(sc_drift + sc_diff, axis=0)
    sc_log = np.vstack([np.zeros(N_SIMULATIONS), sc_log])
    sc_paths = current_price * np.exp(sc_log)
    stress_paths[key] = sc_paths

    sc_terminal = sc_paths[FORECAST_DAYS]
    sc_median_path = np.median(sc_paths, axis=1)
    sc_max_dd = float(
        np.min(sc_median_path
               / np.maximum.accumulate(sc_median_path) - 1) * 100
    )

    stress_results[key] = {
        "name": sc["name"],
        "description": sc["description"],
        "source": sc["source"],
        "mu_daily": round(float(sc_mu), 6),
        "sigma_daily": round(float(sc_sigma), 6),
        "median_2yr": round(float(np.median(sc_terminal)), 2),
        "p10_2yr": round(float(np.percentile(sc_terminal, 10)), 2),
        "p90_2yr": round(float(np.percentile(sc_terminal, 90)), 2),
        "prob_profit": round(
            float(np.mean(sc_terminal > current_price) * 100), 1
        ),
        "max_drawdown_median": round(sc_max_dd, 1),
    }
    print(f"  {sc['name']:35s} -> "
          f"Median 2yr: ${stress_results[key]['median_2yr']:.2f}")


# =====================================================================
# STEP 6: Robustness Analysis
# =====================================================================

print("\n[6/7] Robustness analysis ...")

# Walk-forward validation
lookbacks = [63, 126, 252]
wf_results = {}

for lb in lookbacks:
    n_tests = 0
    errors_gbm = []
    errors_heston = []

    for start in range(lb, len(close) - 21, 63):
        window = log_returns.iloc[start - lb : start]
        if len(window) < lb:
            continue

        w_mu = window.mean()
        w_sigma = window.std()
        actual = close.iloc[min(start + 21, len(close) - 1)]
        start_price = close.iloc[start]

        # GBM prediction
        pred_gbm = float(
            start_price * np.exp((w_mu - 0.5 * w_sigma ** 2) * 21)
        )
        errors_gbm.append(np.log(actual / pred_gbm))

        # Heston prediction (simplified)
        w_var = w_sigma ** 2
        heston_var_21 = w_var + kappa * (theta - w_var) * (21 / 252)
        pred_heston = float(
            start_price * np.exp((w_mu - 0.5 * heston_var_21) * 21)
        )
        errors_heston.append(np.log(actual / pred_heston))

        n_tests += 1

    errors_gbm = np.array(errors_gbm)
    errors_heston = np.array(errors_heston)

    wf_results[f"{lb}_day_lookback"] = {
        "gbm": {
            "mean_error": round(float(np.mean(errors_gbm)), 4),
            "mae": round(float(np.mean(np.abs(errors_gbm))), 4),
            "rmse": round(
                float(np.sqrt(np.mean(errors_gbm ** 2))), 4
            ),
        },
        "heston": {
            "mean_error": round(float(np.mean(errors_heston)), 4),
            "mae": round(float(np.mean(np.abs(errors_heston))), 4),
            "rmse": round(
                float(np.sqrt(np.mean(errors_heston ** 2))), 4
            ),
        },
        "n_tests": n_tests,
    }
    gbm_rmse = wf_results[f"{lb}_day_lookback"]["gbm"]["rmse"]
    heston_rmse = wf_results[f"{lb}_day_lookback"]["heston"]["rmse"]
    print(f"  {lb}-day lookback: GBM RMSE={gbm_rmse:.4f}, "
          f"Heston RMSE={heston_rmse:.4f}  (n={n_tests})")

# Bootstrap confidence intervals
print("\n  Bootstrap confidence intervals ...")
boot_mus = []
boot_sigmas = []
lr_array = log_returns.values
for _ in range(BOOTSTRAP_SAMPLES):
    sample = np.random.choice(lr_array, size=len(lr_array), replace=True)
    boot_mus.append(sample.mean() * 252)
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

# Risk metrics for all three models
milestones = {"6_months": 126, "1_year": 252, "2_years": 504}
all_models = {
    "v1_gbm": paths_v1,
    "v2_heston": paths_heston,
    "v2_jump_diffusion": paths_jd,
}

mc_stats = {}
for model_name, paths in all_models.items():
    mc_stats[model_name] = {}
    for label, day in milestones.items():
        terminal_at = paths[day]
        mc_stats[model_name][label] = {
            "median": round(float(np.median(terminal_at)), 2),
            "mean": round(float(np.mean(terminal_at)), 2),
            "p5": round(float(np.percentile(terminal_at, 5)), 2),
            "p25": round(float(np.percentile(terminal_at, 25)), 2),
            "p75": round(float(np.percentile(terminal_at, 75)), 2),
            "p95": round(float(np.percentile(terminal_at, 95)), 2),
        }

# VaR / CVaR
risk_metrics = {}
for model_name, paths in all_models.items():
    t = paths[FORECAST_DAYS]
    var_95 = round(float(np.percentile(t, 5)), 2)
    var_99 = round(float(np.percentile(t, 1)), 2)
    risk_metrics[model_name] = {
        "var_95": var_95,
        "var_99": var_99,
        "cvar_95": round(float(np.mean(t[t <= var_95])), 2),
        "cvar_99": round(float(np.mean(t[t <= var_99])), 2),
        "prob_profit": round(
            float(np.mean(t > current_price) * 100), 1
        ),
        "prob_double": round(
            float(np.mean(t > 2 * current_price) * 100), 1
        ),
        "prob_below_10": round(float(np.mean(t < 10) * 100), 1),
    }

# Parameter sensitivity
print("\n  Parameter sensitivity grid ...")
mu_shifts = [-0.50, -0.25, 0.0, 0.25, 0.50]
sigma_shifts = [-0.30, 0.0, 0.30]
sensitivity = []
for s_shift in sigma_shifts:
    for m_shift in mu_shifts:
        adj_mu = mu_daily * (1 + m_shift)
        adj_sigma = sigma_daily * (1 + s_shift)
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
# STEP 7: Charts
# =====================================================================

print("\n[7/7] Making charts ...")
plt.style.use("dark_background")

# ── Chart 1: Multi-Model Fan Chart ──────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 7))

hist_dates = close.index
forecast_start = hist_dates[-1]
forecast_dates = pd.bdate_range(
    start=forecast_start, periods=FORECAST_DAYS + 1
)

# V1 GBM bands (faded)
pct_v1 = np.percentile(paths_v1, [5, 25, 50, 75, 95], axis=1)
ax.fill_between(forecast_dates, pct_v1[0], pct_v1[4],
                alpha=0.08, color="#888888", label="V1 GBM 90% CI")
ax.plot(forecast_dates, pct_v1[2], "--", color="#888888",
        linewidth=1, alpha=0.5, label="V1 GBM Median")

# V2 Heston bands
pct_h = np.percentile(paths_heston, [5, 25, 50, 75, 95], axis=1)
ax.fill_between(forecast_dates, pct_h[0], pct_h[4],
                alpha=0.12, color="#00d4aa", label="V2 Heston 90% CI")
ax.fill_between(forecast_dates, pct_h[1], pct_h[3],
                alpha=0.25, color="#00d4aa", label="V2 Heston 50% CI")
ax.plot(forecast_dates, pct_h[2], color="#FFA500",
        linewidth=2, label="V2 Heston Median")

# V2 Jump diffusion median
pct_jd = np.percentile(paths_jd, [50], axis=1)
ax.plot(forecast_dates, pct_jd[0], color="#b388ff",
        linewidth=1.5, linestyle="--", alpha=0.8,
        label="V2 Jump Diffusion Median")

# Historical
ax.plot(hist_dates, close.values, color="#4fc3f7",
        linewidth=1.2, label="Historical Price")
ax.axhline(y=current_price, color="white", linestyle=":", alpha=0.3)

# Annotations
ax.annotate(f"Current: ${current_price:.2f}",
            xy=(forecast_start, current_price),
            fontsize=10, color="#FFA500", fontweight="bold",
            xytext=(-120, 20), textcoords="offset points")

for label_name, day_idx in milestones.items():
    nice = (label_name.replace("_", " ").title()
            .replace("Months", "Mo")
            .replace("Years", "Yr")
            .replace("Year", "Yr"))
    val = pct_h[2][day_idx]
    ax.plot(forecast_dates[day_idx], val, "o", color="white",
            markersize=6, zorder=5)
    ax.annotate(f"{nice}: ${val:.2f}",
                xy=(forecast_dates[day_idx], val),
                fontsize=9, color="#FFA500",
                xytext=(15, 15), textcoords="offset points")

# VaR line
var_95_h = risk_metrics["v2_heston"]["var_95"]
ax.axhline(y=var_95_h, color="#ff6b6b", linestyle="-.",
           alpha=0.4, linewidth=0.8)
ax.annotate(f"Heston VaR 95%: ${var_95_h:.2f}",
            xy=(forecast_dates[FORECAST_DAYS // 2], var_95_h),
            fontsize=8, color="#ff6b6b", alpha=0.7)

ax.set_title(f"Planet Labs ({TICKER}) — Monte Carlo V1 vs V2 "
             f"({N_SIMULATIONS:,} paths)",
             fontsize=16, fontweight="bold", pad=15)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Stock Price (USD)", fontsize=12)
ax.legend(loc="upper left", fontsize=8, framealpha=0.7)
fig.text(0.5, 0.01,
         "V2 adds stochastic volatility (Heston) + jump diffusion "
         "(Merton)  |  Not financial advice.",
         ha="center", fontsize=8, alpha=0.5)
plt.tight_layout()
fig.savefig("fan_chart.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("  Saved: fan_chart.png")


# ── Chart 2: Stress Test Scenarios ──────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(14, 7))
colors_st = {
    "market_crash": "#ff6b6b", "bear_market": "#ff99cc",
    "base_case": "#FFA500", "calm_bull": "#00e676",
    "volatile_bull": "#b388ff",
}
styles_st = {
    "market_crash": "--", "bear_market": "--",
    "base_case": "-", "calm_bull": "-", "volatile_bull": "-",
}

for key in SCENARIOS:
    median_p = np.median(stress_paths[key], axis=1)
    ax2.plot(forecast_dates, median_p, styles_st[key],
             color=colors_st[key], linewidth=2,
             label=stress_results[key]["name"])
    end_val = stress_results[key]["median_2yr"]
    ax2.plot(forecast_dates[-1], end_val, "o",
             color=colors_st[key], markersize=7, zorder=5)
    ax2.annotate(f"${end_val:.2f}",
                 xy=(forecast_dates[-1], end_val),
                 fontsize=10, fontweight="bold", color=colors_st[key],
                 xytext=(10, 0), textcoords="offset points")

hmm_label = ("HMM regime-based" if hmm_available
             else "Static multipliers (V1)")
ax2.axhline(y=current_price, color="white", linestyle=":", alpha=0.3)
ax2.set_yscale("log")
ax2.set_title(f"Planet Labs ({TICKER}) — Stress Tests ({hmm_label})",
              fontsize=16, fontweight="bold", pad=15)
ax2.set_xlabel("Date", fontsize=12)
ax2.set_ylabel("Stock Price (USD)", fontsize=12)
ax2.legend(loc="upper left", fontsize=9, framealpha=0.7)
fig2.text(0.5, 0.01,
          "Median paths per scenario. Log scale. Not financial advice.",
          ha="center", fontsize=8, alpha=0.5)
plt.tight_layout()
fig2.savefig("stress.png", dpi=150, bbox_inches="tight",
             facecolor=fig2.get_facecolor())
plt.close()
print("  Saved: stress.png")


# ── Chart 3: Robustness Dashboard (4-panel) ─────────────────────────

fig3, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel 1: Terminal distributions
ax_dist = axes[0, 0]
bins = np.linspace(0, 150, 200)
ax_dist.hist(v1_terminal, bins=bins, alpha=0.3, color="#888888",
             edgecolor="none",
             label=f"V1 GBM (med ${np.median(v1_terminal):.0f})")
ax_dist.hist(heston_terminal, bins=bins, alpha=0.4, color="#00d4aa",
             edgecolor="none",
             label=f"V2 Heston (med ${np.median(heston_terminal):.0f})")
ax_dist.hist(jd_terminal, bins=bins, alpha=0.3, color="#b388ff",
             edgecolor="none",
             label=f"V2 Jump Diff (med ${np.median(jd_terminal):.0f})")
ax_dist.axvline(current_price, color="#FFA500", linewidth=2,
                label=f"Current ${current_price:.2f}")
ax_dist.set_title("2yr Terminal Distributions: V1 vs V2",
                   fontsize=13, fontweight="bold")
ax_dist.set_xlabel("Price (USD)")
ax_dist.set_ylabel("Frequency")
ax_dist.legend(fontsize=8)
ax_dist.set_xlim(0, 150)

# Panel 2: Bootstrap distributions
ax_boot = axes[0, 1]
ax_boot.hist(boot_mus, bins=80, color="#00e676", alpha=0.6,
             label="Annualized mu")
ax_boot2 = ax_boot.twinx()
ax_boot2.hist(boot_sigmas, bins=80, color="#b388ff", alpha=0.5,
              label="Annualized sigma")
ax_boot.set_title("Bootstrap Parameter Distributions",
                   fontsize=13, fontweight="bold")
ax_boot.set_xlabel("Value")
ax_boot.set_ylabel("mu Frequency", color="#00e676")
ax_boot2.set_ylabel("sigma Frequency", color="#b388ff")
ax_boot.legend(loc="upper left", fontsize=9)
ax_boot2.legend(loc="upper right", fontsize=9)

# Panel 3: Sensitivity heatmap
ax_heat = axes[1, 0]
pivot = sens_df.pivot(
    index="sigma_shift", columns="mu_shift", values="median_2yr"
)
pivot = pivot.reindex(index=["-30%", "+0%", "+30%"])
cmap = LinearSegmentedColormap.from_list(
    "rg", ["#ff6b6b", "#ffeb3b", "#00e676"]
)
im = ax_heat.imshow(pivot.values, cmap=cmap, aspect="auto")
ax_heat.set_xticks(range(len(pivot.columns)))
ax_heat.set_xticklabels(pivot.columns, fontsize=10)
ax_heat.set_yticks(range(len(pivot.index)))
ax_heat.set_yticklabels(pivot.index, fontsize=10)
ax_heat.set_xlabel("Drift (mu) Shift")
ax_heat.set_ylabel("Volatility (sigma) Shift")
ax_heat.set_title("Parameter Sensitivity (2yr Median)",
                   fontsize=13, fontweight="bold")
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        color = "black" if val > 20 else "white"
        ax_heat.text(j, i, f"${val:.0f}", ha="center", va="center",
                     fontsize=12, fontweight="bold", color=color)
plt.colorbar(im, ax=ax_heat, label="Price ($)")

# Panel 4: Model comparison table
ax_tbl = axes[1, 1]
ax_tbl.axis("off")
ax_tbl.set_title("Model Comparison (2-Year)",
                  fontsize=13, fontweight="bold", pad=20)

v1_rm = risk_metrics["v1_gbm"]
h_rm = risk_metrics["v2_heston"]
jd_rm = risk_metrics["v2_jump_diffusion"]

table_data = [
    ["Median Price",
     f"${mc_stats['v1_gbm']['2_years']['median']}",
     f"${mc_stats['v2_heston']['2_years']['median']}",
     f"${mc_stats['v2_jump_diffusion']['2_years']['median']}"],
    ["P5 (Downside)",
     f"${mc_stats['v1_gbm']['2_years']['p5']}",
     f"${mc_stats['v2_heston']['2_years']['p5']}",
     f"${mc_stats['v2_jump_diffusion']['2_years']['p5']}"],
    ["P95 (Upside)",
     f"${mc_stats['v1_gbm']['2_years']['p95']}",
     f"${mc_stats['v2_heston']['2_years']['p95']}",
     f"${mc_stats['v2_jump_diffusion']['2_years']['p95']}"],
    ["VaR 95%",
     f"${v1_rm['var_95']}", f"${h_rm['var_95']}",
     f"${jd_rm['var_95']}"],
    ["CVaR 95%",
     f"${v1_rm['cvar_95']}", f"${h_rm['cvar_95']}",
     f"${jd_rm['cvar_95']}"],
    ["P(Profit)",
     f"{v1_rm['prob_profit']}%", f"{h_rm['prob_profit']}%",
     f"{jd_rm['prob_profit']}%"],
    ["P(Double)",
     f"{v1_rm['prob_double']}%", f"{h_rm['prob_double']}%",
     f"{jd_rm['prob_double']}%"],
]

tbl = ax_tbl.table(
    cellText=table_data,
    colLabels=["Metric", "V1 GBM", "V2 Heston", "V2 Jump Diff"],
    loc="center", cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.0, 1.6)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#555555")
    if r == 0:
        cell.set_facecolor("#1565c0")
        cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor(
            "#1a1a2e" if r % 2 == 0 else "#16213e"
        )
        cell.set_text_props(color="white")

fig3.suptitle(f"Planet Labs ({TICKER}) — Robustness & Risk "
              f"Analysis (V2)",
              fontsize=16, fontweight="bold", y=0.98)
fig3.text(0.5, 0.01,
          f"{N_SIMULATIONS:,} paths per model  |  "
          f"{BOOTSTRAP_SAMPLES:,} bootstrap samples  |  "
          f"Not financial advice.",
          ha="center", fontsize=8, alpha=0.5)
plt.tight_layout(rect=[0, 0.02, 1, 0.96])
fig3.savefig("risk.png", dpi=150, bbox_inches="tight",
             facecolor=fig3.get_facecolor())
plt.close()
print("  Saved: risk.png")


# =====================================================================
# Save All Results
# =====================================================================

summary = {
    "ticker": TICKER,
    "company": stock.info.get("longName", "Planet Labs PBC"),
    "current_price": round(float(current_price), 2),
    "run_date": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
    "version": "v2 — stochastic vol + jump diffusion + regime switching",
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
        "log_return_kurtosis": round(
            float(log_returns.kurtosis()), 4
        ),
    },
    "v2_heston_parameters": {
        "kappa": kappa,
        "theta": round(theta, 6),
        "v0": round(v0, 6),
        "xi": round(xi, 4),
        "rho": round(rho, 4),
    },
    "v2_jump_diffusion_parameters": {
        "jump_intensity_per_year": round(lam, 2),
        "jump_mean": round(jump_mu, 4),
        "jump_std": round(jump_sigma, 4),
        "jumps_detected": n_jumps,
    },
    "regime_detection": (
        hmm_regimes if hmm_available
        else "HMM not available — used V1 fallback"
    ),
    "model_comparison": mc_stats,
    "risk_metrics": risk_metrics,
    "stress_testing": stress_results,
    "robustness": {
        "walk_forward_validation": wf_results,
        "bootstrap_confidence_intervals": {
            "mu_95_ci": mu_ci,
            "sigma_95_ci": sigma_ci,
        },
    },
    "improvements_over_v1": [
        "Stochastic volatility (Heston) — vol is now random and "
        "mean-reverting",
        "Jump diffusion (Merton) — models sudden crashes and spikes",
        "HMM regime detection — stress tests based on real "
        "market regimes",
        "Walk-forward compares GBM vs Heston head-to-head",
        "Multi-model comparison table for risk metrics",
    ],
    "disclaimer": (
        "Educational Monte Carlo simulation. Not financial advice. "
        "Past volatility and drift do not guarantee future results. "
        "Models assume specific parametric forms that may not hold "
        "in practice."
    ),
}

with open("results.json", "w") as f:
    json.dump(summary, f, indent=2)

# Percentile paths CSV (Heston as primary model)
percentiles = [5, 10, 25, 50, 75, 90, 95]
pct_paths_h = np.percentile(paths_heston, percentiles, axis=1)
pct_df = pd.DataFrame(
    {f"P{p}": pct_paths_h[i] for i, p in enumerate(percentiles)},
    index=forecast_dates,
)
pct_df.index.name = "date"
pct_df.to_csv("paths.csv")

# Stress test paths CSV
st_rows = []
for key, res in stress_results.items():
    median_p = np.median(stress_paths[key], axis=1)
    for i, d in enumerate(forecast_dates):
        st_rows.append({
            "date": d,
            "scenario": res["name"],
            "median_price": round(float(median_p[i]), 4),
        })
pd.DataFrame(st_rows).to_csv("stress_paths.csv", index=False)

# Sensitivity CSV
sens_df.to_csv("sensitivity.csv", index=False)

print("\nAll outputs saved:")
print("   fan_chart.png")
print("   stress.png")
print("   risk.png")
print("   results.json")
print("   paths.csv")
print("   stress_paths.csv")
print("   sensitivity.csv")
print(f"\nV1 GBM 2yr median:            ${np.median(v1_terminal):.2f}")
print(f"V2 Heston 2yr median:         "
      f"${np.median(heston_terminal):.2f}")
print(f"V2 Jump Diffusion 2yr median: "
      f"${np.median(jd_terminal):.2f}")
print(f"P(Profit) — GBM: {v1_rm['prob_profit']}%  "
      f"Heston: {h_rm['prob_profit']}%  "
      f"JD: {jd_rm['prob_profit']}%")
