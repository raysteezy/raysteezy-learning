#!/usr/bin/env python3
"""
Planet Labs (PL) — Monte Carlo Simulation V2 (Upgraded)

After I ran V1 through peer reviews, AI tests, as a result I got a C+ for using constant volatility and made-up stress
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
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib.colors import LinearSegmentedColormap

TICKER = "PL"
N_SIMULATIONS = 10_000
FORECAST_DAYS = 504
BOOTSTRAP_SAMPLES = 5_000
RANDOM_SEED = 42

MILESTONES = {"6_months": 126, "1_year": 252, "2_years": 504}


# --- data ---

def fetch_data() -> tuple[pd.Series, pd.Series, float, float, float]:
    """Fetch PL prices and return (close, log_returns, current_price, mu, sigma)."""
    stock = yf.Ticker(TICKER)
    hist = stock.history(period="max")
    hist = hist[hist.index >= "2021-01-01"]
    close = hist["Close"].dropna()
    log_returns = np.log(close / close.shift(1)).dropna()

    current_price = close.iloc[-1]
    mu_daily = log_returns.mean()
    sigma_daily = log_returns.std()

    return close, log_returns, float(current_price), float(mu_daily), float(sigma_daily)


# --- V1 baseline GBM ---

def run_v1_gbm(current_price: float, mu: float, sigma: float) -> np.ndarray:
    """Constant-vol GBM for comparison."""
    Z = np.random.standard_normal((FORECAST_DAYS, N_SIMULATIONS))
    drift = (mu - 0.5 * sigma ** 2)
    diffusion = sigma * Z
    log_paths = np.cumsum(drift + diffusion, axis=0)
    log_paths = np.vstack([np.zeros(N_SIMULATIONS), log_paths])
    return current_price * np.exp(log_paths)


# --- Heston stochastic volatility ---

def calibrate_heston(log_returns: pd.Series, sigma_daily: float) -> dict:
    """Calibrate Heston parameters from historical data."""
    realized_var = log_returns.rolling(21).var().dropna()
    theta = float(sigma_daily ** 2)
    v0 = float(realized_var.iloc[-1])
    kappa = 3.0
    xi = float(realized_var.std() * np.sqrt(252)) * 2
    rho = float(log_returns.corr(
        log_returns.rolling(21).std().pct_change()
    ))
    rho = max(min(rho, -0.1), -0.9)

    return {"theta": theta, "v0": v0, "kappa": kappa, "xi": xi, "rho": rho}


def run_heston(current_price: float, mu: float,
               params: dict) -> np.ndarray:
    """Simulate Heston stochastic volatility model."""
    kappa, theta, xi, rho, v0 = (
        params["kappa"], params["theta"], params["xi"],
        params["rho"], params["v0"],
    )
    dt = 1

    paths = np.zeros((FORECAST_DAYS + 1, N_SIMULATIONS))
    paths[0] = current_price
    variance = np.full(N_SIMULATIONS, v0)

    for t in range(FORECAST_DAYS):
        z1 = np.random.standard_normal(N_SIMULATIONS)
        z2 = (rho * z1
              + np.sqrt(1 - rho ** 2)
              * np.random.standard_normal(N_SIMULATIONS))

        var_pos = np.maximum(variance, 0)
        vol = np.sqrt(var_pos)

        paths[t + 1] = paths[t] * np.exp(
            (mu - 0.5 * var_pos) * dt + vol * np.sqrt(dt) * z1
        )
        variance = (var_pos
                    + kappa * (theta - var_pos) * dt
                    + xi * vol * np.sqrt(dt) * z2)

    return paths


# --- Merton jump diffusion ---

def estimate_jump_params(log_returns: pd.Series,
                         mu: float, sigma: float) -> dict:
    """Estimate jump parameters from tail behavior."""
    threshold = 3 * sigma
    jumps = log_returns[np.abs(log_returns - mu) > threshold]
    n_jumps = len(jumps)
    n_years = len(log_returns) / 252

    return {
        "lam": n_jumps / n_years if n_years > 0 else 2.0,
        "jump_mu": float(jumps.mean()) if n_jumps > 0 else 0.0,
        "jump_sigma": float(jumps.std()) if n_jumps > 1 else sigma,
        "n_jumps": n_jumps,
        "n_years": n_years,
    }


def run_jump_diffusion(current_price: float, mu: float, sigma: float,
                       jump_params: dict) -> np.ndarray:
    """Simulate Merton jump-diffusion model."""
    lam = jump_params["lam"]
    j_mu = jump_params["jump_mu"]
    j_sig = jump_params["jump_sigma"]

    paths = np.zeros((FORECAST_DAYS + 1, N_SIMULATIONS))
    paths[0] = current_price

    jump_compensator = lam * (np.exp(j_mu + 0.5 * j_sig ** 2) - 1) / 252

    for t in range(FORECAST_DAYS):
        z = np.random.standard_normal(N_SIMULATIONS)
        n_jumps_today = np.random.poisson(lam / 252, N_SIMULATIONS)

        jump_sizes = np.zeros(N_SIMULATIONS)
        for i in range(N_SIMULATIONS):
            if n_jumps_today[i] > 0:
                jump_sizes[i] = np.sum(
                    np.random.normal(j_mu, j_sig, n_jumps_today[i])
                )

        continuous = (mu - 0.5 * sigma ** 2 - jump_compensator)
        diffuse = sigma * z
        paths[t + 1] = paths[t] * np.exp(
            continuous + diffuse + jump_sizes
        )

    return paths


# --- HMM regime detection ---

def fit_hmm_regimes(log_returns: pd.Series) -> tuple[dict | None, bool]:
    """Fit a 2-state Gaussian HMM to detect calm/volatile regimes."""
    try:
        from hmmlearn.hmm import GaussianHMM

        returns_2d = log_returns.values.reshape(-1, 1)
        hmm = GaussianHMM(
            n_components=2, covariance_type="full",
            n_iter=2000, tol=1e-4, random_state=RANDOM_SEED,
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
        return hmm_regimes, True

    except Exception as e:
        print(f"  HMM fitting failed ({e}), using V1 multiplier fallback")
        return None, False


def build_scenarios(mu_daily: float, sigma_daily: float,
                    hmm_regimes: dict | None,
                    hmm_available: bool) -> dict:
    """Build stress-test scenarios from HMM regimes or V1 fallback."""
    if hmm_available:
        calm_mu = hmm_regimes["calm"]["mu_daily"]
        calm_sig = hmm_regimes["calm"]["sigma_daily"]
        vol_mu = hmm_regimes["volatile"]["mu_daily"]
        vol_sig = hmm_regimes["volatile"]["sigma_daily"]

        return {
            "market_crash": {
                "name": "Market Crash (2008-style)",
                "description": "Extreme volatile regime with forced negative drift",
                "mu": vol_mu * -3,
                "sigma": vol_sig * 2.0,
                "source": "HMM volatile regime, amplified",
            },
            "bear_market": {
                "name": "Prolonged Bear Market",
                "description": "Volatile regime continues with negative drift",
                "mu": vol_mu if vol_mu < 0 else -abs(vol_mu),
                "sigma": vol_sig * 1.3,
                "source": "HMM volatile regime, slightly amplified",
            },
            "base_case": {
                "name": "Base Case (Historical)",
                "description": "Blend of both regimes weighted by their frequency",
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

    return {
        "market_crash": {
            "name": "Market Crash (2008-style)",
            "description": "Severe downturn",
            "mu": mu_daily * -3.0, "sigma": sigma_daily * 2.5,
            "source": "Static multipliers (V1 fallback)",
        },
        "bear_market": {
            "name": "Prolonged Bear Market",
            "description": "Extended negative sentiment",
            "mu": mu_daily * -1.5, "sigma": sigma_daily * 1.5,
            "source": "Static multipliers (V1 fallback)",
        },
        "base_case": {
            "name": "Base Case",
            "description": "Historical parameters continue",
            "mu": mu_daily, "sigma": sigma_daily,
            "source": "Historical",
        },
        "calm_bull": {
            "name": "Strong Bull Market",
            "description": "Positive sentiment, lower vol",
            "mu": mu_daily * 3.0, "sigma": sigma_daily * 0.8,
            "source": "Static multipliers (V1 fallback)",
        },
        "volatile_bull": {
            "name": "Extreme Bull (Sector Boom)",
            "description": "Parabolic move with high vol",
            "mu": mu_daily * 5.0, "sigma": sigma_daily * 2.0,
            "source": "Static multipliers (V1 fallback)",
        },
    }


def run_stress_tests(scenarios: dict,
                     current_price: float) -> tuple[dict, dict]:
    """Run GBM for each stress scenario."""
    stress_results = {}
    stress_paths = {}

    for key, sc in scenarios.items():
        sc_paths = run_v1_gbm(current_price, sc["mu"], sc["sigma"])
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
            "mu_daily": round(float(sc["mu"]), 6),
            "sigma_daily": round(float(sc["sigma"]), 6),
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

    return stress_results, stress_paths


# --- robustness analysis ---

def walk_forward_validation(close: pd.Series, log_returns: pd.Series,
                            heston_params: dict) -> dict:
    """Compare GBM vs Heston on rolling 21-day predictions."""
    kappa = heston_params["kappa"]
    theta = heston_params["theta"]
    lookbacks = [63, 126, 252]
    wf_results = {}

    for lb in lookbacks:
        errors_gbm, errors_heston = [], []
        n_tests = 0

        for start in range(lb, len(close) - 21, 63):
            window = log_returns.iloc[start - lb : start]
            if len(window) < lb:
                continue

            w_mu = window.mean()
            w_sigma = window.std()
            actual = close.iloc[min(start + 21, len(close) - 1)]
            start_price = close.iloc[start]

            pred_gbm = float(
                start_price * np.exp((w_mu - 0.5 * w_sigma ** 2) * 21)
            )
            errors_gbm.append(np.log(actual / pred_gbm))

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
                "rmse": round(float(np.sqrt(np.mean(errors_gbm ** 2))), 4),
            },
            "heston": {
                "mean_error": round(float(np.mean(errors_heston)), 4),
                "mae": round(float(np.mean(np.abs(errors_heston))), 4),
                "rmse": round(float(np.sqrt(np.mean(errors_heston ** 2))), 4),
            },
            "n_tests": n_tests,
        }
        gbm_rmse = wf_results[f"{lb}_day_lookback"]["gbm"]["rmse"]
        heston_rmse = wf_results[f"{lb}_day_lookback"]["heston"]["rmse"]
        print(f"  {lb}-day lookback: GBM RMSE={gbm_rmse:.4f}, "
              f"Heston RMSE={heston_rmse:.4f}  (n={n_tests})")

    return wf_results


def bootstrap_parameters(log_returns: pd.Series) -> tuple[list, list, list, list]:
    """Bootstrap CIs for annualized drift and volatility."""
    lr_array = log_returns.values
    boot_mus, boot_sigmas = [], []

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

    return boot_mus, boot_sigmas, mu_ci, sigma_ci


def compute_model_stats(all_models: dict[str, np.ndarray],
                        current_price: float) -> tuple[dict, dict]:
    """Compute milestone stats and risk metrics for each model."""
    mc_stats = {}
    for model_name, paths in all_models.items():
        mc_stats[model_name] = {}
        for label, day in MILESTONES.items():
            terminal_at = paths[day]
            mc_stats[model_name][label] = {
                "median": round(float(np.median(terminal_at)), 2),
                "mean": round(float(np.mean(terminal_at)), 2),
                "p5": round(float(np.percentile(terminal_at, 5)), 2),
                "p25": round(float(np.percentile(terminal_at, 25)), 2),
                "p75": round(float(np.percentile(terminal_at, 75)), 2),
                "p95": round(float(np.percentile(terminal_at, 95)), 2),
            }

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
            "prob_profit": round(float(np.mean(t > current_price) * 100), 1),
            "prob_double": round(float(np.mean(t > 2 * current_price) * 100), 1),
            "prob_below_10": round(float(np.mean(t < 10) * 100), 1),
        }

    return mc_stats, risk_metrics


def run_sensitivity(mu_daily: float, sigma_daily: float,
                    current_price: float) -> pd.DataFrame:
    """Parameter sensitivity grid over drift and vol shifts."""
    mu_shifts = [-0.50, -0.25, 0.0, 0.25, 0.50]
    sigma_shifts = [-0.30, 0.0, 0.30]
    rows = []

    for s_shift in sigma_shifts:
        for m_shift in mu_shifts:
            adj_mu = mu_daily * (1 + m_shift)
            adj_sigma = sigma_daily * (1 + s_shift)
            Z = np.random.standard_normal((FORECAST_DAYS, 2000))
            d = (adj_mu - 0.5 * adj_sigma ** 2)
            diff = adj_sigma * Z
            lp = np.cumsum(d + diff, axis=0)
            tp = current_price * np.exp(lp[-1])
            rows.append({
                "mu_shift": f"{m_shift:+.0%}",
                "sigma_shift": f"{s_shift:+.0%}",
                "median_2yr": round(float(np.median(tp)), 2),
            })

    return pd.DataFrame(rows)


# --- charts ---

def plot_fan_chart(close: pd.Series, paths_v1: np.ndarray,
                   paths_heston: np.ndarray, paths_jd: np.ndarray,
                   current_price: float,
                   risk_metrics: dict) -> None:
    """Multi-model fan chart comparing V1 GBM, Heston, and jump diffusion."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(14, 7))

    hist_dates = close.index
    forecast_start = hist_dates[-1]
    forecast_dates = pd.bdate_range(
        start=forecast_start, periods=FORECAST_DAYS + 1
    )

    pct_v1 = np.percentile(paths_v1, [5, 25, 50, 75, 95], axis=1)
    ax.fill_between(forecast_dates, pct_v1[0], pct_v1[4],
                    alpha=0.08, color="#888888", label="V1 GBM 90% CI")
    ax.plot(forecast_dates, pct_v1[2], "--", color="#888888",
            linewidth=1, alpha=0.5, label="V1 GBM Median")

    pct_h = np.percentile(paths_heston, [5, 25, 50, 75, 95], axis=1)
    ax.fill_between(forecast_dates, pct_h[0], pct_h[4],
                    alpha=0.12, color="#00d4aa", label="V2 Heston 90% CI")
    ax.fill_between(forecast_dates, pct_h[1], pct_h[3],
                    alpha=0.25, color="#00d4aa", label="V2 Heston 50% CI")
    ax.plot(forecast_dates, pct_h[2], color="#FFA500",
            linewidth=2, label="V2 Heston Median")

    pct_jd = np.percentile(paths_jd, [50], axis=1)
    ax.plot(forecast_dates, pct_jd[0], color="#b388ff",
            linewidth=1.5, linestyle="--", alpha=0.8,
            label="V2 Jump Diffusion Median")

    ax.plot(hist_dates, close.values, color="#4fc3f7",
            linewidth=1.2, label="Historical Price")
    ax.axhline(y=current_price, color="white", linestyle=":", alpha=0.3)

    ax.annotate(f"Current: ${current_price:.2f}",
                xy=(forecast_start, current_price),
                fontsize=10, color="#FFA500", fontweight="bold",
                xytext=(-120, 20), textcoords="offset points")

    for label_name, day_idx in MILESTONES.items():
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


def plot_stress_chart(close: pd.Series, scenarios: dict,
                      stress_results: dict, stress_paths: dict,
                      current_price: float,
                      hmm_available: bool) -> None:
    """Stress-test scenario median paths."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(14, 7))

    forecast_start = close.index[-1]
    forecast_dates = pd.bdate_range(
        start=forecast_start, periods=FORECAST_DAYS + 1
    )

    colors_st = {
        "market_crash": "#ff6b6b", "bear_market": "#ff99cc",
        "base_case": "#FFA500", "calm_bull": "#00e676",
        "volatile_bull": "#b388ff",
    }
    styles_st = {
        "market_crash": "--", "bear_market": "--",
        "base_case": "-", "calm_bull": "-", "volatile_bull": "-",
    }

    for key in scenarios:
        median_p = np.median(stress_paths[key], axis=1)
        ax.plot(forecast_dates, median_p, styles_st[key],
                color=colors_st[key], linewidth=2,
                label=stress_results[key]["name"])
        end_val = stress_results[key]["median_2yr"]
        ax.plot(forecast_dates[-1], end_val, "o",
                color=colors_st[key], markersize=7, zorder=5)
        ax.annotate(f"${end_val:.2f}",
                    xy=(forecast_dates[-1], end_val),
                    fontsize=10, fontweight="bold", color=colors_st[key],
                    xytext=(10, 0), textcoords="offset points")

    hmm_label = ("HMM regime-based" if hmm_available
                 else "Static multipliers (V1)")
    ax.axhline(y=current_price, color="white", linestyle=":", alpha=0.3)
    ax.set_yscale("log")
    ax.set_title(f"Planet Labs ({TICKER}) — Stress Tests ({hmm_label})",
                 fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Stock Price (USD)", fontsize=12)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.7)
    fig.text(0.5, 0.01,
             "Median paths per scenario. Log scale. Not financial advice.",
             ha="center", fontsize=8, alpha=0.5)
    plt.tight_layout()
    fig.savefig("stress.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print("  Saved: stress.png")


def plot_risk_dashboard(paths_v1: np.ndarray, paths_heston: np.ndarray,
                        paths_jd: np.ndarray, current_price: float,
                        boot_mus: list, boot_sigmas: list,
                        sens_df: pd.DataFrame, mc_stats: dict,
                        risk_metrics: dict) -> None:
    """4-panel robustness dashboard."""
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    v1_terminal = paths_v1[FORECAST_DAYS]
    heston_terminal = paths_heston[FORECAST_DAYS]
    jd_terminal = paths_jd[FORECAST_DAYS]

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

    fig.suptitle(f"Planet Labs ({TICKER}) — Robustness & Risk "
                 f"Analysis (V2)",
                 fontsize=16, fontweight="bold", y=0.98)
    fig.text(0.5, 0.01,
             f"{N_SIMULATIONS:,} paths per model  |  "
             f"{BOOTSTRAP_SAMPLES:,} bootstrap samples  |  "
             f"Not financial advice.",
             ha="center", fontsize=8, alpha=0.5)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig("risk.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print("  Saved: risk.png")


# --- output ---

def save_results(close: pd.Series, log_returns: pd.Series,
                 mu_daily: float, sigma_daily: float,
                 current_price: float, heston_params: dict,
                 jump_params: dict, hmm_regimes: dict | None,
                 hmm_available: bool, mc_stats: dict,
                 risk_metrics: dict, stress_results: dict,
                 wf_results: dict, mu_ci: list, sigma_ci: list,
                 paths_heston: np.ndarray,
                 stress_paths: dict, sens_df: pd.DataFrame) -> None:
    """Save all JSON and CSV outputs."""
    stock = yf.Ticker(TICKER)
    forecast_dates = pd.bdate_range(
        start=close.index[-1], periods=FORECAST_DAYS + 1
    )

    summary = {
        "ticker": TICKER,
        "company": stock.info.get("longName", "Planet Labs PBC"),
        "current_price": round(float(current_price), 2),
        "run_date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "version": "v2 — stochastic vol + jump diffusion + regime switching",
        "historical_parameters": {
            "trading_days": len(log_returns),
            "date_range": (
                f"{log_returns.index[0].strftime('%Y-%m-%d')} to "
                f"{log_returns.index[-1].strftime('%Y-%m-%d')}"
            ),
            "daily_drift_mu": round(float(mu_daily), 6),
            "daily_volatility_sigma": round(float(sigma_daily), 6),
            "annualized_drift": round(float(mu_daily * 252), 4),
            "annualized_volatility": round(float(sigma_daily * np.sqrt(252)), 4),
            "log_return_skewness": round(float(log_returns.skew()), 4),
            "log_return_kurtosis": round(float(log_returns.kurtosis()), 4),
        },
        "v2_heston_parameters": {
            "kappa": heston_params["kappa"],
            "theta": round(heston_params["theta"], 6),
            "v0": round(heston_params["v0"], 6),
            "xi": round(heston_params["xi"], 4),
            "rho": round(heston_params["rho"], 4),
        },
        "v2_jump_diffusion_parameters": {
            "jump_intensity_per_year": round(jump_params["lam"], 2),
            "jump_mean": round(jump_params["jump_mu"], 4),
            "jump_std": round(jump_params["jump_sigma"], 4),
            "jumps_detected": jump_params["n_jumps"],
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
            "Stochastic volatility (Heston) — vol is now random and mean-reverting",
            "Jump diffusion (Merton) — models sudden crashes and spikes",
            "HMM regime detection — stress tests based on real market regimes",
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

    # Heston percentile paths
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    pct_paths = np.percentile(paths_heston, percentiles, axis=1)
    pct_df = pd.DataFrame(
        {f"P{p}": pct_paths[i] for i, p in enumerate(percentiles)},
        index=forecast_dates,
    )
    pct_df.index.name = "date"
    pct_df.to_csv("paths.csv")

    # Stress test median paths
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

    sens_df.to_csv("sensitivity.csv", index=False)

    print("\nAll outputs saved:")
    print("   fan_chart.png")
    print("   stress.png")
    print("   risk.png")
    print("   results.json")
    print("   paths.csv")
    print("   stress_paths.csv")
    print("   sensitivity.csv")


# --- main ---

def main() -> None:
    np.random.seed(RANDOM_SEED)

    print(f"[1/7] Fetching {TICKER} price history ...")
    close, log_returns, current_price, mu_daily, sigma_daily = fetch_data()
    mu_annual = mu_daily * 252
    sigma_annual = sigma_daily * np.sqrt(252)

    print(f"  Current price : ${current_price:.2f}")
    print(f"  Trading days  : {len(log_returns)}")
    print(f"  Daily mu/sigma: {mu_daily:.6f} / {sigma_daily:.6f}")
    print(f"  Annual mu/sigma: {mu_annual:.2%} / {sigma_annual:.2%}")
    print(f"  Skewness      : {log_returns.skew():.4f}")
    print(f"  Kurtosis      : {log_returns.kurtosis():.4f}")

    print(f"\n[2/7] Running V1 baseline GBM ({N_SIMULATIONS:,} paths) ...")
    paths_v1 = run_v1_gbm(current_price, mu_daily, sigma_daily)
    print(f"  V1 GBM 2yr median: ${np.median(paths_v1[FORECAST_DAYS]):.2f}")

    print(f"\n[3/7] Running V2 stochastic volatility model ...")
    heston_params = calibrate_heston(log_returns, sigma_daily)
    print(f"  Long-run variance (theta): {heston_params['theta']:.6f}")
    print(f"  Current variance (v0):     {heston_params['v0']:.6f}")
    print(f"  Mean reversion (kappa):    {heston_params['kappa']:.2f}")
    print(f"  Vol of vol (xi):           {heston_params['xi']:.4f}")
    print(f"  Correlation (rho):         {heston_params['rho']:.4f}")

    paths_heston = run_heston(current_price, mu_daily, heston_params)
    print(f"  Heston 2yr median: ${np.median(paths_heston[FORECAST_DAYS]):.2f}")

    print(f"\n[4/7] Running V2 jump diffusion model ...")
    jump_params = estimate_jump_params(log_returns, mu_daily, sigma_daily)
    print(f"  Detected {jump_params['n_jumps']} jumps in "
          f"{jump_params['n_years']:.1f} years")
    print(f"  Jump intensity (lambda): {jump_params['lam']:.2f}/year")
    print(f"  Jump mean: {jump_params['jump_mu']:.4f}")
    print(f"  Jump std:  {jump_params['jump_sigma']:.4f}")

    paths_jd = run_jump_diffusion(current_price, mu_daily, sigma_daily,
                                  jump_params)
    print(f"  Jump diffusion 2yr median: "
          f"${np.median(paths_jd[FORECAST_DAYS]):.2f}")

    print(f"\n[5/7] Fitting Hidden Markov Model for regime detection ...")
    hmm_regimes, hmm_available = fit_hmm_regimes(log_returns)
    if hmm_available:
        print(f"  Calm regime:     "
              f"mu={hmm_regimes['calm']['mu_annual']:.2%}, "
              f"vol={hmm_regimes['calm']['sigma_annual']:.2%}, "
              f"({hmm_regimes['calm']['frequency']:.0f}% of days)")
        print(f"  Volatile regime: "
              f"mu={hmm_regimes['volatile']['mu_annual']:.2%}, "
              f"vol={hmm_regimes['volatile']['sigma_annual']:.2%}, "
              f"({hmm_regimes['volatile']['frequency']:.0f}% of days)")

    scenarios = build_scenarios(mu_daily, sigma_daily,
                                hmm_regimes, hmm_available)

    print("\n  Running stress-test scenarios ...")
    stress_results, stress_paths = run_stress_tests(scenarios, current_price)

    print("\n[6/7] Robustness analysis ...")
    wf_results = walk_forward_validation(close, log_returns, heston_params)

    print("\n  Bootstrap confidence intervals ...")
    boot_mus, boot_sigmas, mu_ci, sigma_ci = bootstrap_parameters(log_returns)

    all_models = {
        "v1_gbm": paths_v1,
        "v2_heston": paths_heston,
        "v2_jump_diffusion": paths_jd,
    }
    mc_stats, risk_metrics = compute_model_stats(all_models, current_price)

    print("\n  Parameter sensitivity grid ...")
    sens_df = run_sensitivity(mu_daily, sigma_daily, current_price)

    print("\n[7/7] Making charts ...")
    plot_fan_chart(close, paths_v1, paths_heston, paths_jd,
                   current_price, risk_metrics)
    plot_stress_chart(close, scenarios, stress_results, stress_paths,
                      current_price, hmm_available)
    plot_risk_dashboard(paths_v1, paths_heston, paths_jd, current_price,
                        boot_mus, boot_sigmas, sens_df, mc_stats,
                        risk_metrics)

    save_results(close, log_returns, mu_daily, sigma_daily,
                 current_price, heston_params, jump_params,
                 hmm_regimes, hmm_available, mc_stats, risk_metrics,
                 stress_results, wf_results, mu_ci, sigma_ci,
                 paths_heston, stress_paths, sens_df)

    v1_rm = risk_metrics["v1_gbm"]
    h_rm = risk_metrics["v2_heston"]
    jd_rm = risk_metrics["v2_jump_diffusion"]

    print(f"\nV1 GBM 2yr median:            "
          f"${np.median(paths_v1[FORECAST_DAYS]):.2f}")
    print(f"V2 Heston 2yr median:         "
          f"${np.median(paths_heston[FORECAST_DAYS]):.2f}")
    print(f"V2 Jump Diffusion 2yr median: "
          f"${np.median(paths_jd[FORECAST_DAYS]):.2f}")
    print(f"P(Profit) — GBM: {v1_rm['prob_profit']}%  "
          f"Heston: {h_rm['prob_profit']}%  "
          f"JD: {jd_rm['prob_profit']}%")


if __name__ == "__main__":
    main()
