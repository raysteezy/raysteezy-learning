#!/usr/bin/env python3
"""
Planet Labs (PL) — Monte Carlo Simulation V2 (Upgraded)

After I ran V1 through peer reviews, AI tests, as a result I got a C+ for using constant volatility and made-up stress
scenarios, I rebuilt the simulation with proper stochastic models:

  - Stochastic volatility (Heston-inspired, mean-reverting vol)
  - Jump diffusion (Merton model for crashes/spikes)
  - Regime-switching stress tests (HMM-based, not made-up)
  - Walk-forward validation (GBM vs Heston)
  - Bootstrap parameter confidence intervals

What I learned: constant volatility is a terrible assumption.
Real markets have vol clustering, jumps, and regime changes.

Outputs: results.json, paths.csv, stress_paths.csv, sensitivity.csv
Disclaimer: Educational simulation only — not financial advice.
"""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

TICKER = "PL"
N_SIMS = 10_000
FORECAST_DAYS = 504
BOOT_SAMPLES = 5_000
RANDOM_SEED = 42


def fetch_data():
    stock = yf.Ticker(TICKER)
    hist = stock.history(period="max")
    hist = hist[hist.index >= "2021-01-01"]
    close = hist["Close"].dropna()
    log_ret = np.log(close / close.shift(1)).dropna()
    return stock, close, log_ret


def run_v1_gbm(price, mu, sigma):
    np.random.seed(RANDOM_SEED)
    Z = np.random.standard_normal((FORECAST_DAYS, N_SIMS))
    log_paths = np.cumsum((mu - 0.5 * sigma ** 2) + sigma * Z, axis=0)
    return price * np.exp(np.vstack([np.zeros(N_SIMS), log_paths]))


def calibrate_heston(log_ret, sigma):
    rv = log_ret.rolling(21).var().dropna()
    rho = float(log_ret.corr(log_ret.rolling(21).std().pct_change()))
    return {
        "theta": float(sigma ** 2), "v0": float(rv.iloc[-1]), "kappa": 3.0,
        "xi": float(rv.std() * np.sqrt(252)) * 2,
        "rho": max(min(rho, -0.1), -0.9),
    }


def run_heston(price, mu, p):
    paths = np.zeros((FORECAST_DAYS + 1, N_SIMS))
    paths[0] = price
    var = np.full(N_SIMS, p["v0"])

    for t in range(FORECAST_DAYS):
        z1 = np.random.standard_normal(N_SIMS)
        z2 = p["rho"] * z1 + np.sqrt(1 - p["rho"] ** 2) * np.random.standard_normal(N_SIMS)
        vp = np.maximum(var, 0)
        vol = np.sqrt(vp)
        paths[t + 1] = paths[t] * np.exp((mu - 0.5 * vp) + vol * z1)
        var = vp + p["kappa"] * (p["theta"] - vp) / 252 + p["xi"] * vol * z2
    return paths


def estimate_jump_params(log_ret, mu, sigma):
    threshold = 3 * sigma
    jumps = log_ret[np.abs(log_ret - mu) > threshold]
    n_j, n_yr = len(jumps), len(log_ret) / 252
    return {
        "lam": n_j / n_yr if n_yr > 0 else 2.0,
        "jump_mu": float(jumps.mean()) if n_j > 0 else 0.0,
        "jump_sigma": float(jumps.std()) if n_j > 1 else sigma,
        "n_jumps": n_j,
    }


def run_jump_diffusion(price, mu, sigma, jp):
    paths = np.zeros((FORECAST_DAYS + 1, N_SIMS))
    paths[0] = price
    comp = jp["lam"] * (np.exp(jp["jump_mu"] + 0.5 * jp["jump_sigma"] ** 2) - 1) / 252

    for t in range(FORECAST_DAYS):
        z = np.random.standard_normal(N_SIMS)
        nj = np.random.poisson(jp["lam"] / 252, N_SIMS)
        js = np.array([np.sum(np.random.normal(jp["jump_mu"], jp["jump_sigma"], n)) if n > 0 else 0.0 for n in nj])
        paths[t + 1] = paths[t] * np.exp((mu - 0.5 * sigma ** 2 - comp) + sigma * z + js)
    return paths


def fit_hmm_regimes(log_ret):
    try:
        from hmmlearn.hmm import GaussianHMM
        hmm = GaussianHMM(n_components=2, covariance_type="full", n_iter=2000, tol=1e-4, random_state=RANDOM_SEED)
        hmm.fit(log_ret.values.reshape(-1, 1))
        regimes = hmm.predict(log_ret.values.reshape(-1, 1))
        means, stds = hmm.means_.flatten(), np.sqrt(hmm.covars_.flatten())
        calm, vol = np.argmin(stds), np.argmax(stds)
        return {label: {"mu_daily": float(means[i]), "sigma_daily": float(stds[i]),
                        "frequency": float(np.mean(regimes == i) * 100),
                        "mu_annual": float(means[i] * 252), "sigma_annual": float(stds[i] * np.sqrt(252))}
                for label, i in [("calm", calm), ("volatile", vol)]}
    except Exception as e:
        print(f"  HMM failed ({e}), using V1 fallback")
        return None


def build_scenarios(mu, sigma, hmm):
    if hmm:
        cm, cs = hmm["calm"]["mu_daily"], hmm["calm"]["sigma_daily"]
        vm, vs = hmm["volatile"]["mu_daily"], hmm["volatile"]["sigma_daily"]
        return {
            "market_crash":  {"name": "Market Crash (2008-style)", "description": "Extreme volatile regime",
                              "mu": vm * -3, "sigma": vs * 2.0, "source": "HMM amplified"},
            "bear_market":   {"name": "Prolonged Bear Market", "description": "Volatile regime, negative drift",
                              "mu": vm if vm < 0 else -abs(vm), "sigma": vs * 1.3, "source": "HMM volatile"},
            "base_case":     {"name": "Base Case (Historical)", "description": "Blended regimes",
                              "mu": mu, "sigma": sigma, "source": "Full historical"},
            "calm_bull":     {"name": "Calm Bull Market", "description": "Calm regime dominates",
                              "mu": cm if cm > 0 else abs(cm), "sigma": cs, "source": "HMM calm"},
            "volatile_bull": {"name": "Volatile Bull (Sector Boom)", "description": "Strong drift, wild swings",
                              "mu": abs(cm) * 3, "sigma": vs * 1.5, "source": "HMM amplified"},
        }
    fallback = lambda n, d, mm, sm: {"name": n, "description": d, "mu": mu * mm, "sigma": sigma * sm, "source": "Static (V1 fallback)"}
    return {
        "market_crash":  fallback("Market Crash (2008-style)", "Severe downturn", -3.0, 2.5),
        "bear_market":   fallback("Prolonged Bear Market", "Extended negative sentiment", -1.5, 1.5),
        "base_case":     fallback("Base Case", "Historical", 1.0, 1.0),
        "calm_bull":     fallback("Strong Bull Market", "Positive sentiment", 3.0, 0.8),
        "volatile_bull": fallback("Extreme Bull (Sector Boom)", "Parabolic move", 5.0, 2.0),
    }


def run_stress_tests(price, scenarios):
    results, paths_out = {}, {}
    for key, sc in scenarios.items():
        Z = np.random.standard_normal((FORECAST_DAYS, N_SIMS))
        log_p = np.cumsum((sc["mu"] - 0.5 * sc["sigma"] ** 2) + sc["sigma"] * Z, axis=0)
        sp = price * np.exp(np.vstack([np.zeros(N_SIMS), log_p]))
        paths_out[key] = sp
        term = sp[FORECAST_DAYS]
        med_path = np.median(sp, axis=1)
        results[key] = {
            "name": sc["name"], "description": sc["description"], "source": sc["source"],
            "mu_daily": round(float(sc["mu"]), 6), "sigma_daily": round(float(sc["sigma"]), 6),
            "median_2yr": round(float(np.median(term)), 2),
            "p10_2yr": round(float(np.percentile(term, 10)), 2),
            "p90_2yr": round(float(np.percentile(term, 90)), 2),
            "prob_profit": round(float(np.mean(term > price) * 100), 1),
            "max_drawdown_median": round(float(np.min(med_path / np.maximum.accumulate(med_path) - 1) * 100), 1),
        }
        print(f"  {sc['name']:35s} -> Median: ${results[key]['median_2yr']:.2f}")
    return results, paths_out


def walk_forward_validation(close, log_ret, kappa, theta):
    wf = {}
    for lb in (63, 126, 252):
        eg, eh, n = [], [], 0
        for s in range(lb, len(close) - 21, 63):
            w = log_ret.iloc[s - lb:s]
            if len(w) < lb:
                continue
            wm, ws = w.mean(), w.std()
            actual, sp = close.iloc[min(s + 21, len(close) - 1)], close.iloc[s]
            eg.append(np.log(actual / (sp * np.exp((wm - 0.5 * ws ** 2) * 21))))
            hv = ws ** 2 + kappa * (theta - ws ** 2) * (21 / 252)
            eh.append(np.log(actual / (sp * np.exp((wm - 0.5 * hv) * 21))))
            n += 1
        eg, eh = np.array(eg), np.array(eh)
        wf[f"{lb}_day_lookback"] = {
            "gbm": {"mean_error": round(float(eg.mean()), 4), "mae": round(float(np.abs(eg).mean()), 4),
                     "rmse": round(float(np.sqrt((eg ** 2).mean())), 4)},
            "heston": {"mean_error": round(float(eh.mean()), 4), "mae": round(float(np.abs(eh).mean()), 4),
                       "rmse": round(float(np.sqrt((eh ** 2).mean())), 4)},
            "n_tests": n,
        }
        print(f"  {lb}-day: GBM RMSE={wf[f'{lb}_day_lookback']['gbm']['rmse']:.4f}, "
              f"Heston RMSE={wf[f'{lb}_day_lookback']['heston']['rmse']:.4f} (n={n})")
    return wf


def bootstrap_parameters(log_ret):
    lr = log_ret.values
    bm = [np.random.choice(lr, len(lr), replace=True).mean() * 252 for _ in range(BOOT_SAMPLES)]
    bs = [np.random.choice(lr, len(lr), replace=True).std() * np.sqrt(252) for _ in range(BOOT_SAMPLES)]
    mu_ci = [round(float(np.percentile(bm, 2.5)), 6), round(float(np.percentile(bm, 97.5)), 6)]
    sig_ci = [round(float(np.percentile(bs, 2.5)), 6), round(float(np.percentile(bs, 97.5)), 6)]
    print(f"  mu 95% CI: [{mu_ci[0]:.4f}, {mu_ci[1]:.4f}]")
    print(f"  sigma 95% CI: [{sig_ci[0]:.4f}, {sig_ci[1]:.4f}]")
    return mu_ci, sig_ci


def compute_model_stats(models, price):
    milestones = {"6_months": 126, "1_year": 252, "2_years": FORECAST_DAYS}
    mc = {n: {l: {s: round(float(f(p[d])), 2) for s, f in
              [("median", np.median), ("mean", np.mean),
               ("p5", lambda x: np.percentile(x, 5)), ("p25", lambda x: np.percentile(x, 25)),
               ("p75", lambda x: np.percentile(x, 75)), ("p95", lambda x: np.percentile(x, 95))]}
             for l, d in milestones.items()} for n, p in models.items()}
    risk = {}
    for n, p in models.items():
        t = p[FORECAST_DAYS]
        v95 = round(float(np.percentile(t, 5)), 2)
        v99 = round(float(np.percentile(t, 1)), 2)
        below_95 = t[t <= v95]
        below_99 = t[t <= v99]
        risk[n] = {"var_95": v95, "var_99": v99,
                   "cvar_95": round(float(np.mean(below_95)), 2) if len(below_95) > 0 else v95,
                   "cvar_99": round(float(np.mean(below_99)), 2) if len(below_99) > 0 else v99,
                   "prob_profit": round(float(np.mean(t > price) * 100), 1),
                   "prob_double": round(float(np.mean(t > 2 * price) * 100), 1),
                   "prob_below_10": round(float(np.mean(t < 10) * 100), 1)}
    return mc, risk


def run_sensitivity(price, mu, sigma):
    rows = []
    for ss in (-0.30, 0.0, 0.30):
        for ms in (-0.50, -0.25, 0.0, 0.25, 0.50):
            am, asig = mu * (1 + ms), sigma * (1 + ss)
            Z = np.random.standard_normal((FORECAST_DAYS, 2000))
            tp = price * np.exp(np.sum((am - 0.5 * asig ** 2) + asig * Z, axis=0))
            rows.append({"mu_shift": f"{ms:+.0%}", "sigma_shift": f"{ss:+.0%}", "median_2yr": round(float(np.median(tp)), 2)})
    return pd.DataFrame(rows)


def save_results(stock, price, log_ret, mu, sigma, hp, jp, hmm, mc, risk,
                 stress, wf, mu_ci, sig_ci, close, paths_h, sp, sens):
    summary = {
        "ticker": TICKER, "company": stock.info.get("longName", "Planet Labs PBC"),
        "current_price": round(float(price), 2),
        "run_date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "version": "v2 — stochastic vol + jump diffusion + regime switching",
        "historical_parameters": {
            "trading_days": len(log_ret),
            "date_range": f"{log_ret.index[0].strftime('%Y-%m-%d')} to {log_ret.index[-1].strftime('%Y-%m-%d')}",
            "daily_drift_mu": round(float(mu), 6), "daily_volatility_sigma": round(float(sigma), 6),
            "annualized_drift": round(float(mu * 252), 4), "annualized_volatility": round(float(sigma * np.sqrt(252)), 4),
            "log_return_skewness": round(float(log_ret.skew()), 4), "log_return_kurtosis": round(float(log_ret.kurtosis()), 4),
        },
        "v2_heston_parameters": {k: round(v, 6) if isinstance(v, float) else v for k, v in hp.items()},
        "v2_jump_diffusion_parameters": {
            "jump_intensity_per_year": round(jp["lam"], 2), "jump_mean": round(jp["jump_mu"], 4),
            "jump_std": round(jp["jump_sigma"], 4), "jumps_detected": jp["n_jumps"],
        },
        "regime_detection": hmm if hmm else "HMM not available — used V1 fallback",
        "model_comparison": mc, "risk_metrics": risk, "stress_testing": stress,
        "robustness": {"walk_forward_validation": wf, "bootstrap_confidence_intervals": {"mu_95_ci": mu_ci, "sigma_95_ci": sig_ci}},
        "improvements_over_v1": [
            "Stochastic volatility (Heston)", "Jump diffusion (Merton)",
            "HMM regime-based stress tests", "Walk-forward GBM vs Heston",
        ],
        "disclaimer": "Educational Monte Carlo simulation. Not financial advice.",
    }
    with open("results.json", "w") as f:
        json.dump(summary, f, indent=2)

    dates = pd.bdate_range(start=close.index[-1], periods=FORECAST_DAYS + 1)
    pcts = [5, 10, 25, 50, 75, 90, 95]
    pv = np.percentile(paths_h, pcts, axis=1)
    pd.DataFrame({f"P{p}": pv[i] for i, p in enumerate(pcts)}, index=dates).rename_axis("date").to_csv("paths.csv")

    rows = [{"date": d, "scenario": stress[k]["name"], "median_price": round(float(np.median(sp[k], axis=1)[i]), 4)}
            for k in stress for i, d in enumerate(dates)]
    pd.DataFrame(rows).to_csv("stress_paths.csv", index=False)
    sens.to_csv("sensitivity.csv", index=False)
    print("  Saved: results.json, paths.csv, stress_paths.csv, sensitivity.csv")


def main():
    print(f"[1/7] Fetching {TICKER} price history ...")
    stock, close, log_ret = fetch_data()
    price, mu, sigma = close.iloc[-1], log_ret.mean(), log_ret.std()
    print(f"  ${price:.2f}, {len(log_ret)} days, annual: {mu * 252:.2%} / {sigma * np.sqrt(252):.2%}")

    print(f"\n[2/7] V1 baseline GBM ({N_SIMS:,} paths) ...")
    pv1 = run_v1_gbm(price, mu, sigma)
    print(f"  V1 median: ${np.median(pv1[FORECAST_DAYS]):.2f}")

    print("\n[3/7] Heston stochastic volatility ...")
    hp = calibrate_heston(log_ret, sigma)
    ph = run_heston(price, mu, hp)
    print(f"  Heston median: ${np.median(ph[FORECAST_DAYS]):.2f}")

    print("\n[4/7] Jump diffusion ...")
    jp = estimate_jump_params(log_ret, mu, sigma)
    pjd = run_jump_diffusion(price, mu, sigma, jp)
    print(f"  JD median: ${np.median(pjd[FORECAST_DAYS]):.2f}")

    print("\n[5/7] HMM regime detection + stress tests ...")
    hmm = fit_hmm_regimes(log_ret)
    if hmm:
        for r in ("calm", "volatile"):
            h = hmm[r]
            print(f"  {r.title()}: mu={h['mu_annual']:.2%}, vol={h['sigma_annual']:.2%} ({h['frequency']:.0f}%)")
    scenarios = build_scenarios(mu, sigma, hmm)
    stress, sp = run_stress_tests(price, scenarios)

    print("\n[6/7] Robustness ...")
    wf = walk_forward_validation(close, log_ret, hp["kappa"], hp["theta"])
    print("  Bootstrap CIs ...")
    mu_ci, sig_ci = bootstrap_parameters(log_ret)
    models = {"v1_gbm": pv1, "v2_heston": ph, "v2_jump_diffusion": pjd}
    mc, risk = compute_model_stats(models, price)
    print("  Sensitivity grid ...")
    sens = run_sensitivity(price, mu, sigma)

    print("\n[7/7] Saving ...")
    save_results(stock, price, log_ret, mu, sigma, hp, jp, hmm, mc, risk, stress, wf, mu_ci, sig_ci, close, ph, sp, sens)

    print(f"\nV1 GBM: ${np.median(pv1[FORECAST_DAYS]):.2f}  Heston: ${np.median(ph[FORECAST_DAYS]):.2f}  "
          f"JD: ${np.median(pjd[FORECAST_DAYS]):.2f}")
    print(f"P(Profit) — GBM: {risk['v1_gbm']['prob_profit']}%  "
          f"Heston: {risk['v2_heston']['prob_profit']}%  JD: {risk['v2_jump_diffusion']['prob_profit']}%")


if __name__ == "__main__":
    main()
