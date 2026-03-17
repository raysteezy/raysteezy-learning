# Disclaimer

## Not Financial Advice

This repository and all materials contained within it — including but not limited to source code, data files, visualizations, simulation results, price predictions, Monte Carlo analyses, stress tests, and robustness metrics — are provided **strictly for educational and academic purposes only**.

Nothing in this repository constitutes:

- Financial advice or investment advice
- A recommendation or solicitation to buy, sell, or hold any security
- An offer to provide investment advisory services
- A guarantee or prediction of future financial performance
- Professional financial, legal, or tax counsel

**You should not make any investment decision based on the information in this repository.** Always consult a qualified, licensed financial advisor before making any investment decisions.

## No Warranties

All code, models, and analyses are provided "AS IS" without warranty of any kind, express or implied. The author(s) make no representations or warranties regarding the accuracy, completeness, reliability, or suitability of any information, model output, or prediction contained herein.

The Monte Carlo simulations, regression models, stress tests, and all other quantitative analyses are **statistical exercises** based on historical data. They are subject to significant limitations including but not limited to:

- **Model risk** — The Geometric Brownian Motion (GBM) model assumes log-normal returns and constant parameters, which may not hold in practice
- **Parameter uncertainty** — Estimated drift and volatility are subject to sampling error and may not reflect future market conditions
- **Survivorship bias** — Historical data may not account for delisted or bankrupt securities
- **Distributional assumptions** — Real-world returns exhibit fat tails, skewness, and volatility clustering that simple GBM does not capture
- **Black swan events** — Extreme market events are inherently unpredictable and may not be adequately represented in historical data

## Forward-Looking Statements

Any projections, forecasts, or forward-looking statements in this repository are based on assumptions that the author(s) believe are reasonable as of the date they were generated, but actual results may differ materially. Important factors that could cause actual results to differ include:

- Changes in market conditions, interest rates, or macroeconomic environment
- Company-specific events (earnings, management changes, regulatory actions, M&A activity)
- Geopolitical events, natural disasters, or pandemics
- Changes in industry dynamics, competition, or technology
- Legislative or regulatory changes affecting the company or sector

**Past performance does not guarantee future results.**

## Data Sources & Attribution

- **Financial data** is sourced from [Yahoo Finance](https://finance.yahoo.com/) via the [yfinance](https://github.com/ranaroussi/yfinance) Python library (Apache 2.0 License)
- The yfinance library is **not affiliated with, endorsed by, or vetted by Yahoo, Inc.**
- Yahoo Finance data is intended for **personal and educational use** — refer to [Yahoo's Terms of Service](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html) for details on your rights to use the data
- Data files in this repository are snapshots used for **reproducibility of academic exercises** and should not be treated as an authoritative or real-time data source
- Yahoo!, Y!Finance, and Yahoo! Finance are registered trademarks of Yahoo, Inc.

## Academic Use Statement

This project was created as a **personal learning exercise** in quantitative finance, Python programming, and machine learning. It is intended to demonstrate:

- Data pipeline automation with GitHub Actions
- Statistical modeling techniques (regression, Monte Carlo simulation)
- Risk analysis methodologies (VaR, CVaR, stress testing, walk-forward validation)
- Professional software engineering practices (version control, documentation, CI/CD)

This work is **not peer-reviewed** and has **not been submitted to any academic institution** as part of a formal research publication. It represents independent study and self-directed learning.

## Regulatory Notice

The author of this repository is **not a registered investment advisor, broker-dealer, or financial analyst** under any jurisdiction's securities regulations. This repository does not fall under FINRA Rule 2210 (Communications with the Public) because:

1. The author is not a member firm or associated person of a FINRA member
2. No financial product or service is being promoted or sold
3. No compensation is received for any content in this repository
4. All content is clearly labeled as educational and non-advisory

However, the author voluntarily includes these disclaimers as a matter of best practice and transparency.

## Limitation of Liability

In no event shall the author(s) or contributors be liable for any direct, indirect, incidental, special, consequential, or exemplary damages — including but not limited to damages for loss of profits, goodwill, data, or other intangible losses — resulting from the use of or inability to use the materials in this repository.

## Contact

If you have questions or concerns about the content of this repository, please open an issue or contact [@raysteezy](https://github.com/raysteezy).

---

*Last updated: March 17, 2026*
