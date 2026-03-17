# Legal

Everything legal for this project in one place — disclaimer, data attribution, security policy, and the MIT license — so you don't have to hunt through multiple files.

---

## Table of Contents

- [Disclaimer](#disclaimer)
- [Data Notice & Attribution](#data-notice--attribution)
- [Security Policy](#security-policy)
- [MIT License](#mit-license)

---

## Disclaimer

### Not Financial Advice

This repository and all materials contained within it — including but not limited to source code, data files, visualizations, simulation results, price predictions, Monte Carlo analyses, stress tests, and robustness metrics — are provided **strictly for educational and academic purposes only**.

Nothing in this repository constitutes:

- Financial advice or investment advice
- A recommendation or solicitation to buy, sell, or hold any security
- An offer to provide investment advisory services
- A guarantee or prediction of future financial performance
- Professional financial, legal, or tax counsel

**You should not make any investment decision based on the information in this repository.** Always consult a qualified, licensed financial advisor before making any investment decisions.

### No Warranties

All code, models, and analyses are provided "AS IS" without warranty of any kind, express or implied. The author(s) make no representations or warranties regarding the accuracy, completeness, reliability, or suitability of any information, model output, or prediction contained herein.

The Monte Carlo simulations, regression models, stress tests, and all other quantitative analyses are **statistical exercises** based on historical data. They are subject to significant limitations including but not limited to:

- **Model risk** — The Geometric Brownian Motion (GBM) model assumes log-normal returns and constant parameters, which may not hold in practice
- **Parameter uncertainty** — Estimated drift and volatility are subject to sampling error and may not reflect future market conditions
- **Survivorship bias** — Historical data may not account for delisted or bankrupt securities
- **Distributional assumptions** — Real-world returns exhibit fat tails, skewness, and volatility clustering that simple GBM does not capture
- **Black swan events** — Extreme market events are inherently unpredictable and may not be adequately represented in historical data

### Forward-Looking Statements

Any projections, forecasts, or forward-looking statements in this repository are based on assumptions that the author(s) believe are reasonable as of the date they were generated, but actual results may differ materially. Important factors that could cause actual results to differ include:

- Changes in market conditions, interest rates, or macroeconomic environment
- Company-specific events (earnings, management changes, regulatory actions, M&A activity)
- Geopolitical events, natural disasters, or pandemics
- Changes in industry dynamics, competition, or technology
- Legislative or regulatory changes affecting the company or sector

**Past performance does not guarantee future results.**

### Data Sources & Attribution

- **Financial data** is sourced from [Yahoo Finance](https://finance.yahoo.com/) via the [yfinance](https://github.com/ranaroussi/yfinance) Python library (Apache 2.0 License)
- The yfinance library is **not affiliated with, endorsed by, or vetted by Yahoo, Inc.**
- Yahoo Finance data is intended for **personal and educational use** — refer to [Yahoo's Terms of Service](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html) for details on your rights to use the data
- Data files in this repository are snapshots used for **reproducibility of academic exercises** and should not be treated as an authoritative or real-time data source
- Yahoo!, Y!Finance, and Yahoo! Finance are registered trademarks of Yahoo, Inc.

### Academic Use Statement

This project was created as a **personal learning exercise** in quantitative finance, Python programming, and machine learning. It is intended to demonstrate:

- Data pipeline automation with GitHub Actions
- Statistical modeling techniques (regression, Monte Carlo simulation)
- Risk analysis methodologies (VaR, CVaR, stress testing, walk-forward validation)
- Professional software engineering practices (version control, documentation, CI/CD)

This work is **not peer-reviewed** and has **not been submitted to any academic institution** as part of a formal research publication. It represents independent study and self-directed learning.

### Regulatory Notice

The author of this repository is **not a registered investment advisor, broker-dealer, or financial analyst** under any jurisdiction's securities regulations. This repository does not fall under FINRA Rule 2210 (Communications with the Public) because:

1. The author is not a member firm or associated person of a FINRA member
2. No financial product or service is being promoted or sold
3. No compensation is received for any content in this repository
4. All content is clearly labeled as educational and non-advisory

However, the author voluntarily includes these disclaimers as a matter of best practice and transparency.

### Limitation of Liability

In no event shall the author(s) or contributors be liable for any direct, indirect, incidental, special, consequential, or exemplary damages — including but not limited to damages for loss of profits, goodwill, data, or other intangible losses — resulting from the use of or inability to use the materials in this repository.

---

## Data Notice & Attribution

### Data Sources

| Source | License / Terms | Usage in This Repo |
|--------|----------------|-------------------|
| [Yahoo Finance](https://finance.yahoo.com/) | [Yahoo Terms of Service](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html) | Price history, financial statements, key metrics |
| [yfinance](https://github.com/ranaroussi/yfinance) | Apache License 2.0 | Python library for data retrieval |

### Yahoo Finance Data

The financial data in this repository was obtained from Yahoo Finance through the yfinance open-source library. Per Yahoo's terms:

- Yahoo Finance data is provided for **informational purposes only**
- Yahoo Finance data is intended for **personal use only**
- Users should refer to Yahoo's Terms of Service for details on data usage rights

**Yahoo!, Y!Finance, and Yahoo! Finance are registered trademarks of Yahoo, Inc.** The yfinance library and this repository are not affiliated with, endorsed by, or vetted by Yahoo, Inc.

### Data Accuracy

- Data files in this repository represent **point-in-time snapshots** and may not reflect current market conditions
- Historical financial data is subject to retroactive adjustments for stock splits, dividends, and restatements
- No guarantee is made regarding the accuracy, completeness, or timeliness of any data
- Data should **not** be used as the basis for any investment decision

### Derived Outputs

All model outputs (predictions, simulations, charts, risk metrics) are **derived from the source data above** and are subject to:

- The same usage restrictions as the underlying data
- Additional uncertainty introduced by modeling assumptions
- The limitations described in the [Disclaimer](#disclaimer) section above

### Redistribution

If you fork or clone this repository:

- You are responsible for complying with Yahoo's Terms of Service regarding data usage
- The source code is available under the [MIT License](#mit-license)
- Data files carry the restrictions of their original source (Yahoo Finance)
- Model outputs and visualizations are provided for educational purposes only

---

## Security Policy

### Reporting a Vulnerability

If you discover a security issue in this repository, please report it responsibly.

**Contact:** Open an issue or email raymundosanchez2907@gmail.com.

### Supported Versions

| Version | Supported |
|---------|-----------|
| Latest (main branch) | Yes |

### Security Practices

This repository follows these security practices:

- **No secrets in code** — API keys, tokens, and credentials are never committed to this repository
- **Dependabot enabled** — Automated vulnerability alerts and security updates are active
- **Public repository** — All code is open source; no private data is stored here
- **Minimal permissions** — GitHub Actions workflows only write to the `data/` folder
- **`.gitignore` protection** — Environment files, credentials, and sensitive file types are excluded from commits

### Third-Party Integrations

| Service | Authentication | Scope |
|---------|---------------|-------|
| GitHub Actions | Built-in `GITHUB_TOKEN` | Contents write (for auto-commits only) |
| Yahoo Finance | None required | Public financial data via yfinance |

---

## MIT License

```
MIT License

Copyright (c) 2026 raysteezy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

### Contact

If you have questions or concerns about any of the above, please open an issue or contact [@raysteezy](https://github.com/raysteezy).

*Last updated: March 17, 2026*
