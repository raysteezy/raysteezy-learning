# Security Policy

## Reporting a Vulnerability

If you find a security issue in this repository, please report it responsibly:

1. **Do not open a public issue** — email me at raymundosanchez2907@gmail.com instead
2. Describe the issue and how to reproduce it
3. I'll respond within 72 hours and work on a fix

## Security Practices

This is a public educational repository. Here's how I keep it safe:

- **No secrets in code** — API keys, tokens, and credentials are never committed to this repository
- **No hardcoded credentials** — All integrations use environment variables or external dashboards
- **`.gitignore` protection** — Environment files, credentials, and sensitive file types are excluded from commits
- **Dependabot enabled** — Automatically checks for dependency vulnerabilities weekly
- **Minimal permissions** — GitHub Actions workflow only has write access to the `data/` folder

## What's Automated

| Service | How It Authenticates | What It Does |
|---------|---------------------|-------------|
| GitHub Actions | Built-in `GITHUB_TOKEN` | Runs the weekly data pipeline and commits updated files |
| yfinance | No auth needed | Pulls public stock data from Yahoo Finance |

## What's NOT in This Repo

- No API keys or tokens
- No `.env` files
- No personal credentials
- No private data

If you see something that looks like it shouldn't be public, please let me know.
