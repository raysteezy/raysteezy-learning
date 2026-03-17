# Security Policy

## Reporting a Vulnerability

If you discover a security issue in this repository, please report it responsibly.

**Contact:** Open a private issue or reach out to [@raysteezy](https://github.com/raysteezy) directly.

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest (main branch) | Yes |

## Security Practices

This repository follows these security practices:

- **No secrets in code** — API keys, tokens, and credentials are never committed to this repository
- **Dependabot enabled** — Automated vulnerability alerts and security updates are active
- **Private repository** — Access is restricted to authorized collaborators only
- **Minimal permissions** — GitHub Actions workflows use read-only permissions by default
- **`.gitignore` protection** — Environment files, credentials, and sensitive file types are excluded from commits
- **Fine-grained tokens** — All integrations use scoped tokens with minimum required permissions

## Third-Party Integrations

| Service | Authentication | Scope |
|---------|---------------|-------|
| GitHub Actions | Built-in GITHUB_TOKEN | Contents write (for auto-commits only) |
| Airweave | API key (stored in Airweave dashboard) | Read-only sync of repository contents |
| Yahoo Finance | None required | Public financial data via yfinance |
