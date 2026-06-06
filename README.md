# Risk-Aware Portfolio Optimizer

> "Returns matter, but survival matters more."

A production-style Python project for portfolio construction, risk management,
and walk-forward backtesting. The optimizer extends classical mean-variance
portfolio selection with transaction costs, turnover limits, position caps,
sector caps, CVaR controls, volatility targeting, and benchmark comparison.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-unittest-brightgreen.svg)](tests/)

## Why This Project Matters

Classical Markowitz optimization can look excellent in a notebook and fail in
live trading because it ignores implementation constraints. This project makes
those constraints explicit:

- Transaction costs reduce excessive rebalancing.
- Turnover limits keep trades realistic.
- Position and sector caps reduce concentration risk.
- CVaR controls target tail losses instead of only variance.
- Volatility targeting can scale risky exposure into cash.
- Walk-forward backtests use only information available at each rebalance.

The result is a portfolio optimizer built around the idea that risk-adjusted
returns are not enough if the strategy cannot survive drawdowns, trading costs,
and changing market regimes.

## Features

| Area | Implementation |
| --- | --- |
| Optimization | Equal weight, mean-variance, and risk-aware portfolio solvers |
| Solver path | CVXPY first when installed, SciPy SLSQP fallback otherwise |
| Costs | Linear trading cost plus optional quadratic market impact |
| Risk controls | CVaR, volatility targeting, drawdown analytics, sector caps |
| Backtesting | Walk-forward simulation with weight drift and rebalance costs |
| Data | Yahoo Finance loader with cache and deterministic synthetic fallback |
| Covariance | Sample, Ledoit-Wolf, exponentially weighted, constant correlation |
| Reporting | Matplotlib charts saved to `images/` |
| Dashboard | Streamlit app with Plotly diagnostics and parameter controls |
| Tests | Standard-library `unittest` suite, no pytest required |

## Project Structure

```text
risk-aware-portfolio-optimizer/
+-- config/
|   +-- default_config.yaml
+-- dashboard/
|   +-- app.py
+-- src/
|   +-- backtesting/
|   |   +-- backtester.py
|   |   +-- metrics.py
|   +-- data/
|   |   +-- data_loader.py
|   +-- optimization/
|   |   +-- optimizer.py
|   +-- reporting/
|   |   +-- reporter.py
|   +-- risk/
|   |   +-- risk_models.py
|   +-- config.py
|   +-- preprocessing.py
+-- tests/
|   +-- test_backtester.py
|   +-- test_optimizer.py
|   +-- test_risk_models.py
+-- requirements.txt
+-- setup.py
+-- README.md
```

## Pipeline

```text
Price data or synthetic data
        |
        v
DataLoader -> clean returns -> covariance and expected return estimates
        |
        v
RiskAwareOptimizer / MeanVarianceOptimizer / EqualWeightOptimizer
        |
        v
WalkForwardBacktester
        |
        v
Metrics, charts, dashboard, and resume-ready project evidence
```

## Optimization Formulation

At each rebalance, the risk-aware optimizer maximizes expected return net of
risk and trading costs:

```text
maximize    mu.T @ w - (lambda / 2) * w.T @ Sigma @ w - TC(w, w_prev)
subject to  sum(w) <= 1 after optional volatility scaling
            min_weight <= w_i <= max_weight
            one_way_turnover(w, w_prev) <= max_turnover
            CVaR_alpha(-R @ w) <= cvar_limit
            sector_weight_s <= sector_cap_s
```

Transaction cost model:

```text
TC(delta_w) = linear_cost * ||delta_w||_1
              + market_impact_coef * ||delta_w||_2^2
```

CVaR is evaluated from historical scenario returns, and volatility targeting is
implemented as post-solve exposure scaling so the unused allocation behaves like
cash.

## Installation

```bash
git clone https://github.com/yourusername/risk-aware-portfolio-optimizer.git
cd risk-aware-portfolio-optimizer

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

The project can still run core tests in a lighter environment that has NumPy,
Pandas, and SciPy installed. CVXPY, Streamlit, Plotly, and yfinance unlock the
full solver, dashboard, and live-data experience.

## Quickstart

Run the default backtest:

```bash
python3 -m src.backtesting.backtester --config config/default_config.yaml --no-progress
```

Run the dashboard:

```bash
streamlit run dashboard/app.py
```

Run the tests:

```bash
python3 -m unittest discover tests -v
```

## Python API Example

```python
from src.backtesting.backtester import BacktestConfig, run_three_strategy_backtest
from src.data.data_loader import DataLoader
from src.optimization.optimizer import OptimizationConfig
from src.reporting.reporter import PortfolioReporter

loader = DataLoader(fallback_to_synthetic=True)
prices = loader.fetch_prices(
    ["SPY", "QQQ", "IWM", "EFA", "TLT", "GLD"],
    start_date="2018-01-01",
    end_date="2025-12-31",
)
returns = loader.compute_returns(prices)
sector_indices, _, _ = loader.get_sector_info(prices.columns)

opt_config = OptimizationConfig(
    risk_aversion=1.5,
    max_weight=0.30,
    max_turnover=0.35,
    target_volatility=0.12,
    cvar_limit=0.03,
    linear_cost=0.0005,
    market_impact_coef=0.05,
    use_market_impact=True,
)
bt_config = BacktestConfig(
    estimation_window=252,
    rebalance_freq=21,
    cov_estimator="ledoit_wolf",
)

result = run_three_strategy_backtest(
    returns,
    risk_aware_config=opt_config,
    bt_config=bt_config,
    sector_indices=sector_indices,
    show_progress=False,
)

print(result.metrics.round(4))
PortfolioReporter(output_dir="images").generate_all_charts(result)
```

## Required Charts

`PortfolioReporter.generate_all_charts(...)` saves six required charts:

- `images/cumulative_returns.png`
- `images/portfolio_weights.png`
- `images/drawdown.png`
- `images/volatility_comparison.png`
- `images/risk_return_comparison.png`
- `images/turnover_transaction_costs.png`

These files are generated artifacts and can be recreated from the API example or
from a project script that calls the reporter after a backtest.

## Configuration

Edit `config/default_config.yaml` to change the asset universe, date range,
data paths, risk limits, optimizer parameters, backtest settings, and reporting
filenames without changing source code.

Important defaults:

```yaml
data:
  tickers: ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "VNQ"]
  fallback_to_synthetic: true

optimization:
  risk_aware:
    max_weight: 0.25
    max_turnover: 0.35
    target_volatility: 0.12
    cvar_limit: 0.025

backtest:
  estimation_window: 252
  rebalance_freq: 21
  cov_estimator: "ledoit_wolf"
```

## Testing

The project intentionally supports a plain standard-library test command:

```bash
python3 -m unittest discover tests -v
```

Test coverage includes:

- Optimizer constraints, fallback behavior, efficient frontier, and cost helper.
- CVaR, drawdown, volatility targeting, risk budget, and covariance models.
- Walk-forward simulation, strategy comparison, validation, and metrics.

## Dashboard

The Streamlit dashboard provides an interactive demo for:

- Asset universe and date selection.
- Risk aversion, max weight, turnover, CVaR, and target volatility controls.
- Equal-weight, mean-variance, and risk-aware strategy comparison.
- Cumulative returns, drawdowns, rolling volatility, weights, turnover, costs,
  and latest risk budget.

If live Yahoo Finance access is unavailable, the dashboard falls back to
deterministic synthetic ETF-style prices so the demo remains usable offline.

## Employer-Facing Skills Demonstrated

| Skill | Evidence in repo |
| --- | --- |
| Portfolio optimization | `RiskAwareOptimizer`, `MeanVarianceOptimizer`, efficient frontier |
| Convex optimization | CVXPY path with SciPy fallback and constraint validation |
| Risk management | CVaR, volatility targeting, drawdown scale, risk contribution |
| Backtesting | Walk-forward engine with rebalance timing and transaction costs |
| Data engineering | Cached market data loader plus offline synthetic fallback |
| Statistical modeling | Shrinkage covariance, EWM covariance, constant correlation model |
| Software engineering | Modular source layout, typed configs, CLI, tests, dashboard |
| Communication | README, charts, dashboard, and resume materials |

## Design Notes

- The optimizer uses CVXPY when available and SciPy SLSQP as a fallback. This
  keeps the repository runnable in lightweight environments.
- CVaR can be enforced as a hard constraint when volatility targeting is off.
  When volatility targeting is on, CVaR is estimated after scaling and can
  trigger additional exposure reduction.
- Synthetic data is not a performance claim. It is an offline demo path that
  lets reviewers run the system without network access.
- Generated data, cache files, charts, and reports are intentionally ignored by
  version control unless explicitly added as demo artifacts.

## References

- Harry Markowitz, "Portfolio Selection", Journal of Finance, 1952.
- Rockafellar and Uryasev, "Optimization of Conditional Value-at-Risk", Journal
  of Risk, 2000.
- Ledoit and Wolf, "A well-conditioned estimator for large-dimensional
  covariance matrices", Journal of Multivariate Analysis, 2004.
- Boyd et al., "Markowitz Portfolio Construction at Seventy", Stanford, 2024.

---

## License

MIT License. See [LICENSE](LICENSE).

## Author

Thuc Cao
