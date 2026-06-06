"""Performance metrics for portfolio backtests.

The metrics layer converts daily strategy returns into decision-ready measures:
growth, volatility, risk-adjusted return, drawdown survival, turnover, trading
cost drag, benchmark-relative performance, and cross-strategy winners/losers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252
EPSILON = 1e-12


@dataclass(frozen=True)
class StrategyComparison:
    """Summary of best and worst strategies in a comparison table."""

    best_strategy: str
    worst_strategy: str
    ranking_metric: str
    metrics: pd.DataFrame


def cumulative_returns(returns: pd.Series, initial_value: float = 1.0) -> pd.Series:
    """Return a cumulative portfolio value series."""
    clean = _clean_returns(returns)
    if initial_value <= 0:
        raise ValueError("initial_value must be positive.")
    return initial_value * (1.0 + clean).cumprod()


def cumulative_return(returns: pd.Series) -> float:
    """Return total cumulative return over the full backtest."""
    clean = _clean_returns(returns)
    return float((1.0 + clean).prod() - 1.0)


def final_portfolio_value(returns: pd.Series, initial_value: float = 1.0) -> float:
    """Return final portfolio value after applying daily returns."""
    return float(cumulative_returns(returns, initial_value=initial_value).iloc[-1])


def annualized_return(returns: pd.Series, periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """Return geometrically compounded annualized return."""
    clean = _clean_returns(returns)
    _validate_periods(periods)
    years = len(clean) / periods
    total_growth = float((1.0 + clean).prod())
    if years <= 0 or total_growth <= 0:
        return 0.0
    return float(total_growth ** (1.0 / years) - 1.0)


def annualized_volatility(returns: pd.Series, periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """Return annualized standard deviation of daily returns."""
    clean = _clean_returns(returns)
    _validate_periods(periods)
    if len(clean) < 2:
        return 0.0
    return float(clean.std(ddof=1) * np.sqrt(periods))


def sharpe_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Return annualized Sharpe ratio using an annual risk-free rate."""
    clean = _clean_returns(returns)
    _validate_periods(periods)
    excess = clean - risk_free / periods
    volatility = float(excess.std(ddof=1))
    if not np.isfinite(volatility) or volatility <= EPSILON:
        return 0.0
    return float(excess.mean() / volatility * np.sqrt(periods))


def sortino_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Return annualized Sortino ratio using downside deviation."""
    clean = _clean_returns(returns)
    _validate_periods(periods)
    excess = clean - risk_free / periods
    downside = excess[excess < 0]
    downside_deviation = float(downside.std(ddof=1))
    if not np.isfinite(downside_deviation) or downside_deviation <= EPSILON:
        return 0.0
    return float(excess.mean() / downside_deviation * np.sqrt(periods))


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Return drawdown series with the initial capital base as high-water mark."""
    curve = cumulative_returns(returns, initial_value=1.0)
    high_water = curve.cummax().clip(lower=1.0)
    return ((curve - high_water) / high_water).clip(upper=0.0)


def max_drawdown(returns: pd.Series) -> float:
    """Return maximum peak-to-trough drawdown as a negative number."""
    return float(drawdown_series(returns).min())


def calmar_ratio(returns: pd.Series, periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """Return annualized return divided by absolute maximum drawdown."""
    drawdown = abs(max_drawdown(returns))
    if drawdown <= EPSILON:
        return float("inf")
    return float(annualized_return(returns, periods) / drawdown)


def historical_cvar(returns: pd.Series, alpha: float = 0.95) -> float:
    """Return historical conditional value-at-risk as a positive loss value."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    clean = _clean_returns(returns)
    losses = -clean.to_numpy()
    value_at_risk = float(np.quantile(losses, alpha))
    tail_losses = losses[losses >= value_at_risk]
    return float(tail_losses.mean()) if tail_losses.size else value_at_risk


def average_turnover(
    weights_history: pd.DataFrame,
    rebal_freq: int = 21,
    annualize: bool = True,
) -> float:
    """Return average one-way turnover from a rebalancing weight history."""
    _validate_weights_history(weights_history)
    if rebal_freq <= 0:
        raise ValueError("rebal_freq must be positive.")
    if len(weights_history) < 2:
        return 0.0

    diffs = weights_history.diff().abs().sum(axis=1).dropna() / 2.0
    avg_turnover = float(diffs.mean()) if not diffs.empty else 0.0
    return avg_turnover * (TRADING_DAYS_PER_YEAR / rebal_freq) if annualize else avg_turnover


def turnover_series(weights_history: pd.DataFrame) -> pd.Series:
    """Return one-way turnover at each rebalance date."""
    _validate_weights_history(weights_history)
    if len(weights_history) < 2:
        return pd.Series(dtype=float, name="turnover")
    turnover = weights_history.diff().abs().sum(axis=1).dropna() / 2.0
    turnover.name = "turnover"
    return turnover


def total_transaction_cost(transaction_costs: Optional[pd.Series | Sequence[float]]) -> float:
    """Return total transaction-cost drag over a backtest."""
    if transaction_costs is None:
        return 0.0
    costs = pd.Series(transaction_costs, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if costs.empty:
        return 0.0
    if (costs < 0).any():
        raise ValueError("transaction_costs must be non-negative.")
    return float(costs.sum())


def transaction_cost_impact(
    returns: pd.Series,
    transaction_costs: Optional[pd.Series | Sequence[float]] = None,
) -> float:
    """Return cumulative return drag from transaction costs."""
    _clean_returns(returns)
    return total_transaction_cost(transaction_costs)


def information_ratio(
    returns: pd.Series,
    benchmark: pd.Series,
    periods: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Return annualized active return divided by tracking error."""
    active = _align_two_series(returns, benchmark)[0] - _align_two_series(returns, benchmark)[1]
    tracking_error = float(active.std(ddof=1))
    if not np.isfinite(tracking_error) or tracking_error <= EPSILON:
        return 0.0
    return float(active.mean() / tracking_error * np.sqrt(periods))


def beta_alpha(
    returns: pd.Series,
    benchmark: pd.Series,
    risk_free: float = 0.0,
    periods: int = TRADING_DAYS_PER_YEAR,
) -> tuple[float, float]:
    """Return annualized CAPM-style beta and alpha versus a benchmark."""
    strategy, bench = _align_two_series(returns, benchmark)
    if len(strategy) < 2:
        return 0.0, 0.0

    covariance = np.cov(strategy.to_numpy(), bench.to_numpy())
    benchmark_variance = float(covariance[1, 1])
    beta = float(covariance[0, 1] / benchmark_variance) if benchmark_variance > EPSILON else 0.0
    strategy_return = annualized_return(strategy, periods)
    benchmark_return = annualized_return(bench, periods)
    alpha = strategy_return - (risk_free + beta * (benchmark_return - risk_free))
    return beta, float(alpha)


def compute_all_metrics(
    returns: pd.Series,
    weights_history: Optional[pd.DataFrame] = None,
    benchmark: Optional[pd.Series] = None,
    risk_free: float = 0.0,
    periods: int = TRADING_DAYS_PER_YEAR,
    rebal_freq: int = 21,
    transaction_costs: Optional[pd.Series | Sequence[float]] = None,
    initial_value: float = 1.0,
) -> dict[str, float]:
    """Compute a comprehensive metrics dictionary for one strategy."""
    clean = _clean_returns(returns)
    cost_impact = transaction_cost_impact(clean, transaction_costs)

    metrics: dict[str, float] = {
        "cumulative_return": cumulative_return(clean),
        "total_return": cumulative_return(clean),
        "annualized_return": annualized_return(clean, periods),
        "annualized_volatility": annualized_volatility(clean, periods),
        "sharpe_ratio": sharpe_ratio(clean, risk_free, periods),
        "sortino_ratio": sortino_ratio(clean, risk_free, periods),
        "max_drawdown": max_drawdown(clean),
        "calmar_ratio": calmar_ratio(clean, periods),
        "cvar_95": historical_cvar(clean, alpha=0.95),
        "final_portfolio_value": final_portfolio_value(clean, initial_value=initial_value),
        "portfolio_turnover": 0.0,
        "avg_annual_turnover": 0.0,
        "transaction_cost_impact": cost_impact,
        "transaction_cost_total": cost_impact,
        "skewness": float(clean.skew()) if len(clean) > 2 else 0.0,
        "excess_kurtosis": float(clean.kurtosis()) if len(clean) > 3 else 0.0,
    }

    if weights_history is not None and len(weights_history) > 1:
        metrics["portfolio_turnover"] = average_turnover(weights_history, rebal_freq, annualize=False)
        metrics["avg_annual_turnover"] = average_turnover(weights_history, rebal_freq, annualize=True)

    if benchmark is not None:
        strategy_aligned, benchmark_aligned = _align_two_series(clean, benchmark)
        if len(strategy_aligned) >= 2:
            beta, alpha = beta_alpha(strategy_aligned, benchmark_aligned, risk_free, periods)
            metrics["beta"] = beta
            metrics["alpha"] = alpha
            metrics["information_ratio"] = information_ratio(strategy_aligned, benchmark_aligned, periods)
            metrics["benchmark_return"] = annualized_return(benchmark_aligned, periods)
            metrics["active_return"] = (
                annualized_return(strategy_aligned, periods)
                - annualized_return(benchmark_aligned, periods)
            )

    return metrics


def metrics_table(
    strategy_metrics: Mapping[str, float],
    benchmark_metrics: Optional[Mapping[str, float]] = None,
) -> pd.DataFrame:
    """Return a strategy-versus-benchmark metrics table."""
    table = pd.DataFrame({"Strategy": pd.Series(strategy_metrics, dtype=float)})
    if benchmark_metrics is not None:
        table["Benchmark"] = pd.Series(benchmark_metrics, dtype=float)
    return table


def compare_strategies(
    strategy_returns: Mapping[str, pd.Series],
    *,
    weights_history: Optional[Mapping[str, pd.DataFrame]] = None,
    transaction_costs: Optional[Mapping[str, pd.Series | Sequence[float]]] = None,
    benchmark_name: Optional[str] = None,
    risk_free: float = 0.0,
    periods: int = TRADING_DAYS_PER_YEAR,
    rebal_freq: int = 21,
    initial_value: float = 1.0,
    ranking_metric: str = "sharpe_ratio",
) -> StrategyComparison:
    """Compute metrics for multiple strategies and identify best/worst."""
    if not strategy_returns:
        raise ValueError("strategy_returns cannot be empty.")

    benchmark = strategy_returns.get(benchmark_name) if benchmark_name else None
    rows: dict[str, dict[str, float]] = {}

    for name, returns in strategy_returns.items():
        rows[name] = compute_all_metrics(
            returns,
            weights_history=None if weights_history is None else weights_history.get(name),
            benchmark=benchmark if name != benchmark_name else None,
            risk_free=risk_free,
            periods=periods,
            rebal_freq=rebal_freq,
            transaction_costs=None if transaction_costs is None else transaction_costs.get(name),
            initial_value=initial_value,
        )

    metrics = pd.DataFrame.from_dict(rows, orient="index")
    if ranking_metric not in metrics.columns:
        raise KeyError(f"ranking_metric '{ranking_metric}' is not available.")

    ranking = metrics[ranking_metric].replace([np.inf, -np.inf], np.nan).dropna()
    if ranking.empty:
        raise ValueError(f"No finite values available for ranking metric '{ranking_metric}'.")

    best_strategy = str(ranking.idxmax())
    worst_strategy = str(ranking.idxmin())
    return StrategyComparison(
        best_strategy=best_strategy,
        worst_strategy=worst_strategy,
        ranking_metric=ranking_metric,
        metrics=metrics,
    )


def best_worst_strategy(
    metrics: pd.DataFrame,
    ranking_metric: str = "sharpe_ratio",
) -> tuple[str, str]:
    """Return best and worst strategy names from a metrics DataFrame."""
    if ranking_metric not in metrics.columns:
        raise KeyError(f"ranking_metric '{ranking_metric}' is not available.")
    ranking = metrics[ranking_metric].replace([np.inf, -np.inf], np.nan).dropna()
    if ranking.empty:
        raise ValueError(f"No finite values available for ranking metric '{ranking_metric}'.")
    return str(ranking.idxmax()), str(ranking.idxmin())


def format_metrics_for_display(metrics: pd.DataFrame | Mapping[str, float]) -> pd.DataFrame:
    """Return a display-friendly string table for reports and dashboards."""
    table = pd.DataFrame(metrics) if not isinstance(metrics, pd.DataFrame) else metrics.copy()
    percent_rows = {
        "cumulative_return",
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
        "cvar_95",
        "portfolio_turnover",
        "avg_annual_turnover",
        "transaction_cost_impact",
        "transaction_cost_total",
        "alpha",
        "benchmark_return",
        "active_return",
    }

    formatted = table.copy()
    for index_label in formatted.index:
        for column in formatted.columns:
            value = formatted.loc[index_label, column]
            if not isinstance(value, (int, float, np.floating)) or not np.isfinite(value):
                formatted.loc[index_label, column] = str(value)
            elif index_label in percent_rows:
                formatted.loc[index_label, column] = f"{value:.2%}"
            elif index_label == "final_portfolio_value":
                formatted.loc[index_label, column] = f"${value:,.2f}"
            else:
                formatted.loc[index_label, column] = f"{value:.3f}"
    return formatted


def _clean_returns(returns: pd.Series) -> pd.Series:
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns, dtype=float)
    clean = returns.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        raise ValueError("returns cannot be empty.")
    if (clean <= -1.0).any():
        raise ValueError("returns contain values <= -100%, which break compounding.")
    return clean


def _align_two_series(left: pd.Series, right: pd.Series) -> tuple[pd.Series, pd.Series]:
    left_clean = _clean_returns(left)
    right_clean = _clean_returns(right)
    common_index = left_clean.index.intersection(right_clean.index)
    if len(common_index) == 0:
        if len(left_clean) != len(right_clean):
            raise ValueError("Series do not share an index and have different lengths.")
        left_aligned = pd.Series(left_clean.to_numpy(), index=range(len(left_clean)))
        right_aligned = pd.Series(right_clean.to_numpy(), index=range(len(right_clean)))
        return left_aligned, right_aligned
    return left_clean.loc[common_index], right_clean.loc[common_index]


def _validate_weights_history(weights_history: pd.DataFrame) -> None:
    if not isinstance(weights_history, pd.DataFrame):
        raise TypeError("weights_history must be a pandas DataFrame.")
    if weights_history.empty:
        raise ValueError("weights_history cannot be empty.")
    if not np.isfinite(weights_history.to_numpy()).all():
        raise ValueError("weights_history contains non-finite values.")


def _validate_periods(periods: int) -> None:
    if periods <= 0:
        raise ValueError("periods must be positive.")


__all__ = [
    "EPSILON",
    "StrategyComparison",
    "TRADING_DAYS_PER_YEAR",
    "annualized_return",
    "annualized_volatility",
    "average_turnover",
    "best_worst_strategy",
    "beta_alpha",
    "calmar_ratio",
    "compare_strategies",
    "compute_all_metrics",
    "cumulative_return",
    "cumulative_returns",
    "drawdown_series",
    "final_portfolio_value",
    "format_metrics_for_display",
    "historical_cvar",
    "information_ratio",
    "max_drawdown",
    "metrics_table",
    "sharpe_ratio",
    "sortino_ratio",
    "total_transaction_cost",
    "transaction_cost_impact",
    "turnover_series",
]
