"""Walk-forward backtesting framework for portfolio strategies.

The backtester simulates how a portfolio would have behaved if it only used
information available at each rebalance date. It supports the three core
strategies in this project: equal weight, traditional mean-variance, and the
risk-aware optimizer with costs, turnover limits, CVaR, and volatility targeting.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    tqdm = None

from src.backtesting.metrics import (
    StrategyComparison,
    compare_strategies,
    compute_all_metrics,
    drawdown_series,
    metrics_table,
)
from src.data.data_loader import CovarianceEstimator
from src.optimization.optimizer import (
    EqualWeightOptimizer,
    OptimizationConfig,
    OptimizationResult,
    RiskAwareOptimizer,
)


logger = logging.getLogger(__name__)
TRADING_DAYS_PER_YEAR = 252
EPSILON = 1e-12


@dataclass
class BacktestConfig:
    """Configuration for walk-forward portfolio simulations."""

    estimation_window: int = 252
    rebalance_freq: int = 21
    cvar_scenario_window: int = 252
    cov_estimator: str = "ledoit_wolf"
    mu_method: str = "historical"
    mu_halflife: int = 63
    return_method: str = "simple"
    risk_free_rate: float = 0.0
    initial_value: float = 100000.0
    cash_return: float = 0.0

    def __post_init__(self) -> None:
        if self.estimation_window <= 1:
            raise ValueError("estimation_window must be greater than 1.")
        if self.rebalance_freq <= 0:
            raise ValueError("rebalance_freq must be positive.")
        if self.cvar_scenario_window <= 1:
            raise ValueError("cvar_scenario_window must be greater than 1.")
        valid_cov = {"sample", "ledoit_wolf", "ewm", "constant_corr"}
        if self.cov_estimator not in valid_cov:
            raise ValueError(f"cov_estimator must be one of {sorted(valid_cov)}.")
        if self.mu_method not in {"historical", "ewm"}:
            raise ValueError("mu_method must be 'historical' or 'ewm'.")
        if self.mu_halflife <= 0:
            raise ValueError("mu_halflife must be positive.")
        if self.initial_value <= 0:
            raise ValueError("initial_value must be positive.")


@dataclass
class BacktestResult:
    """Complete output of one strategy backtest."""

    portfolio_returns: pd.Series
    weights_history: pd.DataFrame
    optimization_results: list[OptimizationResult]
    benchmark_returns: pd.Series
    tickers: list[str]
    rebal_freq: int = 21
    strategy_name: str = "Risk-Aware Optimizer"
    transaction_costs: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    initial_value: float = 100000.0

    @property
    def cumulative_returns(self) -> pd.Series:
        """Cumulative portfolio value index starting at 1.0."""
        return (1.0 + self.portfolio_returns).cumprod()

    @property
    def benchmark_cumulative(self) -> pd.Series:
        """Cumulative equal-weight benchmark value index starting at 1.0."""
        return (1.0 + self.benchmark_returns).cumprod()

    @property
    def drawdown_series(self) -> pd.Series:
        """Portfolio drawdown series."""
        return drawdown_series(self.portfolio_returns)

    @property
    def turnover_by_rebalance(self) -> pd.Series:
        """One-way turnover recorded at each rebalance date."""
        if not self.optimization_results or self.weights_history.empty:
            return pd.Series(dtype=float, name="turnover")
        values = [result.turnover for result in self.optimization_results[: len(self.weights_history)]]
        return pd.Series(values, index=self.weights_history.index[: len(values)], name="turnover")

    def summary(self, risk_free: float = 0.0) -> dict[str, float]:
        """Return a flat dictionary of strategy metrics."""
        return compute_all_metrics(
            self.portfolio_returns,
            weights_history=self.weights_history,
            benchmark=self.benchmark_returns,
            risk_free=risk_free,
            rebal_freq=self.rebal_freq,
            transaction_costs=self.transaction_costs,
            initial_value=self.initial_value,
        )

    def metrics(self, risk_free: float = 0.0) -> pd.DataFrame:
        """Return a strategy-versus-equal-weight metrics table."""
        strategy_metrics = self.summary(risk_free=risk_free)
        benchmark_metrics = compute_all_metrics(
            self.benchmark_returns,
            risk_free=risk_free,
            initial_value=self.initial_value,
        )
        return metrics_table(strategy_metrics, benchmark_metrics)


@dataclass
class MultiStrategyBacktestResult:
    """Container for equal-weight, MVO, and risk-aware backtest results."""

    results: Mapping[str, BacktestResult]
    comparison: StrategyComparison

    @property
    def metrics(self) -> pd.DataFrame:
        """Metrics table with one row per strategy."""
        return self.comparison.metrics

    @property
    def best_strategy(self) -> str:
        """Best strategy according to the comparison ranking metric."""
        return self.comparison.best_strategy

    @property
    def worst_strategy(self) -> str:
        """Worst strategy according to the comparison ranking metric."""
        return self.comparison.worst_strategy


class WalkForwardBacktester:
    """Walk-forward backtester for optimizer-driven strategies."""

    def __init__(
        self,
        opt_config: Optional[OptimizationConfig] = None,
        bt_config: Optional[BacktestConfig] = None,
        strategy_name: str = "Risk-Aware Optimizer",
    ) -> None:
        self.opt_config = opt_config or OptimizationConfig()
        self.bt_config = bt_config or BacktestConfig()
        self.strategy_name = strategy_name
        self.optimizer = RiskAwareOptimizer(self.opt_config)
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(
        self,
        returns: pd.DataFrame,
        sector_indices: Optional[np.ndarray] = None,
        show_progress: bool = True,
    ) -> BacktestResult:
        """Execute a walk-forward backtest on a return matrix."""
        clean_returns = validate_returns_frame(returns)
        if len(clean_returns) <= self.bt_config.estimation_window:
            raise ValueError("returns length must exceed estimation_window.")

        n_assets = clean_returns.shape[1]
        tickers = list(clean_returns.columns)
        sectors = _validate_sector_indices(sector_indices, n_assets) if sector_indices is not None else None

        current_weights = np.ones(n_assets, dtype=float) / n_assets
        start_idx = self.bt_config.estimation_window
        portfolio_returns: dict[pd.Timestamp, float] = {}
        transaction_costs: dict[pd.Timestamp, float] = {}
        weights_history: dict[pd.Timestamp, np.ndarray] = {}
        optimization_results: list[OptimizationResult] = []
        last_result: Optional[OptimizationResult] = None

        iterator = range(start_idx, len(clean_returns))
        if show_progress and tqdm is not None:
            iterator = tqdm(iterator, desc=f"Backtesting {self.strategy_name}", unit="day")

        for row_idx in iterator:
            date = pd.Timestamp(clean_returns.index[row_idx])
            day_returns = clean_returns.iloc[row_idx].to_numpy(dtype=float)
            is_rebalance = (row_idx - start_idx) % self.bt_config.rebalance_freq == 0

            if is_rebalance:
                history = clean_returns.iloc[row_idx - self.bt_config.estimation_window : row_idx]
                result = self._rebalance(history, current_weights, sectors)
                optimization_results.append(result)
                last_result = result
                if result.feasible:
                    current_weights = result.weights.copy()
                weights_history[date] = current_weights.copy()

            cost = last_result.transaction_cost if is_rebalance and last_result is not None else 0.0
            day_return = self._portfolio_day_return(current_weights, day_returns) - cost
            portfolio_returns[date] = day_return
            transaction_costs[date] = cost
            current_weights = self._drift_weights(current_weights, day_returns, self.bt_config.cash_return)

        portfolio_series = pd.Series(portfolio_returns, name=self.strategy_name)
        cost_series = pd.Series(transaction_costs, name="transaction_cost")
        weights_df = pd.DataFrame.from_dict(weights_history, orient="index", columns=tickers)
        benchmark = equal_weight_returns(clean_returns.iloc[start_idx:], name="Equal Weight")

        return BacktestResult(
            portfolio_returns=portfolio_series,
            weights_history=weights_df,
            optimization_results=optimization_results,
            benchmark_returns=benchmark,
            tickers=tickers,
            rebal_freq=self.bt_config.rebalance_freq,
            strategy_name=self.strategy_name,
            transaction_costs=cost_series,
            initial_value=self.bt_config.initial_value,
        )

    def _rebalance(
        self,
        history: pd.DataFrame,
        current_weights: np.ndarray,
        sector_indices: Optional[np.ndarray],
    ) -> OptimizationResult:
        covariance = self._estimate_covariance(history)
        expected_returns = self._estimate_expected_returns(history)
        scenarios = self._scenario_returns(history) if self.opt_config.cvar_limit is not None else None
        return self.optimizer.optimize(
            expected_returns=expected_returns,
            covariance=covariance,
            current_weights=current_weights,
            scenario_returns=scenarios,
            sector_indices=sector_indices,
        )

    def _estimate_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        estimator = self.bt_config.cov_estimator
        values = returns.to_numpy(dtype=float)
        if estimator == "ledoit_wolf":
            return CovarianceEstimator.ledoit_wolf(values)
        if estimator == "ewm":
            return CovarianceEstimator.ewm(values)
        if estimator == "constant_corr":
            return CovarianceEstimator.constant_corr(values)
        return CovarianceEstimator.sample(values)

    def _estimate_expected_returns(self, history: pd.DataFrame) -> np.ndarray:
        if self.bt_config.mu_method == "ewm":
            daily_mu = history.ewm(halflife=self.bt_config.mu_halflife, adjust=False).mean().iloc[-1]
        else:
            daily_mu = history.mean()
        return daily_mu.to_numpy(dtype=float) * self.opt_config.periods_per_year

    def _scenario_returns(self, history: pd.DataFrame) -> np.ndarray:
        window = min(self.bt_config.cvar_scenario_window, len(history))
        return history.iloc[-window:].to_numpy(dtype=float)

    def _portfolio_day_return(self, weights: np.ndarray, day_returns: np.ndarray) -> float:
        risky_return = float(weights @ day_returns)
        cash_weight = 1.0 - float(np.sum(weights))
        return risky_return + cash_weight * (self.bt_config.cash_return / TRADING_DAYS_PER_YEAR)

    @staticmethod
    def _drift_weights(
        weights: np.ndarray,
        day_returns: np.ndarray,
        cash_return: float = 0.0,
    ) -> np.ndarray:
        """Drift risky weights while preserving any cash allocation."""
        weights_arr = np.asarray(weights, dtype=float)
        day_returns_arr = np.asarray(day_returns, dtype=float)
        cash_weight = 1.0 - float(np.sum(weights_arr))
        risky_values = weights_arr * (1.0 + day_returns_arr)
        cash_value = cash_weight * (1.0 + cash_return / TRADING_DAYS_PER_YEAR)
        total_value = float(risky_values.sum() + cash_value)
        if abs(total_value) <= EPSILON:
            return weights_arr
        return risky_values / total_value


class MVOBacktester(WalkForwardBacktester):
    """Traditional mean-variance baseline with no realistic frictions."""

    def __init__(
        self,
        risk_aversion: float = 4.0,
        bt_config: Optional[BacktestConfig] = None,
        max_weight: float = 1.0,
    ) -> None:
        opt_config = OptimizationConfig(
            risk_aversion=risk_aversion,
            linear_cost=0.0,
            market_impact_coef=0.0,
            use_market_impact=False,
            min_weight=0.0,
            max_weight=max_weight,
            max_turnover=None,
            target_volatility=None,
            cvar_limit=None,
            sector_caps=None,
        )
        super().__init__(
            opt_config=opt_config,
            bt_config=bt_config,
            strategy_name="Mean-Variance Optimization",
        )


class EqualWeightBacktester:
    """Equal-weight baseline backtester."""

    def __init__(self, bt_config: Optional[BacktestConfig] = None) -> None:
        self.bt_config = bt_config or BacktestConfig()
        self.strategy_name = "Equal Weight"

    def run(
        self,
        returns: pd.DataFrame,
        sector_indices: Optional[np.ndarray] = None,
        show_progress: bool = False,
    ) -> BacktestResult:
        """Run a monthly-rebalanced equal-weight baseline."""
        del sector_indices, show_progress
        clean_returns = validate_returns_frame(returns)
        if len(clean_returns) <= self.bt_config.estimation_window:
            raise ValueError("returns length must exceed estimation_window.")

        n_assets = clean_returns.shape[1]
        tickers = list(clean_returns.columns)
        start_idx = self.bt_config.estimation_window
        weights = np.ones(n_assets, dtype=float) / n_assets

        test_returns = clean_returns.iloc[start_idx:]
        portfolio_returns = equal_weight_returns(test_returns, name=self.strategy_name)
        benchmark_returns = portfolio_returns.copy()

        weights_history: dict[pd.Timestamp, np.ndarray] = {}
        optimization_results: list[OptimizationResult] = []
        for offset, date in enumerate(test_returns.index):
            if offset % self.bt_config.rebalance_freq == 0:
                timestamp = pd.Timestamp(date)
                weights_history[timestamp] = weights.copy()
                history = clean_returns.iloc[max(0, start_idx + offset - self.bt_config.estimation_window) : start_idx + offset]
                covariance = CovarianceEstimator.sample(history.to_numpy(dtype=float))
                expected_returns = history.mean().to_numpy(dtype=float) * TRADING_DAYS_PER_YEAR
                optimization_results.append(EqualWeightOptimizer.optimize(n_assets, expected_returns, covariance))

        weights_df = pd.DataFrame.from_dict(weights_history, orient="index", columns=tickers)
        zero_costs = pd.Series(0.0, index=portfolio_returns.index, name="transaction_cost")
        return BacktestResult(
            portfolio_returns=portfolio_returns,
            weights_history=weights_df,
            optimization_results=optimization_results,
            benchmark_returns=benchmark_returns,
            tickers=tickers,
            rebal_freq=self.bt_config.rebalance_freq,
            strategy_name=self.strategy_name,
            transaction_costs=zero_costs,
            initial_value=self.bt_config.initial_value,
        )


def run_three_strategy_backtest(
    returns: pd.DataFrame,
    *,
    risk_aware_config: Optional[OptimizationConfig] = None,
    bt_config: Optional[BacktestConfig] = None,
    sector_indices: Optional[np.ndarray] = None,
    show_progress: bool = True,
) -> MultiStrategyBacktestResult:
    """Run equal-weight, MVO, and risk-aware strategies on the same data."""
    bt_cfg = bt_config or BacktestConfig()
    risk_cfg = risk_aware_config or OptimizationConfig()

    equal_weight = EqualWeightBacktester(bt_cfg).run(returns, show_progress=False)
    mvo = MVOBacktester(risk_aversion=risk_cfg.risk_aversion, bt_config=bt_cfg).run(
        returns,
        sector_indices=sector_indices,
        show_progress=False,
    )
    risk_aware = WalkForwardBacktester(risk_cfg, bt_cfg, strategy_name="Risk-Aware Optimizer").run(
        returns,
        sector_indices=sector_indices,
        show_progress=show_progress,
    )

    results = {
        "Equal Weight": equal_weight,
        "Mean-Variance Optimization": mvo,
        "Risk-Aware Optimizer": risk_aware,
    }
    comparison = compare_strategies(
        {name: result.portfolio_returns for name, result in results.items()},
        weights_history={name: result.weights_history for name, result in results.items()},
        transaction_costs={name: result.transaction_costs for name, result in results.items()},
        benchmark_name="Equal Weight",
        risk_free=bt_cfg.risk_free_rate,
        rebal_freq=bt_cfg.rebalance_freq,
        initial_value=bt_cfg.initial_value,
        ranking_metric="sharpe_ratio",
    )
    return MultiStrategyBacktestResult(results=results, comparison=comparison)


def equal_weight_returns(returns: pd.DataFrame, name: str = "Equal Weight") -> pd.Series:
    """Return daily equal-weight portfolio returns."""
    clean_returns = validate_returns_frame(returns)
    series = clean_returns.mean(axis=1)
    series.name = name
    return series


def validate_returns_frame(returns: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean a return matrix for backtesting."""
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")
    if returns.empty:
        raise ValueError("returns cannot be empty.")
    if returns.shape[1] < 2:
        raise ValueError("at least two assets are required for backtesting.")

    clean = returns.copy()
    clean.index = pd.to_datetime(clean.index)
    clean = clean.sort_index()
    clean = clean.loc[~clean.index.duplicated(keep="last")]
    clean = clean.apply(pd.to_numeric, errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna(how="any")

    if clean.empty:
        raise ValueError("returns are empty after cleaning.")
    if (clean <= -1.0).any().any():
        raise ValueError("returns contain values <= -100%, which break compounding.")
    return clean


def _validate_sector_indices(sector_indices: np.ndarray, n_assets: int) -> np.ndarray:
    sectors = np.asarray(sector_indices)
    if sectors.ndim != 1 or sectors.size != n_assets:
        raise ValueError("sector_indices must be one-dimensional and match asset count.")
    return sectors.astype(int)


def main() -> None:
    """Command-line entry point for a default project backtest."""
    from src.config import load_config
    from src.data.data_loader import DataLoader

    parser = argparse.ArgumentParser(description="Run the Risk-Aware Portfolio Optimizer backtest.")
    parser.add_argument("--config", default="config/default_config.yaml", help="Path to YAML config.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars.")
    args = parser.parse_args()

    config = load_config(args.config)
    config.setup_logging()
    config.ensure_directories()

    loader = DataLoader.from_config(config)
    bundle = loader.load_from_config(config, save_processed=True)
    sector_indices, _, _ = loader.get_sector_info(bundle.tickers)

    result = run_three_strategy_backtest(
        bundle.returns,
        risk_aware_config=config.to_optimizer_config("risk_aware"),
        bt_config=config.to_backtest_config(),
        sector_indices=sector_indices,
        show_progress=not args.no_progress,
    )

    print("\nPerformance metrics")
    print(result.metrics.round(4).to_string())
    print(f"\nBest strategy: {result.best_strategy}")
    print(f"Worst strategy: {result.worst_strategy}")


if __name__ == "__main__":
    main()


__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "EqualWeightBacktester",
    "MVOBacktester",
    "MultiStrategyBacktestResult",
    "WalkForwardBacktester",
    "equal_weight_returns",
    "run_three_strategy_backtest",
    "validate_returns_frame",
]
