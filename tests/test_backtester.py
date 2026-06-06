"""Tests for walk-forward backtesting and performance metrics.

The tests avoid external data and use only ``unittest`` so the project can be
verified in a minimal Python environment.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.backtesting.backtester import (
    BacktestConfig,
    BacktestResult,
    EqualWeightBacktester,
    MVOBacktester,
    WalkForwardBacktester,
    equal_weight_returns,
    run_three_strategy_backtest,
    validate_returns_frame,
)
from src.backtesting.metrics import (
    annualized_return,
    annualized_volatility,
    compute_all_metrics,
    cumulative_returns,
    max_drawdown,
    sharpe_ratio,
)
from src.optimization.optimizer import EqualWeightOptimizer, OptimizationConfig


N_ASSETS = 6
N_DAYS = 180


def make_returns(
    n_days: int = N_DAYS,
    n_assets: int = N_ASSETS,
    *,
    seed: int = 31,
) -> pd.DataFrame:
    """Create deterministic correlated daily returns."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    market = rng.normal(0.00035, 0.008, size=(n_days, 1))
    idiosyncratic = rng.normal(0.00010, 0.009, size=(n_days, n_assets))
    loadings = np.linspace(0.45, 0.95, n_assets)
    returns = market @ loadings.reshape(1, -1) + idiosyncratic
    columns = [f"ETF_{idx + 1}" for idx in range(n_assets)]
    return pd.DataFrame(returns, index=dates, columns=columns)


def fast_backtest_config() -> BacktestConfig:
    """Small but realistic config for fast unit tests."""
    return BacktestConfig(
        estimation_window=50,
        rebalance_freq=20,
        cvar_scenario_window=50,
        cov_estimator="sample",
        initial_value=100000.0,
    )


def fast_optimizer_config() -> OptimizationConfig:
    """Risk-aware config that stays feasible for the synthetic universe."""
    return OptimizationConfig(
        risk_aversion=1.2,
        max_weight=0.45,
        max_turnover=0.35,
        target_volatility=None,
        cvar_limit=None,
        linear_cost=0.0005,
        market_impact_coef=0.02,
        use_market_impact=True,
    )


def make_result() -> BacktestResult:
    """Run one deterministic risk-aware backtest for property tests."""
    return WalkForwardBacktester(fast_optimizer_config(), fast_backtest_config()).run(
        make_returns(),
        show_progress=False,
    )


class BacktestResultTests(unittest.TestCase):
    """BacktestResult convenience properties and summaries."""

    def test_cumulative_returns_start_after_first_daily_return(self) -> None:
        result = make_result()
        expected_first_value = 1.0 + result.portfolio_returns.iloc[0]

        np.testing.assert_allclose(result.cumulative_returns.iloc[0], expected_first_value, rtol=1e-12)

    def test_drawdown_series_is_nonpositive(self) -> None:
        result = make_result()

        self.assertTrue((result.drawdown_series <= 1e-12).all())

    def test_benchmark_cumulative_matches_portfolio_index(self) -> None:
        result = make_result()

        pd.testing.assert_index_equal(result.cumulative_returns.index, result.benchmark_cumulative.index)

    def test_turnover_by_rebalance_matches_weight_history(self) -> None:
        result = make_result()
        turnover = result.turnover_by_rebalance

        self.assertEqual(len(turnover), len(result.weights_history))
        self.assertTrue((turnover >= 0.0).all())

    def test_summary_contains_core_metrics(self) -> None:
        result = make_result()
        summary = result.summary()

        for key in [
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "cvar_95",
            "transaction_cost_total",
        ]:
            self.assertIn(key, summary)

    def test_metrics_table_includes_strategy_and_benchmark(self) -> None:
        table = make_result().metrics()

        self.assertIn("Strategy", table.columns)
        self.assertIn("Benchmark", table.columns)


class WalkForwardBacktesterTests(unittest.TestCase):
    """Walk-forward simulation behavior."""

    def setUp(self) -> None:
        self.returns = make_returns()
        self.bt_config = fast_backtest_config()
        self.opt_config = fast_optimizer_config()

    def test_backtest_return_series_has_expected_length(self) -> None:
        result = WalkForwardBacktester(self.opt_config, self.bt_config).run(self.returns, show_progress=False)

        self.assertEqual(len(result.portfolio_returns), len(self.returns) - self.bt_config.estimation_window)

    def test_portfolio_benchmark_and_costs_share_index(self) -> None:
        result = WalkForwardBacktester(self.opt_config, self.bt_config).run(self.returns, show_progress=False)

        pd.testing.assert_index_equal(result.portfolio_returns.index, result.benchmark_returns.index)
        pd.testing.assert_index_equal(result.portfolio_returns.index, result.transaction_costs.index)

    def test_returns_weights_and_costs_are_finite(self) -> None:
        result = WalkForwardBacktester(self.opt_config, self.bt_config).run(self.returns, show_progress=False)

        self.assertTrue(np.isfinite(result.portfolio_returns.to_numpy()).all())
        self.assertTrue(np.isfinite(result.weights_history.to_numpy()).all())
        self.assertTrue(np.isfinite(result.transaction_costs.to_numpy()).all())
        self.assertTrue((result.transaction_costs >= 0.0).all())

    def test_weights_history_uses_asset_columns(self) -> None:
        result = WalkForwardBacktester(self.opt_config, self.bt_config).run(self.returns, show_progress=False)

        self.assertEqual(list(result.weights_history.columns), list(self.returns.columns))

    def test_weight_sums_stay_within_investment_budget(self) -> None:
        config = OptimizationConfig(
            risk_aversion=1.0,
            max_weight=0.45,
            max_turnover=0.35,
            target_volatility=0.08,
            cvar_limit=None,
        )
        result = WalkForwardBacktester(config, self.bt_config).run(self.returns, show_progress=False)
        weight_sums = result.weights_history.sum(axis=1)

        self.assertTrue((weight_sums <= 1.0 + 1e-8).all())
        self.assertTrue((weight_sums > 0.0).all())

    def test_sector_indices_must_match_asset_count(self) -> None:
        with self.assertRaises(ValueError):
            WalkForwardBacktester(self.opt_config, self.bt_config).run(
                self.returns,
                sector_indices=np.array([0, 1]),
                show_progress=False,
            )

    def test_short_history_raises_clear_error(self) -> None:
        short_returns = self.returns.iloc[: self.bt_config.estimation_window]

        with self.assertRaises(ValueError):
            WalkForwardBacktester(self.opt_config, self.bt_config).run(short_returns, show_progress=False)

    def test_weight_drift_preserves_cash_budget(self) -> None:
        weights = np.array([0.25, 0.25, 0.20])
        day_returns = np.array([0.02, -0.01, 0.00])

        drifted = WalkForwardBacktester._drift_weights(weights, day_returns, cash_return=0.02)

        self.assertLessEqual(drifted.sum(), 1.0 + 1e-12)
        self.assertTrue(np.isfinite(drifted).all())


class BaselineBacktesterTests(unittest.TestCase):
    """Equal-weight, MVO, and combined strategy runs."""

    def setUp(self) -> None:
        self.returns = make_returns(seed=41)
        self.bt_config = fast_backtest_config()

    def test_equal_weight_backtester_matches_manual_equal_weight_returns(self) -> None:
        result = EqualWeightBacktester(self.bt_config).run(self.returns, show_progress=False)
        expected = equal_weight_returns(self.returns.iloc[self.bt_config.estimation_window :])

        pd.testing.assert_series_equal(
            result.portfolio_returns,
            expected.rename("Equal Weight"),
            check_names=True,
        )
        self.assertEqual(result.strategy_name, "Equal Weight")

    def test_mvo_backtester_produces_valid_result(self) -> None:
        result = MVOBacktester(risk_aversion=1.0, bt_config=self.bt_config, max_weight=0.60).run(
            self.returns,
            show_progress=False,
        )

        self.assertEqual(result.strategy_name, "Mean-Variance Optimization")
        self.assertGreater(len(result.portfolio_returns), 0)
        self.assertTrue(np.isfinite(result.portfolio_returns.to_numpy()).all())

    def test_three_strategy_backtest_returns_comparison_table(self) -> None:
        result = run_three_strategy_backtest(
            self.returns,
            risk_aware_config=fast_optimizer_config(),
            bt_config=self.bt_config,
            show_progress=False,
        )

        self.assertEqual(
            set(result.results),
            {"Equal Weight", "Mean-Variance Optimization", "Risk-Aware Optimizer"},
        )
        self.assertEqual(set(result.metrics.index), set(result.results))
        self.assertIn(result.best_strategy, result.results)
        self.assertIn(result.worst_strategy, result.results)


class ValidationTests(unittest.TestCase):
    """Input validation and config validation."""

    def test_validate_returns_sorts_and_deduplicates_index(self) -> None:
        dates = [pd.Timestamp("2022-01-05"), pd.Timestamp("2022-01-03"), pd.Timestamp("2022-01-03")]
        returns = pd.DataFrame(
            [[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]],
            index=dates,
            columns=["A", "B"],
        )

        clean = validate_returns_frame(returns)

        self.assertTrue(clean.index.is_monotonic_increasing)
        self.assertEqual(len(clean), 2)
        np.testing.assert_allclose(clean.iloc[0].to_numpy(), np.array([0.05, 0.06]))

    def test_validate_returns_rejects_single_asset(self) -> None:
        with self.assertRaises(ValueError):
            validate_returns_frame(pd.DataFrame({"A": [0.01, 0.02]}))

    def test_validate_returns_rejects_loss_below_negative_one_hundred_percent(self) -> None:
        returns = pd.DataFrame({"A": [0.01, -1.0], "B": [0.02, 0.03]})

        with self.assertRaises(ValueError):
            validate_returns_frame(returns)

    def test_backtest_config_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            BacktestConfig(estimation_window=1)
        with self.assertRaises(ValueError):
            BacktestConfig(rebalance_freq=0)
        with self.assertRaises(ValueError):
            BacktestConfig(cov_estimator="bad_estimator")


class MetricsTests(unittest.TestCase):
    """Performance metric helpers."""

    def test_cumulative_returns_and_annualized_return_for_positive_series(self) -> None:
        returns = pd.Series([0.001] * 252)

        self.assertGreater(annualized_return(returns), 0.0)
        self.assertGreater(cumulative_returns(returns).iloc[-1], 1.0)

    def test_annualized_volatility_and_sharpe_are_finite(self) -> None:
        returns = pd.Series(np.random.default_rng(51).normal(0.0005, 0.01, 400))

        self.assertGreater(annualized_volatility(returns), 0.0)
        self.assertTrue(np.isfinite(sharpe_ratio(returns)))

    def test_max_drawdown_is_nonpositive(self) -> None:
        returns = pd.Series([0.03, -0.10, 0.02, -0.05, 0.04])

        self.assertLessEqual(max_drawdown(returns), 0.0)

    def test_compute_all_metrics_with_benchmark_includes_relative_fields(self) -> None:
        rng = np.random.default_rng(61)
        returns = pd.Series(rng.normal(0.0006, 0.010, 300))
        benchmark = pd.Series(rng.normal(0.0004, 0.009, 300))
        weights = pd.DataFrame(
            [
                EqualWeightOptimizer.optimize(4).weights,
                np.array([0.30, 0.25, 0.25, 0.20]),
            ],
            columns=["A", "B", "C", "D"],
        )

        metrics = compute_all_metrics(returns, weights_history=weights, benchmark=benchmark, transaction_costs=[0.001])

        for key in ["beta", "alpha", "information_ratio", "avg_annual_turnover", "transaction_cost_total"]:
            self.assertIn(key, metrics)


if __name__ == "__main__":
    unittest.main()
