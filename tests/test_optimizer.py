"""Tests for the portfolio optimization layer.

These tests use only Python's standard ``unittest`` runner so they can run in a
minimal environment. They also remain compatible with pytest discovery.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.optimization.optimizer import (
    EqualWeightOptimizer,
    MeanVarianceOptimizer,
    OptimizationConfig,
    RiskAwareOptimizer,
    estimate_transaction_cost,
    validate_optimization_inputs,
)


N_ASSETS = 8


def make_inputs(
    n_assets: int = N_ASSETS,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create deterministic, well-conditioned optimizer inputs."""
    rng = np.random.default_rng(seed)
    expected_returns = np.linspace(0.05, 0.14, n_assets)
    factor = rng.normal(size=(n_assets, 3))
    covariance = factor @ factor.T / 25.0 + np.eye(n_assets) * 0.04
    current_weights = np.ones(n_assets, dtype=float) / n_assets
    scenarios = rng.multivariate_normal(
        mean=np.zeros(n_assets),
        cov=covariance / 252.0,
        size=300,
    )
    return expected_returns, covariance, current_weights, scenarios


class OptimizerBasicTests(unittest.TestCase):
    """Core optimizer behavior."""

    def setUp(self) -> None:
        self.mu, self.covariance, self.current_weights, self.scenarios = make_inputs()
        self.config = OptimizationConfig(
            risk_aversion=1.0,
            min_weight=0.0,
            max_weight=0.35,
            max_turnover=0.50,
            target_volatility=None,
            cvar_limit=None,
            use_market_impact=True,
            linear_cost=0.0005,
        )
        self.optimizer = RiskAwareOptimizer(self.config)

    def test_risk_aware_optimizer_returns_feasible_weights(self) -> None:
        result = self.optimizer.optimize(self.mu, self.covariance, self.current_weights)

        self.assertTrue(result.feasible, f"Unexpected status: {result.status}")
        self.assertEqual(len(result.weights), len(self.mu))
        self.assertTrue(np.isfinite(result.weights).all())
        np.testing.assert_allclose(result.weights.sum(), 1.0, atol=1e-5)
        self.assertGreaterEqual(result.weights.min(), -1e-8)

    def test_max_weight_is_respected(self) -> None:
        config = OptimizationConfig(
            max_weight=0.20,
            max_turnover=None,
            target_volatility=None,
            cvar_limit=None,
        )
        result = RiskAwareOptimizer(config).optimize(self.mu, self.covariance, self.current_weights)

        self.assertTrue(result.feasible)
        self.assertLessEqual(float(result.weights.max()), 0.20 + 1e-5)

    def test_turnover_constraint_is_respected(self) -> None:
        max_turnover = 0.10
        config = OptimizationConfig(
            max_weight=0.35,
            max_turnover=max_turnover,
            target_volatility=None,
            cvar_limit=None,
            use_market_impact=False,
        )
        result = RiskAwareOptimizer(config).optimize(self.mu, self.covariance, self.current_weights)

        self.assertTrue(result.feasible)
        self.assertLessEqual(result.turnover, max_turnover + 1e-5)

    def test_result_diagnostics_are_finite(self) -> None:
        result = self.optimizer.optimize(self.mu, self.covariance, self.current_weights)

        self.assertTrue(np.isfinite(result.portfolio_return))
        self.assertTrue(np.isfinite(result.portfolio_volatility))
        self.assertTrue(np.isfinite(result.sharpe_ratio))
        self.assertTrue(np.isfinite(result.transaction_cost))
        self.assertGreater(result.portfolio_volatility, 0.0)
        self.assertGreaterEqual(result.transaction_cost, 0.0)


class RiskControlTests(unittest.TestCase):
    """Risk-aware constraints and post-processing."""

    def setUp(self) -> None:
        self.mu, self.covariance, self.current_weights, self.scenarios = make_inputs(seed=11)

    def test_cvar_limit_is_respected_when_enabled(self) -> None:
        cvar_limit = 0.05
        config = OptimizationConfig(
            max_weight=0.35,
            max_turnover=None,
            target_volatility=None,
            cvar_limit=cvar_limit,
            cvar_alpha=0.95,
        )
        result = RiskAwareOptimizer(config).optimize(
            self.mu,
            self.covariance,
            self.current_weights,
            scenario_returns=self.scenarios,
        )

        self.assertTrue(result.feasible)
        self.assertIsNotNone(result.cvar_estimate)
        self.assertLessEqual(float(result.cvar_estimate), cvar_limit + 1e-4)

    def test_sector_cap_is_respected(self) -> None:
        sector_indices = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        config = OptimizationConfig(
            max_weight=0.35,
            max_turnover=None,
            target_volatility=None,
            cvar_limit=None,
            sector_caps={0: 0.45},
        )
        result = RiskAwareOptimizer(config).optimize(
            self.mu,
            self.covariance,
            self.current_weights,
            sector_indices=sector_indices,
        )

        self.assertTrue(result.feasible)
        self.assertLessEqual(float(result.weights[sector_indices == 0].sum()), 0.45 + 1e-5)

    def test_volatility_target_scales_portfolio_down(self) -> None:
        target_volatility = 0.08
        config = OptimizationConfig(
            max_weight=0.35,
            max_turnover=None,
            target_volatility=target_volatility,
            cvar_limit=None,
            max_leverage=1.0,
        )
        result = RiskAwareOptimizer(config).optimize(self.mu, self.covariance, self.current_weights)

        self.assertTrue(result.feasible)
        self.assertLessEqual(result.portfolio_volatility, target_volatility + 1e-5)
        self.assertGreaterEqual(result.vol_scale_factor, 0.0)
        self.assertLessEqual(result.vol_scale_factor, 1.0 + 1e-8)
        self.assertLessEqual(result.weights.sum(), 1.0 + 1e-8)

    def test_infeasible_config_returns_no_trade_fallback(self) -> None:
        config = OptimizationConfig(
            max_weight=0.05,
            min_weight=0.0,
            max_turnover=None,
            target_volatility=None,
            cvar_limit=None,
        )
        result = RiskAwareOptimizer(config).optimize(self.mu, self.covariance, self.current_weights)

        self.assertEqual(result.status, "no_trade_fallback")
        np.testing.assert_allclose(result.weights, self.current_weights, atol=1e-12)
        self.assertFalse(result.feasible)


class BaselineOptimizerTests(unittest.TestCase):
    """Equal-weight and mean-variance baselines."""

    def setUp(self) -> None:
        self.mu, self.covariance, self.current_weights, _ = make_inputs(seed=23)

    def test_equal_weight_optimizer_allocates_evenly(self) -> None:
        result = EqualWeightOptimizer.optimize(len(self.mu), self.mu, self.covariance)

        self.assertTrue(result.feasible)
        np.testing.assert_allclose(result.weights, np.ones(len(self.mu)) / len(self.mu), atol=1e-12)
        self.assertEqual(result.strategy_name, "Equal Weight")

    def test_mean_variance_optimizer_returns_baseline_result(self) -> None:
        result = MeanVarianceOptimizer(risk_aversion=2.0, max_weight=0.40).optimize(
            self.mu,
            self.covariance,
            self.current_weights,
        )

        self.assertTrue(result.feasible)
        self.assertEqual(result.strategy_name, "Mean-Variance Optimization")
        np.testing.assert_allclose(result.weights.sum(), 1.0, atol=1e-5)

    def test_efficient_frontier_restores_original_risk_aversion(self) -> None:
        optimizer = RiskAwareOptimizer(
            OptimizationConfig(
                risk_aversion=1.7,
                max_weight=0.40,
                max_turnover=None,
                target_volatility=None,
                cvar_limit=None,
            )
        )
        frontier = optimizer.efficient_frontier(self.mu, self.covariance, self.current_weights, n_points=8)
        feasible = [result for result in frontier if result.feasible]

        self.assertEqual(len(frontier), 8)
        self.assertGreaterEqual(len(feasible), 5)
        self.assertAlmostEqual(optimizer.config.risk_aversion, 1.7)
        self.assertGreater(max(result.portfolio_volatility for result in feasible), 0.0)


class HelperFunctionTests(unittest.TestCase):
    """Validation and cost helpers."""

    def test_transaction_cost_increases_with_trade_size(self) -> None:
        config = OptimizationConfig(
            linear_cost=0.001,
            market_impact_coef=0.05,
            use_market_impact=True,
            target_volatility=None,
            cvar_limit=None,
        )
        previous = np.array([0.25, 0.25, 0.25, 0.25])
        small_trade = np.array([0.30, 0.20, 0.25, 0.25])
        large_trade = np.array([0.50, 0.10, 0.20, 0.20])

        small_cost = estimate_transaction_cost(small_trade, previous, config)
        large_cost = estimate_transaction_cost(large_trade, previous, config)

        self.assertGreater(small_cost, 0.0)
        self.assertGreater(large_cost, small_cost)

    def test_validate_inputs_rejects_shape_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            validate_optimization_inputs(np.ones(3), np.eye(4))

    def test_validate_inputs_projects_covariance_to_psd(self) -> None:
        covariance = np.array([[1.0, 2.0], [2.0, 1.0]])
        mu, projected = validate_optimization_inputs(np.array([0.1, 0.2]), covariance)

        self.assertEqual(mu.shape, (2,))
        self.assertTrue(np.linalg.eigvalsh(projected).min() >= -1e-8)


if __name__ == "__main__":
    unittest.main()
