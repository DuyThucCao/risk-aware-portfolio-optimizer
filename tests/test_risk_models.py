"""Tests for risk modeling utilities.

The risk layer is where the project turns portfolio survival into measurable
controls: CVaR, drawdown, volatility scaling, risk contribution, and covariance
estimation. The suite uses ``unittest`` so it runs without pytest installed.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.data.data_loader import CovarianceEstimator
from src.risk.risk_models import (
    CVaRCalculator,
    DrawdownAnalyzer,
    RiskBudget,
    VolatilityModel,
    build_risk_snapshot,
)


def make_returns(length: int = 500, *, seed: int = 7) -> pd.Series:
    """Create deterministic daily returns with realistic noise."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=length)
    values = rng.normal(0.0005, 0.012, length)
    return pd.Series(values, index=dates, name="portfolio")


def make_return_matrix(
    n_obs: int = 320,
    n_assets: int = 6,
    *,
    seed: int = 11,
) -> np.ndarray:
    """Create a deterministic return matrix for covariance tests."""
    rng = np.random.default_rng(seed)
    factor = rng.normal(0.0, 0.008, size=(n_obs, 1))
    idiosyncratic = rng.normal(0.0, 0.010, size=(n_obs, n_assets))
    loadings = np.linspace(0.4, 1.0, n_assets)
    return factor @ loadings.reshape(1, -1) + idiosyncratic


def make_portfolio() -> tuple[np.ndarray, np.ndarray]:
    """Create a long-only portfolio with diagonal covariance."""
    weights = np.array([0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.06, 0.04])
    annual_vols = np.array([0.18, 0.20, 0.16, 0.22, 0.14, 0.12, 0.19, 0.15])
    covariance = np.diag(np.square(annual_vols))
    return weights, covariance


class CVaRCalculatorTests(unittest.TestCase):
    """Historical, parametric, and portfolio CVaR behavior."""

    def test_historical_cvar_is_at_least_var(self) -> None:
        returns = make_returns(seed=1)
        var, cvar = CVaRCalculator.historical(returns, alpha=0.95)

        self.assertGreaterEqual(cvar, var - 1e-12)
        self.assertGreater(cvar, 0.0)

    def test_historical_var_matches_loss_quantile(self) -> None:
        returns = make_returns(seed=2)
        var, _ = CVaRCalculator.historical(returns, alpha=0.95)
        expected = float(np.quantile(-returns.to_numpy(), 0.95))

        np.testing.assert_allclose(var, expected, atol=1e-12)

    def test_parametric_cvar_is_close_to_large_gaussian_sample(self) -> None:
        rng = np.random.default_rng(3)
        mu = 0.001
        sigma = 0.015
        sample = pd.Series(rng.normal(mu, sigma, 100000))

        _, historical_cvar = CVaRCalculator.historical(sample, alpha=0.95)
        _, parametric_cvar = CVaRCalculator.parametric_gaussian(mu, sigma, alpha=0.95)

        np.testing.assert_allclose(historical_cvar, parametric_cvar, rtol=0.05)

    def test_portfolio_cvar_and_losses_use_weighted_scenarios(self) -> None:
        scenarios = make_return_matrix(n_obs=250, n_assets=4, seed=4)
        weights = np.ones(4) / 4.0

        losses = CVaRCalculator.scenario_losses(weights, scenarios)
        cvar = CVaRCalculator.portfolio_cvar(weights, scenarios, alpha=0.95)

        self.assertEqual(losses.shape, (250,))
        self.assertTrue(np.isfinite(losses).all())
        self.assertTrue(np.isfinite(cvar))
        self.assertGreater(cvar, 0.0)

    def test_invalid_alpha_raises(self) -> None:
        with self.assertRaises(ValueError):
            CVaRCalculator.historical(make_returns(), alpha=1.0)


class DrawdownAnalyzerTests(unittest.TestCase):
    """Drawdown survival metrics."""

    def test_monotone_uptrend_has_zero_drawdown(self) -> None:
        returns = pd.Series([0.01] * 100)

        np.testing.assert_allclose(DrawdownAnalyzer.max_drawdown(returns), 0.0, atol=1e-12)

    def test_crash_then_recovery_records_initial_capital_drawdown(self) -> None:
        prices = pd.Series([100.0, 50.0, 100.0])
        returns = prices.pct_change().dropna()

        drawdown = DrawdownAnalyzer.series(returns)

        np.testing.assert_allclose(drawdown.min(), -0.50, atol=1e-12)
        np.testing.assert_allclose(DrawdownAnalyzer.max_drawdown(returns), -0.50, atol=1e-12)

    def test_drawdown_series_is_nonpositive(self) -> None:
        drawdown = DrawdownAnalyzer.series(make_returns(seed=5))

        self.assertTrue((drawdown <= 1e-12).all())

    def test_underwater_periods_have_expected_columns(self) -> None:
        returns = pd.Series(
            [0.02, -0.10, 0.03, 0.04, 0.02, -0.05, 0.01],
            index=pd.bdate_range("2023-01-02", periods=7),
        )
        periods = DrawdownAnalyzer.underwater_periods(returns)

        self.assertTrue({"start", "end", "depth", "duration", "recovered"}.issubset(periods.columns))
        self.assertGreaterEqual(len(periods), 1)

    def test_drawdown_scale_reduces_risk_in_deep_drawdown(self) -> None:
        returns = pd.Series([0.01, -0.20, -0.02, 0.01])

        scale = DrawdownAnalyzer.drawdown_scale(
            returns,
            max_drawdown_threshold=-0.15,
            risk_off_scale=0.60,
            recovery_threshold=-0.08,
        )

        self.assertAlmostEqual(scale, 0.60)

    def test_calmar_ratio_positive_for_steady_positive_returns(self) -> None:
        returns = pd.Series([0.001] * 252)

        self.assertTrue(DrawdownAnalyzer.calmar_ratio(returns) > 0.0)


class VolatilityModelTests(unittest.TestCase):
    """Realized volatility and target-volatility scaling."""

    def test_rolling_volatility_is_positive_after_window(self) -> None:
        volatility = VolatilityModel.rolling(make_returns(seed=6), window=21)

        self.assertTrue((volatility.dropna() > 0.0).all())

    def test_ewm_volatility_fills_initial_values(self) -> None:
        volatility = VolatilityModel.ewm(make_returns(seed=7), halflife=21)

        self.assertFalse(volatility.isna().any())
        self.assertTrue((volatility > 0.0).all())

    def test_annualized_volatility_scales_by_sqrt_periods(self) -> None:
        returns = make_returns(seed=8)
        daily = VolatilityModel.rolling(returns, window=21, annualize=False)
        annual = VolatilityModel.rolling(returns, window=21, annualize=True)
        ratio = (annual.dropna() / daily.dropna()).median()

        np.testing.assert_allclose(ratio, np.sqrt(252), rtol=0.01)

    def test_scale_to_target_bounds_result(self) -> None:
        scale = VolatilityModel.scale_to_target(
            realized_volatility=0.24,
            target_volatility=0.12,
            min_scale=0.25,
            max_scale=1.0,
        )

        self.assertAlmostEqual(scale, 0.5)

    def test_trailing_scale_uses_recent_realized_volatility(self) -> None:
        scale = VolatilityModel.trailing_scale(
            make_returns(seed=9),
            target_volatility=0.10,
            window=63,
            min_scale=0.0,
            max_scale=1.0,
        )

        self.assertGreaterEqual(scale, 0.0)
        self.assertLessEqual(scale, 1.0)


class RiskBudgetTests(unittest.TestCase):
    """Portfolio risk contribution and concentration diagnostics."""

    def test_relative_risk_contributions_sum_to_one(self) -> None:
        weights, covariance = make_portfolio()
        contributions = RiskBudget.risk_contribution(weights, covariance, relative=True)

        np.testing.assert_allclose(contributions.sum(), 1.0, atol=1e-12)
        self.assertTrue((contributions >= 0.0).all())

    def test_absolute_risk_contributions_sum_to_portfolio_volatility(self) -> None:
        weights, covariance = make_portfolio()
        contributions = RiskBudget.risk_contribution(weights, covariance, relative=False)
        portfolio_volatility = float(np.sqrt(weights @ covariance @ weights))

        np.testing.assert_allclose(contributions.sum(), portfolio_volatility, rtol=1e-10)

    def test_diversification_ratio_is_at_least_one_for_uncorrelated_assets(self) -> None:
        weights, covariance = make_portfolio()

        self.assertGreaterEqual(RiskBudget.diversification_ratio(weights, covariance), 1.0)

    def test_effective_positions_decline_with_concentration(self) -> None:
        diversified = np.ones(8) / 8.0
        concentrated = np.array([0.70, 0.10, 0.05, 0.05, 0.03, 0.03, 0.02, 0.02])

        self.assertGreater(
            RiskBudget.effective_number_of_positions(diversified),
            RiskBudget.effective_number_of_positions(concentrated),
        )

    def test_invalid_covariance_shape_raises(self) -> None:
        weights, _ = make_portfolio()
        with self.assertRaises(ValueError):
            RiskBudget.risk_contribution(weights, np.eye(3), relative=True)


class CovarianceEstimatorTests(unittest.TestCase):
    """Covariance estimators used by the optimizer and backtester."""

    @staticmethod
    def assert_psd(testcase: unittest.TestCase, matrix: np.ndarray, tol: float = 1e-8) -> None:
        eigenvalues = np.linalg.eigvalsh(matrix)
        testcase.assertGreaterEqual(float(eigenvalues.min()), -tol)

    def test_sample_covariance_is_psd(self) -> None:
        covariance = CovarianceEstimator.sample(make_return_matrix(seed=12))

        self.assertEqual(covariance.shape, (6, 6))
        self.assert_psd(self, covariance)

    def test_ledoit_wolf_covariance_is_psd(self) -> None:
        covariance = CovarianceEstimator.ledoit_wolf(make_return_matrix(seed=13))

        self.assertEqual(covariance.shape, (6, 6))
        self.assert_psd(self, covariance)

    def test_ewm_covariance_is_psd(self) -> None:
        covariance = CovarianceEstimator.ewm(make_return_matrix(seed=14), halflife=60)

        self.assertEqual(covariance.shape, (6, 6))
        self.assert_psd(self, covariance)

    def test_constant_correlation_covariance_is_psd(self) -> None:
        covariance = CovarianceEstimator.constant_corr(make_return_matrix(seed=15))

        self.assertEqual(covariance.shape, (6, 6))
        self.assert_psd(self, covariance)

    def test_sample_covariance_annualization_factor(self) -> None:
        returns = make_return_matrix(seed=16)
        daily = CovarianceEstimator.sample(returns, annualize=False)
        annual = CovarianceEstimator.sample(returns, annualize=True)

        np.testing.assert_allclose(annual, daily * 252, rtol=1e-10)


class RiskSnapshotTests(unittest.TestCase):
    """Integrated risk snapshot used by reports and dashboards."""

    def test_build_risk_snapshot_returns_finite_fields(self) -> None:
        returns = make_returns(length=260, seed=17)
        weights = np.ones(4) / 4.0
        scenarios = make_return_matrix(n_obs=260, n_assets=4, seed=18)

        snapshot = build_risk_snapshot(
            returns,
            weights,
            scenarios,
            target_volatility=0.12,
            alpha=0.95,
            volatility_window=63,
        )

        self.assertTrue(np.isfinite(snapshot.annualized_volatility))
        self.assertLessEqual(snapshot.current_drawdown, 0.0)
        self.assertLessEqual(snapshot.max_drawdown, 0.0)
        self.assertGreaterEqual(snapshot.historical_var, 0.0)
        self.assertGreaterEqual(snapshot.historical_cvar, 0.0)
        self.assertGreaterEqual(snapshot.volatility_scale, 0.0)
        self.assertGreaterEqual(snapshot.drawdown_scale, 0.0)


if __name__ == "__main__":
    unittest.main()
