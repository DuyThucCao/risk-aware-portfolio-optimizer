"""Portfolio optimization engines for the Risk-Aware Portfolio Optimizer.

This module implements the three strategy families used throughout the project:

1. Equal-weight baseline.
2. Traditional mean-variance optimization.
3. Risk-aware optimization with turnover limits, transaction costs, CVaR,
   position limits, sector caps, and volatility targeting.

CVXPY is used when available because the core problem is convex. A SciPy SLSQP
fallback is included so the repository can still run in lighter environments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

try:
    import cvxpy as cp
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    cp = None

try:
    from scipy.optimize import minimize
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    minimize = None

from src.risk.risk_models import CVaRCalculator, VolatilityModel


logger = logging.getLogger(__name__)
TRADING_DAYS_PER_YEAR = 252
EPSILON = 1e-10


@dataclass
class OptimizationConfig:
    """Configuration for a portfolio optimization solve."""

    risk_aversion: float = 1.0
    linear_cost: float = 0.0005
    market_impact_coef: float = 0.05
    use_market_impact: bool = True
    min_weight: float = 0.0
    max_weight: float = 0.25
    max_turnover: Optional[float] = 0.35
    target_volatility: Optional[float] = 0.12
    max_leverage: float = 1.0
    cvar_limit: Optional[float] = 0.025
    cvar_alpha: float = 0.95
    sector_caps: Optional[dict[int, float]] = None
    periods_per_year: int = TRADING_DAYS_PER_YEAR
    solver: str = "CLARABEL"
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.risk_aversion < 0:
            raise ValueError("risk_aversion must be non-negative.")
        if self.linear_cost < 0:
            raise ValueError("linear_cost must be non-negative.")
        if self.market_impact_coef < 0:
            raise ValueError("market_impact_coef must be non-negative.")
        if self.min_weight > self.max_weight:
            raise ValueError("min_weight cannot exceed max_weight.")
        if self.max_weight <= 0:
            raise ValueError("max_weight must be positive.")
        if self.max_turnover is not None and self.max_turnover <= 0:
            raise ValueError("max_turnover must be positive or None.")
        if self.target_volatility is not None and self.target_volatility <= 0:
            raise ValueError("target_volatility must be positive or None.")
        if self.max_leverage <= 0:
            raise ValueError("max_leverage must be positive.")
        if self.cvar_limit is not None and self.cvar_limit <= 0:
            raise ValueError("cvar_limit must be positive or None.")
        if not 0.0 < self.cvar_alpha < 1.0:
            raise ValueError("cvar_alpha must be in (0, 1).")
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive.")
        if self.sector_caps is not None:
            for sector_id, cap in self.sector_caps.items():
                if int(sector_id) < 0:
                    raise ValueError("sector ids must be non-negative integers.")
                if not 0.0 <= cap <= 1.0:
                    raise ValueError("sector caps must be between 0 and 1.")


@dataclass
class OptimizationResult:
    """Output of a single optimization solve."""

    weights: np.ndarray
    status: str
    objective_value: Optional[float]
    portfolio_return: float
    portfolio_volatility: float
    sharpe_ratio: float
    transaction_cost: float
    turnover: float
    cvar_estimate: Optional[float] = None
    vol_scale_factor: float = 1.0
    strategy_name: str = "Risk-Aware Optimizer"

    @property
    def feasible(self) -> bool:
        """True when the result came from a usable optimization solve."""
        return self.status in {
            "optimal",
            "optimal_inaccurate",
            "scipy_optimal",
            "equal_weight",
            "mean_variance",
        }


class EqualWeightOptimizer:
    """Simple equal-weight baseline."""

    @staticmethod
    def optimize(
        n_assets: int,
        expected_returns: Optional[np.ndarray] = None,
        covariance: Optional[np.ndarray] = None,
    ) -> OptimizationResult:
        """Return an equal allocation across all assets."""
        if n_assets <= 0:
            raise ValueError("n_assets must be positive.")
        weights = np.ones(n_assets, dtype=float) / n_assets
        mu = np.zeros(n_assets) if expected_returns is None else np.asarray(expected_returns, dtype=float)
        cov = np.eye(n_assets) * EPSILON if covariance is None else np.asarray(covariance, dtype=float)
        return build_result(
            weights=weights,
            expected_returns=mu,
            covariance=cov,
            current_weights=np.zeros(n_assets),
            status="equal_weight",
            objective_value=None,
            transaction_cost=0.0,
            vol_scale_factor=1.0,
            cvar_estimate=None,
            strategy_name="Equal Weight",
        )


class MeanVarianceOptimizer:
    """Traditional Markowitz mean-variance optimizer without market frictions."""

    def __init__(self, risk_aversion: float = 4.0, min_weight: float = 0.0, max_weight: float = 1.0) -> None:
        self.config = OptimizationConfig(
            risk_aversion=risk_aversion,
            linear_cost=0.0,
            market_impact_coef=0.0,
            use_market_impact=False,
            min_weight=min_weight,
            max_weight=max_weight,
            max_turnover=None,
            target_volatility=None,
            cvar_limit=None,
        )
        self.optimizer = RiskAwareOptimizer(self.config)

    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        current_weights: Optional[np.ndarray] = None,
    ) -> OptimizationResult:
        """Solve a traditional mean-variance portfolio."""
        mu, cov = validate_optimization_inputs(expected_returns, covariance)
        if current_weights is None:
            current_weights = np.ones(len(mu)) / len(mu)
        result = self.optimizer.optimize(mu, cov, current_weights)
        result.strategy_name = "Mean-Variance Optimization"
        if result.status in {"optimal", "optimal_inaccurate", "scipy_optimal"}:
            result.status = "mean_variance"
        return result


class RiskAwareOptimizer:
    """Risk-aware convex optimizer with realistic market frictions."""

    def __init__(self, config: Optional[OptimizationConfig] = None) -> None:
        self.config = config or OptimizationConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        current_weights: np.ndarray,
        scenario_returns: Optional[np.ndarray] = None,
        sector_indices: Optional[np.ndarray] = None,
    ) -> OptimizationResult:
        """Solve the configured portfolio optimization problem."""
        mu, cov, current = validate_optimization_inputs(
            expected_returns,
            covariance,
            current_weights=current_weights,
        )
        scenarios = validate_scenario_returns(scenario_returns, len(mu)) if scenario_returns is not None else None
        sectors = validate_sector_indices(sector_indices, len(mu)) if sector_indices is not None else None

        infeasible_reason = self._basic_infeasibility_reason(len(mu), sectors)
        if infeasible_reason:
            self.logger.warning("Optimization infeasible before solve: %s", infeasible_reason)
            return self._no_trade_result(current, mu, cov, status="no_trade_fallback")

        if cp is not None:
            result = self._optimize_with_cvxpy(mu, cov, current, scenarios, sectors)
            if result is not None:
                return result

        if minimize is not None:
            result = self._optimize_with_scipy(mu, cov, current, scenarios, sectors)
            if result is not None:
                return result

        self.logger.warning("No optimization backend available. Holding current weights.")
        return self._no_trade_result(current, mu, cov, status="no_trade_fallback")

    def optimize_equal_weight(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
    ) -> OptimizationResult:
        """Convenience wrapper for the equal-weight baseline."""
        mu, cov = validate_optimization_inputs(expected_returns, covariance)
        return EqualWeightOptimizer.optimize(len(mu), mu, cov)

    def optimize_mean_variance(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        current_weights: Optional[np.ndarray] = None,
    ) -> OptimizationResult:
        """Convenience wrapper for traditional mean-variance optimization."""
        return MeanVarianceOptimizer(
            risk_aversion=self.config.risk_aversion,
            min_weight=self.config.min_weight,
            max_weight=max(self.config.max_weight, 1.0 / len(expected_returns)),
        ).optimize(expected_returns, covariance, current_weights)

    def efficient_frontier(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        current_weights: np.ndarray,
        n_points: int = 30,
        scenario_returns: Optional[np.ndarray] = None,
        sector_indices: Optional[np.ndarray] = None,
    ) -> list[OptimizationResult]:
        """Trace an efficient frontier by varying risk aversion."""
        if n_points <= 0:
            raise ValueError("n_points must be positive.")
        original_lambda = self.config.risk_aversion
        results: list[OptimizationResult] = []

        for risk_aversion in np.logspace(-1, 2, n_points):
            self.config.risk_aversion = float(risk_aversion)
            result = self.optimize(
                expected_returns=expected_returns,
                covariance=covariance,
                current_weights=current_weights,
                scenario_returns=scenario_returns,
                sector_indices=sector_indices,
            )
            results.append(result)

        self.config.risk_aversion = original_lambda
        return results

    def _optimize_with_cvxpy(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        current_weights: np.ndarray,
        scenario_returns: Optional[np.ndarray],
        sector_indices: Optional[np.ndarray],
    ) -> Optional[OptimizationResult]:
        assert cp is not None
        cfg = self.config
        n_assets = len(expected_returns)
        daily_mu = expected_returns / cfg.periods_per_year
        daily_cov = nearest_psd(covariance / cfg.periods_per_year)

        weights = cp.Variable(n_assets, name="weights")
        trades = weights - current_weights
        transaction_cost = self._cvxpy_transaction_cost(trades)

        objective = cp.Maximize(
            daily_mu @ weights
            - (cfg.risk_aversion / 2.0) * cp.quad_form(weights, daily_cov)
            - transaction_cost
        )
        constraints = self._cvxpy_constraints(
            weights=weights,
            trades=trades,
            daily_cov=daily_cov,
            scenario_returns=scenario_returns,
            sector_indices=sector_indices,
        )
        problem = cp.Problem(objective, constraints)

        solved = self._solve_cvxpy_problem(problem)
        if not solved or weights.value is None or problem.status not in {"optimal", "optimal_inaccurate"}:
            self.logger.warning("CVXPY solve did not return a usable result: %s", problem.status)
            return None

        cleaned_weights, vol_scale = self._post_process_weights(
            np.asarray(weights.value).ravel(),
            covariance,
            scenario_returns=scenario_returns,
        )
        cvar_estimate = self._estimate_cvar(cleaned_weights, scenario_returns)
        transaction_cost_value = estimate_transaction_cost(cleaned_weights, current_weights, cfg)

        return build_result(
            weights=cleaned_weights,
            expected_returns=expected_returns,
            covariance=covariance,
            current_weights=current_weights,
            status=problem.status,
            objective_value=float(problem.value) if problem.value is not None else None,
            transaction_cost=transaction_cost_value,
            vol_scale_factor=vol_scale,
            cvar_estimate=cvar_estimate,
            strategy_name="Risk-Aware Optimizer",
        )

    def _optimize_with_scipy(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        current_weights: np.ndarray,
        scenario_returns: Optional[np.ndarray],
        sector_indices: Optional[np.ndarray],
    ) -> Optional[OptimizationResult]:
        assert minimize is not None
        cfg = self.config
        n_assets = len(expected_returns)
        daily_mu = expected_returns / cfg.periods_per_year
        daily_cov = nearest_psd(covariance / cfg.periods_per_year)
        x0 = self._initial_weights(n_assets, current_weights)

        def objective(weights: np.ndarray) -> float:
            trades = weights - current_weights
            expected_return = float(daily_mu @ weights)
            risk_penalty = float((cfg.risk_aversion / 2.0) * weights @ daily_cov @ weights)
            cost_penalty = estimate_transaction_cost(weights, current_weights, cfg, smooth=True)
            return -(expected_return - risk_penalty - cost_penalty)

        constraints = self._scipy_constraints(daily_cov, current_weights, scenario_returns, sector_indices)
        bounds = [(cfg.min_weight, cfg.max_weight) for _ in range(n_assets)]

        result = minimize(
            objective,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10, "disp": False},
        )
        if not result.success:
            self.logger.warning("SciPy optimizer failed: %s", result.message)
            return None

        cleaned_weights, vol_scale = self._post_process_weights(
            np.asarray(result.x, dtype=float),
            covariance,
            scenario_returns=scenario_returns,
        )
        cvar_estimate = self._estimate_cvar(cleaned_weights, scenario_returns)
        transaction_cost = estimate_transaction_cost(cleaned_weights, current_weights, cfg)

        return build_result(
            weights=cleaned_weights,
            expected_returns=expected_returns,
            covariance=covariance,
            current_weights=current_weights,
            status="scipy_optimal",
            objective_value=float(-result.fun),
            transaction_cost=transaction_cost,
            vol_scale_factor=vol_scale,
            cvar_estimate=cvar_estimate,
            strategy_name="Risk-Aware Optimizer",
        )

    def _cvxpy_transaction_cost(self, trades: object) -> object:
        assert cp is not None
        cfg = self.config
        cost = cfg.linear_cost * cp.norm1(trades)
        if cfg.use_market_impact and cfg.market_impact_coef > 0:
            cost += cfg.market_impact_coef * cp.sum_squares(trades)
        return cost

    def _cvxpy_constraints(
        self,
        weights: object,
        trades: object,
        daily_cov: np.ndarray,
        scenario_returns: Optional[np.ndarray],
        sector_indices: Optional[np.ndarray],
    ) -> list[object]:
        assert cp is not None
        cfg = self.config
        constraints: list[object] = [
            cp.sum(weights) == 1.0,
            weights >= cfg.min_weight,
            weights <= cfg.max_weight,
        ]

        if cfg.max_turnover is not None:
            constraints.append(cp.norm1(trades) <= 2.0 * cfg.max_turnover)

        if cfg.cvar_limit is not None and scenario_returns is not None and cfg.target_volatility is None:
            constraints.extend(self._cvxpy_cvar_constraints(weights, scenario_returns))

        if cfg.sector_caps is not None and sector_indices is not None:
            for sector_id, cap in cfg.sector_caps.items():
                mask = sector_indices == int(sector_id)
                if np.any(mask):
                    constraints.append(cp.sum(weights[mask]) <= float(cap))

        return constraints

    def _cvxpy_cvar_constraints(self, weights: object, scenario_returns: np.ndarray) -> list[object]:
        assert cp is not None
        cfg = self.config
        n_scenarios = scenario_returns.shape[0]
        var_threshold = cp.Variable(name="var_threshold")
        excess_losses = cp.Variable(n_scenarios, nonneg=True, name="excess_losses")
        losses = -scenario_returns @ weights
        cvar = var_threshold + cp.sum(excess_losses) / ((1.0 - cfg.cvar_alpha) * n_scenarios)
        return [
            excess_losses >= losses - var_threshold,
            cvar <= cfg.cvar_limit,
        ]

    def _solve_cvxpy_problem(self, problem: object) -> bool:
        assert cp is not None
        installed = set(cp.installed_solvers())
        candidates = [self.config.solver, "CLARABEL", "ECOS", "SCS"]
        for solver in dict.fromkeys(candidates):
            if solver not in installed:
                continue
            try:
                problem.solve(solver=solver, warm_start=True, verbose=self.config.verbose)
                if problem.status in {"optimal", "optimal_inaccurate"}:
                    return True
            except Exception as exc:
                self.logger.debug("CVXPY solver %s failed: %s", solver, exc)
        return False

    def _scipy_constraints(
        self,
        daily_cov: np.ndarray,
        current_weights: np.ndarray,
        scenario_returns: Optional[np.ndarray],
        sector_indices: Optional[np.ndarray],
    ) -> list[dict[str, object]]:
        cfg = self.config
        constraints: list[dict[str, object]] = [
            {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},
        ]

        if cfg.max_turnover is not None:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w: float(2.0 * cfg.max_turnover - np.sum(np.abs(w - current_weights))),
                }
            )

        if cfg.cvar_limit is not None and scenario_returns is not None and cfg.target_volatility is None:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w: float(
                        cfg.cvar_limit
                        - CVaRCalculator.portfolio_cvar(w, scenario_returns, alpha=cfg.cvar_alpha)
                    ),
                }
            )

        if cfg.sector_caps is not None and sector_indices is not None:
            for sector_id, cap in cfg.sector_caps.items():
                mask = sector_indices == int(sector_id)
                if np.any(mask):
                    constraints.append(
                        {
                            "type": "ineq",
                            "fun": self._make_sector_constraint(mask, float(cap)),
                        }
                    )

        return constraints

    @staticmethod
    def _make_sector_constraint(mask: np.ndarray, cap: float) -> Callable[[np.ndarray], float]:
        return lambda weights: float(cap - np.sum(weights[mask]))

    def _initial_weights(self, n_assets: int, current_weights: np.ndarray) -> np.ndarray:
        cfg = self.config
        x0 = np.asarray(current_weights, dtype=float).copy()
        if x0.size != n_assets or not np.isfinite(x0).all() or x0.sum() <= EPSILON:
            x0 = np.ones(n_assets) / n_assets
        x0 = np.clip(x0, cfg.min_weight, cfg.max_weight)
        if x0.sum() <= EPSILON:
            x0 = np.ones(n_assets) / n_assets
        x0 = x0 / x0.sum()
        return x0

    def _post_process_weights(
        self,
        raw_weights: np.ndarray,
        covariance: np.ndarray,
        scenario_returns: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, float]:
        cfg = self.config
        weights = np.asarray(raw_weights, dtype=float)
        weights = np.where(np.abs(weights) < 1e-9, 0.0, weights)
        weights = np.clip(weights, cfg.min_weight, cfg.max_weight)

        total = float(weights.sum())
        if total <= EPSILON:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / total

        vol_scale = 1.0
        if cfg.target_volatility is not None:
            realized_vol = float(np.sqrt(max(weights @ covariance @ weights, 0.0)))
            vol_scale = VolatilityModel.scale_to_target(
                realized_volatility=realized_vol,
                target_volatility=cfg.target_volatility,
                min_scale=0.0,
                max_scale=cfg.max_leverage,
            )
            weights = weights * vol_scale

            if cfg.cvar_limit is not None and scenario_returns is not None:
                cvar = CVaRCalculator.portfolio_cvar(weights, scenario_returns, alpha=cfg.cvar_alpha)
                if cvar > cfg.cvar_limit > 0:
                    cvar_scale = float(np.clip(cfg.cvar_limit / cvar, 0.0, 1.0))
                    weights = weights * cvar_scale
                    vol_scale *= cvar_scale

        return weights, float(vol_scale)

    def _estimate_cvar(
        self,
        weights: np.ndarray,
        scenario_returns: Optional[np.ndarray],
    ) -> Optional[float]:
        if scenario_returns is None:
            return None
        return CVaRCalculator.portfolio_cvar(weights, scenario_returns, alpha=self.config.cvar_alpha)

    def _basic_infeasibility_reason(self, n_assets: int, sector_indices: Optional[np.ndarray]) -> Optional[str]:
        cfg = self.config
        if n_assets <= 0:
            return "no assets were supplied"
        if cfg.min_weight * n_assets > 1.0 + 1e-8:
            return "minimum weights require more than 100% allocation"
        if cfg.max_weight * n_assets < 1.0 - 1e-8:
            return "maximum weights cannot sum to a fully invested portfolio"
        if cfg.sector_caps is not None and sector_indices is not None:
            cap_total = 0.0
            for sector_id in sorted(set(sector_indices.tolist())):
                cap_total += cfg.sector_caps.get(int(sector_id), 1.0)
            if cap_total < 1.0 - 1e-8:
                return "sector caps cannot sum to a fully invested portfolio"
        return None

    def _no_trade_result(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        status: str,
    ) -> OptimizationResult:
        return build_result(
            weights=weights.copy(),
            expected_returns=expected_returns,
            covariance=covariance,
            current_weights=weights,
            status=status,
            objective_value=None,
            transaction_cost=0.0,
            vol_scale_factor=1.0,
            cvar_estimate=None,
            strategy_name="Risk-Aware Optimizer",
        )


def estimate_transaction_cost(
    weights: np.ndarray,
    current_weights: np.ndarray,
    config: OptimizationConfig,
    *,
    smooth: bool = False,
) -> float:
    """Estimate one-period transaction cost in return units."""
    trades = np.asarray(weights, dtype=float) - np.asarray(current_weights, dtype=float)
    abs_trades = smooth_abs(trades) if smooth else np.abs(trades)
    linear = float(config.linear_cost * np.sum(abs_trades))
    quadratic = 0.0
    if config.use_market_impact and config.market_impact_coef > 0:
        quadratic = float(config.market_impact_coef * np.sum(np.square(trades)))
    return linear + quadratic


def smooth_abs(values: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Smooth absolute value used by the SciPy fallback objective."""
    return np.sqrt(np.square(values) + epsilon)


def build_result(
    *,
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    current_weights: np.ndarray,
    status: str,
    objective_value: Optional[float],
    transaction_cost: float,
    vol_scale_factor: float,
    cvar_estimate: Optional[float],
    strategy_name: str,
) -> OptimizationResult:
    """Build an OptimizationResult with consistent diagnostics."""
    weights_arr = np.asarray(weights, dtype=float)
    mu = np.asarray(expected_returns, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    current = np.asarray(current_weights, dtype=float)

    portfolio_return = float(mu @ weights_arr)
    portfolio_variance = float(weights_arr @ cov @ weights_arr)
    portfolio_volatility = float(np.sqrt(max(portfolio_variance, 0.0)))
    sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > EPSILON else 0.0
    turnover = float(np.sum(np.abs(weights_arr - current))) / 2.0

    return OptimizationResult(
        weights=weights_arr,
        status=status,
        objective_value=objective_value,
        portfolio_return=portfolio_return,
        portfolio_volatility=portfolio_volatility,
        sharpe_ratio=float(sharpe_ratio),
        transaction_cost=float(transaction_cost),
        turnover=turnover,
        cvar_estimate=cvar_estimate,
        vol_scale_factor=float(vol_scale_factor),
        strategy_name=strategy_name,
    )


def validate_optimization_inputs(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    *,
    current_weights: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and standardize optimizer inputs."""
    mu = np.asarray(expected_returns, dtype=float)
    if mu.ndim != 1 or mu.size == 0:
        raise ValueError("expected_returns must be a non-empty one-dimensional array.")
    if not np.isfinite(mu).all():
        raise ValueError("expected_returns contains non-finite values.")

    cov = np.asarray(covariance, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance must be a square matrix.")
    if cov.shape[0] != mu.size:
        raise ValueError("covariance dimensions must match expected_returns length.")
    if not np.isfinite(cov).all():
        raise ValueError("covariance contains non-finite values.")
    cov = nearest_psd(cov)

    if current_weights is None:
        return mu, cov

    current = np.asarray(current_weights, dtype=float)
    if current.ndim != 1 or current.size != mu.size:
        raise ValueError("current_weights must be one-dimensional and match asset count.")
    if not np.isfinite(current).all():
        raise ValueError("current_weights contains non-finite values.")
    if current.sum() > EPSILON:
        current = current / current.sum()
    return mu, cov, current


def validate_scenario_returns(scenario_returns: np.ndarray, n_assets: int) -> np.ndarray:
    """Validate scenario returns for CVaR constraints."""
    scenarios = np.asarray(scenario_returns, dtype=float)
    if scenarios.ndim != 2:
        raise ValueError("scenario_returns must be two-dimensional.")
    if scenarios.shape[1] != n_assets:
        raise ValueError("scenario_returns column count must match asset count.")
    if scenarios.shape[0] < 2:
        raise ValueError("scenario_returns must include at least two scenarios.")
    if not np.isfinite(scenarios).all():
        raise ValueError("scenario_returns contains non-finite values.")
    return scenarios


def validate_sector_indices(sector_indices: np.ndarray, n_assets: int) -> np.ndarray:
    """Validate integer sector or asset-class labels."""
    sectors = np.asarray(sector_indices)
    if sectors.ndim != 1 or sectors.size != n_assets:
        raise ValueError("sector_indices must be one-dimensional and match asset count.")
    return sectors.astype(int)


def nearest_psd(matrix: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Return a symmetric positive semidefinite version of a matrix."""
    arr = np.asarray(matrix, dtype=float)
    symmetric = (arr + arr.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(eigenvalues, epsilon)
    projected = eigenvectors @ np.diag(clipped) @ eigenvectors.T
    return (projected + projected.T) / 2.0


__all__ = [
    "EPSILON",
    "EqualWeightOptimizer",
    "MeanVarianceOptimizer",
    "OptimizationConfig",
    "OptimizationResult",
    "RiskAwareOptimizer",
    "TRADING_DAYS_PER_YEAR",
    "build_result",
    "estimate_transaction_cost",
    "nearest_psd",
    "smooth_abs",
    "validate_optimization_inputs",
    "validate_scenario_returns",
    "validate_sector_indices",
]
