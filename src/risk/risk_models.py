"""Risk modeling utilities for optimization, backtesting, and reporting.

The project message is that returns matter, but survival matters more. This
module turns that idea into measurable controls: tail loss, drawdown, realized
volatility, target-volatility scaling, and risk concentration.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, pi, sqrt
from statistics import NormalDist
from typing import Optional

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252
EPSILON = 1e-12


@dataclass(frozen=True)
class DrawdownPeriod:
    """One continuous underwater period in a portfolio equity curve."""

    start: pd.Timestamp
    end: Optional[pd.Timestamp]
    depth: float
    duration: int
    recovered: bool


@dataclass(frozen=True)
class RiskSnapshot:
    """Point-in-time summary of portfolio risk conditions."""

    annualized_volatility: float
    current_drawdown: float
    max_drawdown: float
    historical_var: float
    historical_cvar: float
    volatility_scale: float
    drawdown_scale: float


class CVaRCalculator:
    """Value-at-Risk and Conditional Value-at-Risk calculations."""

    @staticmethod
    def historical(
        returns: pd.Series | np.ndarray,
        alpha: float = 0.95,
    ) -> tuple[float, float]:
        """Return historical VaR and CVaR as positive loss values."""
        _validate_alpha(alpha)
        clean_returns = _as_1d_array(returns, name="returns")
        losses = -clean_returns
        var = float(np.quantile(losses, alpha))
        tail_losses = losses[losses >= var]
        cvar = float(tail_losses.mean()) if tail_losses.size else var
        return var, cvar

    @staticmethod
    def parametric_gaussian(
        mu: float,
        sigma: float,
        alpha: float = 0.95,
    ) -> tuple[float, float]:
        """Return Gaussian VaR and CVaR as positive loss values."""
        _validate_alpha(alpha)
        if sigma < 0:
            raise ValueError("sigma must be non-negative.")
        if sigma <= EPSILON:
            loss = max(float(-mu), 0.0)
            return loss, loss

        z_score = NormalDist().inv_cdf(alpha)
        pdf_at_z = exp(-0.5 * z_score**2) / sqrt(2.0 * pi)
        var = float(-mu + sigma * z_score)
        cvar = float(-mu + sigma * pdf_at_z / (1.0 - alpha))
        return var, cvar

    @staticmethod
    def portfolio_cvar(
        weights: np.ndarray,
        scenario_returns: np.ndarray,
        alpha: float = 0.95,
    ) -> float:
        """Return historical CVaR for a weighted scenario-return matrix."""
        weights_arr = _as_1d_array(weights, name="weights")
        scenarios = _as_2d_array(scenario_returns, name="scenario_returns")
        if scenarios.shape[1] != weights_arr.size:
            raise ValueError("scenario_returns columns must match weights length.")
        _, cvar = CVaRCalculator.historical(scenarios @ weights_arr, alpha=alpha)
        return cvar

    @staticmethod
    def portfolio_var(
        weights: np.ndarray,
        scenario_returns: np.ndarray,
        alpha: float = 0.95,
    ) -> float:
        """Return historical VaR for a weighted scenario-return matrix."""
        weights_arr = _as_1d_array(weights, name="weights")
        scenarios = _as_2d_array(scenario_returns, name="scenario_returns")
        if scenarios.shape[1] != weights_arr.size:
            raise ValueError("scenario_returns columns must match weights length.")
        var, _ = CVaRCalculator.historical(scenarios @ weights_arr, alpha=alpha)
        return var

    @staticmethod
    def scenario_losses(weights: np.ndarray, scenario_returns: np.ndarray) -> np.ndarray:
        """Return positive-loss scenarios for a proposed portfolio."""
        weights_arr = _as_1d_array(weights, name="weights")
        scenarios = _as_2d_array(scenario_returns, name="scenario_returns")
        if scenarios.shape[1] != weights_arr.size:
            raise ValueError("scenario_returns columns must match weights length.")
        return -(scenarios @ weights_arr)


class DrawdownAnalyzer:
    """Drawdown and survival-risk analysis."""

    @staticmethod
    def equity_curve(returns: pd.Series, initial_value: float = 1.0) -> pd.Series:
        """Convert daily returns into a portfolio value series."""
        clean = _as_series(returns, name="returns")
        if initial_value <= 0:
            raise ValueError("initial_value must be positive.")
        return initial_value * (1.0 + clean).cumprod()

    @staticmethod
    def series(returns: pd.Series) -> pd.Series:
        """Return drawdown series, including the initial capital high-water mark."""
        equity = DrawdownAnalyzer.equity_curve(returns, initial_value=1.0)
        high_water = equity.cummax().clip(lower=1.0)
        drawdown = (equity - high_water) / high_water
        return drawdown.clip(upper=0.0)

    @staticmethod
    def max_drawdown(returns: pd.Series) -> float:
        """Return the maximum peak-to-trough drawdown as a negative value."""
        return float(DrawdownAnalyzer.series(returns).min())

    @staticmethod
    def current_drawdown(returns: pd.Series) -> float:
        """Return the latest drawdown value."""
        drawdown = DrawdownAnalyzer.series(returns)
        return float(drawdown.iloc[-1])

    @staticmethod
    def calmar_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
        """Return annualized return divided by absolute maximum drawdown."""
        clean = _as_series(returns, name="returns")
        if periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive.")
        years = len(clean) / periods_per_year
        ending_value = float((1.0 + clean).prod())
        annualized_return = ending_value ** (1.0 / years) - 1.0 if years > 0 else 0.0
        max_dd = abs(DrawdownAnalyzer.max_drawdown(clean))
        return float(annualized_return / max_dd) if max_dd > EPSILON else float("inf")

    @staticmethod
    def underwater_periods(returns: pd.Series) -> pd.DataFrame:
        """Return a table of continuous drawdown periods."""
        drawdown = DrawdownAnalyzer.series(returns)
        periods: list[DrawdownPeriod] = []
        start: Optional[pd.Timestamp] = None
        previous_date: Optional[pd.Timestamp] = None

        for date, value in drawdown.items():
            timestamp = pd.Timestamp(date)
            if value < 0.0 and start is None:
                start = timestamp
            elif value >= 0.0 and start is not None:
                segment = drawdown.loc[start:previous_date]
                periods.append(
                    DrawdownPeriod(
                        start=start,
                        end=previous_date,
                        depth=float(segment.min()),
                        duration=int(len(segment)),
                        recovered=True,
                    )
                )
                start = None
            previous_date = timestamp

        if start is not None:
            segment = drawdown.loc[start:]
            periods.append(
                DrawdownPeriod(
                    start=start,
                    end=None,
                    depth=float(segment.min()),
                    duration=int(len(segment)),
                    recovered=False,
                )
            )

        return pd.DataFrame(
            [
                {
                    "start": period.start,
                    "end": period.end,
                    "depth": period.depth,
                    "duration": period.duration,
                    "recovered": period.recovered,
                }
                for period in periods
            ],
            columns=["start", "end", "depth", "duration", "recovered"],
        )

    @staticmethod
    def rolling_max_drawdown(
        returns: pd.Series,
        window: int = TRADING_DAYS_PER_YEAR,
    ) -> pd.Series:
        """Return rolling maximum drawdown over a trailing return window."""
        clean = _as_series(returns, name="returns")
        if window <= 0:
            raise ValueError("window must be positive.")
        return clean.rolling(window).apply(
            lambda values: DrawdownAnalyzer.max_drawdown(pd.Series(values)),
            raw=False,
        )

    @staticmethod
    def drawdown_scale(
        returns: pd.Series,
        max_drawdown_threshold: float = -0.15,
        risk_off_scale: float = 0.60,
        recovery_threshold: float = -0.08,
    ) -> float:
        """Return a risk scale based on current drawdown severity."""
        if max_drawdown_threshold >= 0:
            raise ValueError("max_drawdown_threshold must be negative.")
        if recovery_threshold > 0:
            raise ValueError("recovery_threshold must be non-positive.")
        if not 0.0 < risk_off_scale <= 1.0:
            raise ValueError("risk_off_scale must be in (0, 1].")

        current = DrawdownAnalyzer.current_drawdown(returns)
        if current <= max_drawdown_threshold:
            return float(risk_off_scale)
        if current >= recovery_threshold:
            return 1.0

        severity = (recovery_threshold - current) / (recovery_threshold - max_drawdown_threshold)
        return float(1.0 - severity * (1.0 - risk_off_scale))


class VolatilityModel:
    """Realized volatility estimates and volatility-targeting helpers."""

    @staticmethod
    def rolling(
        returns: pd.Series,
        window: int = 21,
        annualize: bool = True,
        periods_per_year: int = TRADING_DAYS_PER_YEAR,
    ) -> pd.Series:
        """Return rolling realized volatility."""
        clean = _as_series(returns, name="returns")
        if window <= 0:
            raise ValueError("window must be positive.")
        volatility = clean.rolling(window).std()
        return volatility * sqrt(periods_per_year) if annualize else volatility

    @staticmethod
    def ewm(
        returns: pd.Series,
        halflife: int = 21,
        annualize: bool = True,
        periods_per_year: int = TRADING_DAYS_PER_YEAR,
    ) -> pd.Series:
        """Return exponentially weighted realized volatility."""
        clean = _as_series(returns, name="returns")
        if halflife <= 0:
            raise ValueError("halflife must be positive.")
        volatility = clean.ewm(halflife=halflife, adjust=False).std(bias=False)
        volatility = volatility.bfill().fillna(clean.std(ddof=1))
        volatility = volatility.fillna(0.0)
        return volatility * sqrt(periods_per_year) if annualize else volatility

    @staticmethod
    def realized_variance(
        returns: pd.Series,
        window: int = 21,
        annualize: bool = True,
        periods_per_year: int = TRADING_DAYS_PER_YEAR,
    ) -> pd.Series:
        """Return rolling realized variance."""
        clean = _as_series(returns, name="returns")
        if window <= 0:
            raise ValueError("window must be positive.")
        variance = clean.rolling(window).var()
        return variance * periods_per_year if annualize else variance

    @staticmethod
    def scale_to_target(
        realized_volatility: float,
        target_volatility: float,
        min_scale: float = 0.0,
        max_scale: float = 1.0,
    ) -> float:
        """Return allocation scale needed to reach a volatility target."""
        if realized_volatility < 0:
            raise ValueError("realized_volatility must be non-negative.")
        if target_volatility <= 0:
            raise ValueError("target_volatility must be positive.")
        if not 0.0 <= min_scale <= max_scale:
            raise ValueError("scale bounds must satisfy 0 <= min <= max.")
        if realized_volatility <= EPSILON:
            return float(max_scale)
        scale = target_volatility / realized_volatility
        return float(np.clip(scale, min_scale, max_scale))

    @staticmethod
    def trailing_scale(
        returns: pd.Series,
        target_volatility: float,
        window: int = 63,
        min_scale: float = 0.25,
        max_scale: float = 1.0,
        periods_per_year: int = TRADING_DAYS_PER_YEAR,
    ) -> float:
        """Return a target-volatility scale from recent realized returns."""
        recent_vol = VolatilityModel.rolling(
            returns,
            window=window,
            annualize=True,
            periods_per_year=periods_per_year,
        ).dropna()
        if recent_vol.empty:
            recent_vol_value = float(_as_series(returns, name="returns").std() * sqrt(periods_per_year))
        else:
            recent_vol_value = float(recent_vol.iloc[-1])
        return VolatilityModel.scale_to_target(
            recent_vol_value,
            target_volatility,
            min_scale=min_scale,
            max_scale=max_scale,
        )


class RiskBudget:
    """Portfolio risk contribution and diversification analysis."""

    @staticmethod
    def marginal_risk_contribution(
        weights: np.ndarray,
        covariance: np.ndarray,
    ) -> np.ndarray:
        """Return marginal contribution of each asset to portfolio volatility."""
        weights_arr, covariance_arr = _validate_weights_and_covariance(weights, covariance)
        variance = float(weights_arr @ covariance_arr @ weights_arr)
        if variance <= EPSILON:
            return np.zeros_like(weights_arr)
        return (covariance_arr @ weights_arr) / sqrt(variance)

    @staticmethod
    def risk_contribution(
        weights: np.ndarray,
        covariance: np.ndarray,
        relative: bool = True,
    ) -> np.ndarray:
        """Return total risk contribution by asset."""
        weights_arr, covariance_arr = _validate_weights_and_covariance(weights, covariance)
        marginal = RiskBudget.marginal_risk_contribution(weights_arr, covariance_arr)
        contribution = weights_arr * marginal
        if not relative:
            return contribution

        total = float(contribution.sum())
        if abs(total) <= EPSILON:
            return np.zeros_like(contribution)
        return contribution / total

    @staticmethod
    def diversification_ratio(
        weights: np.ndarray,
        covariance: np.ndarray,
    ) -> float:
        """Return weighted average asset volatility divided by portfolio volatility."""
        weights_arr, covariance_arr = _validate_weights_and_covariance(weights, covariance)
        asset_volatility = np.sqrt(np.maximum(np.diag(covariance_arr), 0.0))
        weighted_asset_vol = float(weights_arr @ asset_volatility)
        portfolio_variance = float(weights_arr @ covariance_arr @ weights_arr)
        if portfolio_variance <= EPSILON:
            return 1.0
        return float(weighted_asset_vol / sqrt(portfolio_variance))

    @staticmethod
    def concentration_index(weights: np.ndarray) -> float:
        """Return the Herfindahl-Hirschman concentration index for weights."""
        weights_arr = _as_1d_array(weights, name="weights")
        return float(np.sum(np.square(weights_arr)))

    @staticmethod
    def effective_number_of_positions(weights: np.ndarray) -> float:
        """Return the inverse concentration index."""
        concentration = RiskBudget.concentration_index(weights)
        return float(1.0 / concentration) if concentration > EPSILON else 0.0


def build_risk_snapshot(
    returns: pd.Series,
    weights: np.ndarray,
    scenario_returns: np.ndarray,
    *,
    target_volatility: float = 0.12,
    alpha: float = 0.95,
    volatility_window: int = 63,
    max_drawdown_threshold: float = -0.15,
    risk_off_scale: float = 0.60,
    recovery_threshold: float = -0.08,
) -> RiskSnapshot:
    """Build a compact risk summary for reporting or dashboard display."""
    clean_returns = _as_series(returns, name="returns")
    realized_vol = VolatilityModel.trailing_scale(
        clean_returns,
        target_volatility=target_volatility,
        window=volatility_window,
        min_scale=0.0,
        max_scale=10.0,
    )
    rolling_vol = VolatilityModel.rolling(clean_returns, window=volatility_window).dropna()
    annualized_vol = float(rolling_vol.iloc[-1]) if not rolling_vol.empty else float(clean_returns.std() * sqrt(252))
    var, cvar = CVaRCalculator.historical(clean_returns, alpha=alpha)
    return RiskSnapshot(
        annualized_volatility=annualized_vol,
        current_drawdown=DrawdownAnalyzer.current_drawdown(clean_returns),
        max_drawdown=DrawdownAnalyzer.max_drawdown(clean_returns),
        historical_var=var,
        historical_cvar=CVaRCalculator.portfolio_cvar(weights, scenario_returns, alpha=alpha)
        if np.asarray(scenario_returns).ndim == 2
        else cvar,
        volatility_scale=realized_vol,
        drawdown_scale=DrawdownAnalyzer.drawdown_scale(
            clean_returns,
            max_drawdown_threshold=max_drawdown_threshold,
            risk_off_scale=risk_off_scale,
            recovery_threshold=recovery_threshold,
        ),
    )


def _validate_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")


def _as_series(values: pd.Series | np.ndarray, name: str) -> pd.Series:
    if isinstance(values, pd.Series):
        series = values.astype(float).dropna()
    else:
        series = pd.Series(_as_1d_array(values, name=name))
    if series.empty:
        raise ValueError(f"{name} cannot be empty.")
    if not np.isfinite(series.to_numpy()).all():
        raise ValueError(f"{name} contains non-finite values.")
    return series


def _as_1d_array(values: pd.Series | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _as_2d_array(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} cannot be empty.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _validate_weights_and_covariance(weights: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights_arr = _as_1d_array(weights, name="weights")
    covariance_arr = np.asarray(covariance, dtype=float)
    if covariance_arr.ndim != 2 or covariance_arr.shape[0] != covariance_arr.shape[1]:
        raise ValueError("covariance must be a square matrix.")
    if covariance_arr.shape[0] != weights_arr.size:
        raise ValueError("covariance dimensions must match weights length.")
    if not np.isfinite(covariance_arr).all():
        raise ValueError("covariance contains non-finite values.")
    covariance_arr = (covariance_arr + covariance_arr.T) / 2.0
    return weights_arr, covariance_arr


__all__ = [
    "CVaRCalculator",
    "DrawdownAnalyzer",
    "DrawdownPeriod",
    "EPSILON",
    "RiskBudget",
    "RiskSnapshot",
    "TRADING_DAYS_PER_YEAR",
    "VolatilityModel",
    "build_risk_snapshot",
]
