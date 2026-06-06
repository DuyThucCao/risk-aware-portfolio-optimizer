"""Configuration loading and validation for the portfolio optimizer project.

The YAML file is intentionally readable for a recruiter or reviewer. This
module converts it into typed Python objects so the rest of the project can
use validated settings instead of raw dictionaries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Optional

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised only in incomplete envs
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default_config.yaml"


@dataclass(frozen=True)
class ProjectConfig:
    """High-level project metadata used in reports and README output."""

    name: str
    tagline: str
    base_currency: str
    initial_portfolio_value: float

    def __post_init__(self) -> None:
        if self.initial_portfolio_value <= 0:
            raise ValueError("project.initial_portfolio_value must be positive.")


@dataclass(frozen=True)
class DataConfig:
    """Market data settings for downloading and caching ETF prices."""

    tickers: tuple[str, ...]
    start_date: str
    end_date: str
    price_field: str
    auto_adjust: bool
    use_cache: bool
    raw_dir: str
    processed_dir: str
    cache_dir: str
    processed_prices_file: str
    processed_returns_file: str
    min_price_coverage: float
    fallback_to_synthetic: bool
    synthetic_seed: int

    def __post_init__(self) -> None:
        if len(self.tickers) < 3:
            raise ValueError("data.tickers must include at least three assets.")
        if len(set(self.tickers)) != len(self.tickers):
            raise ValueError("data.tickers must not contain duplicates.")
        if _parse_iso_date(self.start_date) >= _parse_iso_date(self.end_date):
            raise ValueError("data.start_date must be before data.end_date.")
        if not 0.0 < self.min_price_coverage <= 1.0:
            raise ValueError("data.min_price_coverage must be in (0, 1].")


@dataclass(frozen=True)
class ReturnsConfig:
    """Return calculation and performance annualization assumptions."""

    method: str
    periods_per_year: int
    risk_free_rate: float

    def __post_init__(self) -> None:
        if self.method not in {"simple", "log"}:
            raise ValueError("returns.method must be 'simple' or 'log'.")
        if self.periods_per_year <= 0:
            raise ValueError("returns.periods_per_year must be positive.")


@dataclass(frozen=True)
class ExpectedReturnsConfig:
    """Expected return estimator settings."""

    method: str
    lookback_days: int
    ewm_halflife: int
    annualize: bool

    def __post_init__(self) -> None:
        if self.method not in {"historical", "ewm"}:
            raise ValueError("expected_returns.method must be 'historical' or 'ewm'.")
        if self.lookback_days <= 0 or self.ewm_halflife <= 0:
            raise ValueError("expected return lookback and halflife must be positive.")


@dataclass(frozen=True)
class CovarianceConfig:
    """Covariance estimator settings."""

    estimator: str
    lookback_days: int
    ewm_halflife: int
    shrinkage_floor: float

    def __post_init__(self) -> None:
        valid = {"sample", "ledoit_wolf", "ewm", "constant_corr"}
        if self.estimator not in valid:
            raise ValueError(f"covariance.estimator must be one of {sorted(valid)}.")
        if self.lookback_days <= 0 or self.ewm_halflife <= 0:
            raise ValueError("covariance lookback and halflife must be positive.")
        if self.shrinkage_floor < 0:
            raise ValueError("covariance.shrinkage_floor must be non-negative.")


@dataclass(frozen=True)
class BacktestSettings:
    """Walk-forward backtest settings."""

    start_after_days: int
    estimation_window: int
    rebalance_frequency: str
    rebalance_frequency_days: int
    cvar_scenario_window: int
    benchmark_name: str
    include_cash: bool

    def __post_init__(self) -> None:
        if self.start_after_days <= 0:
            raise ValueError("backtest.start_after_days must be positive.")
        if self.estimation_window <= 0:
            raise ValueError("backtest.estimation_window must be positive.")
        if self.rebalance_frequency_days <= 0:
            raise ValueError("backtest.rebalance_frequency_days must be positive.")
        if self.cvar_scenario_window <= 0:
            raise ValueError("backtest.cvar_scenario_window must be positive.")


@dataclass(frozen=True)
class DrawdownControlConfig:
    """Drawdown-aware risk scaling settings for the risk-aware strategy."""

    enabled: bool = False
    max_drawdown_threshold: float = -0.15
    risk_off_scale: float = 0.60
    recovery_threshold: float = -0.08

    def __post_init__(self) -> None:
        if self.enabled:
            if self.max_drawdown_threshold >= 0:
                raise ValueError("drawdown threshold must be negative.")
            if not 0.0 < self.risk_off_scale <= 1.0:
                raise ValueError("risk_off_scale must be in (0, 1].")
            if self.recovery_threshold > 0:
                raise ValueError("recovery_threshold must be non-positive.")


@dataclass(frozen=True)
class VolatilityControlConfig:
    """Trailing realized-volatility scaling settings."""

    enabled: bool = False
    target_volatility: Optional[float] = None
    trailing_window: int = 63
    min_scale: float = 0.25
    max_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.enabled:
            if self.target_volatility is None or self.target_volatility <= 0:
                raise ValueError("volatility_control.target_volatility must be positive.")
            if self.trailing_window <= 0:
                raise ValueError("volatility_control.trailing_window must be positive.")
            if not 0.0 < self.min_scale <= self.max_scale:
                raise ValueError("volatility scale bounds must satisfy 0 < min <= max.")


@dataclass(frozen=True)
class StrategyConfig:
    """Configuration for one portfolio strategy."""

    key: str
    name: str
    enabled: bool
    rebalance: bool = True
    risk_aversion: float = 1.0
    min_weight: float = 0.0
    max_weight: float = 1.0
    target_volatility: Optional[float] = None
    max_turnover: Optional[float] = None
    max_leverage: float = 1.0
    linear_transaction_cost: float = 0.0
    quadratic_transaction_cost: float = 0.0
    cvar_alpha: float = 0.95
    cvar_limit: Optional[float] = None
    cash_return: float = 0.0
    drawdown_control: DrawdownControlConfig = field(default_factory=DrawdownControlConfig)
    volatility_control: VolatilityControlConfig = field(default_factory=VolatilityControlConfig)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError(f"strategies.{self.key}.name cannot be empty.")
        if self.risk_aversion < 0:
            raise ValueError(f"strategies.{self.key}.risk_aversion must be non-negative.")
        if self.max_weight <= 0:
            raise ValueError(f"strategies.{self.key}.max_weight must be positive.")
        if self.min_weight > self.max_weight:
            raise ValueError(f"strategies.{self.key}.min_weight cannot exceed max_weight.")
        if self.target_volatility is not None and self.target_volatility <= 0:
            raise ValueError(f"strategies.{self.key}.target_volatility must be positive.")
        if self.max_turnover is not None and self.max_turnover <= 0:
            raise ValueError(f"strategies.{self.key}.max_turnover must be positive.")
        if self.max_leverage <= 0:
            raise ValueError(f"strategies.{self.key}.max_leverage must be positive.")
        if self.linear_transaction_cost < 0 or self.quadratic_transaction_cost < 0:
            raise ValueError(f"strategies.{self.key} transaction costs must be non-negative.")
        if not 0.0 < self.cvar_alpha < 1.0:
            raise ValueError(f"strategies.{self.key}.cvar_alpha must be in (0, 1).")
        if self.cvar_limit is not None and self.cvar_limit <= 0:
            raise ValueError(f"strategies.{self.key}.cvar_limit must be positive.")


@dataclass(frozen=True)
class ConstraintsConfig:
    """Global portfolio construction constraints."""

    long_only: bool
    fully_invested: bool
    allow_cash_buffer: bool
    min_cash_weight: float
    max_cash_weight: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_cash_weight <= self.max_cash_weight <= 1.0:
            raise ValueError("cash weight bounds must satisfy 0 <= min <= max <= 1.")


@dataclass(frozen=True)
class ReportingConfig:
    """Output locations for metrics, reports, and saved images."""

    images_dir: str
    reports_dir: str
    metrics_file: str
    project_report_file: str
    chart_format: str
    dpi: int
    save_charts: bool
    charts: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.chart_format not in {"png", "pdf", "svg"}:
            raise ValueError("reporting.chart_format must be 'png', 'pdf', or 'svg'.")
        if self.dpi <= 0:
            raise ValueError("reporting.dpi must be positive.")
        if not self.charts:
            raise ValueError("reporting.charts cannot be empty.")


@dataclass(frozen=True)
class ValidationConfig:
    """Validation thresholds shared by data and pipeline tests."""

    min_observations: int
    max_missing_fraction: float
    weight_sum_tolerance: float
    min_assets: int

    def __post_init__(self) -> None:
        if self.min_observations <= 0:
            raise ValueError("validation.min_observations must be positive.")
        if not 0.0 <= self.max_missing_fraction < 1.0:
            raise ValueError("validation.max_missing_fraction must be in [0, 1).")
        if self.weight_sum_tolerance <= 0:
            raise ValueError("validation.weight_sum_tolerance must be positive.")
        if self.min_assets < 2:
            raise ValueError("validation.min_assets must be at least 2.")


@dataclass(frozen=True)
class LoggingConfig:
    """Logging settings for scripts and dashboards."""

    level: str
    format: str


@dataclass(frozen=True)
class AppConfig:
    """Top-level project configuration."""

    project: ProjectConfig
    data: DataConfig
    returns: ReturnsConfig
    expected_returns: ExpectedReturnsConfig
    covariance: CovarianceConfig
    backtest: BacktestSettings
    strategies: Mapping[str, StrategyConfig]
    constraints: ConstraintsConfig
    reporting: ReportingConfig
    validation: ValidationConfig
    logging: LoggingConfig
    root_dir: Path = PROJECT_ROOT
    raw: Mapping[str, Any] = field(default_factory=dict)

    def strategy(self, key: str) -> StrategyConfig:
        """Return a strategy by key with a clear error for invalid names."""
        try:
            return self.strategies[key]
        except KeyError as exc:
            available = ", ".join(sorted(self.strategies))
            raise KeyError(f"Unknown strategy '{key}'. Available: {available}") from exc

    def resolve_path(self, path: str | Path) -> Path:
        """Resolve a project-relative path against the repository root."""
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.root_dir / candidate

    def ensure_directories(self) -> None:
        """Create local output folders needed by the project pipeline."""
        dirs = {
            self.data.raw_dir,
            self.data.processed_dir,
            self.data.cache_dir,
            self.reporting.images_dir,
            self.reporting.reports_dir,
        }
        for directory in dirs:
            self.resolve_path(directory).mkdir(parents=True, exist_ok=True)

    def setup_logging(self) -> None:
        """Configure process logging from the YAML settings."""
        level = getattr(logging, self.logging.level.upper(), logging.INFO)
        logging.basicConfig(level=level, format=self.logging.format)

    def to_optimizer_config(self, strategy_key: str = "risk_aware") -> Any:
        """Convert one strategy into the existing optimizer dataclass."""
        from src.optimization.optimizer import OptimizationConfig

        strategy = self.strategy(strategy_key)
        return OptimizationConfig(
            risk_aversion=strategy.risk_aversion,
            linear_cost=strategy.linear_transaction_cost,
            market_impact_coef=strategy.quadratic_transaction_cost,
            use_market_impact=strategy.quadratic_transaction_cost > 0,
            min_weight=strategy.min_weight,
            max_weight=strategy.max_weight,
            max_turnover=strategy.max_turnover,
            target_volatility=strategy.target_volatility,
            max_leverage=strategy.max_leverage,
            cvar_limit=strategy.cvar_limit,
            cvar_alpha=strategy.cvar_alpha,
            periods_per_year=self.returns.periods_per_year,
            solver="CLARABEL",
            verbose=False,
        )

    def to_backtest_config(self) -> Any:
        """Convert project settings into the existing backtester dataclass."""
        from src.backtesting.backtester import BacktestConfig

        return BacktestConfig(
            estimation_window=self.backtest.estimation_window,
            rebalance_freq=self.backtest.rebalance_frequency_days,
            cvar_scenario_window=self.backtest.cvar_scenario_window,
            cov_estimator=self.covariance.estimator,
            mu_method=self.expected_returns.method,
            mu_halflife=self.expected_returns.ewm_halflife,
            return_method=self.returns.method,
            risk_free_rate=self.returns.risk_free_rate,
        )


def load_config(
    path: str | Path | None = None,
    root_dir: str | Path | None = None,
) -> AppConfig:
    """Load, validate, and return the project configuration."""
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to load configuration files. "
            "Install project dependencies with: pip install -r requirements.txt"
        )

    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, Mapping):
        raise ValueError("Configuration root must be a mapping.")

    root = Path(root_dir).resolve() if root_dir is not None else config_path.parents[1]
    return _build_config(raw, root)


def _build_config(raw: Mapping[str, Any], root_dir: Path) -> AppConfig:
    project = _section(raw, "project")
    data = _section(raw, "data")
    returns = _section(raw, "returns")
    expected_returns = _section(raw, "expected_returns")
    covariance = _section(raw, "covariance")
    backtest = _section(raw, "backtest")
    strategies = _section(raw, "strategies")
    constraints = _section(raw, "constraints")
    reporting = _section(raw, "reporting")
    validation = _section(raw, "validation")
    logging_config = _section(raw, "logging")

    parsed_strategies = {
        key: _parse_strategy(key, value)
        for key, value in strategies.items()
    }

    if not any(strategy.enabled for strategy in parsed_strategies.values()):
        raise ValueError("At least one strategy must be enabled.")

    return AppConfig(
        project=ProjectConfig(
            name=_as_str(project, "name"),
            tagline=_as_str(project, "tagline"),
            base_currency=_as_str(project, "base_currency"),
            initial_portfolio_value=_as_float(project, "initial_portfolio_value"),
        ),
        data=DataConfig(
            tickers=_parse_tickers(data),
            start_date=_as_str(data, "start_date"),
            end_date=_as_str(data, "end_date"),
            price_field=_as_str(data, "price_field", default="Close"),
            auto_adjust=_as_bool(data, "auto_adjust", default=True),
            use_cache=_as_bool(data, "use_cache", default=True),
            raw_dir=_as_str(data, "raw_dir"),
            processed_dir=_as_str(data, "processed_dir"),
            cache_dir=_as_str(data, "cache_dir"),
            processed_prices_file=_as_str(data, "processed_prices_file"),
            processed_returns_file=_as_str(data, "processed_returns_file"),
            min_price_coverage=_as_float(data, "min_price_coverage"),
            fallback_to_synthetic=_as_bool(data, "fallback_to_synthetic", default=True),
            synthetic_seed=_as_int(data, "synthetic_seed", default=42),
        ),
        returns=ReturnsConfig(
            method=_as_str(returns, "method"),
            periods_per_year=_as_int(returns, "periods_per_year"),
            risk_free_rate=_as_float(returns, "risk_free_rate", default=0.0),
        ),
        expected_returns=ExpectedReturnsConfig(
            method=_as_str(expected_returns, "method"),
            lookback_days=_as_int(expected_returns, "lookback_days"),
            ewm_halflife=_as_int(expected_returns, "ewm_halflife"),
            annualize=_as_bool(expected_returns, "annualize", default=True),
        ),
        covariance=CovarianceConfig(
            estimator=_as_str(covariance, "estimator"),
            lookback_days=_as_int(covariance, "lookback_days"),
            ewm_halflife=_as_int(covariance, "ewm_halflife"),
            shrinkage_floor=_as_float(covariance, "shrinkage_floor", default=0.0),
        ),
        backtest=BacktestSettings(
            start_after_days=_as_int(backtest, "start_after_days"),
            estimation_window=_as_int(backtest, "estimation_window"),
            rebalance_frequency=_as_str(backtest, "rebalance_frequency"),
            rebalance_frequency_days=_as_int(backtest, "rebalance_frequency_days"),
            cvar_scenario_window=_as_int(backtest, "cvar_scenario_window"),
            benchmark_name=_as_str(backtest, "benchmark_name"),
            include_cash=_as_bool(backtest, "include_cash", default=True),
        ),
        strategies=parsed_strategies,
        constraints=ConstraintsConfig(
            long_only=_as_bool(constraints, "long_only", default=True),
            fully_invested=_as_bool(constraints, "fully_invested", default=True),
            allow_cash_buffer=_as_bool(constraints, "allow_cash_buffer", default=True),
            min_cash_weight=_as_float(constraints, "min_cash_weight", default=0.0),
            max_cash_weight=_as_float(constraints, "max_cash_weight", default=1.0),
        ),
        reporting=ReportingConfig(
            images_dir=_as_str(reporting, "images_dir"),
            reports_dir=_as_str(reporting, "reports_dir"),
            metrics_file=_as_str(reporting, "metrics_file"),
            project_report_file=_as_str(reporting, "project_report_file"),
            chart_format=_as_str(reporting, "chart_format"),
            dpi=_as_int(reporting, "dpi"),
            save_charts=_as_bool(reporting, "save_charts", default=True),
            charts=_section(reporting, "charts"),
        ),
        validation=ValidationConfig(
            min_observations=_as_int(validation, "min_observations"),
            max_missing_fraction=_as_float(validation, "max_missing_fraction"),
            weight_sum_tolerance=_as_float(validation, "weight_sum_tolerance"),
            min_assets=_as_int(validation, "min_assets"),
        ),
        logging=LoggingConfig(
            level=_as_str(logging_config, "level", default="INFO"),
            format=_as_str(
                logging_config,
                "format",
                default="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            ),
        ),
        root_dir=root_dir,
        raw=raw,
    )


def _parse_strategy(key: str, raw_value: Any) -> StrategyConfig:
    data = _ensure_mapping(raw_value, f"strategies.{key}")
    drawdown = _ensure_mapping(data.get("drawdown_control", {}), f"strategies.{key}.drawdown_control")
    volatility = _ensure_mapping(data.get("volatility_control", {}), f"strategies.{key}.volatility_control")

    return StrategyConfig(
        key=key,
        name=_as_str(data, "name", default=key.replace("_", " ").title()),
        enabled=_as_bool(data, "enabled", default=True),
        rebalance=_as_bool(data, "rebalance", default=True),
        risk_aversion=_as_float(data, "risk_aversion", default=1.0),
        min_weight=_as_float(data, "min_weight", default=0.0),
        max_weight=_as_float(data, "max_weight", default=1.0),
        target_volatility=_as_optional_float(data, "target_volatility"),
        max_turnover=_as_optional_float(data, "max_turnover"),
        max_leverage=_as_float(data, "max_leverage", default=1.0),
        linear_transaction_cost=_as_float(data, "linear_transaction_cost", default=0.0),
        quadratic_transaction_cost=_as_float(data, "quadratic_transaction_cost", default=0.0),
        cvar_alpha=_as_float(data, "cvar_alpha", default=0.95),
        cvar_limit=_as_optional_float(data, "cvar_limit"),
        cash_return=_as_float(data, "cash_return", default=0.0),
        drawdown_control=DrawdownControlConfig(
            enabled=_as_bool(drawdown, "enabled", default=False),
            max_drawdown_threshold=_as_float(drawdown, "max_drawdown_threshold", default=-0.15),
            risk_off_scale=_as_float(drawdown, "risk_off_scale", default=0.60),
            recovery_threshold=_as_float(drawdown, "recovery_threshold", default=-0.08),
        ),
        volatility_control=VolatilityControlConfig(
            enabled=_as_bool(volatility, "enabled", default=False),
            target_volatility=_as_optional_float(volatility, "target_volatility"),
            trailing_window=_as_int(volatility, "trailing_window", default=63),
            min_scale=_as_float(volatility, "min_scale", default=0.25),
            max_scale=_as_float(volatility, "max_scale", default=1.0),
        ),
    )


def _parse_tickers(data: Mapping[str, Any]) -> tuple[str, ...]:
    value = data.get("tickers")
    if not isinstance(value, list):
        raise ValueError("data.tickers must be a list of ticker symbols.")

    tickers = tuple(dict.fromkeys(str(ticker).strip().upper() for ticker in value if str(ticker).strip()))
    if not tickers:
        raise ValueError("data.tickers cannot be empty.")
    return tickers


def _section(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    return _ensure_mapping(raw.get(key), key)


def _ensure_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return value


def _as_str(data: Mapping[str, Any], key: str, default: Optional[str] = None) -> str:
    value = data.get(key, default)
    if value is None:
        raise ValueError(f"Missing required string field: {key}")
    return str(value)


def _as_bool(data: Mapping[str, Any], key: str, default: Optional[bool] = None) -> bool:
    value = data.get(key, default)
    if isinstance(value, bool):
        return value
    raise ValueError(f"{key} must be true or false.")


def _as_int(data: Mapping[str, Any], key: str, default: Optional[int] = None) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{key} must be an integer.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer.") from exc


def _as_float(data: Mapping[str, Any], key: str, default: Optional[float] = None) -> float:
    value = data.get(key, default)
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{key} must be numeric.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric.") from exc


def _as_optional_float(data: Mapping[str, Any], key: str) -> Optional[float]:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{key} must be numeric or null.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric or null.") from exc


def _parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid ISO date: {value}") from exc


__all__ = [
    "AppConfig",
    "BacktestSettings",
    "ConstraintsConfig",
    "CovarianceConfig",
    "DataConfig",
    "DEFAULT_CONFIG_PATH",
    "DrawdownControlConfig",
    "ExpectedReturnsConfig",
    "LoggingConfig",
    "PROJECT_ROOT",
    "ProjectConfig",
    "ReportingConfig",
    "ReturnsConfig",
    "StrategyConfig",
    "ValidationConfig",
    "VolatilityControlConfig",
    "load_config",
]
