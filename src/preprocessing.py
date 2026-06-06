"""Price and return preprocessing utilities.

The data loader is responsible for fetching prices. This module is responsible
for making those prices safe for modeling: sorted dates, numeric values,
coverage checks, daily return calculation, and aligned train/test windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd


ReturnMethod = Literal["simple", "log"]


@dataclass(frozen=True)
class PreprocessingReport:
    """Summary of the cleaning decisions applied to a price panel."""

    start_date: pd.Timestamp
    end_date: pd.Timestamp
    observations: int
    assets_before: int
    assets_after: int
    dropped_assets: tuple[str, ...]
    missing_fraction_after: float


@dataclass(frozen=True)
class ProcessedMarketData:
    """Clean prices, daily returns, and an audit-friendly cleaning report."""

    prices: pd.DataFrame
    returns: pd.DataFrame
    report: PreprocessingReport


def clean_price_data(
    prices: pd.DataFrame,
    *,
    min_coverage: float = 0.95,
    max_missing_fraction: float = 0.05,
    min_assets: int = 3,
) -> tuple[pd.DataFrame, PreprocessingReport]:
    """Clean an adjusted-close price panel.

    Args:
        prices: Date-indexed adjusted close prices with one column per asset.
        min_coverage: Minimum fraction of non-missing observations per asset.
        max_missing_fraction: Maximum missing share allowed after filling.
        min_assets: Minimum number of assets that must survive cleaning.

    Returns:
        Cleaned prices and a report describing the cleaning result.
    """
    _require_price_frame(prices)
    if not 0.0 < min_coverage <= 1.0:
        raise ValueError("min_coverage must be in (0, 1].")
    if not 0.0 <= max_missing_fraction < 1.0:
        raise ValueError("max_missing_fraction must be in [0, 1).")
    if min_assets < 1:
        raise ValueError("min_assets must be at least 1.")

    original_columns = tuple(str(col).strip().upper() for col in prices.columns)
    cleaned = prices.copy()
    cleaned.columns = original_columns
    cleaned.index = pd.to_datetime(cleaned.index)
    cleaned = cleaned.sort_index()
    cleaned = cleaned.loc[~cleaned.index.duplicated(keep="last")]
    cleaned = cleaned.apply(pd.to_numeric, errors="coerce")
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.dropna(how="all")
    cleaned = cleaned.loc[:, ~cleaned.columns.duplicated()]

    min_non_missing = int(np.ceil(len(cleaned) * min_coverage))
    cleaned = cleaned.dropna(axis=1, thresh=min_non_missing)
    cleaned = cleaned.ffill().bfill()
    cleaned = cleaned.dropna(axis=0, how="any")

    if cleaned.shape[1] < min_assets:
        raise ValueError(
            f"Only {cleaned.shape[1]} assets remain after cleaning; "
            f"at least {min_assets} are required."
        )
    if cleaned.empty:
        raise ValueError("No price observations remain after cleaning.")
    if (cleaned <= 0).any().any():
        raise ValueError("Cleaned prices contain non-positive values.")

    missing_fraction = float(cleaned.isna().mean().mean())
    if missing_fraction > max_missing_fraction:
        raise ValueError(
            f"Missing fraction after cleaning is {missing_fraction:.2%}, "
            f"above the allowed {max_missing_fraction:.2%}."
        )

    dropped = tuple(col for col in original_columns if col not in cleaned.columns)
    report = PreprocessingReport(
        start_date=pd.Timestamp(cleaned.index.min()),
        end_date=pd.Timestamp(cleaned.index.max()),
        observations=int(len(cleaned)),
        assets_before=len(original_columns),
        assets_after=int(cleaned.shape[1]),
        dropped_assets=dropped,
        missing_fraction_after=missing_fraction,
    )
    return cleaned, report


def calculate_returns(prices: pd.DataFrame, method: ReturnMethod = "simple") -> pd.DataFrame:
    """Calculate daily simple or log returns from cleaned prices."""
    _require_price_frame(prices)
    if method == "simple":
        returns = prices.pct_change()
    elif method == "log":
        returns = np.log(prices / prices.shift(1))
    else:
        raise ValueError("method must be 'simple' or 'log'.")

    returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if returns.empty:
        raise ValueError("Return calculation produced no rows.")
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Returns contain non-finite values.")
    return returns


def preprocess_market_data(
    prices: pd.DataFrame,
    *,
    return_method: ReturnMethod = "simple",
    min_coverage: float = 0.95,
    max_missing_fraction: float = 0.05,
    min_assets: int = 3,
) -> ProcessedMarketData:
    """Clean prices and compute model-ready daily returns."""
    clean_prices, report = clean_price_data(
        prices,
        min_coverage=min_coverage,
        max_missing_fraction=max_missing_fraction,
        min_assets=min_assets,
    )
    returns = calculate_returns(clean_prices, method=return_method)
    clean_prices, returns = align_price_and_return_index(clean_prices, returns)
    return ProcessedMarketData(prices=clean_prices, returns=returns, report=report)


def align_price_and_return_index(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align price and return panels to the same asset columns and return dates."""
    _require_price_frame(prices)
    _require_return_frame(returns)

    common_columns = [col for col in prices.columns if col in returns.columns]
    if not common_columns:
        raise ValueError("Prices and returns do not share any asset columns.")

    aligned_returns = returns.loc[:, common_columns].copy()
    aligned_prices = prices.loc[aligned_returns.index, common_columns].copy()
    if aligned_prices.empty or aligned_returns.empty:
        raise ValueError("Aligned price/return data is empty.")
    return aligned_prices, aligned_returns


def split_train_test(
    returns: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split returns into chronological train and test sets."""
    _require_return_frame(returns)
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1).")

    split_idx = int(len(returns) * train_fraction)
    if split_idx < 1 or split_idx >= len(returns):
        raise ValueError("Not enough rows to create non-empty train and test sets.")
    return returns.iloc[:split_idx].copy(), returns.iloc[split_idx:].copy()


def rolling_window_slices(
    data: pd.DataFrame,
    *,
    estimation_window: int,
    step: int,
) -> Iterable[tuple[pd.Timestamp, pd.DataFrame]]:
    """Yield chronological rolling windows ending before each rebalance date."""
    _require_return_frame(data)
    if estimation_window <= 0:
        raise ValueError("estimation_window must be positive.")
    if step <= 0:
        raise ValueError("step must be positive.")
    if len(data) <= estimation_window:
        raise ValueError("data length must exceed estimation_window.")

    for end_idx in range(estimation_window, len(data), step):
        rebalance_date = pd.Timestamp(data.index[end_idx])
        window = data.iloc[end_idx - estimation_window : end_idx].copy()
        yield rebalance_date, window


def save_processed_data(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    prices_path: str | Path,
    returns_path: str | Path,
) -> None:
    """Save cleaned prices and returns to CSV files."""
    _require_price_frame(prices)
    _require_return_frame(returns)

    prices_output = Path(prices_path)
    returns_output = Path(returns_path)
    prices_output.parent.mkdir(parents=True, exist_ok=True)
    returns_output.parent.mkdir(parents=True, exist_ok=True)

    prices.to_csv(prices_output, index_label="Date")
    returns.to_csv(returns_output, index_label="Date")


def load_processed_data(
    *,
    prices_path: str | Path,
    returns_path: str | Path,
) -> ProcessedMarketData:
    """Load previously saved processed prices and returns."""
    prices_file = Path(prices_path)
    returns_file = Path(returns_path)
    if not prices_file.exists():
        raise FileNotFoundError(f"Processed prices file not found: {prices_file}")
    if not returns_file.exists():
        raise FileNotFoundError(f"Processed returns file not found: {returns_file}")

    prices = pd.read_csv(prices_file, index_col=0, parse_dates=True)
    returns = pd.read_csv(returns_file, index_col=0, parse_dates=True)
    clean_prices, report = clean_price_data(prices, min_assets=1)
    _, clean_returns = align_price_and_return_index(clean_prices, returns)
    return ProcessedMarketData(prices=clean_prices, returns=clean_returns, report=report)


def winsorize_returns(
    returns: pd.DataFrame,
    *,
    lower_quantile: float = 0.001,
    upper_quantile: float = 0.999,
) -> pd.DataFrame:
    """Cap extreme return observations without changing the return index."""
    _require_return_frame(returns)
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1.")

    lower = returns.quantile(lower_quantile)
    upper = returns.quantile(upper_quantile)
    return returns.clip(lower=lower, upper=upper, axis=1)


def select_columns(data: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Select an ordered asset subset with a clear error for missing columns."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    requested = [str(col).strip().upper() for col in columns]
    missing = [col for col in requested if col not in data.columns]
    if missing:
        raise KeyError(f"Missing requested columns: {missing}")
    return data.loc[:, requested].copy()


def _require_price_frame(prices: pd.DataFrame) -> None:
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")
    if prices.empty:
        raise ValueError("prices cannot be empty.")
    if prices.shape[1] == 0:
        raise ValueError("prices must contain at least one asset column.")


def _require_return_frame(returns: pd.DataFrame) -> None:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")
    if returns.empty:
        raise ValueError("returns cannot be empty.")
    if returns.shape[1] == 0:
        raise ValueError("returns must contain at least one asset column.")
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("returns contains non-finite values.")


__all__ = [
    "PreprocessingReport",
    "ProcessedMarketData",
    "ReturnMethod",
    "align_price_and_return_index",
    "calculate_returns",
    "clean_price_data",
    "load_processed_data",
    "preprocess_market_data",
    "rolling_window_slices",
    "save_processed_data",
    "select_columns",
    "split_train_test",
    "winsorize_returns",
]
