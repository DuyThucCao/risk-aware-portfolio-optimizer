"""Market data ingestion and covariance estimation.

This module owns the first stage of the project pipeline:

1. Download public ETF prices from Yahoo Finance through ``yfinance``.
2. Extract adjusted close prices into a clean date-by-ticker DataFrame.
3. Cache raw and processed data locally for reproducible reruns.
4. Provide a synthetic fallback so the GitHub demo still runs offline.
5. Estimate covariance matrices used by the optimizer and backtester.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    yf = None

try:
    from sklearn.covariance import LedoitWolf
except ModuleNotFoundError:  # pragma: no cover - fallback is tested indirectly
    LedoitWolf = None


logger = logging.getLogger(__name__)

DEFAULT_TICKERS: list[str] = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "VNQ"]

ETF_ASSET_CLASS: dict[str, str] = {
    "SPY": "US Large Cap Equity",
    "QQQ": "US Growth Equity",
    "IWM": "US Small Cap Equity",
    "EFA": "Developed International Equity",
    "EEM": "Emerging Markets Equity",
    "TLT": "Long Duration Treasury Bonds",
    "GLD": "Gold",
    "VNQ": "US Real Estate",
}

# Kept for compatibility with the existing optimizer/dashboard terminology.
SECTOR_MAP: dict[str, str] = ETF_ASSET_CLASS.copy()


@dataclass(frozen=True)
class PriceDataBundle:
    """Container returned by the high-level data-loading workflow."""

    prices: pd.DataFrame
    returns: pd.DataFrame
    tickers: tuple[str, ...]
    source: str


class DataLoader:
    """Download, validate, cache, and transform ETF price data.

    Args:
        cache_dir: Directory for downloaded price CSV cache files.
        use_cache: Load existing cache files before attempting a new download.
        raw_dir: Directory for raw downloaded price snapshots.
        processed_dir: Directory for cleaned price and return files.
        min_price_coverage: Minimum non-missing observation share per ticker.
        auto_adjust: Pass-through flag for ``yfinance.download``.
        fallback_to_synthetic: Use generated price data if live download fails.
        synthetic_seed: Random seed used by the offline fallback.
        root_dir: Project root used for resolving relative directories.
    """

    def __init__(
        self,
        cache_dir: str | Path = ".cache",
        use_cache: bool = True,
        raw_dir: str | Path = "data/raw",
        processed_dir: str | Path = "data/processed",
        min_price_coverage: float = 0.95,
        auto_adjust: bool = True,
        fallback_to_synthetic: bool = False,
        synthetic_seed: int = 42,
        root_dir: str | Path | None = None,
    ) -> None:
        if not 0.0 < min_price_coverage <= 1.0:
            raise ValueError("min_price_coverage must be in (0, 1].")

        self.root_dir = Path(root_dir).resolve() if root_dir is not None else Path.cwd()
        self.cache_dir = self._resolve(cache_dir)
        self.raw_dir = self._resolve(raw_dir)
        self.processed_dir = self._resolve(processed_dir)
        self.use_cache = use_cache
        self.min_price_coverage = float(min_price_coverage)
        self.auto_adjust = auto_adjust
        self.fallback_to_synthetic = fallback_to_synthetic
        self.synthetic_seed = int(synthetic_seed)
        self.logger = logging.getLogger(self.__class__.__name__)

        for directory in (self.cache_dir, self.raw_dir, self.processed_dir):
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config: object) -> "DataLoader":
        """Build a loader from ``src.config.AppConfig`` or its data section."""
        data = getattr(config, "data", config)
        root_dir = getattr(config, "root_dir", None)
        return cls(
            cache_dir=getattr(data, "cache_dir"),
            use_cache=getattr(data, "use_cache"),
            raw_dir=getattr(data, "raw_dir"),
            processed_dir=getattr(data, "processed_dir"),
            min_price_coverage=getattr(data, "min_price_coverage"),
            auto_adjust=getattr(data, "auto_adjust"),
            fallback_to_synthetic=getattr(data, "fallback_to_synthetic"),
            synthetic_seed=getattr(data, "synthetic_seed"),
            root_dir=root_dir,
        )

    def fetch_prices(
        self,
        tickers: Sequence[str] | None = None,
        start_date: str = "2018-01-01",
        end_date: str = "2025-12-31",
        *,
        auto_adjust: Optional[bool] = None,
        min_price_coverage: Optional[float] = None,
    ) -> pd.DataFrame:
        """Fetch adjusted close prices for the requested ticker universe.

        The method first checks the local CSV cache. If no cache is available,
        it attempts a live Yahoo Finance download. If that fails and synthetic
        fallback is enabled, it returns deterministic generated ETF-like data.
        """
        clean_tickers = self._clean_tickers(tickers or DEFAULT_TICKERS)
        cache_path = self._cache_path(clean_tickers, start_date, end_date)

        if self.use_cache and cache_path.exists():
            self.logger.info("Loading cached prices from %s", cache_path)
            cached = self._read_price_csv(cache_path)
            return self._validate_prices(cached, clean_tickers, min_price_coverage)

        try:
            prices = self._download_prices(
                clean_tickers,
                start_date,
                end_date,
                auto_adjust=self.auto_adjust if auto_adjust is None else auto_adjust,
            )
            prices = self._validate_prices(prices, clean_tickers, min_price_coverage)
            self._write_price_csv(prices, cache_path)
            self._write_price_csv(
                prices,
                self.raw_dir / f"raw_prices_{self._cache_key(clean_tickers)}_{start_date}_{end_date}.csv",
            )
            return prices
        except Exception as exc:
            if not self.fallback_to_synthetic:
                raise RuntimeError(
                    "Unable to load market data. Enable fallback_to_synthetic "
                    "or install dependencies and check network access."
                ) from exc

            self.logger.warning("Using synthetic fallback data because price download failed: %s", exc)
            prices = self.generate_synthetic_prices(clean_tickers, start_date, end_date)
            prices.attrs["source"] = "synthetic"
            return prices

    def load_price_data(
        self,
        tickers: Sequence[str] | None = None,
        start_date: str = "2018-01-01",
        end_date: str = "2025-12-31",
        return_method: str = "simple",
        save_processed: bool = True,
    ) -> PriceDataBundle:
        """Fetch prices, compute returns, and optionally save processed CSVs."""
        prices = self.fetch_prices(tickers, start_date, end_date)
        returns = self.compute_returns(prices, method=return_method)
        source = str(prices.attrs.get("source", "yfinance"))

        if save_processed:
            self._write_price_csv(prices, self.processed_dir / "adjusted_close_prices.csv")
            returns.to_csv(self.processed_dir / "daily_returns.csv", index_label="Date")

        return PriceDataBundle(
            prices=prices,
            returns=returns,
            tickers=tuple(prices.columns),
            source=source,
        )

    def load_from_config(self, config: object, save_processed: bool = True) -> PriceDataBundle:
        """Run the data workflow using a loaded ``AppConfig`` object."""
        data = getattr(config, "data")
        returns = getattr(config, "returns")
        prices = self.fetch_prices(
            tickers=getattr(data, "tickers"),
            start_date=getattr(data, "start_date"),
            end_date=getattr(data, "end_date"),
        )
        daily_returns = self.compute_returns(prices, method=getattr(returns, "method"))

        if save_processed:
            self._write_price_csv(prices, self._resolve(getattr(data, "processed_prices_file")))
            daily_returns.to_csv(self._resolve(getattr(data, "processed_returns_file")), index_label="Date")

        return PriceDataBundle(
            prices=prices,
            returns=daily_returns,
            tickers=tuple(prices.columns),
            source=str(prices.attrs.get("source", "yfinance")),
        )

    def compute_returns(self, prices: pd.DataFrame, method: str = "simple") -> pd.DataFrame:
        """Compute clean daily returns from adjusted close prices."""
        self._require_price_frame(prices)
        method_normalized = method.lower().strip()

        if method_normalized == "simple":
            returns = prices.pct_change()
        elif method_normalized == "log":
            returns = np.log(prices / prices.shift(1))
        else:
            raise ValueError("method must be 'simple' or 'log'.")

        returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
        if returns.empty:
            raise ValueError("Return calculation produced an empty DataFrame.")
        return returns

    def estimate_expected_returns(
        self,
        returns: pd.DataFrame,
        method: str = "historical",
        halflife: int = 63,
        annualize: bool = True,
        periods_per_year: int = 252,
    ) -> np.ndarray:
        """Estimate expected returns for the optimizer objective."""
        self._require_returns_frame(returns)
        method_normalized = method.lower().strip()

        if method_normalized == "historical":
            mu = returns.mean().to_numpy()
        elif method_normalized == "ewm":
            if halflife <= 0:
                raise ValueError("halflife must be positive.")
            mu = returns.ewm(halflife=halflife).mean().iloc[-1].to_numpy()
        else:
            raise ValueError("method must be 'historical' or 'ewm'.")

        return mu * periods_per_year if annualize else mu

    def get_sector_info(
        self,
        tickers: Sequence[str],
        sector_map: Optional[Mapping[str, str]] = None,
    ) -> tuple[np.ndarray, dict[str, int], dict[int, str]]:
        """Encode ETF asset classes as integer labels for grouped constraints."""
        clean_tickers = self._clean_tickers(tickers)
        mapping = dict(sector_map or SECTOR_MAP)
        labels = [mapping.get(ticker, "Other") for ticker in clean_tickers]
        unique_labels = sorted(set(labels))
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        id_to_label = {idx: label for label, idx in label_to_id.items()}
        encoded = np.array([label_to_id[label] for label in labels], dtype=int)
        return encoded, label_to_id, id_to_label

    def generate_synthetic_prices(
        self,
        tickers: Sequence[str] | None = None,
        start_date: str = "2018-01-01",
        end_date: str = "2025-12-31",
    ) -> pd.DataFrame:
        """Generate deterministic ETF-like prices for offline demos and tests."""
        clean_tickers = self._clean_tickers(tickers or DEFAULT_TICKERS)
        dates = pd.bdate_range(start=start_date, end=end_date)
        if len(dates) < 2:
            raise ValueError("Synthetic data requires at least two business days.")

        annual_mu = np.array([0.085, 0.105, 0.075, 0.060, 0.065, 0.030, 0.045, 0.055])
        annual_vol = np.array([0.18, 0.23, 0.24, 0.19, 0.25, 0.13, 0.16, 0.22])
        mu = np.resize(annual_mu, len(clean_tickers)) / 252.0
        vol = np.resize(annual_vol, len(clean_tickers)) / np.sqrt(252.0)

        corr = np.full((len(clean_tickers), len(clean_tickers)), 0.35)
        np.fill_diagonal(corr, 1.0)
        for i, ticker_i in enumerate(clean_tickers):
            for j, ticker_j in enumerate(clean_tickers):
                if i == j:
                    continue
                corr[i, j] = self._synthetic_correlation(ticker_i, ticker_j)

        cov = corr * np.outer(vol, vol)
        cov = nearest_psd(cov)
        rng = np.random.default_rng(self.synthetic_seed)
        daily_returns = rng.multivariate_normal(mu, cov, size=len(dates))
        daily_returns = np.clip(daily_returns, -0.12, 0.12)
        prices = 100.0 * pd.DataFrame(1.0 + daily_returns, index=dates, columns=clean_tickers).cumprod()
        prices.attrs["source"] = "synthetic"
        return prices.round(4)

    def _download_prices(
        self,
        tickers: Sequence[str],
        start_date: str,
        end_date: str,
        auto_adjust: bool,
    ) -> pd.DataFrame:
        if yf is None:
            raise RuntimeError(
                "yfinance is not installed. Install dependencies with: "
                "pip install -r requirements.txt"
            )

        self.logger.info("Downloading %d tickers from Yahoo Finance.", len(tickers))
        raw = yf.download(
            tickers=list(tickers),
            start=start_date,
            end=end_date,
            auto_adjust=auto_adjust,
            progress=False,
            group_by="column",
            threads=True,
        )

        if raw is None or raw.empty:
            raise RuntimeError("Yahoo Finance returned no rows.")

        prices = self._extract_adjusted_close(raw, tickers)
        prices.attrs["source"] = "yfinance"
        return prices

    def _extract_adjusted_close(self, raw: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
        """Extract the adjusted close field from yfinance's varied output shapes."""
        if isinstance(raw.columns, pd.MultiIndex):
            field_candidates = ["Adj Close", "Close"]
            for field in field_candidates:
                if field in raw.columns.get_level_values(0):
                    prices = raw[field].copy()
                    prices = prices.reindex(columns=list(tickers))
                    return prices
            available = sorted(set(str(level) for level in raw.columns.get_level_values(0)))
            raise ValueError(f"No close price field found in download. Available fields: {available}")

        if "Adj Close" in raw.columns:
            prices = raw[["Adj Close"]].copy()
        elif "Close" in raw.columns:
            prices = raw[["Close"]].copy()
        else:
            raise ValueError("No 'Adj Close' or 'Close' column found in Yahoo Finance response.")

        prices.columns = [tickers[0]]
        return prices

    def _validate_prices(
        self,
        prices: pd.DataFrame,
        requested_tickers: Sequence[str],
        min_price_coverage: Optional[float] = None,
    ) -> pd.DataFrame:
        self._require_price_frame(prices)
        coverage_threshold = self.min_price_coverage if min_price_coverage is None else min_price_coverage
        if not 0.0 < coverage_threshold <= 1.0:
            raise ValueError("min_price_coverage must be in (0, 1].")

        cleaned = prices.copy()
        cleaned.index = pd.to_datetime(cleaned.index)
        cleaned = cleaned.sort_index()
        cleaned = cleaned.apply(pd.to_numeric, errors="coerce")
        cleaned = cleaned.loc[:, ~cleaned.columns.duplicated()]
        cleaned = cleaned.reindex(columns=[ticker for ticker in requested_tickers if ticker in cleaned.columns])
        cleaned = cleaned.dropna(how="all")

        min_observations = int(np.ceil(len(cleaned) * coverage_threshold))
        cleaned = cleaned.dropna(axis=1, thresh=min_observations)
        cleaned = cleaned.ffill().bfill().dropna(axis=0, how="any")

        missing = [ticker for ticker in requested_tickers if ticker not in cleaned.columns]
        if missing:
            self.logger.warning("Dropped tickers with insufficient data: %s", ", ".join(missing))

        if cleaned.empty:
            raise ValueError("No usable price data remains after cleaning.")
        if cleaned.shape[1] < 2:
            raise ValueError("At least two assets are required after price cleaning.")
        if (cleaned <= 0).any().any():
            raise ValueError("Price data contains non-positive values after cleaning.")

        return cleaned

    def _require_price_frame(self, prices: pd.DataFrame) -> None:
        if not isinstance(prices, pd.DataFrame):
            raise TypeError("prices must be a pandas DataFrame.")
        if prices.empty:
            raise ValueError("prices cannot be empty.")
        if prices.shape[1] == 0:
            raise ValueError("prices must contain at least one asset column.")

    def _require_returns_frame(self, returns: pd.DataFrame) -> None:
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns must be a pandas DataFrame.")
        if returns.empty:
            raise ValueError("returns cannot be empty.")
        if not np.isfinite(returns.to_numpy()).all():
            raise ValueError("returns contains non-finite values.")

    def _read_price_csv(self, path: Path) -> pd.DataFrame:
        try:
            prices = pd.read_csv(path, index_col=0, parse_dates=True)
        except Exception as exc:
            raise RuntimeError(f"Failed to read cached price file: {path}") from exc
        prices.columns = [str(col).upper() for col in prices.columns]
        prices.attrs["source"] = "cache"
        return prices

    def _write_price_csv(self, prices: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(path, index_label="Date")

    def _cache_path(self, tickers: Sequence[str], start_date: str, end_date: str) -> Path:
        return self.cache_dir / f"prices_{self._cache_key(tickers)}_{start_date}_{end_date}.csv"

    @staticmethod
    def _cache_key(tickers: Sequence[str]) -> str:
        return "_".join(sorted(tickers))

    def _resolve(self, path: str | Path) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.root_dir / candidate

    @staticmethod
    def _clean_tickers(tickers: Sequence[str]) -> tuple[str, ...]:
        cleaned = tuple(dict.fromkeys(str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()))
        if not cleaned:
            raise ValueError("At least one ticker is required.")
        return cleaned

    @staticmethod
    def _synthetic_correlation(ticker_a: str, ticker_b: str) -> float:
        equity = {"SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ"}
        defensive = {"TLT", "GLD"}
        if ticker_a in equity and ticker_b in equity:
            return 0.68
        if ticker_a in defensive and ticker_b in defensive:
            return 0.18
        if "TLT" in {ticker_a, ticker_b}:
            return -0.20
        if "GLD" in {ticker_a, ticker_b}:
            return 0.05
        return 0.30


class CovarianceEstimator:
    """Covariance estimators used by optimization and backtesting."""

    @staticmethod
    def sample(returns: np.ndarray, annualize: bool = True) -> np.ndarray:
        """Return the sample covariance matrix."""
        matrix = _validate_return_matrix(returns)
        cov = np.cov(matrix, rowvar=False)
        cov = nearest_psd(cov)
        return cov * 252 if annualize else cov

    @staticmethod
    def ledoit_wolf(returns: np.ndarray, annualize: bool = True) -> np.ndarray:
        """Return a Ledoit-Wolf shrinkage covariance matrix.

        If scikit-learn is unavailable, the method falls back to a transparent
        diagonal-target shrinkage estimator so the offline demo remains usable.
        """
        matrix = _validate_return_matrix(returns)
        if LedoitWolf is not None:
            estimator = LedoitWolf(assume_centered=False)
            estimator.fit(matrix)
            cov = estimator.covariance_
        else:
            sample = np.cov(matrix, rowvar=False)
            diagonal_target = np.diag(np.diag(sample))
            cov = 0.50 * sample + 0.50 * diagonal_target
        cov = nearest_psd(cov)
        return cov * 252 if annualize else cov

    @staticmethod
    def ewm(
        returns: np.ndarray,
        halflife: int = 60,
        annualize: bool = True,
    ) -> np.ndarray:
        """Return an exponentially weighted covariance matrix."""
        matrix = _validate_return_matrix(returns)
        if halflife <= 0:
            raise ValueError("halflife must be positive.")

        n_obs = matrix.shape[0]
        decay = 0.5 ** (1.0 / halflife)
        weights = np.array([decay ** (n_obs - idx - 1) for idx in range(n_obs)])
        weights = weights / weights.sum()
        mean = np.sum(weights[:, None] * matrix, axis=0)
        centered = matrix - mean
        cov = (weights[:, None] * centered).T @ centered
        cov = nearest_psd(cov)
        return cov * 252 if annualize else cov

    @staticmethod
    def constant_corr(returns: np.ndarray, annualize: bool = True) -> np.ndarray:
        """Return a constant-correlation shrinkage covariance matrix."""
        matrix = _validate_return_matrix(returns)
        sample = np.cov(matrix, rowvar=False)
        std = np.sqrt(np.maximum(np.diag(sample), 0.0))
        outer_std = np.outer(std, std)
        corr = np.divide(sample, outer_std, out=np.zeros_like(sample), where=outer_std > 0)
        np.fill_diagonal(corr, 1.0)

        n_assets = corr.shape[0]
        avg_corr = (corr.sum() - n_assets) / max(n_assets * (n_assets - 1), 1)
        target_corr = np.full_like(corr, avg_corr)
        np.fill_diagonal(target_corr, 1.0)

        shrunk_corr = 0.50 * corr + 0.50 * target_corr
        cov = shrunk_corr * outer_std
        cov = nearest_psd(cov)
        return cov * 252 if annualize else cov


def nearest_psd(matrix: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Project a symmetric matrix to the nearest positive semidefinite matrix."""
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("matrix must be square.")

    symmetric = (arr + arr.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(symmetric)
    clipped = np.maximum(eigvals, epsilon)
    projected = eigvecs @ np.diag(clipped) @ eigvecs.T
    return (projected + projected.T) / 2.0


def _validate_return_matrix(returns: np.ndarray) -> np.ndarray:
    matrix = np.asarray(returns, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("returns must be a two-dimensional array.")
    if matrix.shape[0] < 2:
        raise ValueError("at least two observations are required.")
    if matrix.shape[1] < 1:
        raise ValueError("at least one asset is required.")
    if not np.isfinite(matrix).all():
        raise ValueError("returns contains non-finite values.")
    return matrix


__all__ = [
    "CovarianceEstimator",
    "DEFAULT_TICKERS",
    "DataLoader",
    "ETF_ASSET_CLASS",
    "PriceDataBundle",
    "SECTOR_MAP",
    "nearest_psd",
]
