"""Visualization and reporting utilities for portfolio backtests.

The reporter writes the project's required charts into the ``images/`` folder:

1. Cumulative returns chart.
2. Portfolio weights over time.
3. Drawdown chart.
4. Volatility comparison chart.
5. Risk-return comparison chart.
6. Turnover and transaction-cost analysis chart.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Mapping, Optional

if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path(tempfile.gettempdir()) / "risk_aware_optimizer_mpl"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtesting.backtester import BacktestResult, MultiStrategyBacktestResult
from src.backtesting.metrics import (
    annualized_return,
    annualized_volatility,
    cumulative_returns,
    drawdown_series,
    format_metrics_for_display,
)
from src.optimization.optimizer import OptimizationResult
from src.risk.risk_models import RiskBudget, VolatilityModel


logger = logging.getLogger(__name__)

PALETTE = {
    "Risk-Aware Optimizer": "#1565C0",
    "Mean-Variance Optimization": "#C62828",
    "Equal Weight": "#2E7D32",
    "SPY": "#1F77B4",
    "QQQ": "#FF7F0E",
    "IWM": "#2CA02C",
    "EFA": "#D62728",
    "EEM": "#9467BD",
    "TLT": "#8C564B",
    "GLD": "#BCBD22",
    "VNQ": "#17BECF",
    "Cash": "#7F7F7F",
    "neutral": "#5F6368",
    "drawdown_fill": "#F4A6A6",
    "cost": "#B26A00",
}

REQUIRED_CHART_FILENAMES = {
    "cumulative_returns": "cumulative_returns.png",
    "portfolio_weights": "portfolio_weights.png",
    "drawdown": "drawdown.png",
    "volatility_comparison": "volatility_comparison.png",
    "risk_return_comparison": "risk_return_comparison.png",
    "turnover_transaction_costs": "turnover_transaction_costs.png",
}

ResultsInput = MultiStrategyBacktestResult | Mapping[str, BacktestResult]


class PortfolioReporter:
    """Generate publication-ready charts and metrics tables.

    Args:
        output_dir: Directory where charts are saved. Defaults to ``images``.
        fmt: File format for saved charts.
        dpi: Image resolution.
        style: Matplotlib style name.
    """

    def __init__(
        self,
        output_dir: str | Path = "images",
        fmt: str = "png",
        dpi: int = 150,
        style: str = "seaborn-v0_8-whitegrid",
    ) -> None:
        if fmt not in {"png", "pdf", "svg"}:
            raise ValueError("fmt must be 'png', 'pdf', or 'svg'.")
        if dpi <= 0:
            raise ValueError("dpi must be positive.")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fmt = fmt
        self.dpi = dpi
        self.logger = logging.getLogger(self.__class__.__name__)

        try:
            plt.style.use(style)
        except OSError:
            plt.style.use("default")

    def generate_all_charts(
        self,
        results: ResultsInput,
        *,
        primary_strategy: str = "Risk-Aware Optimizer",
        focus_strategy: Optional[str] = None,
    ) -> dict[str, Path]:
        """Generate the six required project charts and return their paths."""
        result_map = self._coerce_results(results)
        focus_name = focus_strategy or primary_strategy
        focus_result = self._focus_result(result_map, focus_name)

        chart_paths = {
            "cumulative_returns": self.plot_cumulative_returns(result_map),
            "portfolio_weights": self.plot_portfolio_weights(focus_result),
            "drawdown": self.plot_drawdown(result_map),
            "volatility_comparison": self.plot_volatility_comparison(result_map),
            "risk_return_comparison": self.plot_risk_return_comparison(result_map),
            "turnover_transaction_costs": self.plot_turnover_transaction_costs(result_map),
        }
        return {name: path for name, path in chart_paths.items() if path is not None}

    def plot_backtest_summary(
        self,
        result: BacktestResult,
        mvo_result: Optional[BacktestResult] = None,
        save: bool = True,
    ) -> Optional[Path]:
        """Generate a compact four-panel summary for backward compatibility."""
        result_map: dict[str, BacktestResult] = {result.strategy_name: result}
        if mvo_result is not None:
            result_map[mvo_result.strategy_name] = mvo_result
        if "Equal Weight" not in result_map and not result.benchmark_returns.empty:
            result_map["Equal Weight"] = self._benchmark_as_result(result)

        fig, axes = plt.subplots(2, 2, figsize=(15, 9))
        self._draw_cumulative_returns(axes[0, 0], result_map)
        self._draw_drawdowns(axes[0, 1], result_map)
        self._draw_volatility_comparison(axes[1, 0], result_map)
        self._draw_turnover_costs(axes[1, 1], result_map)
        fig.suptitle("Risk-Aware Portfolio Optimizer: Backtest Summary", fontsize=15, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        return self._save_or_close(fig, "backtest_summary", save)

    def plot_cumulative_returns(
        self,
        results: ResultsInput,
        *,
        filename: str = REQUIRED_CHART_FILENAMES["cumulative_returns"],
        save: bool = True,
    ) -> Optional[Path]:
        """Save cumulative return comparison across strategies."""
        result_map = self._coerce_results(results)
        fig, ax = plt.subplots(figsize=(13, 6))
        self._draw_cumulative_returns(ax, result_map)
        fig.tight_layout()
        return self._save_or_close(fig, filename, save)

    def plot_portfolio_weights(
        self,
        result: BacktestResult,
        *,
        filename: str = REQUIRED_CHART_FILENAMES["portfolio_weights"],
        save: bool = True,
    ) -> Optional[Path]:
        """Save a stacked area chart of portfolio weights over time."""
        weights = self._weights_with_cash(result.weights_history)
        if weights.empty:
            return self._empty_plot(
                "Portfolio Weights Over Time",
                "No weight history available for this strategy.",
                filename,
                save,
            )

        colors = [self._color_for(column) for column in weights.columns]
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.stackplot(
            weights.index,
            [weights[column].to_numpy(dtype=float) for column in weights.columns],
            labels=list(weights.columns),
            colors=colors,
            alpha=0.88,
        )
        ax.set_title(f"Portfolio Weights Over Time: {result.strategy_name}", fontweight="bold")
        ax.set_ylabel("Portfolio Weight")
        ax.set_ylim(0.0, max(1.0, float(weights.sum(axis=1).max()) * 1.05))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
        self._format_date_axis(ax)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        return self._save_or_close(fig, filename, save)

    def plot_drawdown(
        self,
        results: ResultsInput,
        *,
        filename: str = REQUIRED_CHART_FILENAMES["drawdown"],
        save: bool = True,
    ) -> Optional[Path]:
        """Save drawdown comparison across strategies."""
        result_map = self._coerce_results(results)
        fig, ax = plt.subplots(figsize=(13, 6))
        self._draw_drawdowns(ax, result_map)
        fig.tight_layout()
        return self._save_or_close(fig, filename, save)

    def plot_drawdowns(
        self,
        results: ResultsInput,
        *,
        filename: str = REQUIRED_CHART_FILENAMES["drawdown"],
        save: bool = True,
    ) -> Optional[Path]:
        """Backward-compatible alias for ``plot_drawdown``."""
        return self.plot_drawdown(results, filename=filename, save=save)

    def plot_volatility_comparison(
        self,
        results: ResultsInput,
        *,
        filename: str = REQUIRED_CHART_FILENAMES["volatility_comparison"],
        window: int = 63,
        save: bool = True,
    ) -> Optional[Path]:
        """Save rolling volatility comparison across strategies."""
        if window <= 0:
            raise ValueError("window must be positive.")

        result_map = self._coerce_results(results)
        fig, ax = plt.subplots(figsize=(13, 6))
        self._draw_volatility_comparison(ax, result_map, window=window)
        fig.tight_layout()
        return self._save_or_close(fig, filename, save)

    def plot_risk_return_comparison(
        self,
        results: ResultsInput,
        *,
        filename: str = REQUIRED_CHART_FILENAMES["risk_return_comparison"],
        save: bool = True,
    ) -> Optional[Path]:
        """Save risk-return scatter chart across strategies."""
        result_map = self._coerce_results(results)
        rows = []
        for name, result in result_map.items():
            returns = self._clean_returns(result.portfolio_returns)
            if returns.empty:
                continue
            summary = result.summary()
            rows.append(
                {
                    "strategy": name,
                    "return": summary.get("annualized_return", annualized_return(returns)),
                    "volatility": summary.get("annualized_volatility", annualized_volatility(returns)),
                    "sharpe": summary.get("sharpe_ratio", 0.0),
                    "max_drawdown": summary.get("max_drawdown", float("nan")),
                }
            )

        if not rows:
            return self._empty_plot(
                "Risk-Return Comparison",
                "No strategy return series available.",
                filename,
                save,
            )

        metrics = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(10, 7))
        for _, row in metrics.iterrows():
            sharpe = float(row["sharpe"]) if np.isfinite(row["sharpe"]) else 0.0
            size = 220 + max(sharpe, 0.0) * 140
            strategy = str(row["strategy"])
            ax.scatter(
                row["volatility"],
                row["return"],
                s=size,
                color=self._color_for(strategy),
                edgecolor="white",
                linewidth=1.0,
                alpha=0.9,
            )
            ax.annotate(
                strategy,
                (row["volatility"], row["return"]),
                xytext=(8, 6),
                textcoords="offset points",
                fontsize=9,
            )

        ax.set_title("Risk-Return Comparison", fontweight="bold")
        ax.set_xlabel("Annualized Volatility")
        ax.set_ylabel("Annualized Return")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return self._save_or_close(fig, filename, save)

    def plot_turnover_transaction_costs(
        self,
        results: ResultsInput,
        *,
        filename: str = REQUIRED_CHART_FILENAMES["turnover_transaction_costs"],
        save: bool = True,
    ) -> Optional[Path]:
        """Save turnover and transaction-cost analysis chart."""
        result_map = self._coerce_results(results)
        fig, ax = plt.subplots(figsize=(13, 6))
        self._draw_turnover_costs(ax, result_map)
        fig.tight_layout()
        return self._save_or_close(fig, filename, save)

    def plot_efficient_frontier(
        self,
        frontier_results: list[OptimizationResult],
        highlight_result: Optional[OptimizationResult] = None,
        save: bool = True,
        filename: str = "efficient_frontier",
    ) -> Optional[Path]:
        """Plot an efficient frontier from optimization results."""
        feasible = [result for result in frontier_results if result.feasible]
        if not feasible:
            return self._empty_plot(
                "Efficient Frontier",
                "No feasible optimization results available.",
                filename,
                save,
            )

        fig, ax = plt.subplots(figsize=(9, 7))
        volatility = np.array([result.portfolio_volatility for result in feasible], dtype=float)
        returns = np.array([result.portfolio_return for result in feasible], dtype=float)
        sharpe = np.array([result.sharpe_ratio for result in feasible], dtype=float)
        scatter = ax.scatter(volatility, returns, c=sharpe, cmap="viridis", s=70)
        fig.colorbar(scatter, ax=ax, label="Sharpe Ratio")

        if highlight_result is not None and highlight_result.feasible:
            ax.scatter(
                highlight_result.portfolio_volatility,
                highlight_result.portfolio_return,
                marker="*",
                s=280,
                color=PALETTE["Risk-Aware Optimizer"],
                edgecolor="black",
                label="Selected Portfolio",
            )
            ax.legend()

        ax.set_title("Efficient Frontier", fontweight="bold")
        ax.set_xlabel("Annualized Volatility")
        ax.set_ylabel("Annualized Expected Return")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return self._save_or_close(fig, filename, save)

    def plot_weight_heatmap(
        self,
        result: BacktestResult,
        top_n: int = 20,
        save: bool = True,
        filename: str = "weight_heatmap",
    ) -> Optional[Path]:
        """Plot a heatmap-like view of portfolio weights over rebalances."""
        if top_n <= 0:
            raise ValueError("top_n must be positive.")

        weights = self._weights_with_cash(result.weights_history)
        if weights.empty:
            return self._empty_plot(
                "Portfolio Weight Heatmap",
                "No weight history available for this strategy.",
                filename,
                save,
            )

        top_columns = list(weights.mean(axis=0).sort_values(ascending=False).head(top_n).index)
        data = weights[top_columns].T

        fig, ax = plt.subplots(figsize=(13, max(5, len(top_columns) * 0.45)))
        image = ax.imshow(data.to_numpy(dtype=float), aspect="auto", cmap="Blues", vmin=0.0)
        fig.colorbar(image, ax=ax, label="Portfolio Weight")
        ax.set_yticks(range(len(top_columns)))
        ax.set_yticklabels(top_columns)
        step = max(1, len(data.columns) // 8)
        tick_positions = list(range(0, len(data.columns), step))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            [pd.Timestamp(data.columns[position]).strftime("%Y-%m") for position in tick_positions],
            rotation=45,
            ha="right",
        )
        ax.set_title(f"Portfolio Weight Heatmap: {result.strategy_name}", fontweight="bold")
        fig.tight_layout()
        return self._save_or_close(fig, filename, save)

    def plot_risk_contribution(
        self,
        weights: np.ndarray,
        covariance: np.ndarray,
        tickers: list[str],
        top_n: int = 15,
        save: bool = True,
        filename: str = "risk_contribution",
    ) -> Optional[Path]:
        """Plot risk contribution by asset."""
        if top_n <= 0:
            raise ValueError("top_n must be positive.")
        weights_arr = np.asarray(weights, dtype=float)
        if len(tickers) != len(weights_arr):
            raise ValueError("tickers length must match weights length.")

        contributions = RiskBudget.risk_contribution(weights_arr, covariance, relative=True)
        order = np.argsort(contributions)[::-1][:top_n]
        labels = [tickers[idx] for idx in order]
        values = contributions[order]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(labels[::-1], values[::-1], color=PALETTE["Risk-Aware Optimizer"], alpha=0.85)
        ax.set_title("Risk Contribution by Asset", fontweight="bold")
        ax.set_xlabel("Share of Portfolio Risk")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        return self._save_or_close(fig, filename, save)

    def print_metrics_table(self, result: BacktestResult) -> None:
        """Print a formatted strategy-versus-benchmark metrics table."""
        table = format_metrics_for_display(result.metrics())
        print("\nPerformance Metrics")
        print(table.to_string())

    def _draw_cumulative_returns(
        self,
        ax: plt.Axes,
        results: Mapping[str, BacktestResult],
    ) -> None:
        plotted_any = False
        for name, result in results.items():
            returns = self._clean_returns(result.portfolio_returns)
            if returns.empty:
                continue
            curve = cumulative_returns(returns, initial_value=1.0)
            ax.plot(curve.index, curve.values, label=name, color=self._color_for(name), linewidth=2.1)
            plotted_any = True

        ax.set_title("Cumulative Returns", fontweight="bold")
        ax.set_ylabel("Growth of $1")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"${value:.2f}"))
        self._format_date_axis(ax)
        if plotted_any:
            ax.legend(loc="best")
        else:
            self._write_empty_axis_message(ax, "No return series available.")
        ax.grid(True, alpha=0.3)

    def _draw_drawdowns(
        self,
        ax: plt.Axes,
        results: Mapping[str, BacktestResult],
    ) -> None:
        plotted_any = False
        for name, result in results.items():
            returns = self._clean_returns(result.portfolio_returns)
            if returns.empty:
                continue
            dd = drawdown_series(returns)
            ax.plot(dd.index, dd.values, label=name, color=self._color_for(name), linewidth=1.8)
            if name == "Risk-Aware Optimizer":
                ax.fill_between(dd.index, dd.values, 0, color=PALETTE["drawdown_fill"], alpha=0.22)
            plotted_any = True

        ax.set_title("Drawdown Comparison", fontweight="bold")
        ax.set_ylabel("Drawdown")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
        self._format_date_axis(ax)
        ax.axhline(0.0, color="black", linewidth=0.8)
        if plotted_any:
            ax.legend(loc="best")
        else:
            self._write_empty_axis_message(ax, "No return series available.")
        ax.grid(True, alpha=0.3)

    def _draw_volatility_comparison(
        self,
        ax: plt.Axes,
        results: Mapping[str, BacktestResult],
        window: int = 63,
    ) -> None:
        plotted_any = False
        for name, result in results.items():
            returns = self._clean_returns(result.portfolio_returns)
            if returns.empty:
                continue
            if len(returns) < max(2, window):
                realized_vol = pd.Series(
                    annualized_volatility(returns),
                    index=returns.index,
                    name=name,
                )
            else:
                realized_vol = VolatilityModel.rolling(returns, window=window).bfill().fillna(0.0)
            ax.plot(
                realized_vol.index,
                realized_vol.values,
                label=name,
                color=self._color_for(name),
                linewidth=1.8,
            )
            plotted_any = True

        ax.set_title(f"Rolling {window}-Day Volatility", fontweight="bold")
        ax.set_ylabel("Annualized Volatility")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
        self._format_date_axis(ax)
        if plotted_any:
            ax.legend(loc="best")
        else:
            self._write_empty_axis_message(ax, "No return series available.")
        ax.grid(True, alpha=0.3)

    def _draw_turnover_costs(
        self,
        ax: plt.Axes,
        results: Mapping[str, BacktestResult],
    ) -> None:
        width = 0.8 / max(len(results), 1)
        plotted_turnover = False
        plotted_costs = False

        for idx, (name, result) in enumerate(results.items()):
            turnover = result.turnover_by_rebalance.replace([np.inf, -np.inf], np.nan).dropna()
            if turnover.empty:
                continue
            offset_days = int((idx - (len(results) - 1) / 2.0) * 5)
            x_values = pd.to_datetime(turnover.index) + pd.to_timedelta(offset_days, unit="D")
            ax.bar(
                x_values,
                turnover.values,
                width=max(width * result.rebal_freq, 1.0),
                label=f"{name} turnover",
                color=self._color_for(name),
                alpha=0.65,
            )
            plotted_turnover = True

        cost_axis = ax.twinx()
        for name, result in results.items():
            costs = self._clean_costs(result.transaction_costs)
            if costs.empty:
                continue
            cumulative_costs = costs.cumsum()
            cost_axis.plot(
                cumulative_costs.index,
                cumulative_costs.values,
                color=self._color_for(name),
                linestyle="--",
                linewidth=1.8,
                label=f"{name} cumulative costs",
            )
            plotted_costs = True

        if not plotted_turnover and not plotted_costs:
            self._write_empty_axis_message(ax, "No turnover or transaction-cost events recorded.")

        ax.set_title("Turnover and Transaction Cost Analysis", fontweight="bold")
        ax.set_ylabel("One-Way Turnover")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
        cost_axis.set_ylabel("Cumulative Cost Impact")
        cost_axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.2%}"))
        self._format_date_axis(ax)
        ax.grid(True, axis="y", alpha=0.3)

        handles, labels = ax.get_legend_handles_labels()
        cost_handles, cost_labels = cost_axis.get_legend_handles_labels()
        if handles or cost_handles:
            ax.legend(handles + cost_handles, labels + cost_labels, loc="best", fontsize=8)

    def _empty_plot(
        self,
        title: str,
        message: str,
        filename: str,
        save: bool,
    ) -> Optional[Path]:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title(title, fontweight="bold")
        self._write_empty_axis_message(ax, message)
        fig.tight_layout()
        return self._save_or_close(fig, filename, save)

    def _save_or_close(self, fig: plt.Figure, filename: str, save: bool) -> Optional[Path]:
        if save:
            return self._save_figure(fig, filename)
        plt.close(fig)
        return None

    def _save_figure(self, fig: plt.Figure, filename: str) -> Path:
        output_name = filename if Path(filename).suffix else f"{filename}.{self.fmt}"
        path = self.output_dir / output_name
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        self.logger.info("Saved chart: %s", path)
        plt.close(fig)
        return path

    @staticmethod
    def _coerce_results(results: ResultsInput) -> Mapping[str, BacktestResult]:
        if isinstance(results, MultiStrategyBacktestResult):
            if not results.results:
                raise ValueError("MultiStrategyBacktestResult.results cannot be empty.")
            return results.results
        if not results:
            raise ValueError("results mapping cannot be empty.")
        return results

    _result_mapping = _coerce_results

    @staticmethod
    def _focus_result(results: Mapping[str, BacktestResult], focus_strategy: str) -> BacktestResult:
        if focus_strategy in results:
            return results[focus_strategy]
        return next(iter(results.values()))

    @staticmethod
    def _benchmark_as_result(result: BacktestResult) -> BacktestResult:
        return BacktestResult(
            portfolio_returns=result.benchmark_returns.copy(),
            weights_history=pd.DataFrame(),
            optimization_results=[],
            benchmark_returns=result.benchmark_returns.copy(),
            tickers=result.tickers,
            rebal_freq=result.rebal_freq,
            strategy_name="Equal Weight",
            transaction_costs=pd.Series(0.0, index=result.benchmark_returns.index),
            initial_value=result.initial_value,
        )

    @staticmethod
    def _clean_returns(returns: pd.Series) -> pd.Series:
        if returns is None:
            return pd.Series(dtype=float)
        clean = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
        if clean.empty:
            return pd.Series(dtype=float)
        clean.index = pd.to_datetime(clean.index)
        return clean.sort_index()

    @staticmethod
    def _clean_costs(costs: pd.Series) -> pd.Series:
        if costs is None:
            return pd.Series(dtype=float)
        clean = pd.Series(costs, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
        clean = clean[clean > 0.0]
        if clean.empty:
            return pd.Series(dtype=float)
        clean.index = pd.to_datetime(clean.index)
        return clean.sort_index()

    @staticmethod
    def _weights_with_cash(weights_history: pd.DataFrame) -> pd.DataFrame:
        if weights_history is None or weights_history.empty:
            return pd.DataFrame()
        weights = weights_history.copy()
        weights.index = pd.to_datetime(weights.index)
        weights = weights.sort_index().apply(pd.to_numeric, errors="coerce")
        weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        cash = 1.0 - weights.sum(axis=1)
        if (cash.abs() > 1e-8).any():
            weights["Cash"] = cash.clip(lower=0.0)
        return weights

    @staticmethod
    def _format_date_axis(ax: plt.Axes) -> None:
        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    @staticmethod
    def _write_empty_axis_message(ax: plt.Axes, message: str) -> None:
        ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center", color=PALETTE["neutral"])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    @staticmethod
    def _color_for(strategy_name: str) -> str:
        return PALETTE.get(strategy_name, PALETTE.get(strategy_name.title(), PALETTE["neutral"]))


__all__ = [
    "PALETTE",
    "PortfolioReporter",
    "REQUIRED_CHART_FILENAMES",
]
