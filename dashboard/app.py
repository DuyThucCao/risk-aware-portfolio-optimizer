"""Streamlit dashboard for the Risk-Aware Portfolio Optimizer.

Launch from the project root:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtesting.backtester import BacktestConfig, BacktestResult, run_three_strategy_backtest
from src.data.data_loader import DEFAULT_TICKERS, DataLoader
from src.optimization.optimizer import OptimizationConfig
from src.risk.risk_models import RiskBudget


PALETTE = {
    "Risk-Aware Optimizer": "#1565C0",
    "Mean-Variance Optimization": "#C62828",
    "Equal Weight": "#2E7D32",
    "neutral": "#5F6368",
    "cash": "#9E9E9E",
}

METRIC_CARDS = [
    ("Annual Return", "annualized_return", "{:.1%}", True),
    ("Annual Volatility", "annualized_volatility", "{:.1%}", False),
    ("Sharpe Ratio", "sharpe_ratio", "{:.2f}", True),
    ("Max Drawdown", "max_drawdown", "{:.1%}", True),
    ("CVaR 95%", "cvar_95", "{:.2%}", False),
]


def main() -> None:
    """Render the Streamlit dashboard."""
    st.set_page_config(
        page_title="Risk-Aware Portfolio Optimizer",
        page_icon="R",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()

    settings = _sidebar_controls()
    st.title("Risk-Aware Portfolio Optimizer")
    st.caption("Portfolio construction with transaction costs, turnover limits, CVaR controls, and volatility targeting.")

    if not settings["run"]:
        _render_landing_state()
        return

    try:
        bundle = _load_data(settings)
        result = _run_backtest(bundle["returns"], bundle["tickers"], settings)
    except Exception as exc:
        st.error(f"Unable to complete the backtest: {exc}")
        return

    st.success(
        f"Loaded {len(bundle['tickers'])} assets across {len(bundle['returns']):,} daily return observations "
        f"from {bundle['source']} data."
    )
    _render_summary(result.results)
    _render_charts(result.results)
    _render_risk_budget(result.results["Risk-Aware Optimizer"], bundle["returns"])
    _render_metrics_table(result.metrics, result.best_strategy, result.worst_strategy)


def _sidebar_controls() -> dict[str, object]:
    with st.sidebar:
        st.header("Controls")
        tickers = st.multiselect(
            "Asset universe",
            options=DEFAULT_TICKERS,
            default=DEFAULT_TICKERS,
        )

        date_cols = st.columns(2)
        with date_cols[0]:
            start_date = st.date_input("Start", value=pd.Timestamp("2018-01-01"))
        with date_cols[1]:
            end_date = st.date_input("End", value=pd.Timestamp("2025-12-31"))

        st.divider()
        st.subheader("Optimization")
        risk_aversion = st.slider("Risk aversion", 0.1, 10.0, 1.5, 0.1)
        target_volatility = st.slider("Target volatility", 0.05, 0.30, 0.12, 0.01, format="%.2f")
        max_weight = st.slider("Maximum asset weight", 0.05, 1.00, 0.25, 0.05, format="%.2f")
        max_turnover = st.slider("Maximum turnover", 0.05, 1.00, 0.35, 0.05, format="%.2f")
        cvar_limit = st.slider("Daily CVaR limit", 0.00, 0.10, 0.03, 0.005, format="%.3f")

        st.divider()
        st.subheader("Backtest")
        estimation_window = st.select_slider(
            "Estimation window",
            options=[63, 126, 252, 504],
            value=252,
            format_func=lambda days: f"{days} trading days",
        )
        rebalance_freq = st.selectbox(
            "Rebalance frequency",
            options=[5, 21, 63],
            index=1,
            format_func=lambda days: {5: "Weekly", 21: "Monthly", 63: "Quarterly"}[days],
        )
        cov_estimator = st.selectbox(
            "Covariance estimator",
            options=["ledoit_wolf", "sample", "ewm", "constant_corr"],
            index=0,
        )
        run = st.button("Run backtest", type="primary", use_container_width=True)

    return {
        "tickers": tickers or DEFAULT_TICKERS,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "risk_aversion": float(risk_aversion),
        "target_volatility": float(target_volatility),
        "max_weight": float(max_weight),
        "max_turnover": float(max_turnover),
        "cvar_limit": None if cvar_limit <= 0 else float(cvar_limit),
        "estimation_window": int(estimation_window),
        "rebalance_freq": int(rebalance_freq),
        "cov_estimator": str(cov_estimator),
        "run": bool(run),
    }


def _load_data(settings: Mapping[str, object]) -> dict[str, object]:
    loader = DataLoader(
        cache_dir=PROJECT_ROOT / ".cache",
        raw_dir=PROJECT_ROOT / "data/raw",
        processed_dir=PROJECT_ROOT / "data/processed",
        use_cache=True,
        fallback_to_synthetic=True,
        root_dir=PROJECT_ROOT,
    )
    prices = loader.fetch_prices(
        settings["tickers"],
        start_date=str(settings["start_date"]),
        end_date=str(settings["end_date"]),
    )
    returns = loader.compute_returns(prices)
    return {
        "loader": loader,
        "prices": prices,
        "returns": returns,
        "tickers": list(prices.columns),
        "source": str(prices.attrs.get("source", "yfinance")),
    }


def _run_backtest(
    returns: pd.DataFrame,
    tickers: list[str],
    settings: Mapping[str, object],
):
    loader = DataLoader(root_dir=PROJECT_ROOT, fallback_to_synthetic=True)
    sector_indices, _, _ = loader.get_sector_info(tickers)
    opt_config = OptimizationConfig(
        risk_aversion=float(settings["risk_aversion"]),
        target_volatility=float(settings["target_volatility"]),
        max_weight=float(settings["max_weight"]),
        max_turnover=float(settings["max_turnover"]),
        cvar_limit=settings["cvar_limit"],
        cvar_alpha=0.95,
        linear_cost=0.0005,
        market_impact_coef=0.05,
        use_market_impact=True,
    )
    bt_config = BacktestConfig(
        estimation_window=int(settings["estimation_window"]),
        rebalance_freq=int(settings["rebalance_freq"]),
        cvar_scenario_window=int(settings["estimation_window"]),
        cov_estimator=str(settings["cov_estimator"]),
        initial_value=100000.0,
    )
    if len(returns) <= bt_config.estimation_window:
        raise ValueError("Date range is too short for the selected estimation window.")

    with st.spinner("Running walk-forward comparison..."):
        return run_three_strategy_backtest(
            returns,
            risk_aware_config=opt_config,
            bt_config=bt_config,
            sector_indices=sector_indices,
            show_progress=False,
        )


def _render_landing_state() -> None:
    st.info("Choose settings in the sidebar, then run the backtest.")
    cols = st.columns(3)
    cards = [
        ("Risk controls", "CVaR limits, position caps, sector caps, and volatility targeting."),
        ("Trading realism", "Turnover constraints and explicit transaction-cost drag."),
        ("Strategy comparison", "Equal weight, classical mean-variance, and risk-aware optimization side by side."),
    ]
    for col, (title, body) in zip(cols, cards):
        with col:
            st.markdown(f"**{title}**")
            st.write(body)


def _render_summary(results: Mapping[str, BacktestResult]) -> None:
    risk_aware = results["Risk-Aware Optimizer"].summary()
    mvo = results["Mean-Variance Optimization"].summary()

    st.subheader("Performance Summary")
    columns = st.columns(len(METRIC_CARDS))
    for col, (label, key, fmt, higher_is_better) in zip(columns, METRIC_CARDS):
        value = float(risk_aware.get(key, 0.0))
        baseline = float(mvo.get(key, 0.0))
        better = value > baseline if higher_is_better else value < baseline
        delta = value - baseline
        col.metric(
            label,
            fmt.format(value),
            delta=fmt.format(delta),
            delta_color="normal" if better else "inverse",
        )

    turnover = float(risk_aware.get("avg_annual_turnover", 0.0))
    mvo_turnover = float(mvo.get("avg_annual_turnover", 0.0))
    cost = float(risk_aware.get("transaction_cost_total", 0.0))
    st.caption(
        f"Risk-aware annual turnover: {turnover:.1%}. "
        f"MVO annual turnover: {mvo_turnover:.1%}. "
        f"Total transaction-cost drag: {cost:.2%}."
    )


def _render_charts(results: Mapping[str, BacktestResult]) -> None:
    st.subheader("Portfolio Diagnostics")
    tab_growth, tab_risk, tab_weights, tab_turnover = st.tabs(
        ["Growth", "Risk-return", "Weights", "Turnover and costs"]
    )

    with tab_growth:
        st.plotly_chart(_cumulative_return_figure(results), use_container_width=True)
        st.plotly_chart(_drawdown_figure(results), use_container_width=True)

    with tab_risk:
        st.plotly_chart(_volatility_figure(results), use_container_width=True)
        st.plotly_chart(_risk_return_figure(results), use_container_width=True)

    with tab_weights:
        st.plotly_chart(_weights_figure(results["Risk-Aware Optimizer"]), use_container_width=True)

    with tab_turnover:
        st.plotly_chart(_turnover_cost_figure(results), use_container_width=True)


def _cumulative_return_figure(results: Mapping[str, BacktestResult]) -> go.Figure:
    fig = go.Figure()
    for name, result in results.items():
        curve = result.cumulative_returns
        fig.add_trace(
            go.Scatter(
                x=curve.index,
                y=curve.values,
                mode="lines",
                name=name,
                line={"color": _color_for(name), "width": 2.4},
            )
        )
    fig.update_layout(
        title="Cumulative Returns: Growth of $1",
        yaxis_tickprefix="$",
        yaxis_tickformat=".2f",
        hovermode="x unified",
        height=420,
        legend={"orientation": "h", "y": 1.08},
    )
    return fig


def _drawdown_figure(results: Mapping[str, BacktestResult]) -> go.Figure:
    fig = go.Figure()
    for name, result in results.items():
        drawdown = result.drawdown_series
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode="lines",
                name=name,
                fill="tozeroy" if name == "Risk-Aware Optimizer" else None,
                line={"color": _color_for(name), "width": 2.0},
            )
        )
    fig.update_layout(
        title="Drawdown Comparison",
        yaxis_tickformat=".0%",
        hovermode="x unified",
        height=360,
        legend={"orientation": "h", "y": 1.08},
    )
    return fig


def _volatility_figure(results: Mapping[str, BacktestResult], window: int = 63) -> go.Figure:
    fig = go.Figure()
    for name, result in results.items():
        volatility = result.portfolio_returns.rolling(window).std().bfill().fillna(0.0) * np.sqrt(252)
        fig.add_trace(
            go.Scatter(
                x=volatility.index,
                y=volatility.values,
                mode="lines",
                name=name,
                line={"color": _color_for(name), "width": 2.0},
            )
        )
    fig.update_layout(
        title=f"Rolling {window}-Day Annualized Volatility",
        yaxis_tickformat=".0%",
        hovermode="x unified",
        height=360,
        legend={"orientation": "h", "y": 1.08},
    )
    return fig


def _risk_return_figure(results: Mapping[str, BacktestResult]) -> go.Figure:
    fig = go.Figure()
    for name, result in results.items():
        summary = result.summary()
        fig.add_trace(
            go.Scatter(
                x=[summary["annualized_volatility"]],
                y=[summary["annualized_return"]],
                mode="markers+text",
                name=name,
                text=[name],
                textposition="top center",
                marker={
                    "size": max(14, 14 + summary["sharpe_ratio"] * 5),
                    "color": _color_for(name),
                    "line": {"width": 1, "color": "white"},
                },
            )
        )
    fig.update_layout(
        title="Risk-Return Comparison",
        xaxis_title="Annualized Volatility",
        yaxis_title="Annualized Return",
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
        height=420,
        showlegend=False,
    )
    return fig


def _weights_figure(result: BacktestResult) -> go.Figure:
    weights = result.weights_history.copy()
    fig = go.Figure()
    if weights.empty:
        fig.update_layout(title="Portfolio Weights", annotations=[{"text": "No weight history available.", "showarrow": False}])
        return fig

    weights.index = pd.to_datetime(weights.index)
    cash = 1.0 - weights.sum(axis=1)
    if (cash.abs() > 1e-8).any():
        weights["Cash"] = cash.clip(lower=0.0)

    for column in weights.columns:
        fig.add_trace(
            go.Scatter(
                x=weights.index,
                y=weights[column],
                mode="lines",
                name=column,
                stackgroup="one",
                line={"width": 0.5},
            )
        )
    fig.update_layout(
        title="Risk-Aware Portfolio Weights Over Time",
        yaxis_tickformat=".0%",
        hovermode="x unified",
        height=460,
        legend={"orientation": "h", "y": -0.2},
    )
    return fig


def _turnover_cost_figure(results: Mapping[str, BacktestResult]) -> go.Figure:
    fig = go.Figure()
    for name, result in results.items():
        turnover = result.turnover_by_rebalance
        if not turnover.empty:
            fig.add_trace(
                go.Bar(
                    x=turnover.index,
                    y=turnover.values,
                    name=f"{name} turnover",
                    marker_color=_color_for(name),
                    opacity=0.65,
                    yaxis="y",
                )
            )
        costs = result.transaction_costs[result.transaction_costs > 0].cumsum()
        if not costs.empty:
            fig.add_trace(
                go.Scatter(
                    x=costs.index,
                    y=costs.values,
                    name=f"{name} cumulative costs",
                    mode="lines",
                    line={"color": _color_for(name), "dash": "dash", "width": 2.2},
                    yaxis="y2",
                )
            )
    fig.update_layout(
        title="Turnover and Transaction-Cost Analysis",
        yaxis={"title": "One-way turnover", "tickformat": ".0%"},
        yaxis2={"title": "Cumulative cost drag", "tickformat": ".2%", "overlaying": "y", "side": "right"},
        barmode="group",
        hovermode="x unified",
        height=430,
        legend={"orientation": "h", "y": -0.25},
    )
    return fig


def _render_risk_budget(result: BacktestResult, returns: pd.DataFrame) -> None:
    st.subheader("Latest Risk Budget")
    if result.weights_history.empty:
        st.warning("No weight history is available for risk-budget decomposition.")
        return

    latest_weights = result.weights_history.iloc[-1].to_numpy(dtype=float)
    recent_returns = returns[result.weights_history.columns].iloc[-252:].to_numpy(dtype=float)
    covariance = np.cov(recent_returns, rowvar=False) * 252
    contributions = RiskBudget.risk_contribution(latest_weights, covariance, relative=True)
    order = np.argsort(contributions)[::-1]

    fig = go.Figure(
        go.Bar(
            x=contributions[order],
            y=[result.weights_history.columns[idx] for idx in order],
            orientation="h",
            marker_color=PALETTE["Risk-Aware Optimizer"],
        )
    )
    fig.update_layout(
        title="Risk Contribution by Holding",
        xaxis_tickformat=".0%",
        yaxis={"autorange": "reversed"},
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_metrics_table(metrics: pd.DataFrame, best_strategy: str, worst_strategy: str) -> None:
    st.subheader("Full Metrics")
    st.caption(f"Best Sharpe profile: {best_strategy}. Most challenged profile: {worst_strategy}.")
    display = metrics.copy()
    percent_columns = [
        "cumulative_return",
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
        "cvar_95",
        "avg_annual_turnover",
        "transaction_cost_total",
        "transaction_cost_impact",
    ]
    formatters = {column: "{:.2%}" for column in percent_columns if column in display.columns}
    st.dataframe(display.style.format(formatters).format(precision=3), use_container_width=True)


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; }
        div[data-testid="stMetric"] {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _color_for(name: str) -> str:
    return PALETTE.get(name, PALETTE["neutral"])


if __name__ == "__main__":
    main()
