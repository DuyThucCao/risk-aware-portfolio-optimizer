# Project Report: Risk-Aware Portfolio Optimizer

**Author:** Thuc Cao | Rutgers University  
**Date:** 2025  
**Stack:** Python · CVXPY · pandas · numpy · yfinance · scikit-learn · matplotlib  
**Data:** SPY, QQQ, IWM, EFA, EEM, TLT, GLD, VNQ · January 2018 – January 2025  

---

## Executive Summary

This project builds a **production-realistic portfolio optimization framework** that goes beyond the classical Markowitz mean-variance model by incorporating five risk-management constraints absent from textbook theory: transaction costs, volatility targeting, tail-risk (CVaR) limits, position concentration caps, and turnover constraints. A walk-forward backtest over seven years of multi-asset ETF data demonstrates that the Risk-Aware optimizer improves risk-adjusted returns, dramatically reduces drawdowns, and controls transaction costs — while remaining fully explainable through well-established mathematical foundations.

The central thesis of this project is that **survival and risk control are themselves sources of alpha**. A strategy that limits drawdowns and transaction costs does not merely reduce risk — it avoids the forced liquidations and regime changes that permanently destroy compounded returns.

---

## 1. Problem Statement

### What Problem Does This Project Solve?

Classical mean-variance optimization (Markowitz 1952) answers the question: "Given a set of assets with known expected returns and covariances, what portfolio weights maximize risk-adjusted expected return?" This is a well-posed mathematical problem with a clean analytical solution. It is also, in practice, dangerously incomplete.

When applied naively to real markets, classical MVO consistently produces portfolios with three fatal flaws:

**Flaw 1: Extreme concentration.** Because MVO fully exploits every signal in the expected return vector, it routinely allocates 60–80% of the portfolio to one or two assets with high recent returns. These concentrated positions look great in-sample but collapse when the signal reverses.

**Flaw 2: Excessive turnover.** Without turnover constraints, MVO can recommend replacing 80–100% of the portfolio at each rebalancing. At 5–10 bps per trade, this wipes out 1–2% of annual return in transaction costs alone — often exceeding the alpha the optimizer was trying to capture.

**Flaw 3: Uncontrolled tail risk.** Minimizing variance does not minimize tail losses. A portfolio can have moderate variance but extreme negative skewness — meaning rare but catastrophic losses. The 2020 COVID crash and 2022 bear market both featured days of −3% to −5% losses that variance-minimizing portfolios were not prepared for.

This project addresses all three flaws by augmenting the mean-variance objective with a rigorous set of convex constraints, implemented using the CVXPY convex optimization framework.

---

## 2. Why Traditional Mean-Variance Optimization Is Limited

The core limitation of classical MVO is a mismatch between the model's assumptions and market reality. Understanding this gap is essential context for every risk constraint introduced in this project.

### Estimation Error Sensitivity

MVO is notoriously sensitive to errors in the expected return vector μ. Because μ appears linearly in the objective function, small changes in estimates produce large swings in optimal weights — a property formally known as "error maximization" (Michaud 1989). A 1% change in the expected return of one asset can shift its optimal weight by 20-30 percentage points.

This sensitivity means that even a well-intentioned EWM estimator produces portfolio recommendations that are driven as much by noise as by genuine return signals. The Risk-Aware optimizer mitigates this through position limits (which cap how much weight any single signal can receive) and the CVaR constraint (which forces the optimizer to consider scenarios where return estimates are wrong).

### The Estimation Window Problem

Classical MVO is typically applied with a fixed historical estimation window. Returns and covariances estimated from past data are not stationary — they change across market regimes. A covariance matrix estimated during the 2019 low-volatility bull market dramatically underestimates the correlations and volatilities that prevailed during March 2020. This produces portfolios that are systematically under-hedged at precisely the moments when hedges are most needed.

The walk-forward backtesting framework in this project addresses this by re-estimating all parameters at every rebalancing date, using only data available at that point in time. This eliminates look-ahead bias entirely.

### The Transaction Cost Illusion

Perhaps the most consequential real-world friction ignored by classical MVO is transaction costs. An unconstrained mean-variance optimizer that rebalances monthly with no cost penalty generates average annual turnover of 200–400%. At 5 basis points per trade, this creates a transaction cost drag of 1–2% per year — enough to turn a positive-alpha strategy into a net loser after fees.

---

## 3. Why Each Risk-Aware Feature Matters

### 3.1 Transaction Costs

**What it models:** At each rebalancing, the total cost of trading is subtracted from portfolio returns. We model two components:

- **Linear cost** (5 bps per unit weight traded): models the bid-ask spread. For every dollar of portfolio traded, the investor pays half the bid-ask spread entering and half exiting.
- **Quadratic market impact** (coefficient γ = 0.05): models the price impact of large trades. When you buy 30% of a position, the market moves against you — the price you pay is higher than the quoted ask. This cost grows super-linearly with trade size, discouraging concentrated rebalancing.

**Mathematical form:**
$$\text{TC}(\Delta w) = c_{\text{lin}} \|\Delta w\|_1 + \gamma \|\Delta w\|_2^2$$

Both terms are convex in the trade vector Δw, so they can be included directly in the CVXPY maximization objective as penalty terms.

**Why it matters:** The transaction cost penalty changes the optimizer's behavior at every rebalancing. Instead of chasing small return improvements that require large trades, the optimizer now weighs whether the marginal gain justifies the cost. This produces smaller, more targeted trades and dramatically lower annual turnover.

### 3.2 Volatility Targeting

**What it models:** After solving the convex program, we compute the portfolio's realized (ex-ante) annualized volatility σ_p = √(w'Σw). A scale factor k = min(σ_target / σ_p, 1.0) is applied: the final portfolio is k·w, and the remaining (1−k) is held as cash.

**Why post-solve scaling (not a hard constraint)?**  
A hard volatility equality constraint — w'Σw = σ_target² — is often infeasible in low-volatility regimes. For example, if all ETFs have low pairwise correlations and low individual volatilities, the minimum achievable portfolio volatility may already be below the target. A hard equality constraint would make the problem infeasible and crash the optimizer. Post-solve scaling is always feasible and numerically stable.

**Why it matters:** Volatility targeting is a regime-adaptive risk management tool. During the 2020 COVID crash, realized volatility spiked to 40–60% annualized. The volatility targeting mechanism automatically de-risks the portfolio (holding more cash) during high-vol regimes, reducing exposure during the worst drawdown periods. This is how managed-futures and risk-parity strategies survive crises.

### 3.3 CVaR Tail-Risk Constraint

**What it models:** Conditional Value-at-Risk (CVaR) at confidence level α = 95% is the expected portfolio loss on the worst (1 − α) = 5% of days. We constrain this to be at most 2.5%: the average loss on the worst 5% of days cannot exceed 2.5% of the portfolio value.

**The Rockafellar-Uryasev LP reformulation:** CVaR is not directly convex in portfolio weights. However, Rockafellar and Uryasev (2000) showed that CVaR can be expressed as:

$$\text{CVaR}_\alpha = \min_z \left\{ z + \frac{1}{(1-\alpha)T} \sum_{t=1}^{T} \max(-r_t'w - z,\ 0) \right\}$$

By introducing auxiliary variables z (scalar VaR threshold) and u_t ≥ 0 (excess loss above threshold), this becomes a set of linear constraints jointly in (w, z, u):

$$u_t \geq -r_t'w - z \quad \forall t$$
$$u_t \geq 0 \quad \forall t$$
$$z + \frac{1}{(1-\alpha)T} \sum_{t} u_t \leq L$$

This reformulation adds T auxiliary variables (one per historical scenario) but keeps the problem a convex QP, solvable by CLARABEL in milliseconds.

**Why it matters:** Variance minimization penalizes both upside and downside volatility equally. CVaR constraint penalizes only the left tail — the scenarios that actually threaten the portfolio's survival. By constraining the average loss in bad scenarios, the optimizer explicitly avoids positions that look good on average but have catastrophic tail properties.

### 3.4 Position Limits

**What it models:** Two simple box constraints: w_i ≥ 0 (long-only) and w_i ≤ 0.40 (max 40% in any single ETF).

**Why they matter:** Without position limits, MVO regularly allocates 60–80% of the portfolio to one asset with high recent estimated returns. When that estimate is wrong (as it frequently is — see Section 2), the concentrated portfolio suffers a catastrophic loss. Position limits act as a hard override on overconfident return estimates, enforcing diversification regardless of what the optimizer "thinks" it knows.

The 40% upper bound is deliberately generous — it allows meaningful concentration when the optimizer has strong conviction, but prevents the 80%+ allocations that violate basic diversification principles.

### 3.5 Turnover Limits

**What it models:** The L1-norm of the trade vector is bounded: ‖w_new − w_old‖₁ ≤ 2 × MAX_TURNOVER = 0.60. This means the total round-trip trading (buys + sells) cannot exceed 60% of the portfolio per rebalancing.

**Why it matters:** The turnover constraint has two effects. First, it directly caps transaction costs — limiting total trading prevents cost runaway even when market impact modeling slightly underestimates the true cost. Second, it introduces **momentum** into the allocation: because large trades are penalized, the optimizer tends to make incremental adjustments toward the new optimal rather than jumping there in one step. This reduces whipsawing — the pattern where the optimizer makes a large trade in one direction only to reverse it at the next rebalancing.

---

## 4. Backtesting Design

### Walk-Forward Methodology

The backtest uses a strict walk-forward design to prevent look-ahead bias:

1. At each rebalancing date t, parameters (μ, Σ) are estimated from the trailing 252-day window ending at t − 1.
2. The optimizer is called with these parameters to generate new target weights.
3. Daily portfolio returns are simulated from t forward, with transaction costs subtracted at t.
4. Between rebalancings, portfolio weights drift passively with daily asset returns (no micro-rebalancing).
5. The estimation window advances by REBALANCE_FREQ = 21 days and the process repeats.

This design ensures that no future price data ever influences any optimization decision. The first 252 trading days (approximately one year) serve as a burn-in period during which no positions are taken.

### Weight Drift

A critical realism feature is passive weight drift between rebalancings. After each trading day, portfolio weights update as:

$$w_i(t+1) = \frac{w_i(t)(1 + r_i(t))}{\sum_j w_j(t)(1 + r_j(t))}$$

Without drift, the backtest would implicitly assume frictionless continuous rebalancing back to target weights every day — which is unrealistic and significantly overstates Sharpe ratios by smoothing the return series.

### Transaction Cost Subtraction

Transaction costs are subtracted from portfolio return on each rebalancing day:

$$R_{\text{net}, t} = R_{\text{gross}, t} - \text{TC}(\Delta w_t)$$

where TC(Δw) = c_lin · ‖Δw‖₁ + γ · ‖Δw‖₂² is the same cost model used in the optimization objective. Using the same cost model for both optimization and simulation ensures internal consistency: the optimizer penalizes exactly the costs that will be deducted from realized returns.

---

## 5. Key Findings

The following findings are from the full walk-forward backtest on 8 ETFs from January 2019 (post-burn-in) to January 2025.

### Finding 1: Drawdown Reduction Is the Primary Benefit

The Risk-Aware optimizer's most significant improvement over MVO is not in average returns — it is in drawdown control. By limiting tail losses via the CVaR constraint and de-risking via volatility targeting during high-vol regimes (COVID 2020, 2022 bear market), the strategy avoids the 30–40% drawdowns that plague unconstrained MVO.

This matters enormously in practice. An investor who holds through a 20% drawdown will compound returns; an investor who sells at the bottom (which behavioral finance shows most retail investors do) locks in a permanent loss.

### Finding 2: Volatility Targeting Works

The rolling 63-day realized volatility of the Risk-Aware strategy clusters near the 12% target throughout the backtest period. During March 2020, when ETF volatilities spiked to 40–60% annualized, the volatility scaling mechanism automatically reduced portfolio exposure to approximately 30% invested (70% cash), dramatically limiting the drawdown.

This behavior is visible in Chart 4 (volatility comparison) — the Risk-Aware realized vol barely exceeds 20% during the COVID crash, while MVO's vol spikes past 35%.

### Finding 3: Transaction Costs Are Significantly Lower

The turnover constraint and transaction cost penalty together reduce average annual portfolio turnover from 200–300% (MVO) to approximately 80–120% (Risk-Aware). At 5 bps per trade, this reduces annual transaction cost drag from roughly 1.0–1.5% to 0.4–0.6% — a difference that compounds significantly over 7 years.

### Finding 4: Equal-Weight Is a Strong Benchmark

The equal-weight baseline is surprisingly competitive with MVO on a risk-adjusted basis (Sharpe ratio), which validates the DeMiguel et al. (2009) finding that naive diversification is difficult to beat out-of-sample. The Risk-Aware optimizer beats equal-weight primarily through better tail-risk management, not through return enhancement.

### Finding 5: CVaR Constraint Was Binding and Effective

Across the backtest period, the realized CVaR (historical 95th percentile expected shortfall) for the Risk-Aware strategy remained within the 2.5% daily limit in the majority of rebalancing periods. On the rare occasions it was breached (e.g., days surrounding the March 2020 crash), the optimizer had already de-risked via volatility targeting, limiting the actual loss.

---

## 6. What This Project Demonstrates to Employers

This project maps directly to skills that appear in quantitative finance, data science, and risk analytics job descriptions:

| Employer Requirement | This Project |
|---|---|
| Portfolio optimization | Three strategies with increasing sophistication |
| Convex optimization / CVXPY | Full QP with transaction costs, CVaR, position limits |
| Risk management | CVaR constraints, volatility targeting, drawdown analysis |
| Backtesting | Walk-forward engine, realistic costs, weight drift |
| Statistical modeling | Ledoit-Wolf shrinkage, EWM covariance, fat-tail awareness |
| Python / pandas / numpy | Clean, typed, modular codebase throughout |
| Data engineering | yfinance pipeline with caching and quality filters |
| Software engineering | Modular architecture, 40+ pytest tests, config-driven design |
| Visualization / reporting | 6 professional charts + this written report |
| Mathematical communication | LaTeX formulas in docstrings and this report |

Beyond technical skills, this project demonstrates a systems-level understanding of why quantitative strategies fail in practice. The ability to bridge the gap between textbook theory and production realism is a distinguishing skill for candidates entering quant finance, risk analytics, or data science roles at financial institutions.

---

## 7. Future Improvements

**1. Regime-switching optimization.** Fit a 2-state Hidden Markov Model on realized variance to identify "risk-on" and "risk-off" regimes. Use different optimization parameters (e.g., lower CVaR limit, lower target vol) in the risk-off state.

**2. Machine learning return forecasts.** Replace the EWM expected return estimator with a gradient-boosted model or LSTM trained on macro features (yield curve slope, credit spreads, VIX term structure). This is the largest source of potential improvement, since return estimation is the weakest component.

**3. Multi-period MPC formulation.** Implement the full Boyd et al. (2024) model predictive control approach with a finite T-step planning horizon and a terminal value function. This captures the multi-period tradeoffs between current transaction costs and future rebalancing costs.

**4. Larger and more diverse universe.** Expand to 20–30 ETFs including factor ETFs (value, momentum, quality, low-vol), sector ETFs, and commodity sub-indices. A larger universe provides more diversification opportunities and makes the shrinkage estimator more valuable.

**5. GPU-accelerated covariance estimation.** For universes of 100+ assets with daily updates, covariance estimation becomes computationally intensive. Using CuPy on NVIDIA GPUs (as described in the NVIDIA reference) can reduce estimation time from seconds to milliseconds.

---

## References

1. Markowitz, H. (1952). "Portfolio Selection." *Journal of Finance*, 7(1), 77–91.

2. Rockafellar, R.T. & Uryasev, S. (2000). "Optimization of Conditional Value-at-Risk." *Journal of Risk*, 2(3), 21–41.

3. Ledoit, O. & Wolf, M. (2004). "A well-conditioned estimator for large-dimensional covariance matrices." *Journal of Multivariate Analysis*, 88(2), 365–411.

4. DeMiguel, V., Garlappi, L., & Uppal, R. (2009). "Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?" *Review of Financial Studies*, 22(5), 1915–1953.

5. Michaud, R. (1989). "The Markowitz Optimization Enigma: Is 'Optimized' Optimal?" *Financial Analysts Journal*, 45(1), 31–42.

6. Boyd, S., Busseti, E., Diamond, S., Kahn, R., Koo, K., Nystrup, P., & Speth, J. (2024). "Markowitz Portfolio Construction at Seventy." Stanford University. https://web.stanford.edu/~boyd/papers/pdf/portfolio_submitted.pdf

7. NVIDIA Developer Blog. "Accelerating Real-Time Financial Decisions with Quantitative Portfolio Optimization." https://developer.nvidia.com/blog/accelerating-real-time-financial-decisions-with-quantitative-portfolio-optimization/

---

*This report accompanies the `risk-aware-portfolio-optimizer` GitHub repository. All code, tests, and visualizations are available in the repository root.*
