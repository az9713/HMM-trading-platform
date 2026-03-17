# HMM Regime Terminal

A Python-based stock research platform using **Hidden Markov Models** for market regime detection, multi-confirmation trading signals, and walk-forward backtesting with bootstrap confidence intervals.

## Features

- **BIC-optimal HMM fitting** — tests 2–8 states with 20 random restarts, selects by Bayesian Information Criterion
- **5 engineered features** — log return, rolling volatility, volume change, intraday range, RSI
- **8-condition signal gating** — RSI bounds, momentum, volatility range, volume, ADX, EMA trend, MACD, regime confidence
- **Walk-forward backtesting** — no lookahead bias, with commission/slippage modeling
- **Bootstrap confidence intervals** — 1000-sample CIs on Sharpe, return, and drawdown
- **Kelly + entropy-scaled sizing** — bet less when the model is uncertain
- **Streamlit dashboard** — 5 interactive tabs

## Dashboard Tabs

| Tab | Contents |
|-----|----------|
| **Current Signal** | Regime banner, confidence %, confirmation breakdown, position size |
| **Regime Analysis** | Price chart with regime overlay, transition heatmap, return distributions, BIC curve |
| **Backtest Results** | Metric cards with 90% CIs, equity curve vs benchmark, drawdown chart |
| **Trade Log** | Sortable table with entry/exit/PnL/regime/size |
| **Model Diagnostics** | Rolling log-likelihood, entropy time series, feature correlation matrix |

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
config.yaml      # All tunable parameters
data_loader.py   # yfinance fetch + feature engineering
hmm_engine.py    # HMM fitting, BIC selection, regime analysis
strategy.py      # Confirmation signals, entry/exit, Kelly sizing
backtester.py    # Walk-forward engine, metrics, bootstrap CIs
app.py           # Streamlit dashboard (5 tabs)
```

## Key Math

| Concept | Formula | Purpose |
|---------|---------|---------|
| BIC model selection | `BIC = -2·LL + k·ln(T)` | Data-driven state count |
| Shannon entropy | `H = -Σ pᵢ log₂(pᵢ)` | Regime uncertainty per bar |
| Expected duration | `E[dᵢ] = 1/(1-aᵢᵢ)` | How long regimes persist |
| Kelly criterion | `f* = (p·b-q)/b` | Optimal sizing from edge |
| CVaR (5%) | `E[r \| r ≤ VaR₅%]` | Tail risk beyond VaR |
| Entropy scaling | `size × (1 - H/H_max)` | Reduce bets when uncertain |

## Configuration

All parameters are in `config.yaml` — data settings, HMM hyperparameters, strategy thresholds, risk limits, and backtest windows. The sidebar in the Streamlit app overrides these at runtime.

## Research

The `docs/` folder contains 11 academic papers and a YouTube transcript that informed the mathematical approach.
