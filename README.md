# HMM Regime Terminal

A Python-based stock research platform using **Hidden Markov Models** for market regime detection, multi-confirmation trading signals, and walk-forward backtesting with bootstrap confidence intervals.

## Features

- **BIC-optimal HMM fitting** -- tests 2-8 states with 20 random restarts, selects by Bayesian Information Criterion
- **5 engineered features** -- log return, rolling volatility, volume change, intraday range, RSI
- **8-condition signal gating** -- RSI bounds, momentum, volatility range, volume, ADX, EMA trend, MACD, regime confidence
- **Walk-forward backtesting** -- no lookahead bias, with commission/slippage modeling
- **Bootstrap confidence intervals** -- 1000-sample CIs on Sharpe, return, and drawdown
- **Kelly + entropy-scaled sizing** -- bet less when the model is uncertain
- **Streamlit dashboard** -- 5 interactive tabs

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
git clone https://github.com/az9713/HMM-regime-terminal.git
cd HMM-regime-terminal

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Git Bash (Windows):
source .venv/Scripts/activate
# cmd / PowerShell (Windows):
# .venv\Scripts\activate
# Linux / macOS:
# source .venv/bin/activate

# Install dependencies and run
pip install -r requirements.txt
python -m streamlit run app.py
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

## Documentation

| Document | Description |
|----------|-------------|
| [User Guide](docs/USER_GUIDE.md) | Complete onboarding guide from installation to productive use, covering every tab, parameter, workflow, and troubleshooting step |
| [Architecture](docs/ARCHITECTURE.md) | System architecture with data flow diagrams, module dependency graph, configuration cascade, state management, and performance model |
| [Theory](docs/THEORY.md) | Mathematical reference covering HMMs, Baum-Welch, Viterbi, BIC, Shannon entropy, Kelly criterion, CVaR, bootstrap CIs, and walk-forward validation |
| [Implementation](docs/IMPLEMENTATION.md) | Developer deep-dive into code design decisions, hmmlearn API usage, signal generation state machine, trade simulation, and testing strategy |

## Key Math

| Concept | Formula | Purpose |
|---------|---------|---------|
| BIC model selection | `BIC = -2*LL + k*ln(T)` | Data-driven state count |
| Shannon entropy | `H = -sum(p_i * log2(p_i))` | Regime uncertainty per bar |
| Expected duration | `E[d_i] = 1/(1-a_ii)` | How long regimes persist |
| Kelly criterion | `f* = (p*b-q)/b` | Optimal sizing from edge |
| CVaR (5%) | `E[r \| r <= VaR_5%]` | Tail risk beyond VaR |
| Entropy scaling | `size * (1 - H/H_max)` | Reduce bets when uncertain |

## Configuration

All parameters are in `config.yaml` -- data settings, HMM hyperparameters, strategy thresholds, risk limits, and backtest windows. The sidebar in the Streamlit app overrides these at runtime. See the [User Guide](docs/USER_GUIDE.md) for a complete parameter reference.

## Acknowledgements

### Inspiration

This project was inspired by the YouTube video **["How To Actually Use Claude Code for Trading Strategies (Like a Quant)"](https://www.youtube.com/watch?v=EUSXhJNwRqI&t=459s)**, which demonstrates building a "Regime Terminal" using HMMs for market regime detection. This implementation goes significantly deeper into the mathematics, adding BIC model selection, Shannon entropy confidence, Kelly criterion sizing, walk-forward backtesting, and bootstrap confidence intervals.

### Academic Papers

The following papers in `docs/` informed the mathematical approach and implementation:

| Paper | Authors | Topic |
|-------|---------|-------|
| `1199600.pdf` | Ji, Neerchal | Creating stock portfolios using Hidden Markov Models |
| `2310.03775v2.pdf` | Catello, Ruggiero, Schiavone, Valentino | HMMs for stock market prediction (Univ. Federico II) |
| `2407.19858v7.pdf` | Monteiro | AI-powered energy algorithmic trading: integrating HMMs with neural networks |
| `AdrovicCinoProenca.pdf` | Adrovic, Di Cino, Proenca | Stock trading with Hidden Markov Models (experimental mathematics) |
| `Trading_Strategy_for_Market_Situation_Estimation_B.pdf` | Chen, Yi, Zhao | Trading strategy for market situation estimation based on HMM |
| `Wisebourt_Shaul.pdf` | Wisebourt | Hierarchical HMM of high-frequency market regimes (Univ. Waterloo, MQF thesis) |
| `djk_20190217.pdf` | Daniel, Jagannathan, Kim | A Hidden Markov Model of momentum |
| `jfallon_hmm_stock.pdf` | Fallon | Making profit in the stock market using HMMs (UMass Lowell) |
| `p3D_2.pdf` | Wu, Siwasarit | Capturing order imbalance with HMM: SET50 and KOSPI50 (Thammasat Univ.) |
| `resumo.pdf` | (authors in PDF) | Algorithmic stock index trading using discrete HMMs with daily/weekly windows |
| `stock_hmm.pdf` | (authors in PDF) | Stock market prediction using Hidden Markov Models |

See the [Theory](docs/THEORY.md) document for details on how each mathematical concept from these papers is applied in the codebase.
