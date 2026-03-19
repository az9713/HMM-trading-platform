# HMM Regime Terminal — User Guide

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Understanding the Dashboard](#understanding-the-dashboard)
4. [Sidebar Controls Reference](#sidebar-controls-reference)
5. [Tab 1: Current Signal](#tab-1-current-signal)
6. [Tab 2: Regime Analysis](#tab-2-regime-analysis)
7. [Tab 3: Backtest Results](#tab-3-backtest-results)
8. [Tab 4: Trade Log](#tab-4-trade-log)
9. [Tab 5: Model Diagnostics](#tab-5-model-diagnostics)
10. [Configuration File](#configuration-file)
11. [Interpreting Results](#interpreting-results)
12. [Common Workflows](#common-workflows)
13. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Internet connection (for yfinance data)

### Setup

```bash
git clone https://github.com/az9713/HMM-trading-platform.git
cd HMM-trading-platform
python -m venv .venv

# Activate the virtual environment
# Git Bash on Windows:
source .venv/Scripts/activate
# cmd/PowerShell on Windows:
.venv\Scripts\activate
# Linux / Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `yfinance` | OHLCV market data from Yahoo Finance |
| `hmmlearn` | Hidden Markov Model fitting (Baum-Welch, Viterbi) |
| `numpy` | Numerical computation |
| `pandas` | Data manipulation and time series |
| `scipy` | Statistical functions and linear algebra |
| `ta` | Technical analysis indicators (RSI, ADX, MACD, EMA) |
| `streamlit` | Web dashboard framework |
| `plotly` | Interactive charts |
| `pyyaml` | Configuration file parsing |

---

## Quick Start

```bash
python -m streamlit run app.py
```

This opens the dashboard in your browser (default: `http://localhost:8501`).

1. **Configure** parameters in the left sidebar (or leave defaults)
2. **Click "Run Analysis"** to fetch data, fit the HMM, and generate signals
3. **Browse the 5 tabs** to explore regimes, backtest results, and diagnostics

The default configuration analyzes BTC-USD on 1-hour bars over 90 days, which runs in approximately 30–60 seconds depending on your machine.

---

## Understanding the Dashboard

The dashboard is organized into a **sidebar** for parameter control and **5 tabs** for analysis output.

### Execution Flow

When you click **Run Analysis**, the system executes these steps in order:

1. **Data fetch** — Downloads OHLCV data from Yahoo Finance for the selected ticker/interval/lookback
2. **Feature engineering** — Computes 5 features: log return, rolling volatility, volume change, intraday range, RSI
3. **Standardization** — Z-score normalizes features using training set statistics only
4. **HMM fitting** — Tests 2–8 states (configurable) with 20 random restarts each, selects the model with lowest BIC
5. **Regime decoding** — Runs Viterbi algorithm for most-likely state sequence and forward-backward for posterior probabilities
6. **Signal generation** — Evaluates 8 confirmation conditions, applies cooldown/hysteresis, outputs LONG/SHORT/FLAT
7. **Walk-forward backtest** — Trains and tests in rolling windows with no lookahead bias
8. **Bootstrap CIs** — Resamples returns 1,000 times to compute 90% confidence intervals on key metrics

---

## Sidebar Controls Reference

### Data Settings

| Control | Default | Description |
|---------|---------|-------------|
| **Ticker** | `BTC-USD` | Any Yahoo Finance ticker symbol. Stocks: `AAPL`, `TSLA`. Crypto: `ETH-USD`. Indices: `^SPX`. |
| **Interval** | `1h` | Bar size: `1m`, `5m`, `15m`, `1h`, `1d`. Smaller intervals have shorter max lookback periods. |
| **Lookback (days)** | 90 | How far back to fetch data. yfinance limits: 1m=7d, 5m/15m=60d, 1h=730d, 1d=unlimited. |

### HMM Settings

| Control | Default | Description |
|---------|---------|-------------|
| **Min states** | 2 | Minimum number of hidden states to test |
| **Max states** | 8 | Maximum number of hidden states to test. Higher = slower fitting. |
| **Model type** | `gaussian` | `gaussian` (single Gaussian per state) or `gmm` (Gaussian mixture per state) |
| **Covariance** | `full` | Covariance structure: `full` (most flexible), `diag`, `tied`, `spherical` |
| **Random restarts** | 20 | Number of random initializations per state count. More = better optima but slower. |

**Guidance**: Start with defaults. If the BIC curve is noisy or flat, try increasing restarts to 30–50. Use `diag` covariance if you have very few data points relative to features.

### Strategy Settings

| Control | Default | Description |
|---------|---------|-------------|
| **Min confirmations** | 4 | Number of the 8 conditions that must be true to generate a signal (out of 8) |
| **Cooldown bars** | 5 | After closing a position, wait this many bars before opening a new one |
| **Min hold bars** | 10 | Minimum bars to hold a position before allowing exit |
| **Min regime confidence** | 0.6 | Entropy-based confidence threshold — lower = more signals, higher = more selective |

**Guidance**: Setting min confirmations to 6–7 produces very few but high-conviction signals. Setting it to 2–3 produces many signals with lower average quality. The confidence threshold is the single most impactful filter.

### Risk Settings

| Control | Default | Description |
|---------|---------|-------------|
| **Kelly sizing** | On | Use Kelly criterion for position sizing (vs. fixed 100%) |
| **Entropy scaling** | On | Scale position size down when regime confidence is low |
| **Max leverage** | 2.0x | Maximum allowed position size as a multiple of equity |

**Guidance**: Half-Kelly (the default `kelly_fraction: 0.5` in config.yaml) is standard practice — full Kelly is theoretically optimal but produces extreme drawdowns in practice.

### Backtest Settings

| Control | Default | Description |
|---------|---------|-------------|
| **Train window** | 500 bars | Number of bars used to fit the HMM in each walk-forward fold |
| **Test window** | 100 bars | Number of bars to generate signals on after training |
| **Step size** | 50 bars | How far to advance the window between folds |

**Guidance**: The train window should be at least 200 bars for stable HMM estimation. The test window should be shorter than the train window. Step size controls overlap — smaller step = more folds but slower.

**Important**: You need at least `train_window + test_window` total bars of data. If you see an error about insufficient bars, either increase lookback or decrease these windows.

---

## Tab 1: Current Signal

This tab shows the **latest regime classification and trading signal** based on the most recent bar of data.

### Components

- **Regime Banner** — Large colored indicator showing the current regime (e.g., BULL, BEAR, CRASH, NEUTRAL, BULL_RUN). Color-coded: red=crash, orange=bear, gray=neutral, green=bull, blue=bull_run.

- **Confidence** — Percentage derived from Shannon entropy of the posterior probabilities. 100% = the model is completely certain about the current regime. 0% = maximum uncertainty (uniform distribution over states).

- **Signal** — Current trading recommendation: LONG (buy/hold), SHORT (sell/short), or FLAT (no position).

- **Position Size** — Recommended allocation as a percentage of equity, computed from Kelly criterion scaled by confidence.

- **Confirmation Breakdown** — Table showing which of the 8 conditions are currently met:

| Condition | What it checks |
|-----------|---------------|
| RSI Not Overbought | RSI < 70 (avoids entering at tops) |
| RSI Not Oversold | RSI > 30 (avoids catching falling knives) |
| Momentum | Price > price N bars ago |
| Vol Range | Rolling volatility between 20th and 80th percentile |
| Volume | Current volume > 1.2x the 20-bar average |
| ADX | ADX > 20 (market is trending, not ranging) |
| Above EMA | Price > 50-period EMA (uptrend) |
| MACD | MACD line > signal line (bullish crossover) |

- **Total Confirmations** — Count of conditions met vs. total (e.g., "5 / 8")

### How to read it

A high-conviction long signal looks like: BULL or BULL_RUN regime, confidence > 80%, 6+ confirmations met, meaningful position size. If the signal is FLAT despite a bullish regime, check which confirmations are failing — they explain why the system is hesitant.

---

## Tab 2: Regime Analysis

This tab provides deep analysis of the detected market regimes.

### Price with Regime Overlay

An interactive price chart where each point is colored by its assigned regime. The black line shows the continuous price, while colored dots show regime assignments. Hover over any point to see the exact price and regime.

### Transition Matrix

A heatmap showing the probability of transitioning between regimes. Read it as "probability of moving FROM row TO column in one bar."

**What to look for**:
- High diagonal values (e.g., 0.95+) mean regimes are **sticky** — they persist for many bars
- Off-diagonal spikes show common regime transitions (e.g., bull → neutral happens 8% of the time)
- Asymmetry reveals directional bias (e.g., it may be easier to go from bull to bear than bear to bull)

### Regime Statistics

A table with per-regime metrics:

| Column | Meaning |
|--------|---------|
| **mean_return** | Average log return per bar in this regime |
| **volatility** | Standard deviation of returns (from emission covariance) |
| **expected_duration** | Average number of bars a regime persists: `1 / (1 - self_transition_prob)` |
| **stationary_weight** | Long-run proportion of time spent in this regime |

### Return Distributions

Overlaid histograms showing the distribution of log returns for each regime. Well-separated distributions indicate the HMM is finding genuinely different market conditions. Overlapping distributions suggest the model may be over-fitting or the states aren't meaningfully different.

### BIC / AIC Model Selection

A line chart showing BIC and AIC scores for each tested number of states. The selected model is the one with the **lowest BIC**. Look for a clear "elbow" — if the curve is flat, the data doesn't strongly prefer any particular number of states.

**BIC vs AIC**: BIC penalizes model complexity more heavily than AIC. If BIC and AIC disagree on the optimal state count, BIC's choice is usually more robust (less prone to over-fitting).

---

## Tab 3: Backtest Results

This tab shows walk-forward backtest performance with bootstrap confidence intervals.

### Performance Metrics

Each metric card shows the point estimate. Hover over metrics marked with (i) to see the 90% confidence interval from bootstrap resampling.

| Metric | What it means | Good values |
|--------|--------------|-------------|
| **Total Return** | Cumulative return over the backtest period | Positive, and ideally > benchmark |
| **Sharpe Ratio** | Risk-adjusted return (annualized) | > 1.0 is good, > 2.0 is excellent |
| **Sortino Ratio** | Like Sharpe but only penalizes downside volatility | > 1.5 is good |
| **Calmar Ratio** | Annualized return / max drawdown | > 1.0 means you earn more than your worst loss |
| **Max Drawdown** | Largest peak-to-trough decline | Closer to 0% is better; -20% is concerning |
| **Max DD Duration** | Longest period spent in drawdown (bars) | Shorter is better |
| **CVaR (5%)** | Expected loss in the worst 5% of bars | How bad can a bad bar get? |
| **Win Rate** | Percentage of trades that were profitable | > 50% with positive profit factor |
| **Profit Factor** | Gross profit / gross loss | > 1.5 is good, > 2.0 is strong |
| **Alpha** | Excess return vs. buy-and-hold benchmark | Positive = you beat the benchmark |
| **Total Trades** | Number of completed round-trip trades | Too few = not enough data to judge |

### Equity Curve vs Benchmark

Blue line = strategy equity, gray dashed = buy-and-hold the same asset. The strategy should ideally stay above the benchmark during drawdowns (demonstrating risk management) even if it lags during strong bull runs.

### Drawdown Chart

Red filled area showing ongoing drawdown from peak equity. Deep, prolonged drawdowns indicate periods where the model was wrong or the market was in an unfavorable regime.

### Interpreting Bootstrap CIs

The 90% CI means: "If we resample the return series 1,000 times, 90% of simulated outcomes fall within this range." Wide CIs mean the result is uncertain and could easily be luck. Narrow CIs mean the result is more reliable.

**Red flag**: If the CI for Sharpe includes 0, or the CI for total return includes negative values, the strategy may not have a genuine edge.

---

## Tab 4: Trade Log

A sortable table of every round-trip trade executed during the backtest.

| Column | Meaning |
|--------|---------|
| **Entry/Exit Bar** | Bar indices of trade open and close |
| **Direction** | Long or Short |
| **Entry/Exit Price** | Prices including slippage |
| **PnL ($)** | Dollar profit/loss |
| **PnL (%)** | Percentage return on the trade |
| **Position Size** | Kelly/entropy-scaled allocation |
| **Bars Held** | Duration of the trade |

Click any column header to sort. Look for patterns: are losses clustered in a particular regime? Do short trades perform differently than longs? Are longer-held trades more profitable?

---

## Tab 5: Model Diagnostics

This tab helps you assess whether the HMM is behaving well or degrading.

### Rolling Log-Likelihood

A time series of the model's per-bar log-likelihood computed over a rolling 50-bar window. This measures how well the fitted model explains recent data.

**What to look for**:
- **Stable values** = the model generalizes well
- **Downward trend** = model degradation; the market structure may have changed since fitting
- **Sudden drops** = regime breaks or black-swan events the model can't explain

### Shannon Entropy / Confidence

Dual-axis chart showing:
- **Entropy** (orange, left axis) — Higher entropy = more regime uncertainty. Maximum = log2(n_states) bits.
- **Confidence** (teal, right axis) — 1 - normalized entropy. Higher = more certain about the current regime.

**What to look for**:
- Entropy spikes often coincide with regime transitions
- Persistently high entropy suggests the model can't distinguish regimes well
- Low confidence periods should correlate with FLAT signals (if entropy scaling is working)

### Feature Correlation Matrix

Heatmap showing pairwise Pearson correlations between the 5 input features. Ideally, features should have low mutual correlation (providing independent information). High correlation (|r| > 0.7) between features means they're partially redundant.

---

## Configuration File

All defaults live in `config.yaml`. The sidebar overrides these at runtime but doesn't modify the file. To change defaults permanently, edit `config.yaml` directly.

### Full Parameter Reference

```yaml
data:
  default_ticker: "BTC-USD"       # Any yfinance ticker
  default_interval: "1h"          # 1m, 5m, 15m, 1h, 1d
  default_lookback_days: 90       # Days of history to fetch
  features:                       # Feature columns used by HMM
    - log_return
    - rolling_vol
    - volume_change
    - intraday_range
    - rsi
  rolling_vol_window: 21          # Window for rolling volatility
  rsi_period: 14                  # RSI calculation period

hmm:
  min_states: 2                   # Minimum states to test
  max_states: 8                   # Maximum states to test
  n_restarts: 20                  # Random seeds per state count
  n_iter: 200                     # Max Baum-Welch iterations
  tol: 1.0e-4                    # Convergence tolerance
  model_type: "gaussian"          # gaussian or gmm
  covariance_type: "full"         # full, diag, tied, spherical
  gmm_n_mix: 2                   # GMM mixtures per state

strategy:
  confirmations:
    rsi_oversold: 30              # RSI level for oversold
    rsi_overbought: 70            # RSI level for overbought
    momentum_window: 10           # Bars for momentum check
    vol_low_pct: 20               # Volatility percentile lower bound
    vol_high_pct: 80              # Volatility percentile upper bound
    volume_threshold: 1.2         # Volume ratio vs 20-bar SMA
    adx_threshold: 20             # ADX trending threshold
    ema_period: 50                # EMA period for trend filter
    macd_fast: 12                 # MACD fast period
    macd_slow: 26                 # MACD slow period
    macd_signal: 9                # MACD signal period
    min_confidence: 0.6           # Minimum entropy confidence
  min_confirmations: 4            # Conditions needed out of 8
  cooldown_bars: 5                # Wait after closing a position
  min_hold_bars: 10               # Minimum position duration
  hysteresis_bars: 3              # Regime must persist this long

risk:
  use_kelly: true                 # Enable Kelly criterion sizing
  kelly_fraction: 0.5             # Fraction of full Kelly (0.5 = half-Kelly)
  use_entropy_scaling: true       # Scale size by confidence
  max_leverage: 2.0               # Position size cap
  max_position_pct: 1.0           # Max % of equity per position
  stop_loss_pct: 0.05             # 5% stop loss
  take_profit_pct: 0.15           # 15% take profit

backtest:
  train_window_bars: 500          # Training window size
  test_window_bars: 100           # Test window size
  step_bars: 50                   # Window advance step
  initial_capital: 100000         # Starting equity ($)
  commission_pct: 0.001           # 10 bps per trade
  slippage_pct: 0.0005            # 5 bps slippage
  bootstrap_samples: 1000         # Bootstrap iterations
  bootstrap_ci: 0.90              # Confidence interval level
  benchmark_ticker: null           # null = buy-and-hold same asset
```

---

## Interpreting Results

### What "good" looks like

- BIC curve has a clear minimum (the model found meaningful regimes)
- Regimes are well-separated in the return distribution histograms
- Walk-forward Sharpe > 1.0 with bootstrap CI not crossing zero
- Strategy equity stays above or near benchmark during drawdowns
- Rolling log-likelihood is stable (model isn't degrading)
- Entropy is low during strong regimes, high during transitions

### Warning signs

- **BIC curve is flat**: The data doesn't have distinct regimes, or you need more data
- **Bootstrap CI for Sharpe crosses zero**: The strategy may not have a real edge
- **Very few trades**: Not enough statistical power to trust the metrics
- **High entropy everywhere**: The model can't distinguish regimes — try fewer states or more data
- **Rolling LL declining**: The market has changed since the model was fit — the walk-forward mechanism should handle this, but check the window sizes
- **All signals are FLAT**: Confidence threshold or min confirmations may be too strict

### Comparing in-sample vs walk-forward

The Regime Analysis tab shows in-sample results (HMM fit on all data). The Backtest tab shows walk-forward results (HMM fit only on past data at each step). Walk-forward metrics should be **lower but positive** — if in-sample looks amazing but walk-forward is flat or negative, the model is overfitting.

---

## Common Workflows

### Workflow 1: Quick regime check for a stock

1. Enter ticker (e.g., `AAPL`), set interval to `1d`, lookback to `365`
2. Set HMM states to 2–4 (simpler for stocks)
3. Click Run Analysis
4. Check Tab 1 (Current Signal) and Tab 2 (Regime Analysis)
5. Focus on the regime overlay chart and current signal

### Workflow 2: Crypto scalping analysis

1. Enter `BTC-USD` or `ETH-USD`, interval `15m`, lookback `30`
2. Keep HMM defaults (2–8 states)
3. Lower min confirmations to 3, cooldown to 2, min hold to 5
4. Run and examine Tab 3 (Backtest) for viability
5. Tune strategy settings based on trade count and win rate

### Workflow 3: Robust strategy validation

1. Set a large lookback (e.g., 365 days, 1d interval)
2. Use train window 500, test window 100, step 50
3. Run analysis and check:
   - Bootstrap CIs don't cross zero for Sharpe
   - Walk-forward return is positive
   - Profit factor > 1.5
4. Examine Tab 5 (Diagnostics) for model stability

### Workflow 4: Comparing different state counts

1. Run with min_states=2, max_states=2 (force 2 states)
2. Note the metrics
3. Run with min_states=3, max_states=3 (force 3 states)
4. Compare BIC scores and backtest metrics
5. Let BIC decide: run with min_states=2, max_states=8

---

## Troubleshooting

### "No data returned for TICKER"

- Check that the ticker symbol is valid on Yahoo Finance
- Some tickers require specific suffixes (e.g., `AAPL` not `APPLE`)
- Very recent IPOs may not have enough history

### "Need at least N bars, got M"

- Your lookback period doesn't provide enough bars for the train + test windows
- Solution: increase lookback days or decrease train_window_bars and test_window_bars in the sidebar

### "All HMM fits failed"

- Usually happens with very few data points or extreme outliers
- Try: reduce max_states, increase lookback, or use `diag` covariance instead of `full`

### Dashboard is slow

- HMM fitting with 20 restarts x 7 state counts = 140 model fits. This is the bottleneck.
- Reduce restarts to 10 or narrow the state range (e.g., 2–4)
- Use `diag` covariance (faster than `full`)
- Reduce lookback period (fewer bars = faster fitting)

### Signals are always FLAT

- Min confirmations may be too high — try 3 instead of 4
- Min confidence may be too high — try 0.4 instead of 0.6
- Hysteresis bars may be too high — try 1 instead of 3
- The model may genuinely be uncertain — check entropy in Tab 5
