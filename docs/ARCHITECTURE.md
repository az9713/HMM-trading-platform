# HMM Regime Terminal — Architecture Document

## Table of Contents

1. [System Overview](#system-overview)
2. [Module Dependency Graph](#module-dependency-graph)
3. [Data Flow](#data-flow)
4. [Module Deep Dives](#module-deep-dives)
   - [data_loader.py](#data_loaderpy)
   - [hmm_engine.py](#hmm_enginepy)
   - [strategy.py](#strategypy)
   - [backtester.py](#backtesterpy)
   - [app.py](#apppy)
   - [config.yaml](#configyaml)
5. [Mathematical Foundations](#mathematical-foundations)
6. [Walk-Forward Protocol](#walk-forward-protocol)
7. [Data Integrity Guarantees](#data-integrity-guarantees)
8. [Performance Characteristics](#performance-characteristics)
9. [Extension Points](#extension-points)

---

## System Overview

The HMM Regime Terminal is a **6-module Python application** that detects market regimes using Hidden Markov Models, generates multi-confirmation trading signals, and evaluates them through walk-forward backtesting with statistical confidence bounds.

### Design Principles

- **No lookahead bias** — All standardization uses training-set statistics only. Walk-forward backtesting strictly separates train/test periods.
- **BIC-driven model selection** — The number of hidden states is determined by data, not guessed.
- **Multi-confirmation gating** — Signals require convergence of regime classification, technical indicators, and confidence thresholds.
- **Statistical rigor** — Bootstrap confidence intervals quantify uncertainty in all key metrics.
- **Configuration-driven** — Every tunable parameter lives in `config.yaml` with sidebar overrides.

### Technology Stack

| Layer | Technology |
|-------|-----------|
| Data source | Yahoo Finance via `yfinance` |
| HMM fitting | `hmmlearn` (Baum-Welch EM, Viterbi, forward-backward) |
| Technical indicators | `ta` library (RSI, ADX, EMA, MACD) |
| Numerical compute | `numpy`, `scipy`, `pandas` |
| Visualization | `plotly` (interactive charts) |
| Dashboard | `streamlit` (reactive web UI) |
| Configuration | `pyyaml` |

---

## Module Dependency Graph

```
config.yaml
    │
    ▼
┌──────────────┐
│ data_loader   │  ← yfinance, ta, numpy, pandas
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ hmm_engine    │  ← hmmlearn, numpy, pandas, scipy
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ strategy      │  ← ta, numpy, pandas
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ backtester    │  ← data_loader, hmm_engine, strategy, numpy, pandas
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ app           │  ← all modules, streamlit, plotly, yaml
└──────────────┘
```

**Key dependency rule**: Modules only import downward. `data_loader` has no internal imports. `hmm_engine` is independent. `strategy` is independent. `backtester` orchestrates the lower three. `app` orchestrates everything.

---

## Data Flow

### Full Analysis Pipeline

```
User clicks "Run Analysis"
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│ 1. FETCH                                                │
│    fetch_ohlcv(ticker, interval, lookback)               │
│    → OHLCV DataFrame (Open, High, Low, Close, Volume)    │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│ 2. FEATURE ENGINEERING                                   │
│    compute_features(df, config)                          │
│    → adds: log_return, rolling_vol, volume_change,       │
│            intraday_range, rsi                           │
└─────────────────────┬───────────────────────────────────┘
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
┌──────────────────┐    ┌──────────────────────────────┐
│ 3a. FULL-SAMPLE  │    │ 3b. WALK-FORWARD             │
│ (Tabs 1, 2, 5)  │    │ (Tabs 3, 4)                  │
│                  │    │                              │
│ standardize(all) │    │ for each fold:               │
│ → fit HMM       │    │   standardize(train, test)   │
│ → decode all     │    │   → fit HMM on train         │
│ → label regimes  │    │   → decode test              │
│ → compute signals│    │   → generate signals         │
│ → entropy        │    │   → simulate trades          │
│ → rolling LL     │    │   → advance window           │
└──────────────────┘    │                              │
                        │ aggregate → equity curve     │
                        │ compute metrics + bootstrap  │
                        └──────────────────────────────┘
```

### Walk-Forward Window Progression

```
Bar index:  0        500       600   650       750   800
            │─────────│─────────│     │─────────│     │
Fold 1:     [  TRAIN (500)    ][TEST(100)]
Fold 2:          [  TRAIN (500)    ][TEST(100)]
                 ↑ shifted by step_bars=50
Fold 3:               [  TRAIN (500)    ][TEST(100)]
...
```

---

## Module Deep Dives

### data_loader.py

**Responsibility**: Data acquisition and feature engineering. No model logic.

#### Functions

**`fetch_ohlcv(ticker, interval, lookback_days) → DataFrame`**

Wraps `yf.download()` with automatic interval-based lookback clamping. yfinance imposes maximum lookback limits per interval (e.g., 7 days for 1-minute data, 730 days for hourly). This function silently clamps to the maximum if the user requests more.

Handles multi-level column flattening (yfinance sometimes returns MultiIndex columns depending on version).

**`compute_features(df, config) → DataFrame`**

Adds 5 columns to the OHLCV DataFrame:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `log_return` | `ln(Close_t / Close_{t-1})` | Stationary, additive, approximately normal — ideal for Gaussian HMM |
| `rolling_vol` | `std(log_return, window=21)` | Captures volatility clustering — key regime differentiator |
| `volume_change` | `Volume / SMA(Volume, 20) - 1` | Relative volume surge/drought — normalized to be scale-free |
| `intraday_range` | `(High - Low) / Close` | Price dispersion within a bar — captures realized volatility |
| `rsi` | RSI(14) via `ta` library | Mean-reversion signal, bounded [0, 100] |

Drops NaN rows introduced by rolling calculations, then resets the index.

**`standardize(train, test, cols) → (train_z, test_z, stats)`**

Z-score normalization: `x_z = (x - mean) / std`, where mean and std come from the **training set only**. This is critical for preventing lookahead bias — the test set is standardized using training statistics.

Returns the stats dict so downstream code can inspect or reuse the parameters.

**`get_feature_matrix(df, cols) → ndarray`**

Trivial extraction of feature columns into a 2D numpy array shaped `(T, d)` for hmmlearn's API.

---

### hmm_engine.py

**Responsibility**: HMM fitting, model selection, regime decoding and analysis. The mathematical core.

#### Class: `RegimeDetector`

**Constructor**: Reads HMM config (state range, restarts, iterations, tolerance, model type, covariance type). Initializes empty result containers.

**`_count_params(n, d) → int`**

Counts the free parameters of a Gaussian HMM with `n` states and `d` features:

```
k = (n-1)           # initial state distribution (π)
  + n*(n-1)         # transition matrix (A) — each row sums to 1, so n-1 free per row
  + n*d             # emission means (μ)
  + covariance_params  # emission covariances (Σ)
```

Covariance parameters depend on structure:
- `full`: `n * d*(d+1)/2` — full symmetric matrix, one per state
- `diag`: `n * d` — diagonal only
- `spherical`: `n` — single scalar per state
- `tied`: `d*(d+1)/2` — one matrix shared across states

This count is used in the BIC formula: `BIC = -2*LL + k*ln(T)`.

**`fit_and_select(X_train) → dict`**

The core model selection loop:

```python
for n in range(min_states, max_states + 1):     # e.g., 2 to 8
    for seed in range(n_restarts):               # e.g., 20 seeds
        model = GaussianHMM(n, random_state=seed)
        model.fit(X_train)                       # Baum-Welch EM
        ll = model.score(X_train)                # log-likelihood
        keep best ll for this n

    BIC_n = -2 * best_ll_n + k * ln(T)
    keep model with lowest BIC across all n
```

**Why 20 restarts?** Baum-Welch (the EM algorithm for HMMs) is sensitive to initialization and converges to local optima. Multiple random restarts increase the probability of finding the global optimum. The best model (highest log-likelihood) across all restarts for a given `n` is kept.

**Why BIC over AIC?** BIC penalizes complexity as `k*ln(T)` vs AIC's `2k`. For large T (typical in financial time series), BIC more aggressively penalizes overfitting, leading to simpler, more generalizable models.

**`decode(X) → (states, posteriors)`**

Two decoding passes:
1. `model.predict(X)` — **Viterbi algorithm**: finds the single most-likely state sequence. Used for regime assignment.
2. `model.predict_proba(X)` — **Forward-backward algorithm**: computes posterior probability of each state at each time step. Used for confidence/entropy.

**`label_regimes(X) → dict`**

States are arbitrary integers (0, 1, 2...). This method sorts them by their emission mean for the first feature (log_return) and assigns semantic labels:

| States | Labels (lowest to highest mean return) |
|--------|----------------------------------------|
| 2 | bear, bull |
| 3 | bear, neutral, bull |
| 4 | crash, bear, bull, bull_run |
| 5 | crash, bear, neutral, bull, bull_run |
| 6+ | crash, regime_1, ..., regime_{n-2}, bull_run |

**`regime_statistics() → DataFrame`**

Extracts from the fitted model:
- **Mean return**: `model.means_[i, 0]` — emission mean of log_return for state i
- **Volatility**: `sqrt(model.covars_[i][0, 0])` — emission std dev (from covariance matrix)
- **Expected duration**: `1 / (1 - A[i,i])` where `A` is the transition matrix. This is the expected number of bars before leaving state i, derived from the geometric distribution of self-transitions.
- **Stationary weight**: Left eigenvector of the transition matrix corresponding to eigenvalue 1. Represents the long-run fraction of time spent in each state.

**`shannon_entropy(posteriors) → (entropy, confidence)`**

Per-bar entropy: `H_t = -Σ_i p_i(t) * log2(p_i(t))`

- Minimum = 0 bits (model is 100% certain about the state)
- Maximum = log2(n) bits (uniform distribution — maximum uncertainty)

Confidence is normalized: `confidence = 1 - H / log2(n)`, ranging from 0 (total uncertainty) to 1 (total certainty).

**`log_likelihood_series(X, window) → ndarray`**

Computes `model.score(X[t-window:t]) / window` for each bar. This rolling per-bar log-likelihood detects model degradation: a declining trend means the current model explains recent data poorly, suggesting the market regime structure has shifted since fitting.

---

### strategy.py

**Responsibility**: Signal generation with multi-confirmation gating and position sizing.

#### Class: `SignalGenerator`

**`compute_confirmations(df) → DataFrame`**

Evaluates 8 independent boolean conditions on the OHLCV + feature DataFrame:

| # | Column | Condition | Rationale |
|---|--------|-----------|-----------|
| 1 | `conf_rsi_not_overbought` | RSI < 70 | Avoid entering longs at overbought levels |
| 2 | `conf_rsi_not_oversold` | RSI > 30 | Avoid catching falling knives |
| 3 | `conf_momentum` | Close > Close[N bars ago] | Price trending upward |
| 4 | `conf_vol_range` | Vol between 20th-80th pctl | Avoid extreme low-vol (chop) and high-vol (chaos) |
| 5 | `conf_volume` | Volume/SMA20 > 1.2 | Confirm with above-average participation |
| 6 | `conf_adx` | ADX > 20 | Market is trending (not ranging) |
| 7 | `conf_above_ema` | Close > EMA(50) | Intermediate-term uptrend |
| 8 | `conf_macd` | MACD > Signal | Bullish momentum crossover |

Adds `n_confirmations` column = sum of all 8 booleans.

**`generate_signals(df, states, posteriors, labels, confidence) → Series`**

State machine that produces signal values: 1 (long), -1 (short), 0 (flat).

Logic per bar (in priority order):

1. **Min hold check**: If in a position and fewer than `min_hold_bars` have passed → maintain current position
2. **Cooldown check**: If exited a position within `cooldown_bars` → stay flat
3. **Hysteresis check**: If the current regime hasn't persisted for `hysteresis_bars` → maintain current position
4. **Confidence check**: If `confidence[t] < min_confidence` → maintain current position
5. **Signal generation**:
   - State in {bull, bull_run} AND n_confirmations >= min_confirmations → LONG
   - State in {bear, crash} AND n_confirmations >= min_confirmations → SHORT
   - Otherwise → FLAT

The hysteresis filter prevents whipsawing on regime boundaries. The cooldown prevents overtrading after exits. Min hold ensures trades have time to develop.

**`compute_position_size(confidence, win_rate, avg_win, avg_loss) → float`**

Three-stage sizing:

1. **Kelly criterion**: `f* = (p*b - q) / b` where `p` = win rate, `q` = 1-p, `b` = avg_win/avg_loss
2. **Half-Kelly**: `size = f* * kelly_fraction` (default 0.5 — reduces variance at the cost of lower expected growth)
3. **Entropy scaling**: `size *= confidence` (when enabled — reduces allocation when the model is uncertain)
4. **Cap**: `min(size, max_leverage * max_position_pct)`

---

### backtester.py

**Responsibility**: Walk-forward orchestration, trade simulation, metric computation, bootstrap CIs.

#### Data Structures

**`TradeRecord`** (dataclass): Stores entry/exit bar, prices, direction, PnL (dollar and percentage), regime, confirmations, and position size for each round-trip trade.

**`BacktestResult`** (dataclass): Contains equity curve, benchmark curve, trade list, metrics dict, regime/confidence series, and CI bounds.

#### Class: `WalkForwardBacktester`

**`run(df) → BacktestResult`**

The walk-forward loop:

```
for each fold (start advancing by step_bars):
    train_df = df[start : start + train_window]
    test_df  = df[start + train_window : start + train_window + test_window]

    train_z, test_z = standardize(train_df, test_df)  # train stats only!

    detector = RegimeDetector(config)
    detector.fit_and_select(X_train)    # fresh HMM each fold
    states, posteriors = detector.decode(X_test)

    sig_gen = SignalGenerator(config)
    signals = sig_gen.generate_signals(...)

    # accumulate signals, sizes, regimes for test period
    start += step_bars
```

After all folds, calls `simulate_trades()` on the aggregated signal series, then `compute_metrics()` and `bootstrap_confidence_intervals()`.

**Critical design choice**: A **new HMM is fit from scratch each fold**. This means:
- The model adapts to changing market structure over time
- No single model fit contaminates the entire backtest
- Computational cost is O(n_folds * n_states * n_restarts)

**`simulate_trades(df, signals, sizes) → (equity, trades)`**

Bar-by-bar simulation:

1. **Mark-to-market**: If in a position, update capital by `(price_t / price_{t-1} - 1) * direction * size`
2. **Position changes**: When signal differs from current position:
   - Close existing: apply slippage, deduct commission, record TradeRecord
   - Open new: apply slippage, deduct commission, record entry
3. **Costs**: Commission = `capital * commission_pct` per trade side. Slippage = `price * slippage_pct` applied adversely.

**`compute_metrics(equity, trades, benchmark) → dict`**

| Metric | Formula | Notes |
|--------|---------|-------|
| Sharpe | `mean(r) / std(r) * sqrt(8760)` | Annualized assuming hourly bars |
| Sortino | `mean(r) / std(r_negative) * sqrt(8760)` | Only penalizes downside |
| Calmar | `annualized_return / abs(max_drawdown)` | Return per unit of worst-case loss |
| Max DD | `min((equity - cummax) / cummax)` | Deepest trough from peak |
| Max DD Duration | Longest consecutive bars in drawdown | Measured in bars |
| CVaR (5%) | `mean(r | r <= VaR_5%)` | Expected loss in worst 5% of bars |
| Win Rate | `n_winning_trades / n_total_trades` | Percentage |
| Profit Factor | `gross_profit / gross_loss` | > 1 = net profitable |
| Alpha | `strategy_return - benchmark_return` | Excess return vs buy-and-hold |

**`bootstrap_confidence_intervals(equity) → (ci_lower, ci_upper)`**

Non-parametric bootstrap:

1. Compute bar-by-bar returns from equity curve
2. For each of 1,000 iterations:
   - Resample returns with replacement (same length)
   - Reconstruct synthetic equity curve: `cumprod(1 + resampled_returns)`
   - Compute Sharpe, total return, max drawdown on synthetic curve
3. Take the 5th and 95th percentiles (for 90% CI) across all bootstrap samples

This gives confidence intervals that account for the actual return distribution (fat tails, skew, autocorrelation broken by resampling).

---

### app.py

**Responsibility**: Streamlit dashboard — user interface, parameter controls, visualization.

#### Architecture

The app follows Streamlit's **top-down execution model**: the entire script reruns on every interaction. State is managed through Streamlit's widget return values (sidebar controls) and the `run_btn` flag.

**Execution guard**: All analysis code is inside `if run_btn:`, so it only executes when the user clicks "Run Analysis".

**Config override pattern**: `build_config()` creates a runtime config dict that merges `config.yaml` defaults with sidebar widget values. This config is passed to all downstream modules.

#### Tab Structure

| Tab | Data Sources | Key Visualizations |
|-----|-------------|-------------------|
| Current Signal | Last row of decoded DataFrame | Regime banner, confirmation table, position size |
| Regime Analysis | Full decoded DataFrame, transition matrix, BIC scores | Price overlay, heatmap, histograms, BIC curve |
| Backtest Results | BacktestResult from walk-forward | Metric cards, equity curve, drawdown |
| Trade Log | BacktestResult.trades | Sortable table |
| Model Diagnostics | Rolling LL, entropy, feature correlations | Time series, correlation heatmap |

#### Color Scheme

```python
REGIME_COLORS = {
    "crash":    "#d32f2f",  # red
    "bear":     "#f57c00",  # orange
    "neutral":  "#9e9e9e",  # gray
    "bull":     "#388e3c",  # green
    "bull_run": "#1565c0",  # blue
    "unknown":  "#e0e0e0",  # light gray
}
```

---

### config.yaml

**Responsibility**: Single source of truth for all tunable parameters.

#### Section Hierarchy

```
config.yaml
├── data          # Ticker, interval, lookback, feature params
├── hmm           # State range, restarts, iterations, model type
├── strategy      # Confirmation thresholds, cooldown, hysteresis
│   └── confirmations  # Individual indicator thresholds
├── risk          # Kelly, entropy scaling, leverage caps
└── backtest      # Window sizes, costs, bootstrap settings
```

Every parameter has a sensible default. The sidebar in `app.py` exposes the most frequently tuned parameters. Less common parameters (e.g., `kelly_fraction`, `commission_pct`, `rsi_period`) require editing the YAML file directly.

---

## Mathematical Foundations

### Hidden Markov Model (HMM)

An HMM assumes the observed market features `X_t` are generated by a latent (hidden) state `S_t` that follows a Markov chain:

```
S_1 → S_2 → S_3 → ... → S_T     (hidden states — regimes)
 ↓      ↓      ↓            ↓
X_1    X_2    X_3    ...   X_T     (observed features)
```

**Parameters**:
- `π` — Initial state distribution: `P(S_1 = i)`
- `A` — Transition matrix: `A[i,j] = P(S_{t+1} = j | S_t = i)`
- `μ_i, Σ_i` — Emission parameters: `X_t | S_t = i ~ N(μ_i, Σ_i)`

**Training (Baum-Welch)**: EM algorithm that iterates between:
- E-step: Compute expected state occupancies given current parameters (forward-backward)
- M-step: Update parameters to maximize expected log-likelihood

**Decoding (Viterbi)**: Dynamic programming to find `argmax_{S_1:T} P(S_1:T | X_1:T)` — the most likely state sequence.

**Posteriors (Forward-Backward)**: Computes `P(S_t = i | X_1:T)` for every state at every time step. Unlike Viterbi (which gives a single sequence), this gives a probability distribution over states per bar.

### BIC Model Selection

```
BIC = -2 * LL + k * ln(T)
```

- `LL` = log-likelihood of the data under the model
- `k` = number of free parameters
- `T` = number of observations

Lower BIC = better balance of fit and parsimony. The `k * ln(T)` penalty grows with both model complexity and data size, preventing complex models from winning just because they memorize the data.

### Shannon Entropy

```
H_t = -Σ_i P(S_t = i | X) * log₂(P(S_t = i | X))
```

Measures uncertainty in bits. For `n` states:
- H = 0: one state has probability 1 (perfect certainty)
- H = log₂(n): all states equally likely (maximum uncertainty)

Normalized confidence: `c = 1 - H / log₂(n)`, ranging [0, 1].

### Kelly Criterion

```
f* = (p * b - q) / b
```

- `p` = win probability
- `q` = 1 - p
- `b` = ratio of average win to average loss

`f*` is the fraction of capital to bet that maximizes long-run geometric growth rate. In practice, half-Kelly (`0.5 * f*`) is used because:
- It gives ~75% of Kelly's growth rate
- It reduces variance by ~50%
- It's more robust to estimation error in `p` and `b`

### Walk-Forward Validation

Unlike in-sample fitting (which sees the future), walk-forward strictly enforces temporal ordering:

1. Train on bars `[0, T_train)`
2. Generate signals on bars `[T_train, T_train + T_test)`
3. Advance by `step` bars
4. Repeat

**Why not simple train/test split?** A single split gives one sample of out-of-sample performance. Walk-forward gives many overlapping samples, providing a more robust estimate.

**Why not cross-validation?** Time series data has autocorrelation — random K-fold splits would leak future information into training folds. Walk-forward respects the arrow of time.

### Bootstrap Confidence Intervals

1. Given observed returns `r_1, r_2, ..., r_T`
2. Resample with replacement to create `r*_1, r*_2, ..., r*_T`
3. Compute metric (e.g., Sharpe) on the resampled series
4. Repeat 1,000 times
5. Take percentiles for CI bounds

**Limitation**: Resampling breaks autocorrelation structure. Block bootstrap would preserve it but adds complexity. For a first-order assessment, iid bootstrap is adequate.

---

## Data Integrity Guarantees

### No Lookahead Bias

| Stage | Guarantee |
|-------|-----------|
| Standardization | `standardize()` computes mean/std from `train` only, applies to `test` |
| HMM fitting | `fit_and_select()` only sees `X_train` |
| HMM decoding | `decode()` runs on `X_test` using the model fit on `X_train` |
| Confirmation signals | `compute_confirmations()` uses only past/current data (RSI, MACD, etc. are causal) |
| Signal generation | `generate_signals()` iterates forward in time, using only past states/confidence |

### Trade Simulation Realism

| Cost | Implementation |
|------|---------------|
| Commission | `capital * commission_pct` deducted on each trade open and close |
| Slippage | Price adjusted adversely by `slippage_pct` on entry and exit |
| No partial fills | Assumes full execution at the adjusted price |
| Bar-level execution | Trades execute at the close price of the signal bar (not intra-bar) |

---

## Performance Characteristics

### Computational Bottlenecks

| Operation | Complexity | Typical Time |
|-----------|-----------|-------------|
| Data fetch | O(T) network | 1-3 seconds |
| Feature engineering | O(T) | < 100ms |
| HMM fitting (per model) | O(T * n² * d² * n_iter) | 0.2-2 seconds |
| Total HMM selection | Above × n_states × n_restarts | 20-120 seconds |
| Walk-forward backtest | HMM selection × n_folds | 2-10 minutes |
| Bootstrap CIs | O(n_samples * T) | 1-5 seconds |

### Memory Usage

- OHLCV data: ~500 bytes/bar × T bars
- Feature matrix: ~40 bytes/bar × T bars × 5 features
- HMM model: ~n² * d² * 8 bytes (negligible)
- Posteriors: T × n × 8 bytes
- Equity curve: T × 8 bytes
- Bootstrap: n_samples × T × 8 bytes (peak ~8 MB for 1000 samples × 1000 bars)

Total memory for a typical run (2000 bars, 3 states): < 50 MB.

---

## Extension Points

### Adding New Features

1. Add computation in `compute_features()` in `data_loader.py`
2. Add feature name to `config.yaml` → `data.features` list
3. The rest of the pipeline picks it up automatically

### Adding New Confirmation Conditions

1. Add boolean column computation in `compute_confirmations()` in `strategy.py`
2. Name it with the `conf_` prefix
3. It's automatically counted in `n_confirmations`
4. Update `min_confirmations` if needed

### Supporting New Model Types

1. Import the model class in `hmm_engine.py`
2. Add a branch in `fit_and_select()` based on `self.model_type`
3. Adjust `_count_params()` for the new model's parameter count
4. Ensure `decode()`, `label_regimes()`, and `regime_statistics()` work with the new model's API

### Adding New Metrics

1. Add computation in `compute_metrics()` in `backtester.py`
2. Add to the return dict
3. Add display in the relevant tab in `app.py`
4. Optionally add to `bootstrap_confidence_intervals()` for CI bounds

### Multi-Asset Support

The current architecture processes one ticker at a time. To support portfolios:
1. Run `fetch_ohlcv()` and `compute_features()` per asset
2. Fit separate HMMs per asset (regime structure is asset-specific)
3. Combine signals in a portfolio-level allocation layer (new module)
4. Feed combined signals into `backtester` with per-asset position tracking
