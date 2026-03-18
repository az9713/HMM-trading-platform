"""
app.py — Streamlit dashboard with 5 tabs for HMM Regime Terminal.

Tabs: Current Signal, Regime Analysis, Backtest Results, Trade Log, Model Diagnostics.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml

from data_loader import fetch_ohlcv, compute_features, standardize, get_feature_matrix
from hmm_engine import RegimeDetector
from strategy import SignalGenerator
from backtester import WalkForwardBacktester

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="HMM Regime Terminal", layout="wide")

# ── Global CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

:root {
    --term-bg: #0a0e17;
    --term-surface: #111827;
    --term-border: #1e293b;
    --term-green: #00e599;
    --term-amber: #f59e0b;
    --term-red: #ef4444;
    --term-blue: #3b82f6;
    --term-cyan: #06b6d4;
    --term-text: #e2e8f0;
    --term-muted: #64748b;
    --term-glow: rgba(0, 229, 153, 0.15);
}

/* Hide default Streamlit title */
.stApp > header { background: transparent; }

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1729 0%, #0a0e17 100%);
    border-right: 1px solid var(--term-border);
}
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'JetBrains Mono', monospace;
    color: var(--term-green);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    border-bottom: 1px solid var(--term-border);
    padding-bottom: 0.4rem;
    margin-top: 1.2rem;
}

/* Sidebar logo area */
.sidebar-brand {
    text-align: center;
    padding: 1.5rem 0.5rem 1rem;
    border-bottom: 1px solid var(--term-border);
    margin-bottom: 0.5rem;
}
.sidebar-brand .logo-icon {
    font-size: 2rem;
    display: block;
    margin-bottom: 0.3rem;
    line-height: 1;
}
.sidebar-brand .logo-wordmark {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 1.05rem;
    color: #f1f5f9;
    letter-spacing: 0.08em;
}
.sidebar-brand .logo-sub {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 300;
    font-size: 0.6rem;
    color: var(--term-muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 0.15rem;
}
.sidebar-brand .version-badge {
    display: inline-block;
    margin-top: 0.5rem;
    background: rgba(0, 229, 153, 0.1);
    border: 1px solid rgba(0, 229, 153, 0.25);
    border-radius: 9999px;
    padding: 0.15rem 0.55rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem;
    color: var(--term-green);
    letter-spacing: 0.08em;
}

/* ── Landing page styles ── */
.landing-hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
    position: relative;
}
.landing-hero::before {
    content: '';
    position: absolute;
    top: 0; left: 50%;
    transform: translateX(-50%);
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(0,229,153,0.06) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}
.hero-overline {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--term-green);
    letter-spacing: 0.35em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    position: relative; z-index: 1;
}
.hero-title {
    font-family: 'DM Sans', sans-serif;
    font-weight: 700;
    font-size: 3.2rem;
    line-height: 1.1;
    color: #f8fafc;
    margin-bottom: 0.5rem;
    position: relative; z-index: 1;
}
.hero-title .accent {
    background: linear-gradient(135deg, var(--term-green) 0%, var(--term-cyan) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    font-size: 1.15rem;
    color: var(--term-muted);
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.6;
    position: relative; z-index: 1;
}
.hero-divider {
    width: 60px;
    height: 2px;
    background: linear-gradient(90deg, var(--term-green), var(--term-cyan));
    margin: 2rem auto;
    border-radius: 1px;
}

/* Pipeline steps */
.pipeline-section { margin: 1rem auto 2.5rem; max-width: 1000px; }
.pipeline-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--term-muted);
    letter-spacing: 0.25em;
    text-transform: uppercase;
    text-align: center;
    margin-bottom: 1rem;
}
.pipeline-row {
    display: flex;
    align-items: center;
    justify-content: center;
    flex-wrap: wrap;
    gap: 0;
}
.pipe-step {
    display: flex;
    align-items: center;
    gap: 0;
}
.pipe-node {
    background: var(--term-surface);
    border: 1px solid var(--term-border);
    border-radius: 8px;
    padding: 0.65rem 1.1rem;
    text-align: center;
    transition: border-color 0.3s, box-shadow 0.3s;
    min-width: 120px;
}
.pipe-node:hover {
    border-color: var(--term-green);
    box-shadow: 0 0 20px var(--term-glow);
}
.pipe-node .step-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem;
    color: var(--term-green);
    letter-spacing: 0.15em;
    margin-bottom: 0.2rem;
}
.pipe-node .step-name {
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    color: #f1f5f9;
}
.pipe-node .step-detail {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem;
    color: var(--term-muted);
    margin-top: 0.15rem;
}
.pipe-arrow {
    font-family: 'JetBrains Mono', monospace;
    color: var(--term-border);
    font-size: 1.2rem;
    padding: 0 0.35rem;
    user-select: none;
}

/* Math concept cards */
.concepts-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem;
    max-width: 1100px;
    margin: 0 auto 2.5rem;
}
.concept-card {
    background: var(--term-surface);
    border: 1px solid var(--term-border);
    border-radius: 10px;
    padding: 1.25rem 1.1rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.25s, border-color 0.3s, box-shadow 0.3s;
}
.concept-card:hover {
    transform: translateY(-3px);
    border-color: var(--term-green);
    box-shadow: 0 8px 30px rgba(0,0,0,0.3), 0 0 20px var(--term-glow);
}
.concept-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.concept-card.card-green::before { background: linear-gradient(90deg, var(--term-green), transparent); }
.concept-card.card-amber::before { background: linear-gradient(90deg, var(--term-amber), transparent); }
.concept-card.card-blue::before  { background: linear-gradient(90deg, var(--term-blue), transparent); }
.concept-card.card-cyan::before  { background: linear-gradient(90deg, var(--term-cyan), transparent); }
.concept-card.card-red::before   { background: linear-gradient(90deg, var(--term-red), transparent); }

.concept-icon {
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
    display: block;
}
.concept-title {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 0.85rem;
    color: #f1f5f9;
    margin-bottom: 0.3rem;
}
.concept-formula {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--term-green);
    background: rgba(0,229,153,0.06);
    padding: 0.3rem 0.5rem;
    border-radius: 4px;
    display: inline-block;
    margin-bottom: 0.4rem;
    border: 1px solid rgba(0,229,153,0.1);
}
.concept-desc {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
    color: var(--term-muted);
    line-height: 1.5;
}

/* Getting started section */
.getting-started {
    max-width: 700px;
    margin: 0 auto 3rem;
    text-align: center;
    padding: 2rem 1.5rem;
    background: var(--term-surface);
    border: 1px solid var(--term-border);
    border-radius: 12px;
    position: relative;
    overflow: hidden;
}
.getting-started::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--term-green), var(--term-cyan), var(--term-blue));
}
.gs-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--term-green);
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.gs-steps {
    text-align: left;
    display: inline-block;
}
.gs-step {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    margin-bottom: 0.85rem;
}
.gs-step-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--term-bg);
    background: var(--term-green);
    width: 20px; height: 20px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    font-weight: 700;
    margin-top: 1px;
}
.gs-step-text {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.88rem;
    color: var(--term-text);
    line-height: 1.45;
}
.gs-step-text strong {
    color: #f1f5f9;
}

/* Footer */
.landing-footer {
    text-align: center;
    padding: 1.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: var(--term-muted);
    letter-spacing: 0.1em;
}
.landing-footer .dot {
    display: inline-block;
    width: 4px; height: 4px;
    background: var(--term-green);
    border-radius: 50%;
    margin: 0 0.6rem;
    vertical-align: middle;
}
</style>
""", unsafe_allow_html=True)

st.title("HMM Regime Terminal")

# ── Load config ──────────────────────────────────────────────────────────────

@st.cache_data
def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

base_config = load_config()

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <span class="logo-icon">
            <svg width="36" height="36" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect x="1" y="1" width="34" height="34" rx="8" stroke="#1e293b" stroke-width="1" fill="#111827"/>
                <path d="M8 24 L12 18 L16 20 L20 12 L24 16 L28 10" stroke="#00e599" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
                <circle cx="12" cy="18" r="2" fill="#00e599" opacity="0.6"/>
                <circle cx="20" cy="12" r="2" fill="#00e599" opacity="0.6"/>
                <circle cx="28" cy="10" r="2" fill="#00e599"/>
            </svg>
        </span>
        <div class="logo-wordmark">HMM REGIME</div>
        <div class="logo-sub">Terminal</div>
        <span class="version-badge">v1.0.0</span>
    </div>
    """, unsafe_allow_html=True)

    st.header("Data Settings")
    ticker = st.text_input("Ticker", base_config["data"]["default_ticker"])
    interval = st.selectbox("Interval", ["1m", "5m", "15m", "1h", "1d"],
                            index=["1m", "5m", "15m", "1h", "1d"].index(base_config["data"]["default_interval"]))
    lookback = st.slider("Lookback (days)", 7, 730, base_config["data"]["default_lookback_days"])

    st.header("HMM Settings")
    min_states = st.slider("Min states", 2, 6, base_config["hmm"]["min_states"])
    max_states = st.slider("Max states", 3, 10, base_config["hmm"]["max_states"])
    model_type = st.selectbox("Model type", ["gaussian", "gmm"],
                              index=0 if base_config["hmm"]["model_type"] == "gaussian" else 1)
    cov_type = st.selectbox("Covariance", ["full", "diag", "tied", "spherical"],
                            index=["full", "diag", "tied", "spherical"].index(base_config["hmm"]["covariance_type"]))
    n_restarts = st.slider("Random restarts", 5, 50, base_config["hmm"]["n_restarts"])

    st.header("Strategy Settings")
    min_conf = st.slider("Min confirmations", 1, 8, base_config["strategy"]["min_confirmations"])
    cooldown = st.slider("Cooldown bars", 0, 20, base_config["strategy"]["cooldown_bars"])
    min_hold = st.slider("Min hold bars", 1, 50, base_config["strategy"]["min_hold_bars"])
    min_regime_conf = st.slider("Min regime confidence", 0.0, 1.0,
                                base_config["strategy"]["confirmations"]["min_confidence"], 0.05)

    st.header("Risk Settings")
    use_kelly = st.checkbox("Kelly sizing", base_config["risk"]["use_kelly"])
    use_entropy = st.checkbox("Entropy scaling", base_config["risk"]["use_entropy_scaling"])
    max_leverage = st.slider("Max leverage", 0.5, 5.0, base_config["risk"]["max_leverage"], 0.5)

    st.header("Backtest Settings")
    train_window = st.slider("Train window (bars)", 100, 2000,
                             base_config["backtest"]["train_window_bars"], 50)
    test_window = st.slider("Test window (bars)", 20, 500,
                            base_config["backtest"]["test_window_bars"], 10)
    step_bars = st.slider("Step size (bars)", 10, 200,
                          base_config["backtest"]["step_bars"], 10)

    run_btn = st.button("Run Analysis", type="primary", use_container_width=True)

# ── Build runtime config ─────────────────────────────────────────────────────

def build_config():
    cfg = base_config.copy()
    cfg["data"] = {**base_config["data"], "default_ticker": ticker,
                   "default_interval": interval, "default_lookback_days": lookback}
    cfg["hmm"] = {**base_config["hmm"], "min_states": min_states, "max_states": max_states,
                  "model_type": model_type, "covariance_type": cov_type, "n_restarts": n_restarts}
    cfg["strategy"] = {**base_config["strategy"], "min_confirmations": min_conf,
                       "cooldown_bars": cooldown, "min_hold_bars": min_hold,
                       "confirmations": {**base_config["strategy"]["confirmations"],
                                         "min_confidence": min_regime_conf}}
    cfg["risk"] = {**base_config["risk"], "use_kelly": use_kelly,
                   "use_entropy_scaling": use_entropy, "max_leverage": max_leverage}
    cfg["backtest"] = {**base_config["backtest"], "train_window_bars": train_window,
                       "test_window_bars": test_window, "step_bars": step_bars}
    return cfg

# ── Regime color map ─────────────────────────────────────────────────────────

REGIME_COLORS = {
    "crash": "#d32f2f",
    "bear": "#f57c00",
    "neutral": "#9e9e9e",
    "bull": "#388e3c",
    "bull_run": "#1565c0",
    "unknown": "#e0e0e0",
}

def get_regime_color(label: str) -> str:
    for key, color in REGIME_COLORS.items():
        if key in label:
            return color
    return "#e0e0e0"

# ── Main analysis ────────────────────────────────────────────────────────────

if run_btn:
    config = build_config()
    feature_cols = config["data"]["features"]

    with st.spinner("Fetching data..."):
        df = fetch_ohlcv(ticker, interval, lookback)
        df = compute_features(df, config["data"])

    st.success(f"Loaded {len(df)} bars for {ticker} ({interval})")

    # ── Fit HMM on full dataset for current signal / regime analysis ──
    with st.spinner("Fitting HMM (BIC selection with random restarts)..."):
        train_z, _, stats = standardize(df, cols=feature_cols)
        X = get_feature_matrix(train_z, feature_cols)

        detector = RegimeDetector(config)
        bic_scores = detector.fit_and_select(X)
        states, posteriors = detector.decode(X)
        labels = detector.label_regimes(X)
        entropy, confidence = detector.shannon_entropy(posteriors)
        regime_stats = detector.regime_statistics()
        transmat = detector.transition_matrix()
        ll_series = detector.log_likelihood_series(X, window=50)

    st.success(f"Selected {detector.n_states} states (BIC)")

    # ── Compute confirmations and signals ──
    sig_gen = SignalGenerator(config)
    df_conf = sig_gen.compute_confirmations(df)
    signals = sig_gen.generate_signals(df_conf, states, posteriors, labels, confidence)

    # Add regime labels to df
    df["regime"] = [labels.get(s, "unknown") for s in states]
    df["confidence"] = confidence
    df["entropy"] = entropy
    df["signal"] = signals.values

    # ── Tabs ─────────────────────────────────────────────────────────────

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Current Signal", "Regime Analysis", "Backtest Results",
        "Trade Log", "Model Diagnostics"
    ])

    # ── Tab 1: Current Signal ────────────────────────────────────────────

    with tab1:
        last = df.iloc[-1]
        regime = last["regime"]
        conf_pct = last["confidence"] * 100
        signal_val = int(last["signal"])
        signal_text = {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(signal_val, "FLAT")
        signal_color = {1: "green", -1: "red", 0: "gray"}.get(signal_val, "gray")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"### Regime: **{regime.upper()}**")
            st.markdown(f"<div style='background:{get_regime_color(regime)};padding:20px;"
                        f"border-radius:10px;text-align:center;color:white;font-size:24px'>"
                        f"{regime.upper()}</div>", unsafe_allow_html=True)
        with col2:
            st.metric("Confidence", f"{conf_pct:.1f}%")
            st.metric("Signal", signal_text)
        with col3:
            pos_size = sig_gen.compute_position_size(last["confidence"])
            st.metric("Position Size", f"{pos_size:.1%}")

        st.subheader("Confirmation Breakdown")
        conf_cols = [c for c in df_conf.columns if c.startswith("conf_")]
        conf_data = []
        for c in conf_cols:
            name = c.replace("conf_", "").replace("_", " ").title()
            val = bool(df_conf[c].iloc[-1])
            conf_data.append({"Condition": name, "Met": val})
        conf_df = pd.DataFrame(conf_data)
        st.dataframe(conf_df, use_container_width=True, hide_index=True)

        st.metric("Total Confirmations", f"{int(df_conf['n_confirmations'].iloc[-1])} / {len(conf_cols)}")

    # ── Tab 2: Regime Analysis ───────────────────────────────────────────

    with tab2:
        # Price chart with regime overlay
        st.subheader("Price with Regime Overlay")
        fig = go.Figure()

        date_col = "Date" if "Date" in df.columns else "Datetime" if "Datetime" in df.columns else df.index.name
        x_vals = df["Date"] if "Date" in df.columns else df.index

        # Add regime background shading
        unique_regimes = df["regime"].unique()
        for regime_label in unique_regimes:
            mask = df["regime"] == regime_label
            indices = df.index[mask]
            if len(indices) == 0:
                continue
            # Create regime overlay as scatter with fill
            fig.add_trace(go.Scatter(
                x=x_vals[mask], y=df["Close"][mask],
                mode="markers", marker=dict(size=3, color=get_regime_color(regime_label)),
                name=regime_label, legendgroup=regime_label,
            ))

        fig.add_trace(go.Scatter(
            x=x_vals, y=df["Close"], mode="lines",
            line=dict(color="black", width=1), name="Price",
            showlegend=True,
        ))
        fig.update_layout(height=500, xaxis_title="", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)

        # Transition heatmap
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Transition Matrix")
            state_labels = [labels.get(i, f"S{i}") for i in range(detector.n_states)]
            fig_tm = px.imshow(
                transmat, x=state_labels, y=state_labels,
                color_continuous_scale="Blues", text_auto=".2f",
                labels=dict(x="To", y="From", color="P"),
            )
            fig_tm.update_layout(height=400)
            st.plotly_chart(fig_tm, use_container_width=True)

        with col2:
            st.subheader("Regime Statistics")
            st.dataframe(regime_stats.round(4), use_container_width=True, hide_index=True)

        # Per-regime return distributions
        st.subheader("Return Distributions by Regime")
        fig_dist = go.Figure()
        for regime_label in unique_regimes:
            mask = df["regime"] == regime_label
            rets = df.loc[mask, "log_return"]
            fig_dist.add_trace(go.Histogram(
                x=rets, name=regime_label, opacity=0.6,
                marker_color=get_regime_color(regime_label),
            ))
        fig_dist.update_layout(barmode="overlay", height=400,
                               xaxis_title="Log Return", yaxis_title="Count")
        st.plotly_chart(fig_dist, use_container_width=True)

        # BIC/AIC curve
        st.subheader("BIC / AIC Model Selection")
        bic_df = pd.DataFrame({
            "States": list(bic_scores.keys()),
            "BIC": list(bic_scores.values()),
            "AIC": [detector.aic_scores[k] for k in bic_scores.keys()],
        })
        fig_bic = go.Figure()
        fig_bic.add_trace(go.Scatter(x=bic_df["States"], y=bic_df["BIC"],
                                     mode="lines+markers", name="BIC"))
        fig_bic.add_trace(go.Scatter(x=bic_df["States"], y=bic_df["AIC"],
                                     mode="lines+markers", name="AIC"))
        fig_bic.update_layout(height=350, xaxis_title="Number of States",
                              yaxis_title="Information Criterion")
        st.plotly_chart(fig_bic, use_container_width=True)

    # ── Tab 3: Backtest Results ──────────────────────────────────────────

    with tab3:
        with st.spinner("Running walk-forward backtest..."):
            backtester = WalkForwardBacktester(config)
            try:
                bt_result = backtester.run(df)

                # Metric cards with CIs
                st.subheader("Performance Metrics (with 90% Bootstrap CIs)")
                m = bt_result.metrics
                ci_lo = bt_result.ci_lower
                ci_hi = bt_result.ci_upper

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Total Return", f"{m['total_return']:.2%}",
                              help=f"CI: [{ci_lo.get('total_return',0):.2%}, {ci_hi.get('total_return',0):.2%}]")
                    st.metric("Win Rate", f"{m['win_rate']:.1%}")
                with c2:
                    st.metric("Sharpe Ratio", f"{m['sharpe_ratio']:.2f}",
                              help=f"CI: [{ci_lo.get('sharpe_ratio',0):.2f}, {ci_hi.get('sharpe_ratio',0):.2f}]")
                    st.metric("Sortino Ratio", f"{m['sortino_ratio']:.2f}")
                with c3:
                    st.metric("Max Drawdown", f"{m['max_drawdown']:.2%}",
                              help=f"CI: [{ci_lo.get('max_drawdown',0):.2%}, {ci_hi.get('max_drawdown',0):.2%}]")
                    st.metric("Max DD Duration", f"{m['max_dd_duration']} bars")
                with c4:
                    st.metric("Calmar Ratio", f"{m['calmar_ratio']:.2f}")
                    st.metric("CVaR (5%)", f"{m['cvar_5pct']:.4f}")

                c5, c6, c7 = st.columns(3)
                with c5:
                    st.metric("Alpha", f"{m['alpha']:.2%}")
                with c6:
                    st.metric("Profit Factor", f"{m['profit_factor']:.2f}")
                with c7:
                    st.metric("Total Trades", m["n_trades"])

                # Equity curve vs benchmark
                st.subheader("Equity Curve vs Benchmark")
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(
                    x=x_vals, y=bt_result.equity_curve,
                    mode="lines", name="Strategy", line=dict(color="blue"),
                ))
                fig_eq.add_trace(go.Scatter(
                    x=x_vals, y=bt_result.benchmark_curve,
                    mode="lines", name="Buy & Hold", line=dict(color="gray", dash="dash"),
                ))
                fig_eq.update_layout(height=450, yaxis_title="Equity ($)")
                st.plotly_chart(fig_eq, use_container_width=True)

                # Drawdown chart
                st.subheader("Drawdown")
                cummax = bt_result.equity_curve.cummax()
                dd = (bt_result.equity_curve - cummax) / cummax
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=x_vals, y=dd, fill="tozeroy",
                    line=dict(color="red"), name="Drawdown",
                ))
                fig_dd.update_layout(height=300, yaxis_title="Drawdown %")
                st.plotly_chart(fig_dd, use_container_width=True)

            except ValueError as e:
                st.error(str(e))

    # ── Tab 4: Trade Log ─────────────────────────────────────────────────

    with tab4:
        st.subheader("Trade Log")
        if "bt_result" in dir() and bt_result.trades:
            trade_data = []
            for t in bt_result.trades:
                trade_data.append({
                    "Entry Bar": t.entry_bar,
                    "Exit Bar": t.exit_bar,
                    "Direction": "Long" if t.direction == 1 else "Short",
                    "Entry Price": f"{t.entry_price:.2f}",
                    "Exit Price": f"{t.exit_price:.2f}",
                    "PnL ($)": f"{t.pnl:.2f}",
                    "PnL (%)": f"{t.pnl_pct:.4f}",
                    "Position Size": f"{t.position_size:.2%}",
                    "Bars Held": t.exit_bar - t.entry_bar,
                })
            trade_df = pd.DataFrame(trade_data)
            st.dataframe(trade_df, use_container_width=True, hide_index=True)
        else:
            st.info("Run backtest to see trade log.")

    # ── Tab 5: Model Diagnostics ─────────────────────────────────────────

    with tab5:
        col1, col2 = st.columns(2)

        with col1:
            # Rolling log-likelihood
            st.subheader("Rolling Log-Likelihood")
            fig_ll = go.Figure()
            fig_ll.add_trace(go.Scatter(
                x=x_vals, y=ll_series, mode="lines",
                line=dict(color="purple"), name="LL/bar",
            ))
            fig_ll.update_layout(height=350, yaxis_title="Log-Likelihood per bar")
            st.plotly_chart(fig_ll, use_container_width=True)

        with col2:
            # Entropy time series
            st.subheader("Shannon Entropy / Confidence")
            fig_ent = make_subplots(specs=[[{"secondary_y": True}]])
            fig_ent.add_trace(go.Scatter(
                x=x_vals, y=entropy, mode="lines",
                line=dict(color="orange"), name="Entropy",
            ), secondary_y=False)
            fig_ent.add_trace(go.Scatter(
                x=x_vals, y=confidence, mode="lines",
                line=dict(color="teal"), name="Confidence",
            ), secondary_y=True)
            fig_ent.update_yaxes(title_text="Entropy (bits)", secondary_y=False)
            fig_ent.update_yaxes(title_text="Confidence", secondary_y=True)
            fig_ent.update_layout(height=350)
            st.plotly_chart(fig_ent, use_container_width=True)

        # Feature correlation matrix
        st.subheader("Feature Correlation Matrix")
        corr = df[feature_cols].corr()
        fig_corr = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
        )
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)

else:
    # ── Landing Page ─────────────────────────────────────────────────────

    # Hero section
    st.markdown("""
    <div class="landing-hero">
        <div class="hero-overline">Quantitative Regime Detection</div>
        <div class="hero-title">
            HMM Regime <span class="accent">Terminal</span>
        </div>
        <div class="hero-subtitle">
            Decode hidden market states with Bayesian model selection,
            walk-forward backtesting, and entropy-weighted position sizing.
        </div>
        <div class="hero-divider"></div>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline section
    st.markdown("""
    <div class="pipeline-section">
        <div class="pipeline-label">Analysis Pipeline</div>
        <div class="pipeline-row">
            <div class="pipe-step">
                <div class="pipe-node">
                    <div class="step-num">01</div>
                    <div class="step-name">Fetch Data</div>
                    <div class="step-detail">OHLCV via yfinance</div>
                </div>
                <span class="pipe-arrow">&rarr;</span>
            </div>
            <div class="pipe-step">
                <div class="pipe-node">
                    <div class="step-num">02</div>
                    <div class="step-name">Engineer Features</div>
                    <div class="step-detail">Returns, vol, momentum</div>
                </div>
                <span class="pipe-arrow">&rarr;</span>
            </div>
            <div class="pipe-step">
                <div class="pipe-node">
                    <div class="step-num">03</div>
                    <div class="step-name">Fit HMM</div>
                    <div class="step-detail">BIC model selection</div>
                </div>
                <span class="pipe-arrow">&rarr;</span>
            </div>
            <div class="pipe-step">
                <div class="pipe-node">
                    <div class="step-num">04</div>
                    <div class="step-name">Detect Regimes</div>
                    <div class="step-detail">Viterbi decoding</div>
                </div>
                <span class="pipe-arrow">&rarr;</span>
            </div>
            <div class="pipe-step">
                <div class="pipe-node">
                    <div class="step-num">05</div>
                    <div class="step-name">Generate Signals</div>
                    <div class="step-detail">Confirmed entries</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Math concepts section
    st.markdown('<div class="pipeline-label">Core Mathematical Concepts</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="concepts-grid">
        <div class="concept-card card-green">
            <span class="concept-icon">&#x1D6CC;</span>
            <div class="concept-title">Hidden Markov Model</div>
            <div class="concept-formula">P(X|Z) = &prod; P(x_t | z_t) P(z_t | z_{t-1})</div>
            <div class="concept-desc">
                Latent-state time-series model. Observable features are
                generated by unobserved regime states with Markov transitions.
            </div>
        </div>
        <div class="concept-card card-amber">
            <span class="concept-icon">&#x0042;</span>
            <div class="concept-title">Bayesian Info Criterion</div>
            <div class="concept-formula">BIC = -2 ln(L) + k ln(n)</div>
            <div class="concept-desc">
                Penalizes model complexity to prevent overfitting.
                Selects optimal number of hidden states automatically.
            </div>
        </div>
        <div class="concept-card card-cyan">
            <span class="concept-icon">&#x0048;</span>
            <div class="concept-title">Shannon Entropy</div>
            <div class="concept-formula">H = -&sum; p_i log&sub2;(p_i)</div>
            <div class="concept-desc">
                Measures regime uncertainty from posterior probabilities.
                Low entropy = high conviction, used for confidence scaling.
            </div>
        </div>
        <div class="concept-card card-blue">
            <span class="concept-icon">&#x0066;</span>
            <div class="concept-title">Kelly Criterion</div>
            <div class="concept-formula">f* = (bp - q) / b</div>
            <div class="concept-desc">
                Optimal fraction of capital to risk per trade.
                Maximizes long-run geometric growth rate of portfolio.
            </div>
        </div>
        <div class="concept-card card-red">
            <span class="concept-icon">&#x03B1;</span>
            <div class="concept-title">Walk-Forward Validation</div>
            <div class="concept-formula">train &rarr; test &rarr; step &rarr; repeat</div>
            <div class="concept-desc">
                Anchored expanding-window backtest. Prevents look-ahead
                bias by re-fitting the model at each fold boundary.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Getting started section
    st.markdown("""
    <div class="getting-started">
        <div class="gs-title">Getting Started</div>
        <div class="gs-steps">
            <div class="gs-step">
                <span class="gs-step-num">1</span>
                <span class="gs-step-text">
                    Open the <strong>sidebar</strong> and enter a ticker symbol (e.g. SPY, AAPL, BTC-USD).
                </span>
            </div>
            <div class="gs-step">
                <span class="gs-step-num">2</span>
                <span class="gs-step-text">
                    Choose your <strong>interval</strong> and <strong>lookback period</strong> for the data window.
                </span>
            </div>
            <div class="gs-step">
                <span class="gs-step-num">3</span>
                <span class="gs-step-text">
                    Adjust HMM, strategy, and risk parameters &mdash; or keep the defaults.
                </span>
            </div>
            <div class="gs-step">
                <span class="gs-step-num">4</span>
                <span class="gs-step-text">
                    Click <strong>Run Analysis</strong> to fetch data, fit the model, and generate regime signals.
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="landing-footer">
        Hidden Markov Models <span class="dot"></span> Baum-Welch EM
        <span class="dot"></span> Viterbi Decoding <span class="dot"></span> Walk-Forward Backtesting
    </div>
    """, unsafe_allow_html=True)
