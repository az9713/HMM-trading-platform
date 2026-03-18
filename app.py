"""
app.py — Streamlit dashboard with 6 tabs for HMM Regime Terminal.

Tabs: Current Signal, Regime Analysis, Backtest Results, Trade Log,
      Model Diagnostics, Fundamentals.
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
from fundamentals import FundamentalAnalyzer

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="HMM Regime Terminal", layout="wide")
st.title("HMM Regime Terminal")

# ── Load config ──────────────────────────────────────────────────────────────

@st.cache_data
def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

base_config = load_config()

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
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

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Current Signal", "Regime Analysis", "Backtest Results",
        "Trade Log", "Model Diagnostics", "Fundamentals"
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

    # ── Tab 6: Fundamentals ─────────────────────────────────────────────

    with tab6:
        if FundamentalAnalyzer.is_crypto(ticker):
            st.warning("Fundamental analysis is not available for crypto assets. "
                       "Crypto tickers do not have traditional financial statements, "
                       "ratios, or analyst coverage.")
        else:
            fa = FundamentalAnalyzer()

            with st.spinner("Fetching fundamental data..."):
                overview = fa.get_company_overview(ticker)
                ratios = fa.get_financial_ratios(ticker)
                statements = fa.get_financial_statements(ticker)
                analyst = fa.get_analyst_data(ticker)
                score = fa.health_score(ratios)

            # ── Company Overview Card ────────────────────────────────
            st.subheader("Company Overview")
            ov_c1, ov_c2, ov_c3, ov_c4 = st.columns(4)
            with ov_c1:
                st.metric("Company", overview["name"])
            with ov_c2:
                st.metric("Sector", overview["sector"])
            with ov_c3:
                st.metric("Industry", overview["industry"])
            with ov_c4:
                st.metric("Market Cap", overview["market_cap_fmt"])

            price_c1, price_c2, price_c3 = st.columns(3)
            with price_c1:
                cp = overview["current_price"]
                st.metric("Current Price",
                          f"${cp:,.2f}" if cp else "N/A")
            with price_c2:
                high52 = overview["fifty_two_week_high"]
                st.metric("52W High",
                          f"${high52:,.2f}" if high52 else "N/A")
            with price_c3:
                low52 = overview["fifty_two_week_low"]
                st.metric("52W Low",
                          f"${low52:,.2f}" if low52 else "N/A")

            desc = overview.get("description", "")
            if desc and desc != "No description available.":
                with st.expander("Business Description"):
                    st.write(desc)

            # ── Financial Health Score ────────────────────────────────
            st.subheader("Financial Health Score")
            score_color = fa.health_color(score)
            score_label = "Strong" if score >= 70 else "Moderate" if score >= 40 else "Weak"

            sc_c1, sc_c2 = st.columns([1, 3])
            with sc_c1:
                st.markdown(
                    f"<div style='background:{score_color};padding:30px;"
                    f"border-radius:15px;text-align:center;color:white;"
                    f"font-size:36px;font-weight:bold'>{score}/100</div>"
                    f"<p style='text-align:center;font-size:18px;margin-top:8px'>"
                    f"{score_label}</p>",
                    unsafe_allow_html=True,
                )
            with sc_c2:
                # Score gauge using plotly
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    title={"text": "Health Score"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": score_color},
                        "steps": [
                            {"range": [0, 40], "color": "#ffcdd2"},
                            {"range": [40, 70], "color": "#fff9c4"},
                            {"range": [70, 100], "color": "#c8e6c9"},
                        ],
                    },
                ))
                fig_gauge.update_layout(height=250, margin=dict(t=40, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)

            # ── Financial Ratios Grid ────────────────────────────────
            st.subheader("Key Financial Ratios")

            ratio_display = {
                "pe_trailing": ("P/E (TTM)", None),
                "pe_forward": ("P/E (Fwd)", None),
                "pb": ("P/B", None),
                "ps": ("P/S", None),
                "peg": ("PEG", None),
                "ev_ebitda": ("EV/EBITDA", None),
                "debt_to_equity": ("D/E Ratio", None),
                "current_ratio": ("Current Ratio", None),
                "roe": ("ROE", "%"),
                "roa": ("ROA", "%"),
                "profit_margin": ("Profit Margin", "%"),
                "operating_margin": ("Operating Margin", "%"),
                "gross_margin": ("Gross Margin", "%"),
                "revenue_growth": ("Revenue Growth", "%"),
                "earnings_growth": ("Earnings Growth", "%"),
                "dividend_yield": ("Dividend Yield", "%"),
                "payout_ratio": ("Payout Ratio", "%"),
                "beta": ("Beta", None),
            }

            # Render in rows of 4
            ratio_keys = list(ratio_display.keys())
            for i in range(0, len(ratio_keys), 4):
                cols = st.columns(4)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx >= len(ratio_keys):
                        break
                    key = ratio_keys[idx]
                    label, fmt = ratio_display[key]
                    val = ratios.get(key)
                    color = fa.ratio_color(key, val)

                    if val is not None:
                        if fmt == "%":
                            display_val = f"{val * 100:.2f}%"
                        else:
                            display_val = f"{val:.2f}"
                    else:
                        display_val = "N/A"
                        color = "#9e9e9e"

                    with col:
                        st.markdown(
                            f"<div style='border-left:4px solid {color};"
                            f"padding:8px 12px;margin-bottom:8px'>"
                            f"<span style='color:#888;font-size:12px'>{label}</span><br>"
                            f"<span style='font-size:20px;font-weight:bold'>{display_val}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            # ── Income Statement Trend ───────────────────────────────
            inc_summary = statements.get("income_summary", pd.DataFrame())
            if not inc_summary.empty:
                st.subheader("Income Statement Trend")
                fig_inc = go.Figure()

                for row_name in ["Total Revenue", "Net Income", "EBITDA"]:
                    if row_name in inc_summary.index:
                        row = inc_summary.loc[row_name]
                        periods = [str(c.date()) if hasattr(c, "date") else str(c)
                                   for c in row.index]
                        vals = row.values.astype(float)
                        fig_inc.add_trace(go.Bar(
                            x=periods, y=vals, name=row_name,
                        ))

                fig_inc.update_layout(
                    barmode="group", height=400,
                    xaxis_title="Period", yaxis_title="USD",
                    xaxis=dict(categoryorder="array",
                               categoryarray=sorted(periods)),
                )
                st.plotly_chart(fig_inc, use_container_width=True)

            # ── Balance Sheet Composition ────────────────────────────
            bal_summary = statements.get("balance_summary", pd.DataFrame())
            if not bal_summary.empty:
                st.subheader("Balance Sheet Composition")
                fig_bal = go.Figure()

                for row_name, color in [
                    ("Total Assets", "#1565c0"),
                    ("Total Liabilities Net Minority Interest", "#d32f2f"),
                    ("Stockholders Equity", "#388e3c"),
                    ("Total Debt", "#f57c00"),
                    ("Cash And Cash Equivalents", "#00897b"),
                ]:
                    if row_name in bal_summary.index:
                        row = bal_summary.loc[row_name]
                        periods = [str(c.date()) if hasattr(c, "date") else str(c)
                                   for c in row.index]
                        vals = row.values.astype(float)
                        fig_bal.add_trace(go.Bar(
                            x=periods, y=vals,
                            name=row_name.replace("Net Minority Interest", ""),
                            marker_color=color,
                        ))

                fig_bal.update_layout(
                    barmode="group", height=400,
                    xaxis_title="Period", yaxis_title="USD",
                    xaxis=dict(categoryorder="array",
                               categoryarray=sorted(periods) if periods else []),
                )
                st.plotly_chart(fig_bal, use_container_width=True)

            # ── Cash Flow Chart ──────────────────────────────────────
            cf_summary = statements.get("cashflow_summary", pd.DataFrame())
            if not cf_summary.empty:
                st.subheader("Cash Flow Summary")
                fig_cf = go.Figure()

                cf_colors = {
                    "Operating Cash Flow": "#388e3c",
                    "Investing Cash Flow": "#f57c00",
                    "Financing Cash Flow": "#1565c0",
                    "Free Cash Flow": "#7b1fa2",
                }
                for row_name in ["Operating Cash Flow", "Investing Cash Flow",
                                 "Financing Cash Flow", "Free Cash Flow"]:
                    if row_name in cf_summary.index:
                        row = cf_summary.loc[row_name]
                        periods = [str(c.date()) if hasattr(c, "date") else str(c)
                                   for c in row.index]
                        vals = row.values.astype(float)
                        fig_cf.add_trace(go.Bar(
                            x=periods, y=vals, name=row_name,
                            marker_color=cf_colors.get(row_name, "#9e9e9e"),
                        ))

                fig_cf.update_layout(
                    barmode="group", height=400,
                    xaxis_title="Period", yaxis_title="USD",
                    xaxis=dict(categoryorder="array",
                               categoryarray=sorted(periods) if periods else []),
                )
                st.plotly_chart(fig_cf, use_container_width=True)

            # ── Analyst Consensus ────────────────────────────────────
            st.subheader("Analyst Consensus")
            an_c1, an_c2, an_c3, an_c4 = st.columns(4)
            with an_c1:
                rec_key = analyst.get("recommendation_key", "N/A")
                rec_colors = {
                    "buy": "#388e3c", "strong_buy": "#1b5e20",
                    "hold": "#f57c00", "sell": "#d32f2f",
                    "strong_sell": "#b71c1c",
                }
                rec_color = rec_colors.get(rec_key, "#9e9e9e")
                st.markdown(
                    f"<div style='background:{rec_color};padding:15px;"
                    f"border-radius:10px;text-align:center;color:white;"
                    f"font-size:18px;font-weight:bold'>"
                    f"{rec_key.upper().replace('_', ' ')}</div>",
                    unsafe_allow_html=True,
                )
            with an_c2:
                st.metric("Analysts", analyst.get("number_of_analysts") or "N/A")
            with an_c3:
                target = analyst.get("target_mean")
                st.metric("Avg Target",
                          f"${target:,.2f}" if target else "N/A")
            with an_c4:
                curr = analyst.get("current_price")
                target_m = analyst.get("target_mean")
                if curr and target_m and curr > 0:
                    upside = (target_m - curr) / curr * 100
                    st.metric("Upside/Downside", f"{upside:+.1f}%")
                else:
                    st.metric("Upside/Downside", "N/A")

            # Price target range
            tgt_low = analyst.get("target_low")
            tgt_high = analyst.get("target_high")
            tgt_med = analyst.get("target_median")
            curr_price = analyst.get("current_price")
            if tgt_low and tgt_high:
                fig_tgt = go.Figure()
                fig_tgt.add_trace(go.Scatter(
                    x=["Target Range"], y=[tgt_low],
                    mode="markers", marker=dict(size=12, color="#d32f2f"),
                    name=f"Low: ${tgt_low:,.2f}",
                ))
                if tgt_med:
                    fig_tgt.add_trace(go.Scatter(
                        x=["Target Range"], y=[tgt_med],
                        mode="markers", marker=dict(size=14, color="#f57c00",
                                                     symbol="diamond"),
                        name=f"Median: ${tgt_med:,.2f}",
                    ))
                fig_tgt.add_trace(go.Scatter(
                    x=["Target Range"], y=[tgt_high],
                    mode="markers", marker=dict(size=12, color="#388e3c"),
                    name=f"High: ${tgt_high:,.2f}",
                ))
                if curr_price:
                    fig_tgt.add_hline(
                        y=curr_price, line_dash="dash", line_color="blue",
                        annotation_text=f"Current: ${curr_price:,.2f}",
                    )
                fig_tgt.update_layout(height=250, showlegend=True,
                                       yaxis_title="Price ($)")
                st.plotly_chart(fig_tgt, use_container_width=True)

            # Recent recommendations table
            recs_df = analyst.get("recommendations", pd.DataFrame())
            if not recs_df.empty:
                with st.expander("Recent Analyst Recommendations"):
                    st.dataframe(recs_df, use_container_width=True)

            # Earnings dates
            earnings_df = analyst.get("earnings_dates", pd.DataFrame())
            if not earnings_df.empty:
                with st.expander("Earnings Dates"):
                    st.dataframe(earnings_df, use_container_width=True)

else:
    st.info("Configure parameters in the sidebar and click **Run Analysis** to start.")
