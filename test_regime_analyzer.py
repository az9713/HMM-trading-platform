"""
Tests for regime_analyzer.py — RegimeTransitionAnalyzer.

Tests cover: transition detection, empirical transition matrix,
forward return computation, early warning signals, regime attribution,
and transition timing analysis.
"""

import numpy as np
import pandas as pd
import pytest

from regime_analyzer import RegimeTransitionAnalyzer, TransitionEvent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_synthetic_data(n=200, seed=42):
    """Create synthetic regime data with known transitions."""
    rng = np.random.default_rng(seed)

    # 4 regimes: 0=bear, 1=neutral, 2=bull, 3=bull_run
    # Transitions at bars 50, 100, 150
    states = np.zeros(n, dtype=int)
    states[50:100] = 1
    states[100:150] = 2
    states[150:] = 0

    labels = {0: "bear", 1: "neutral", 2: "bull", 3: "bull_run"}

    # Prices: random walk with regime-dependent drift
    drift = {0: -0.002, 1: 0.0, 2: 0.003, 3: 0.005}
    log_returns = np.array([drift[states[t]] + rng.normal(0, 0.01) for t in range(n)])
    prices = 100.0 * np.exp(np.cumsum(log_returns))

    # Posteriors: high confidence in current regime
    posteriors = np.full((n, 4), 0.05)
    for t in range(n):
        posteriors[t, states[t]] = 0.85

    # Entropy from posteriors
    eps = 1e-12
    p = np.clip(posteriors, eps, 1.0)
    entropy = -np.sum(p * np.log2(p), axis=1)

    confidence = 1.0 - entropy / np.log2(4)

    signals = np.zeros(n, dtype=int)
    signals[states == 2] = 1    # long in bull
    signals[states == 0] = -1   # short in bear (except first segment)
    signals[:50] = -1

    returns = np.diff(prices) / prices[:-1]
    returns = np.concatenate([[0.0], returns])

    return states, labels, prices, posteriors, entropy, confidence, signals, returns


# ---------------------------------------------------------------------------
# Tests: transition detection
# ---------------------------------------------------------------------------

class TestTransitionDetection:
    def test_detects_correct_number(self):
        states, labels, prices, posteriors, entropy, confidence, _, _ = make_synthetic_data()
        analyzer = RegimeTransitionAnalyzer()
        transitions = analyzer.detect_transitions(states, labels, prices, entropy, confidence)
        # Transitions at bars 50, 100, 150
        assert len(transitions) == 3

    def test_transition_labels(self):
        states, labels, prices, posteriors, entropy, confidence, _, _ = make_synthetic_data()
        analyzer = RegimeTransitionAnalyzer()
        transitions = analyzer.detect_transitions(states, labels, prices, entropy, confidence)

        assert transitions[0].from_regime == "bear"
        assert transitions[0].to_regime == "neutral"
        assert transitions[0].bar == 50

        assert transitions[1].from_regime == "neutral"
        assert transitions[1].to_regime == "bull"
        assert transitions[1].bar == 100

        assert transitions[2].from_regime == "bull"
        assert transitions[2].to_regime == "bear"
        assert transitions[2].bar == 150

    def test_forward_returns_computed(self):
        states, labels, prices, posteriors, entropy, confidence, _, _ = make_synthetic_data()
        analyzer = RegimeTransitionAnalyzer()
        transitions = analyzer.detect_transitions(states, labels, prices, entropy, confidence)

        # First transition at bar 50 should have valid forward returns
        tr = transitions[0]
        assert not np.isnan(tr.forward_return_5)
        assert not np.isnan(tr.forward_return_10)
        assert not np.isnan(tr.forward_return_20)

    def test_forward_returns_nan_at_end(self):
        """Transitions near end should have NaN for long lookforwards."""
        states, labels, prices, posteriors, entropy, confidence, _, _ = make_synthetic_data(n=60)
        # Force a transition at bar 50
        states[:50] = 0
        states[50:] = 1
        analyzer = RegimeTransitionAnalyzer()
        transitions = analyzer.detect_transitions(states, labels, prices, entropy, confidence)
        tr = transitions[-1]
        # 9 bars after transition (indices 51-59) — 5-bar OK, 10 and 20 NaN
        assert not np.isnan(tr.forward_return_5)  # 50+5=55 < 60
        assert np.isnan(tr.forward_return_10)     # 50+10=60, not < 60
        assert np.isnan(tr.forward_return_20)

    def test_no_transitions(self):
        """All same state should yield no transitions."""
        states = np.zeros(100, dtype=int)
        labels = {0: "neutral"}
        prices = np.linspace(100, 110, 100)
        entropy = np.ones(100) * 0.1
        confidence = np.ones(100) * 0.9
        analyzer = RegimeTransitionAnalyzer()
        transitions = analyzer.detect_transitions(states, labels, prices, entropy, confidence)
        assert len(transitions) == 0

    def test_entropy_context(self):
        states, labels, prices, posteriors, entropy, confidence, _, _ = make_synthetic_data()
        analyzer = RegimeTransitionAnalyzer()
        transitions = analyzer.detect_transitions(states, labels, prices, entropy, confidence)
        for tr in transitions:
            assert tr.entropy_before >= 0
            assert tr.entropy_after >= 0


# ---------------------------------------------------------------------------
# Tests: empirical transition matrix
# ---------------------------------------------------------------------------

class TestTransitionMatrix:
    def test_matrix_shape(self):
        states, labels, prices, posteriors, entropy, confidence, _, _ = make_synthetic_data()
        analyzer = RegimeTransitionAnalyzer()
        transitions = analyzer.detect_transitions(states, labels, prices, entropy, confidence)
        regime_labels = ["bear", "neutral", "bull", "bull_run"]
        matrix = analyzer.transition_matrix_empirical(transitions, regime_labels)

        assert matrix.shape == (4, 4)
        assert list(matrix.index) == regime_labels
        assert list(matrix.columns) == regime_labels

    def test_matrix_counts(self):
        states, labels, prices, posteriors, entropy, confidence, _, _ = make_synthetic_data()
        analyzer = RegimeTransitionAnalyzer()
        transitions = analyzer.detect_transitions(states, labels, prices, entropy, confidence)
        regime_labels = ["bear", "neutral", "bull", "bull_run"]
        matrix = analyzer.transition_matrix_empirical(transitions, regime_labels)

        # bear -> neutral: 1 transition
        assert matrix.loc["bear", "neutral"] == 1
        # neutral -> bull: 1
        assert matrix.loc["neutral", "bull"] == 1
        # bull -> bear: 1
        assert matrix.loc["bull", "bear"] == 1
        # Total should equal number of transitions
        assert matrix.values.sum() == 3

    def test_empty_transitions(self):
        analyzer = RegimeTransitionAnalyzer()
        matrix = analyzer.transition_matrix_empirical([], ["bear", "bull"])
        assert matrix.values.sum() == 0


# ---------------------------------------------------------------------------
# Tests: transition forward returns
# ---------------------------------------------------------------------------

class TestTransitionForwardReturns:
    def test_returns_table_shape(self):
        states, labels, prices, posteriors, entropy, confidence, _, _ = make_synthetic_data()
        analyzer = RegimeTransitionAnalyzer()
        transitions = analyzer.detect_transitions(states, labels, prices, entropy, confidence)
        fwd_df = analyzer.transition_forward_returns(transitions)

        assert "from_regime" in fwd_df.columns
        assert "to_regime" in fwd_df.columns
        assert "count" in fwd_df.columns
        assert "mean_fwd_5" in fwd_df.columns
        assert "hit_rate_5" in fwd_df.columns
        assert len(fwd_df) == 3  # 3 unique transition types

    def test_hit_rate_bounded(self):
        states, labels, prices, posteriors, entropy, confidence, _, _ = make_synthetic_data()
        analyzer = RegimeTransitionAnalyzer()
        transitions = analyzer.detect_transitions(states, labels, prices, entropy, confidence)
        fwd_df = analyzer.transition_forward_returns(transitions)
        for _, row in fwd_df.iterrows():
            if not np.isnan(row["hit_rate_5"]):
                assert 0.0 <= row["hit_rate_5"] <= 1.0

    def test_empty_transitions(self):
        analyzer = RegimeTransitionAnalyzer()
        fwd_df = analyzer.transition_forward_returns([])
        assert len(fwd_df) == 0


# ---------------------------------------------------------------------------
# Tests: early warning signals
# ---------------------------------------------------------------------------

class TestEarlyWarning:
    def test_warning_output_shape(self):
        states, labels, prices, posteriors, entropy, confidence, _, _ = make_synthetic_data()
        analyzer = RegimeTransitionAnalyzer()
        warnings = analyzer.early_warning_signals(posteriors, entropy, states, labels)

        assert "bar" in warnings.columns
        assert "entropy_gradient" in warnings.columns
        assert "posterior_gap" in warnings.columns
        assert "warning_level" in warnings.columns
        # Should have T - gradient_window rows
        assert len(warnings) == len(entropy) - 5

    def test_warning_levels_bounded(self):
        states, labels, prices, posteriors, entropy, confidence, _, _ = make_synthetic_data()
        analyzer = RegimeTransitionAnalyzer()
        warnings = analyzer.early_warning_signals(posteriors, entropy, states, labels)

        assert warnings["warning_level"].min() >= 0
        assert warnings["warning_level"].max() <= 3

    def test_posterior_gap_positive(self):
        states, labels, prices, posteriors, entropy, confidence, _, _ = make_synthetic_data()
        analyzer = RegimeTransitionAnalyzer()
        warnings = analyzer.early_warning_signals(posteriors, entropy, states, labels)

        assert (warnings["posterior_gap"] >= 0).all()

    def test_custom_thresholds(self):
        states, labels, prices, posteriors, entropy, confidence, _, _ = make_synthetic_data()
        analyzer = RegimeTransitionAnalyzer()
        # Very low thresholds should generate more warnings
        warnings_low = analyzer.early_warning_signals(
            posteriors, entropy, states, labels,
            entropy_spike_threshold=0.01, posterior_shift_threshold=0.99,
        )
        warnings_high = analyzer.early_warning_signals(
            posteriors, entropy, states, labels,
            entropy_spike_threshold=10.0, posterior_shift_threshold=0.01,
        )
        assert warnings_low["warning_level"].sum() >= warnings_high["warning_level"].sum()


# ---------------------------------------------------------------------------
# Tests: regime attribution
# ---------------------------------------------------------------------------

class TestRegimeAttribution:
    def test_attribution_shape(self):
        states, labels, prices, posteriors, entropy, confidence, signals, returns = make_synthetic_data()
        analyzer = RegimeTransitionAnalyzer()
        attr = analyzer.regime_attribution(returns, states, labels, signals)

        assert "regime" in attr.columns
        assert "n_bars" in attr.columns
        assert "sharpe" in attr.columns
        assert "pnl_contribution" in attr.columns
        # Should have entries for regimes present in data
        assert len(attr) >= 2

    def test_pct_time_sums_to_one(self):
        states, labels, prices, posteriors, entropy, confidence, signals, returns = make_synthetic_data()
        analyzer = RegimeTransitionAnalyzer()
        attr = analyzer.regime_attribution(returns, states, labels, signals)

        assert abs(attr["pct_time"].sum() - 1.0) < 1e-6

    def test_pnl_contribution_sums_to_one(self):
        states, labels, prices, posteriors, entropy, confidence, signals, returns = make_synthetic_data()
        analyzer = RegimeTransitionAnalyzer()
        attr = analyzer.regime_attribution(returns, states, labels, signals)

        total_contrib = attr["pnl_contribution"].sum()
        # Should sum close to 1 (or -1 if negative total)
        if not np.isnan(total_contrib) and total_contrib != 0:
            assert abs(abs(total_contrib) - 1.0) < 0.1

    def test_win_rate_bounded(self):
        states, labels, prices, posteriors, entropy, confidence, signals, returns = make_synthetic_data()
        analyzer = RegimeTransitionAnalyzer()
        attr = analyzer.regime_attribution(returns, states, labels, signals)

        for _, row in attr.iterrows():
            assert 0.0 <= row["win_rate"] <= 1.0

    def test_all_flat_signals(self):
        """No active positions should yield zero returns."""
        states, labels, prices, posteriors, entropy, confidence, _, returns = make_synthetic_data()
        signals = np.zeros(len(states), dtype=int)
        analyzer = RegimeTransitionAnalyzer()
        attr = analyzer.regime_attribution(returns, states, labels, signals)

        assert (attr["cumulative_return"] == 0).all()


# ---------------------------------------------------------------------------
# Tests: transition timing
# ---------------------------------------------------------------------------

class TestTransitionTiming:
    def test_timing_keys(self):
        states, labels, prices, posteriors, entropy, confidence, _, _ = make_synthetic_data()
        analyzer = RegimeTransitionAnalyzer()
        transitions = analyzer.detect_transitions(states, labels, prices, entropy, confidence)
        timing = analyzer.transition_timing_analysis(transitions)

        assert "avg_bars_between" in timing
        assert "median_bars_between" in timing
        assert "entropy_precedes_transition" in timing
        assert "avg_confidence_at_transition" in timing
        assert "n_transitions" in timing

    def test_timing_values(self):
        states, labels, prices, posteriors, entropy, confidence, _, _ = make_synthetic_data()
        analyzer = RegimeTransitionAnalyzer()
        transitions = analyzer.detect_transitions(states, labels, prices, entropy, confidence)
        timing = analyzer.transition_timing_analysis(transitions)

        # Transitions at 50, 100, 150 → gaps of 50
        assert timing["avg_bars_between"] == 50.0
        assert timing["n_transitions"] == 3

    def test_single_transition(self):
        states = np.zeros(100, dtype=int)
        states[50:] = 1
        labels = {0: "bear", 1: "bull"}
        prices = np.linspace(100, 110, 100)
        entropy = np.ones(100) * 0.1
        confidence = np.ones(100) * 0.9
        analyzer = RegimeTransitionAnalyzer()
        transitions = analyzer.detect_transitions(states, labels, prices, entropy, confidence)
        timing = analyzer.transition_timing_analysis(transitions)

        assert timing["n_transitions"] == 1
        assert np.isnan(timing["avg_bars_between"])
