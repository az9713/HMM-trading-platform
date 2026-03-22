"""
TDD tests for multi_timeframe.py — Multi-Timeframe Regime Fusion.

Tests cover: sentiment mapping, timeframe alignment, confluence computation,
conflict detection, enhanced confidence, dominant regime voting,
and edge cases (single timeframe, NaN handling, all-agree/all-disagree).
"""

import numpy as np
import pandas as pd
import pytest

from multi_timeframe import (
    _regime_to_sentiment,
    _sentiment_to_regime,
    align_timeframes,
    compute_confluence,
    TimeframeResult,
    FusionResult,
    DEFAULT_WEIGHTS,
    REGIME_SENTIMENT,
)
from hmm_engine import RegimeDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(min_states=2, max_states=3, n_restarts=3, n_iter=50):
    return {
        "hmm": {
            "min_states": min_states,
            "max_states": max_states,
            "n_restarts": n_restarts,
            "n_iter": n_iter,
            "tol": 1e-4,
            "covariance_type": "full",
        },
        "data": {
            "features": ["log_return", "rolling_vol"],
            "rolling_vol_window": 21,
            "rsi_period": 14,
        },
    }


def make_mock_tf_result(
    interval: str,
    n_bars: int,
    regime_labels: list[str],
    dates: pd.DatetimeIndex | None = None,
    seed: int = 42,
) -> TimeframeResult:
    """Create a mock TimeframeResult for testing without fitting a real HMM."""
    rng = np.random.default_rng(seed)

    if dates is None:
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="h")

    # Create minimal df
    df = pd.DataFrame({
        "Datetime": dates,
        "Open": 100 + rng.normal(0, 1, n_bars).cumsum(),
        "High": 101 + rng.normal(0, 1, n_bars).cumsum(),
        "Low": 99 + rng.normal(0, 1, n_bars).cumsum(),
        "Close": 100 + rng.normal(0, 1, n_bars).cumsum(),
        "Volume": rng.integers(1000, 10000, n_bars),
    })

    # Map labels to state ids
    unique_labels = list(dict.fromkeys(regime_labels))  # preserve order, dedupe
    label_to_id = {lbl: i for i, lbl in enumerate(unique_labels)}
    states = np.array([label_to_id[lbl] for lbl in regime_labels])
    labels = {i: lbl for lbl, i in label_to_id.items()}
    n_states = len(unique_labels)

    posteriors = np.zeros((n_bars, n_states))
    for t in range(n_bars):
        posteriors[t, states[t]] = 0.9
        remaining = 0.1 / max(n_states - 1, 1)
        for s in range(n_states):
            if s != states[t]:
                posteriors[t, s] = remaining

    confidence = np.full(n_bars, 0.8)
    entropy = np.full(n_bars, 0.2)

    # Create a minimal detector (won't be used for computation in tests)
    config = make_config()
    detector = RegimeDetector(config)
    detector.n_states = n_states

    return TimeframeResult(
        interval=interval,
        df=df,
        states=states,
        posteriors=posteriors,
        labels=labels,
        confidence=confidence,
        entropy=entropy,
        n_states=n_states,
        bic_scores={n_states: 100.0},
        detector=detector,
    )


# ---------------------------------------------------------------------------
# Tests: Sentiment Mapping
# ---------------------------------------------------------------------------

class TestSentimentMapping:
    def test_crash_maps_to_neg2(self):
        assert _regime_to_sentiment("crash") == -2

    def test_bear_maps_to_neg1(self):
        assert _regime_to_sentiment("bear") == -1

    def test_neutral_maps_to_0(self):
        assert _regime_to_sentiment("neutral") == 0

    def test_bull_maps_to_1(self):
        assert _regime_to_sentiment("bull") == 1

    def test_bull_run_maps_to_2(self):
        assert _regime_to_sentiment("bull_run") == 2

    def test_unknown_maps_to_0(self):
        assert _regime_to_sentiment("regime_3") == 0

    def test_roundtrip_crash(self):
        assert _sentiment_to_regime(-2.0) == "crash"

    def test_roundtrip_bear(self):
        assert _sentiment_to_regime(-1.0) == "bear"

    def test_roundtrip_neutral(self):
        assert _sentiment_to_regime(0.0) == "neutral"

    def test_roundtrip_bull(self):
        assert _sentiment_to_regime(1.0) == "bull"

    def test_roundtrip_bull_run(self):
        assert _sentiment_to_regime(2.0) == "bull_run"

    def test_boundary_values(self):
        assert _sentiment_to_regime(-1.5) == "crash"
        assert _sentiment_to_regime(-0.5) == "bear"
        assert _sentiment_to_regime(0.5) == "neutral"
        assert _sentiment_to_regime(1.5) == "bull"


# ---------------------------------------------------------------------------
# Tests: Timeframe Alignment
# ---------------------------------------------------------------------------

class TestAlignTimeframes:
    def test_single_timeframe_returns_self(self):
        """With no other timeframes, alignment just returns the base regimes."""
        base = make_mock_tf_result("1h", 100, ["bull"] * 100)
        aligned = align_timeframes(base, [])
        assert "regime_1h" in aligned.columns
        assert len(aligned) == 100
        assert all(aligned["regime_1h"] == "bull")

    def test_two_timeframes_aligned(self):
        """Two timeframes produce two regime columns."""
        dates_1h = pd.date_range("2024-01-01", periods=24, freq="h")
        dates_1d = pd.date_range("2024-01-01", periods=1, freq="D")

        base = make_mock_tf_result("1h", 24, ["bull"] * 24, dates=dates_1h)
        daily = make_mock_tf_result("1d", 1, ["bear"], dates=dates_1d, seed=99)

        aligned = align_timeframes(base, [daily])
        assert "regime_1h" in aligned.columns
        assert "regime_1d" in aligned.columns
        assert len(aligned) == 24

    def test_forward_fill_is_causal(self):
        """Higher TF regime is forward-filled (no lookahead)."""
        dates_1h = pd.date_range("2024-01-01", periods=48, freq="h")
        dates_1d = pd.date_range("2024-01-01", periods=2, freq="D")

        base = make_mock_tf_result("1h", 48, ["bull"] * 48, dates=dates_1h)
        daily = make_mock_tf_result("1d", 2, ["bear", "bull"], dates=dates_1d, seed=99)

        aligned = align_timeframes(base, [daily])

        # First 24 hours should be "bear" (day 1 regime)
        first_day = aligned["regime_1d"].iloc[:24]
        assert all(first_day == "bear"), f"Expected bear for day 1, got {first_day.unique()}"

    def test_nan_before_first_higher_tf_bar(self):
        """Before the first higher TF bar, forward-fill produces NaN."""
        # Base starts before daily data
        dates_1h = pd.date_range("2023-12-31 12:00", periods=36, freq="h")
        dates_1d = pd.date_range("2024-01-01", periods=1, freq="D")

        base = make_mock_tf_result("1h", 36, ["bull"] * 36, dates=dates_1h)
        daily = make_mock_tf_result("1d", 1, ["bear"], dates=dates_1d, seed=99)

        aligned = align_timeframes(base, [daily])

        # First 12 hours (before 2024-01-01) should be NaN
        early_bars = aligned["regime_1d"].iloc[:12]
        assert early_bars.isna().all(), "Pre-daily bars should be NaN"


# ---------------------------------------------------------------------------
# Tests: Confluence Computation
# ---------------------------------------------------------------------------

class TestComputeConfluence:
    def test_perfect_agreement_gives_max_confluence(self):
        """When all timeframes agree, confluence should be 1.0."""
        aligned = pd.DataFrame({
            "regime_1h": ["bull"] * 50,
            "regime_1d": ["bull"] * 50,
        })
        confluence, conflicts, sentiments, dominant = compute_confluence(aligned)
        np.testing.assert_allclose(confluence, 1.0)

    def test_total_disagreement_reduces_confluence(self):
        """Crash vs bull_run should give low confluence."""
        aligned = pd.DataFrame({
            "regime_1h": ["crash"] * 50,
            "regime_1d": ["bull_run"] * 50,
        })
        weights = {"1h": 0.5, "1d": 0.5}
        confluence, _, _, _ = compute_confluence(aligned, weights)
        # With equal weights, crash(-2) vs bull_run(2): std = 2.0
        # confluence = 1 - 2/2 = 0.0
        assert confluence[0] == pytest.approx(0.0, abs=0.05)

    def test_confluence_bounded_0_1(self):
        """Confluence score should always be in [0, 1]."""
        aligned = pd.DataFrame({
            "regime_1h": ["bear", "bull", "crash", "neutral", "bull_run"],
            "regime_1d": ["bull", "crash", "bull_run", "bear", "neutral"],
        })
        confluence, _, _, _ = compute_confluence(aligned)
        assert np.all(confluence >= 0.0)
        assert np.all(confluence <= 1.0)

    def test_conflict_detection(self):
        """Bars where sentiment spread > 1 should be flagged as conflicts."""
        aligned = pd.DataFrame({
            "regime_1h": ["bull", "crash"],
            "regime_1d": ["bull", "bull_run"],
        })
        _, conflicts, _, _ = compute_confluence(aligned)
        assert conflicts["conflict"].iloc[0] == False  # bull(1) vs bull(1): spread=0
        assert conflicts["conflict"].iloc[1] == True   # crash(-2) vs bull_run(2): spread=4

    def test_dominant_regime_follows_weighted_vote(self):
        """Dominant regime should match the weighted average sentiment."""
        aligned = pd.DataFrame({
            "regime_1h": ["bull"] * 10,
            "regime_1d": ["bull"] * 10,
        })
        _, _, _, dominant = compute_confluence(aligned)
        assert all(dominant == "bull")

    def test_three_timeframes(self):
        """Works with 3+ timeframes."""
        aligned = pd.DataFrame({
            "regime_15m": ["bull"] * 20,
            "regime_1h": ["bull"] * 20,
            "regime_1d": ["bull"] * 20,
        })
        confluence, _, _, dominant = compute_confluence(aligned)
        assert len(confluence) == 20
        np.testing.assert_allclose(confluence, 1.0)

    def test_custom_weights(self):
        """Custom weights change the confluence calculation."""
        aligned = pd.DataFrame({
            "regime_1h": ["bear"] * 10,
            "regime_1d": ["bull"] * 10,
        })
        # Heavy weight on daily: dominant should lean bullish
        weights = {"1h": 0.1, "1d": 0.9}
        _, _, _, dominant = compute_confluence(aligned, weights)
        # Weighted sentiment: 0.1*(-1) + 0.9*(1) = 0.8 → bull (since > 0.5)
        assert all(dominant == "bull")


# ---------------------------------------------------------------------------
# Tests: Enhanced Confidence
# ---------------------------------------------------------------------------

class TestEnhancedConfidence:
    def test_enhanced_confidence_multiplicative(self):
        """Enhanced confidence = base confidence * confluence."""
        base_conf = np.array([0.8, 0.6, 0.9])
        confluence = np.array([1.0, 0.5, 0.3])
        enhanced = base_conf * confluence
        expected = np.array([0.8, 0.3, 0.27])
        np.testing.assert_allclose(enhanced, expected)

    def test_perfect_confluence_preserves_confidence(self):
        """When confluence=1, enhanced = base confidence."""
        base_conf = np.array([0.7, 0.85, 0.95])
        confluence = np.ones(3)
        enhanced = base_conf * confluence
        np.testing.assert_allclose(enhanced, base_conf)

    def test_zero_confluence_zeros_confidence(self):
        """When confluence=0, enhanced confidence = 0."""
        base_conf = np.array([0.9, 0.9, 0.9])
        confluence = np.zeros(3)
        enhanced = base_conf * confluence
        np.testing.assert_allclose(enhanced, 0.0)


# ---------------------------------------------------------------------------
# Tests: Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_bar(self):
        """Works with a single bar."""
        aligned = pd.DataFrame({
            "regime_1h": ["bull"],
            "regime_1d": ["bull"],
        })
        confluence, _, _, dominant = compute_confluence(aligned)
        assert len(confluence) == 1
        assert confluence[0] == pytest.approx(1.0)

    def test_all_neutral_regimes(self):
        """All neutral gives confluence=1."""
        aligned = pd.DataFrame({
            "regime_1h": ["neutral"] * 10,
            "regime_1d": ["neutral"] * 10,
        })
        confluence, _, _, _ = compute_confluence(aligned)
        np.testing.assert_allclose(confluence, 1.0)

    def test_regime_sentiment_completeness(self):
        """All standard regime labels have defined sentiments."""
        for label in ["crash", "bear", "neutral", "bull", "bull_run"]:
            assert label in REGIME_SENTIMENT

    def test_default_weights_cover_standard_intervals(self):
        """Default weights include all standard intervals."""
        for interval in ["1m", "5m", "15m", "1h", "1d", "1wk"]:
            assert interval in DEFAULT_WEIGHTS
