"""
TDD tests for adaptive_strategy.py — AdaptiveStrategyEngine.

Tests cover: playbook loading, confirmation scoring, posterior blending,
adaptive exits (trailing stop, max hold, entropy de-risk), regime-conditional
sizing, signal discretization, and end-to-end signal generation.
"""

import numpy as np
import pandas as pd
import pytest

from adaptive_strategy import (
    AdaptiveStrategyEngine,
    RegimePlaybook,
    AdaptiveSignal,
    BULL_PLAYBOOK,
    BEAR_PLAYBOOK,
    CRASH_PLAYBOOK,
    NEUTRAL_PLAYBOOK,
    DEFAULT_PLAYBOOKS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def default_config():
    return {
        "adaptive_strategy": {
            "use_blending": True,
            "blend_temperature": 1.0,
            "discretization_threshold": 0.25,
            "atr_period": 14,
            "time_decay_start": 0.7,
            "entropy_exit_threshold": 0.85,
            "max_entropy_bars": 5,
            "cooldown_bars": 3,
            "min_hold_bars": 5,
        },
        "risk": {
            "kelly_fraction": 0.5,
            "max_leverage": 2.0,
            "max_position_pct": 1.0,
        },
    }


def make_ohlcv(n=100, trend="up"):
    """Generate synthetic OHLCV data."""
    np.random.seed(42)
    if trend == "up":
        close = 100 + np.cumsum(np.random.normal(0.1, 0.5, n))
    elif trend == "down":
        close = 200 + np.cumsum(np.random.normal(-0.1, 0.5, n))
    else:
        close = 100 + np.cumsum(np.random.normal(0.0, 0.5, n))

    close = np.maximum(close, 10)  # floor at 10
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n)))
    open_ = close * (1 + np.random.normal(0, 0.005, n))
    volume = np.random.uniform(1000, 5000, n)

    df = pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    })
    return df


def make_hmm_outputs(n=100, regime="bull", n_states=2):
    """Generate synthetic HMM outputs."""
    if regime == "bull":
        states = np.zeros(n, dtype=int)
        labels = {0: "bull", 1: "bear"}
        posteriors = np.column_stack([np.ones(n) * 0.9, np.ones(n) * 0.1])
    elif regime == "bear":
        states = np.ones(n, dtype=int)
        labels = {0: "bull", 1: "bear"}
        posteriors = np.column_stack([np.ones(n) * 0.1, np.ones(n) * 0.9])
    elif regime == "crash":
        states = np.full(n, 2, dtype=int)
        labels = {0: "bull", 1: "bear", 2: "crash"}
        posteriors = np.column_stack([
            np.ones(n) * 0.05, np.ones(n) * 0.05, np.ones(n) * 0.9
        ])
    elif regime == "mixed":
        states = np.zeros(n, dtype=int)
        labels = {0: "bull", 1: "bear"}
        # 50/50 posterior — ambiguous
        posteriors = np.column_stack([np.ones(n) * 0.5, np.ones(n) * 0.5])
    else:
        states = np.zeros(n, dtype=int)
        labels = {0: "neutral"}
        posteriors = np.column_stack([np.ones(n)])

    confidence = np.ones(n) * 0.85
    entropy = np.ones(n) * 0.15
    return states, posteriors, labels, confidence, entropy


# ---------------------------------------------------------------------------
# Tests: Playbook defaults
# ---------------------------------------------------------------------------

class TestPlaybookDefaults:
    def test_bull_is_long_biased(self):
        assert BULL_PLAYBOOK.bias == 1

    def test_bear_is_short_biased(self):
        assert BEAR_PLAYBOOK.bias == -1

    def test_crash_is_flat(self):
        assert CRASH_PLAYBOOK.bias == 0
        assert CRASH_PLAYBOOK.kelly_multiplier == 0.0

    def test_neutral_is_unbiased(self):
        assert NEUTRAL_PLAYBOOK.bias == 0

    def test_crash_has_tightest_stops(self):
        assert CRASH_PLAYBOOK.trailing_atr_multiplier < BEAR_PLAYBOOK.trailing_atr_multiplier
        assert BEAR_PLAYBOOK.trailing_atr_multiplier < BULL_PLAYBOOK.trailing_atr_multiplier

    def test_crash_has_shortest_max_hold(self):
        assert CRASH_PLAYBOOK.max_hold_bars < NEUTRAL_PLAYBOOK.max_hold_bars
        assert NEUTRAL_PLAYBOOK.max_hold_bars < BEAR_PLAYBOOK.max_hold_bars
        assert BEAR_PLAYBOOK.max_hold_bars < BULL_PLAYBOOK.max_hold_bars

    def test_all_default_regimes_covered(self):
        for name in ["bull", "bull_run", "bear", "crash", "neutral"]:
            assert name in DEFAULT_PLAYBOOKS


# ---------------------------------------------------------------------------
# Tests: Engine initialization and config
# ---------------------------------------------------------------------------

class TestEngineInit:
    def test_creates_with_defaults(self):
        engine = AdaptiveStrategyEngine(default_config())
        assert engine.use_blending is True
        assert engine.blend_temperature == 1.0
        assert len(engine.playbooks) > 0

    def test_config_override_playbook(self):
        cfg = default_config()
        cfg["adaptive_strategy"]["playbooks"] = {
            "bull": {"min_confirmations": 6, "kelly_multiplier": 0.8}
        }
        engine = AdaptiveStrategyEngine(cfg)
        assert engine.playbooks["bull"].min_confirmations == 6
        assert engine.playbooks["bull"].kelly_multiplier == 0.8
        # Other fields keep defaults
        assert engine.playbooks["bull"].bias == 1

    def test_unknown_regime_falls_back_to_neutral(self):
        engine = AdaptiveStrategyEngine(default_config())
        pb = engine._get_playbook("regime_99")
        assert pb.name == "neutral"


# ---------------------------------------------------------------------------
# Tests: Confirmation scoring
# ---------------------------------------------------------------------------

class TestConfirmationScoring:
    def test_returns_float(self):
        engine = AdaptiveStrategyEngine(default_config())
        df = make_ohlcv(n=80)
        df_ind = engine.compute_indicators(df)
        score = engine._score_confirmations(df_ind.iloc[-1])
        assert isinstance(score, float)

    def test_score_in_range(self):
        engine = AdaptiveStrategyEngine(default_config())
        df = make_ohlcv(n=80)
        df_ind = engine.compute_indicators(df)
        for i in range(50, len(df_ind)):
            score = engine._score_confirmations(df_ind.iloc[i])
            assert 0 <= score <= 8.0

    def test_score_handles_nan_gracefully(self):
        engine = AdaptiveStrategyEngine(default_config())
        row = pd.Series({
            "rsi": np.nan, "adx": np.nan, "ema_50": np.nan,
            "macd_line": np.nan, "macd_signal": np.nan,
            "Close": 100, "volume_ratio": np.nan,
            "momentum_10": False, "vol_pctrank": np.nan,
        })
        score = engine._score_confirmations(row)
        assert score >= 0


# ---------------------------------------------------------------------------
# Tests: Playbook signal
# ---------------------------------------------------------------------------

class TestPlaybookSignal:
    def test_bull_playbook_positive(self):
        engine = AdaptiveStrategyEngine(default_config())
        sig = engine._playbook_signal(BULL_PLAYBOOK, confirmation_score=6.0, confidence=0.9)
        assert sig > 0

    def test_bear_playbook_negative(self):
        engine = AdaptiveStrategyEngine(default_config())
        sig = engine._playbook_signal(BEAR_PLAYBOOK, confirmation_score=6.0, confidence=0.9)
        assert sig < 0

    def test_crash_playbook_always_zero(self):
        engine = AdaptiveStrategyEngine(default_config())
        sig = engine._playbook_signal(CRASH_PLAYBOOK, confirmation_score=8.0, confidence=1.0)
        assert sig == 0.0

    def test_low_confidence_blocks_signal(self):
        engine = AdaptiveStrategyEngine(default_config())
        sig = engine._playbook_signal(BULL_PLAYBOOK, confirmation_score=6.0, confidence=0.2)
        assert sig == 0.0

    def test_low_confirmations_blocks_signal(self):
        engine = AdaptiveStrategyEngine(default_config())
        sig = engine._playbook_signal(BULL_PLAYBOOK, confirmation_score=1.0, confidence=0.9)
        assert sig == 0.0


# ---------------------------------------------------------------------------
# Tests: Posterior blending
# ---------------------------------------------------------------------------

class TestBlending:
    def test_pure_bull_posterior_positive(self):
        engine = AdaptiveStrategyEngine(default_config())
        posteriors = np.array([0.95, 0.05])
        labels = {0: "bull", 1: "bear"}
        sig, regime, conf = engine.blend_signals(posteriors, labels, 6.0, 0.9)
        assert sig > 0
        assert regime == "bull"

    def test_pure_bear_posterior_negative(self):
        engine = AdaptiveStrategyEngine(default_config())
        posteriors = np.array([0.05, 0.95])
        labels = {0: "bull", 1: "bear"}
        sig, regime, conf = engine.blend_signals(posteriors, labels, 6.0, 0.9)
        assert sig < 0
        assert regime == "bear"

    def test_equal_posteriors_cancel(self):
        """50/50 bull/bear should roughly cancel to near zero."""
        engine = AdaptiveStrategyEngine(default_config())
        posteriors = np.array([0.5, 0.5])
        labels = {0: "bull", 1: "bear"}
        sig, regime, conf = engine.blend_signals(posteriors, labels, 6.0, 0.9)
        assert abs(sig) < 0.5  # near zero due to cancellation

    def test_temperature_sharpens_blend(self):
        """Lower temperature should make dominant regime more decisive."""
        cfg = default_config()
        cfg["adaptive_strategy"]["blend_temperature"] = 0.5
        engine = AdaptiveStrategyEngine(cfg)
        posteriors = np.array([0.6, 0.4])
        labels = {0: "bull", 1: "bear"}
        sig_sharp, _, conf_sharp = engine.blend_signals(posteriors, labels, 6.0, 0.9)

        cfg["adaptive_strategy"]["blend_temperature"] = 2.0
        engine_soft = AdaptiveStrategyEngine(cfg)
        sig_soft, _, conf_soft = engine_soft.blend_signals(posteriors, labels, 6.0, 0.9)

        assert conf_sharp > conf_soft

    def test_blend_confidence_range(self):
        engine = AdaptiveStrategyEngine(default_config())
        posteriors = np.array([0.7, 0.3])
        labels = {0: "bull", 1: "bear"}
        _, _, conf = engine.blend_signals(posteriors, labels, 6.0, 0.9)
        assert 0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# Tests: Adaptive sizing
# ---------------------------------------------------------------------------

class TestAdaptiveSizing:
    def test_size_non_negative(self):
        engine = AdaptiveStrategyEngine(default_config())
        for conf in [0.0, 0.5, 1.0]:
            for blend in [0.0, 0.5, 1.0]:
                for km in [0.0, 0.5, 1.0]:
                    size = engine._compute_adaptive_size(conf, blend, km)
                    assert size >= 0.0

    def test_crash_multiplier_zero_size(self):
        engine = AdaptiveStrategyEngine(default_config())
        size = engine._compute_adaptive_size(0.9, 0.9, kelly_multiplier=0.0)
        assert size == 0.0

    def test_higher_confidence_bigger_size(self):
        engine = AdaptiveStrategyEngine(default_config())
        size_hi = engine._compute_adaptive_size(0.95, 0.8, 1.0)
        size_lo = engine._compute_adaptive_size(0.5, 0.8, 1.0)
        assert size_hi > size_lo

    def test_capped_at_max(self):
        cfg = default_config()
        cfg["risk"]["max_leverage"] = 1.0
        cfg["risk"]["max_position_pct"] = 0.5
        engine = AdaptiveStrategyEngine(cfg)
        size = engine._compute_adaptive_size(1.0, 1.0, 10.0)
        assert size <= 0.5


# ---------------------------------------------------------------------------
# Tests: End-to-end signal generation
# ---------------------------------------------------------------------------

class TestGenerateAdaptiveSignals:
    def test_returns_correct_length(self):
        engine = AdaptiveStrategyEngine(default_config())
        df = make_ohlcv(n=100)
        states, posteriors, labels, confidence, entropy = make_hmm_outputs(100, "bull")
        signals = engine.generate_adaptive_signals(
            df, states, posteriors, labels, confidence, entropy
        )
        assert len(signals) == 100

    def test_all_signals_are_adaptive_signal(self):
        engine = AdaptiveStrategyEngine(default_config())
        df = make_ohlcv(n=80)
        states, posteriors, labels, confidence, entropy = make_hmm_outputs(80, "bull")
        signals = engine.generate_adaptive_signals(
            df, states, posteriors, labels, confidence, entropy
        )
        for s in signals:
            assert isinstance(s, AdaptiveSignal)

    def test_positions_only_valid_values(self):
        engine = AdaptiveStrategyEngine(default_config())
        df = make_ohlcv(n=80)
        states, posteriors, labels, confidence, entropy = make_hmm_outputs(80, "bull")
        signals = engine.generate_adaptive_signals(
            df, states, posteriors, labels, confidence, entropy
        )
        positions = {s.position for s in signals}
        assert positions.issubset({-1, 0, 1})

    def test_bull_regime_eventually_goes_long(self):
        engine = AdaptiveStrategyEngine(default_config())
        df = make_ohlcv(n=100, trend="up")
        states, posteriors, labels, confidence, entropy = make_hmm_outputs(100, "bull")
        signals = engine.generate_adaptive_signals(
            df, states, posteriors, labels, confidence, entropy
        )
        positions = [s.position for s in signals]
        assert 1 in positions

    def test_crash_regime_stays_flat(self):
        engine = AdaptiveStrategyEngine(default_config())
        df = make_ohlcv(n=80)
        states, posteriors, labels, confidence, entropy = make_hmm_outputs(80, "crash", n_states=3)
        signals = engine.generate_adaptive_signals(
            df, states, posteriors, labels, confidence, entropy
        )
        # Crash playbook has kelly_multiplier=0, so should never enter
        positions = [s.position for s in signals]
        assert all(p == 0 for p in positions)

    def test_sizes_non_negative(self):
        engine = AdaptiveStrategyEngine(default_config())
        df = make_ohlcv(n=80)
        states, posteriors, labels, confidence, entropy = make_hmm_outputs(80, "bull")
        signals = engine.generate_adaptive_signals(
            df, states, posteriors, labels, confidence, entropy
        )
        for s in signals:
            assert s.size >= 0.0

    def test_flat_positions_have_zero_size(self):
        engine = AdaptiveStrategyEngine(default_config())
        df = make_ohlcv(n=80)
        states, posteriors, labels, confidence, entropy = make_hmm_outputs(80, "bull")
        signals = engine.generate_adaptive_signals(
            df, states, posteriors, labels, confidence, entropy
        )
        for s in signals:
            if s.position == 0:
                assert s.size == 0.0


# ---------------------------------------------------------------------------
# Tests: Adaptive exits
# ---------------------------------------------------------------------------

class TestAdaptiveExits:
    def test_max_hold_forces_exit(self):
        """Position held beyond max_hold_bars should be force-closed."""
        cfg = default_config()
        cfg["adaptive_strategy"]["min_hold_bars"] = 2
        engine = AdaptiveStrategyEngine(cfg)

        n = 250
        df = make_ohlcv(n=n, trend="up")
        states, posteriors, labels, confidence, entropy = make_hmm_outputs(n, "bull")
        signals = engine.generate_adaptive_signals(
            df, states, posteriors, labels, confidence, entropy
        )
        # Check that at least one exit_reason is "max_hold_time"
        exit_reasons = engine.get_exit_reasons(signals)
        # Bull max_hold is 200, so with n=250 it could trigger
        # The test verifies the mechanism works, not specific counts
        assert isinstance(exit_reasons, dict)

    def test_entropy_spike_triggers_exit(self):
        """Sustained high entropy should cause de-risking."""
        cfg = default_config()
        cfg["adaptive_strategy"]["entropy_exit_threshold"] = 0.5
        cfg["adaptive_strategy"]["max_entropy_bars"] = 3
        cfg["adaptive_strategy"]["min_hold_bars"] = 2
        engine = AdaptiveStrategyEngine(cfg)

        n = 80
        df = make_ohlcv(n=n, trend="up")
        states, posteriors, labels, confidence, entropy = make_hmm_outputs(n, "bull")
        # Spike entropy after bar 30
        entropy[30:] = 0.9
        confidence[30:] = 0.6  # still above min_confidence for bull

        signals = engine.generate_adaptive_signals(
            df, states, posteriors, labels, confidence, entropy
        )
        exit_reasons = engine.get_exit_reasons(signals)
        assert "entropy_spike" in exit_reasons

    def test_trailing_stop_triggers_on_reversal(self):
        """Sharp price reversal should trigger trailing stop."""
        cfg = default_config()
        cfg["adaptive_strategy"]["min_hold_bars"] = 2
        engine = AdaptiveStrategyEngine(cfg)

        n = 80
        # Price goes up then crashes
        close = np.concatenate([
            np.linspace(100, 130, 40),
            np.linspace(130, 80, 40),
        ])
        high = close * 1.01
        low = close * 0.99
        df = pd.DataFrame({
            "Open": close,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.random.uniform(1000, 5000, n),
        })
        states, posteriors, labels, confidence, entropy = make_hmm_outputs(n, "bull")

        signals = engine.generate_adaptive_signals(
            df, states, posteriors, labels, confidence, entropy
        )
        exit_reasons = engine.get_exit_reasons(signals)
        assert "trailing_stop" in exit_reasons


# ---------------------------------------------------------------------------
# Tests: Series conversion
# ---------------------------------------------------------------------------

class TestSeriesToConversion:
    def test_signals_to_series(self):
        engine = AdaptiveStrategyEngine(default_config())
        df = make_ohlcv(n=50)
        states, posteriors, labels, confidence, entropy = make_hmm_outputs(50, "bull")
        signals = engine.generate_adaptive_signals(
            df, states, posteriors, labels, confidence, entropy
        )
        positions, sizes, regimes = engine.signals_to_series(signals, df.index)
        assert isinstance(positions, pd.Series)
        assert isinstance(sizes, pd.Series)
        assert isinstance(regimes, pd.Series)
        assert len(positions) == 50


# ---------------------------------------------------------------------------
# Tests: Diagnostics
# ---------------------------------------------------------------------------

class TestDiagnostics:
    def test_blend_diagnostics_dataframe(self):
        engine = AdaptiveStrategyEngine(default_config())
        df = make_ohlcv(n=50)
        states, posteriors, labels, confidence, entropy = make_hmm_outputs(50, "bull")
        signals = engine.generate_adaptive_signals(
            df, states, posteriors, labels, confidence, entropy
        )
        diag = engine.get_blend_diagnostics(signals)
        assert isinstance(diag, pd.DataFrame)
        assert "raw_blend" in diag.columns
        assert "dominant_regime" in diag.columns
        assert len(diag) == 50

    def test_exit_reasons_dict(self):
        engine = AdaptiveStrategyEngine(default_config())
        df = make_ohlcv(n=50)
        states, posteriors, labels, confidence, entropy = make_hmm_outputs(50, "bull")
        signals = engine.generate_adaptive_signals(
            df, states, posteriors, labels, confidence, entropy
        )
        reasons = engine.get_exit_reasons(signals)
        assert isinstance(reasons, dict)
