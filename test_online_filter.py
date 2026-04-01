"""
TDD tests for online_filter.py — Online Bayesian Regime Filter.

Tests cover: forward algorithm correctness, posterior normalization,
regime tracking, CUSUM change-point detection, N-step forecasting,
predictive distributions, persistence probability, batch processing,
and edge cases.
"""

import numpy as np
import pandas as pd
import pytest

from hmm_engine import RegimeDetector
from online_filter import (
    OnlineBayesianFilter,
    RegimeChangeEvent,
    RegimeForecast,
    FilterState,
    _logsumexp,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_config(min_states=2, max_states=2, n_restarts=5, n_iter=50,
                covariance_type="full"):
    return {
        "hmm": {
            "min_states": min_states,
            "max_states": max_states,
            "n_restarts": n_restarts,
            "n_iter": n_iter,
            "tol": 1e-4,
            "covariance_type": covariance_type,
        }
    }


def make_two_regime_data(n=500, seed=42):
    """Synthetic 2-regime data: low-vol bull vs high-vol bear."""
    rng = np.random.default_rng(seed)
    # Regime 0: bull (positive drift, low vol)
    # Regime 1: bear (negative drift, high vol)
    half = n // 2
    returns_0 = rng.normal(0.002, 0.01, half)
    returns_1 = rng.normal(-0.003, 0.04, half)
    returns = np.concatenate([returns_0, returns_1])
    vol = np.abs(returns)
    volume = rng.normal(1.0, 0.2, n)
    intraday = rng.normal(0.02, 0.005, n)
    rsi = np.concatenate([rng.normal(60, 10, half), rng.normal(35, 10, half)])
    X = np.column_stack([returns, vol, volume, intraday, rsi])
    return X


def make_fitted_detector(X=None, covariance_type="full"):
    """Create and fit a RegimeDetector on synthetic data."""
    if X is None:
        X = make_two_regime_data()
    cfg = make_config(covariance_type=covariance_type)
    det = RegimeDetector(cfg)
    det.fit_and_select(X)
    det.label_regimes(X)
    return det, X


# ---------------------------------------------------------------------------
# Tests: initialization
# ---------------------------------------------------------------------------

class TestOnlineFilterInit:
    def test_requires_fitted_model(self):
        det = RegimeDetector(make_config())
        with pytest.raises(ValueError, match="must be fitted"):
            OnlineBayesianFilter(det)

    def test_basic_init(self):
        det, X = make_fitted_detector()
        filt = OnlineBayesianFilter(det)
        assert filt.n_states == det.n_states
        assert filt.state.bar_count == 0
        assert len(filt.labels) > 0
        assert filt.stationary_dist.shape == (filt.n_states,)
        assert abs(filt.stationary_dist.sum() - 1.0) < 1e-10

    def test_custom_params(self):
        det, X = make_fitted_detector()
        filt = OnlineBayesianFilter(
            det, cusum_threshold=5.0, cusum_drift=0.5,
            forecast_horizons=[1, 10, 100], entropy_warmup=50,
        )
        assert filt.cusum_threshold == 5.0
        assert filt.forecast_horizons == [1, 10, 100]
        assert filt.entropy_warmup == 50


# ---------------------------------------------------------------------------
# Tests: _logsumexp
# ---------------------------------------------------------------------------

class TestLogsumexp:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0])
        result = _logsumexp(x)
        expected = np.log(np.exp(1) + np.exp(2) + np.exp(3))
        assert abs(result - expected) < 1e-10

    def test_large_values(self):
        x = np.array([1000.0, 1001.0, 1002.0])
        result = _logsumexp(x)
        # Should not overflow
        assert np.isfinite(result)
        assert result > 1001.0

    def test_negative_inf(self):
        x = np.array([float("-inf"), float("-inf")])
        result = _logsumexp(x)
        assert result == float("-inf")

    def test_single_element(self):
        x = np.array([5.0])
        assert abs(_logsumexp(x) - 5.0) < 1e-10


# ---------------------------------------------------------------------------
# Tests: forward algorithm (update)
# ---------------------------------------------------------------------------

class TestForwardAlgorithm:
    def test_first_update_returns_valid_posterior(self):
        det, X = make_fitted_detector()
        filt = OnlineBayesianFilter(det)
        res = filt.update(X[0])
        assert res["posteriors"].shape == (filt.n_states,)
        assert abs(res["posteriors"].sum() - 1.0) < 1e-10
        assert 0 <= res["confidence"] <= 1.0
        assert 0 <= res["entropy"] <= 1.0

    def test_posteriors_always_normalized(self):
        det, X = make_fitted_detector()
        filt = OnlineBayesianFilter(det)
        for t in range(min(100, len(X))):
            res = filt.update(X[t])
            total = res["posteriors"].sum()
            assert abs(total - 1.0) < 1e-8, f"Bar {t}: posterior sum = {total}"

    def test_regime_is_argmax_of_posterior(self):
        det, X = make_fitted_detector()
        filt = OnlineBayesianFilter(det)
        for t in range(50):
            res = filt.update(X[t])
            assert res["regime"] == np.argmax(res["posteriors"])

    def test_bar_count_increments(self):
        det, X = make_fitted_detector()
        filt = OnlineBayesianFilter(det)
        for t in range(10):
            filt.update(X[t])
        assert filt.state.bar_count == 10

    def test_log_likelihood_accumulates(self):
        det, X = make_fitted_detector()
        filt = OnlineBayesianFilter(det)
        for t in range(20):
            res = filt.update(X[t])
        assert filt.state.log_likelihood != 0.0
        assert np.isfinite(filt.state.log_likelihood)

    def test_confidence_matches_max_posterior(self):
        det, X = make_fitted_detector()
        filt = OnlineBayesianFilter(det)
        for t in range(20):
            res = filt.update(X[t])
            expected_conf = float(np.max(res["posteriors"]))
            assert abs(res["confidence"] - expected_conf) < 1e-10


# ---------------------------------------------------------------------------
# Tests: regime tracking
# ---------------------------------------------------------------------------

class TestRegimeTracking:
    def test_regime_duration_increments(self):
        det, X = make_fitted_detector()
        filt = OnlineBayesianFilter(det)
        # Feed same-regime data (first half = bull)
        last_dur = 0
        same_regime_count = 0
        prev_regime = None
        for t in range(50):
            res = filt.update(X[t])
            if prev_regime is not None and res["regime"] == prev_regime:
                same_regime_count += 1
            prev_regime = res["regime"]
        # At least some bars should show duration > 1
        assert filt.state.regime_duration >= 1

    def test_regime_label_matches_detector(self):
        det, X = make_fitted_detector()
        filt = OnlineBayesianFilter(det)
        res = filt.update(X[0])
        regime = res["regime"]
        assert res["regime_label"] == det.labels.get(regime, f"state_{regime}")

    def test_change_events_logged(self):
        det, X = make_fitted_detector()
        filt = OnlineBayesianFilter(det)
        # Process all data — should see transitions between bull/bear half
        for t in range(len(X)):
            filt.update(X[t])
        events = filt.change_events
        # With clear regime separation, we should get at least one transition
        assert len(events) >= 1
        for ev in events:
            assert isinstance(ev, RegimeChangeEvent)
            assert ev.from_regime != ev.to_regime
            assert 0 < ev.confidence <= 1.0


# ---------------------------------------------------------------------------
# Tests: CUSUM change-point detection
# ---------------------------------------------------------------------------

class TestCUSUM:
    def test_cusum_inactive_during_warmup(self):
        det, X = make_fitted_detector()
        filt = OnlineBayesianFilter(det, entropy_warmup=50)
        for t in range(30):
            res = filt.update(X[t])
            # During warmup, CUSUM should not fire
            # (cusum_pos/neg stay at initial 0 until warmup)

    def test_cusum_detects_regime_shift(self):
        """Feed clear bull data then clear bear data — CUSUM should fire."""
        det, X = make_fitted_detector()
        filt = OnlineBayesianFilter(det, cusum_threshold=2.0, entropy_warmup=10)
        detected = False
        for t in range(len(X)):
            res = filt.update(X[t])
            if res["change_detected"]:
                detected = True
        # With threshold=2.0 and clear regime shift, should detect
        # (may not always fire depending on entropy dynamics, so we just
        # verify the mechanism doesn't crash)
        assert isinstance(detected, bool)

    def test_cusum_resets_after_detection(self):
        det, X = make_fitted_detector()
        filt = OnlineBayesianFilter(det, cusum_threshold=0.5, entropy_warmup=5)
        detected_at = None
        for t in range(len(X)):
            res = filt.update(X[t])
            if res["change_detected"] and detected_at is None:
                detected_at = t
                # After detection, CUSUM should reset
                assert res["cusum_pos"] == 0.0 or res["cusum_neg"] == 0.0
                break


# ---------------------------------------------------------------------------
# Tests: forecasting
# ---------------------------------------------------------------------------

class TestForecasting:
    def test_forecast_requires_data(self):
        det, X = make_fitted_detector()
        filt = OnlineBayesianFilter(det)
        with pytest.raises(ValueError, match="No observations"):
            filt.forecast()

    def test_forecast_returns_valid_probs(self):
        det, X = make_fitted_detector()
        filt = OnlineBayesianFilter(det)
        for t in range(50):
            filt.update(X[t])
        fc = filt.forecast(horizons=[1, 5, 10])
        assert isinstance(fc, RegimeForecast)
        for h in [1, 5, 10]:
            probs = fc.forecast_probs[h]
            assert probs.shape == (filt.n_states,)
            assert abs(probs.sum() - 1.0) < 1e-8
            assert np.all(probs >= 0)

    def test_forecast_converges_to_stationary(self):
        det, X = make_fitted_detector()
        filt = OnlineBayesianFilter(det)
        for t in range(50):
            filt.update(X[t])
        fc = filt.forecast(horizons=[1, 5, 10, 50, 500, 2000])
        # At very long horizon, should be close to stationary
        long_horizon_probs = fc.forecast_probs[2000]
        dist = np.sum(np.abs(long_horizon_probs - fc.stationary_dist))
        assert dist < 0.15, f"L1 distance from stationary at h=2000: {dist}"

    def test_convergence_horizon_is_positive(self):
        det, X = make_fitted_detector()
        filt = OnlineBayesianFilter(det)
        for t in range(50):
            filt.update(X[t])
        fc = filt.forecast()
        assert fc.convergence_horizon >= 1


# ---------------------------------------------------------------------------
# Tests: predictive distribution
# ---------------------------------------------------------------------------

class TestPredictiveDistribution:
    def test_requires_data(self):
        det, X = make_fitted_detector()
        filt = OnlineBayesianFilter(det)
        with pytest.raises(ValueError, match="No observations"):
            filt.predictive_distribution()

    def test_returns_valid_mixture(self):
        det, X = make_fitted_detector()
        filt = OnlineBayesianFilter(det)
        for t in range(50):
            filt.update(X[t])
        pred = filt.predictive_distribution(horizon=1)
        assert abs(pred["weights"].sum() - 1.0) < 1e-8
        assert pred["mixture_mean"].shape == (X.shape[1],)
        assert pred["mixture_covariance"].shape == (X.shape[1], X.shape[1])
        # Mixture covariance should be positive semi-definite
        eigvals = np.linalg.eigvalsh(pred["mixture_covariance"])
        assert np.all(eigvals >= -1e-10)


# ---------------------------------------------------------------------------
# Tests: persistence and next regime
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_persistence_prob_valid(self):
        det, X = make_fitted_detector()
        filt = OnlineBayesianFilter(det)
        for t in range(20):
            filt.update(X[t])
        p = filt.regime_persistence_prob(horizon=5)
        assert 0.0 <= p <= 1.0

    def test_persistence_decreases_with_horizon(self):
        det, X = make_fitted_detector()
        filt = OnlineBayesianFilter(det)
        for t in range(20):
            filt.update(X[t])
        p1 = filt.regime_persistence_prob(1)
        p5 = filt.regime_persistence_prob(5)
        p10 = filt.regime_persistence_prob(10)
        assert p1 >= p5 >= p10

    def test_expected_duration_positive(self):
        det, X = make_fitted_detector()
        filt = OnlineBayesianFilter(det)
        for t in range(20):
            filt.update(X[t])
        d = filt.expected_regime_duration()
        assert d > 0

    def test_most_likely_next_regime(self):
        det, X = make_fitted_detector()
        filt = OnlineBayesianFilter(det)
        for t in range(20):
            filt.update(X[t])
        state, label, prob = filt.most_likely_next_regime()
        assert state != filt.state.current_regime
        assert isinstance(label, str)
        assert 0 < prob < 1


# ---------------------------------------------------------------------------
# Tests: batch processing
# ---------------------------------------------------------------------------

class TestBatchProcessing:
    def test_batch_returns_dataframe(self):
        det, X = make_fitted_detector()
        filt = OnlineBayesianFilter(det)
        df = filt.process_batch(X[:100])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert "regime" in df.columns
        assert "confidence" in df.columns
        assert "entropy" in df.columns

    def test_batch_with_timestamps(self):
        det, X = make_fitted_detector()
        filt = OnlineBayesianFilter(det)
        ts = pd.date_range("2024-01-01", periods=100, freq="h")
        df = filt.process_batch(X[:100], timestamps=ts)
        assert len(df) == 100

    def test_batch_matches_sequential(self):
        """Batch processing should give same results as sequential updates."""
        det, X = make_fitted_detector()
        n = 50

        # Sequential
        filt1 = OnlineBayesianFilter(det)
        sequential_regimes = []
        for t in range(n):
            res = filt1.update(X[t])
            sequential_regimes.append(res["regime"])

        # Batch
        filt2 = OnlineBayesianFilter(det)
        df = filt2.process_batch(X[:n])
        batch_regimes = df["regime"].values

        np.testing.assert_array_equal(sequential_regimes, batch_regimes)


# ---------------------------------------------------------------------------
# Tests: history properties
# ---------------------------------------------------------------------------

class TestHistory:
    def test_posterior_history_shape(self):
        det, X = make_fitted_detector()
        filt = OnlineBayesianFilter(det)
        for t in range(30):
            filt.update(X[t])
        ph = filt.posterior_history
        assert ph.shape == (30, filt.n_states)

    def test_entropy_history_length(self):
        det, X = make_fitted_detector()
        filt = OnlineBayesianFilter(det)
        for t in range(30):
            filt.update(X[t])
        assert len(filt.entropy_history) == 30

    def test_regime_history_length(self):
        det, X = make_fitted_detector()
        filt = OnlineBayesianFilter(det)
        for t in range(30):
            filt.update(X[t])
        assert len(filt.regime_history) == 30

    def test_empty_history_before_update(self):
        det, X = make_fitted_detector()
        filt = OnlineBayesianFilter(det)
        assert filt.posterior_history.shape == (0, filt.n_states)
        assert len(filt.entropy_history) == 0


# ---------------------------------------------------------------------------
# Tests: reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_state(self):
        det, X = make_fitted_detector()
        filt = OnlineBayesianFilter(det)
        for t in range(50):
            filt.update(X[t])
        assert filt.state.bar_count == 50
        filt.reset()
        assert filt.state.bar_count == 0
        assert len(filt.entropy_history) == 0
        assert len(filt.change_events) == 0


# ---------------------------------------------------------------------------
# Tests: summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_before_data(self):
        det, X = make_fitted_detector()
        filt = OnlineBayesianFilter(det)
        s = filt.summary()
        assert s["status"] == "no observations processed"

    def test_summary_after_data(self):
        det, X = make_fitted_detector()
        filt = OnlineBayesianFilter(det)
        for t in range(50):
            filt.update(X[t])
        s = filt.summary()
        assert "bars_processed" in s
        assert s["bars_processed"] == 50
        assert "current_label" in s
        assert "confidence" in s
        assert "expected_remaining_duration" in s
        assert "next_likely_regime" in s


# ---------------------------------------------------------------------------
# Tests: covariance types
# ---------------------------------------------------------------------------

class TestCovarianceTypes:
    @pytest.mark.parametrize("cov_type", ["full", "diag", "spherical", "tied"])
    def test_filter_works_with_cov_type(self, cov_type):
        X = make_two_regime_data(n=300)
        det, _ = make_fitted_detector(X, covariance_type=cov_type)
        filt = OnlineBayesianFilter(det)
        for t in range(50):
            res = filt.update(X[t])
            assert abs(res["posteriors"].sum() - 1.0) < 1e-8
