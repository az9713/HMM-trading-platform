"""
test_regime_predictor.py — Tests for Bayesian Online Change Point Detection
and regime forecasting engine.
"""

import numpy as np
import pandas as pd
import pytest

from regime_predictor import RegimePredictor, ChangePoint


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def default_config():
    return {
        "prediction": {
            "hazard_rate": 1 / 50,
            "forecast_horizon": 20,
            "changepoint_threshold": 0.3,
            "max_run_length": 200,
            "blend_weight_bocd": 0.6,
            "blend_weight_entropy": 0.4,
        }
    }


@pytest.fixture
def predictor(default_config):
    return RegimePredictor(default_config)


@pytest.fixture
def two_regime_hmm_params():
    """Emission params for a simple 2-state HMM (bull/bear)."""
    means = np.array([
        [-0.02, 0.03],  # bear: negative return, high vol
        [0.01, 0.01],   # bull: positive return, low vol
    ])
    covars = np.array([
        [[0.001, 0.0], [0.0, 0.0005]],  # bear covariance
        [[0.0005, 0.0], [0.0, 0.0002]], # bull covariance
    ])
    transmat = np.array([
        [0.95, 0.05],  # bear -> bear 95%, bear -> bull 5%
        [0.03, 0.97],  # bull -> bear 3%, bull -> bull 97%
    ])
    return means, covars, transmat


@pytest.fixture
def synthetic_regime_data(two_regime_hmm_params):
    """Generate synthetic data with a clear regime change at bar 100."""
    means, covars, _ = two_regime_hmm_params
    np.random.seed(42)

    # 100 bars of bull regime, then 100 bars of bear regime
    n_bull, n_bear = 100, 100
    X_bull = np.random.multivariate_normal(means[1], covars[1], n_bull)
    X_bear = np.random.multivariate_normal(means[0], covars[0], n_bear)
    X = np.vstack([X_bull, X_bear])

    states = np.array([1] * n_bull + [0] * n_bear)
    return X, states


# ── BOCD Core Tests ──────────────────────────────────────────────────────────

class TestBOCDCore:
    """Test the Bayesian Online Change Point Detection algorithm."""

    def test_run_bocd_output_shapes(self, predictor, synthetic_regime_data, two_regime_hmm_params):
        X, _ = synthetic_regime_data
        means, covars, transmat = two_regime_hmm_params
        stationary = np.array([0.375, 0.625])

        result = predictor.run_bocd(X, means, covars, "full", transmat, stationary)

        T = len(X)
        assert result["changepoint_prob"].shape == (T,)
        assert result["map_run_length"].shape == (T,)
        assert result["regime_log_liks"].shape == (T, 2)
        assert result["run_length_posteriors"].shape[0] == T

    def test_changepoint_prob_in_range(self, predictor, synthetic_regime_data, two_regime_hmm_params):
        X, _ = synthetic_regime_data
        means, covars, transmat = two_regime_hmm_params
        stationary = np.array([0.375, 0.625])

        result = predictor.run_bocd(X, means, covars, "full", transmat, stationary)
        cp_prob = result["changepoint_prob"]

        assert np.all(cp_prob >= 0)
        assert np.all(cp_prob <= 1.0 + 1e-10)

    def test_changepoint_detected_near_true_break(self, predictor, synthetic_regime_data, two_regime_hmm_params):
        """BOCD should detect elevated change point probability near bar 100."""
        X, _ = synthetic_regime_data
        means, covars, transmat = two_regime_hmm_params
        stationary = np.array([0.375, 0.625])

        result = predictor.run_bocd(X, means, covars, "full", transmat, stationary)
        cp_prob = result["changepoint_prob"]

        # The CP probability should be higher in the region around the true break
        # than in the stable middle of the first regime
        region_around_break = cp_prob[90:120]
        stable_region = cp_prob[20:60]

        assert np.max(region_around_break) > np.mean(stable_region)

    def test_run_length_posteriors_sum_to_one(self, predictor, synthetic_regime_data, two_regime_hmm_params):
        X, _ = synthetic_regime_data
        means, covars, transmat = two_regime_hmm_params
        stationary = np.array([0.375, 0.625])

        result = predictor.run_bocd(X, means, covars, "full", transmat, stationary)
        rl_post = result["run_length_posteriors"]

        # Each row should sum to approximately 1
        row_sums = rl_post.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=0.05)

    def test_map_run_length_nonnegative(self, predictor, synthetic_regime_data, two_regime_hmm_params):
        X, _ = synthetic_regime_data
        means, covars, transmat = two_regime_hmm_params
        stationary = np.array([0.375, 0.625])

        result = predictor.run_bocd(X, means, covars, "full", transmat, stationary)
        assert np.all(result["map_run_length"] >= 0)

    def test_regime_log_liks_finite(self, predictor, synthetic_regime_data, two_regime_hmm_params):
        X, _ = synthetic_regime_data
        means, covars, transmat = two_regime_hmm_params
        stationary = np.array([0.375, 0.625])

        result = predictor.run_bocd(X, means, covars, "full", transmat, stationary)
        ll = result["regime_log_liks"]

        # Should be finite (not NaN or +inf)
        assert np.all(np.isfinite(ll))

    def test_bocd_with_diag_covariance(self, predictor):
        """Test BOCD with diagonal covariance type."""
        np.random.seed(123)
        means = np.array([[0.01, 0.01], [-0.01, 0.03]])
        covars = np.array([[0.001, 0.0005], [0.002, 0.001]])  # diag
        transmat = np.array([[0.95, 0.05], [0.05, 0.95]])
        stationary = np.array([0.5, 0.5])

        X = np.random.randn(50, 2) * 0.03
        result = predictor.run_bocd(X, means, covars, "diag", transmat, stationary)

        assert result["changepoint_prob"].shape == (50,)
        assert np.all(result["changepoint_prob"] >= 0)

    def test_bocd_with_spherical_covariance(self, predictor):
        """Test BOCD with spherical covariance type."""
        np.random.seed(456)
        means = np.array([[0.01, 0.01], [-0.01, 0.03]])
        covars = np.array([0.001, 0.002])  # spherical
        transmat = np.array([[0.95, 0.05], [0.05, 0.95]])
        stationary = np.array([0.5, 0.5])

        X = np.random.randn(50, 2) * 0.03
        result = predictor.run_bocd(X, means, covars, "spherical", transmat, stationary)

        assert result["changepoint_prob"].shape == (50,)


# ── Change Point Detection Tests ─────────────────────────────────────────────

class TestChangePointDetection:

    def test_detect_changepoints_returns_list(self, predictor):
        cp_prob = np.array([0.0, 0.1, 0.5, 0.8, 0.1, 0.0])
        states = np.array([0, 0, 0, 1, 1, 1])
        labels = {0: "bear", 1: "bull"}

        cps = predictor.detect_changepoints(cp_prob, states, labels)
        assert isinstance(cps, list)
        assert all(isinstance(cp, ChangePoint) for cp in cps)

    def test_detect_threshold_filtering(self, predictor):
        """Only points above threshold should be returned."""
        cp_prob = np.array([0.0, 0.1, 0.2, 0.5, 0.8, 0.1])
        states = np.array([0, 0, 0, 1, 1, 1])
        labels = {0: "bear", 1: "bull"}

        cps = predictor.detect_changepoints(cp_prob, states, labels)

        # With threshold 0.3, bars 3 (0.5) and 4 (0.8) should be detected
        detected_bars = {cp.bar for cp in cps}
        assert 3 in detected_bars
        assert 4 in detected_bars
        assert 1 not in detected_bars

    def test_detection_lag_negative_means_early(self, predictor):
        """Negative detection_lag means BOCD caught it before Viterbi."""
        cp_prob = np.zeros(20)
        cp_prob[8] = 0.6  # BOCD detects at bar 8
        states = np.array([0]*10 + [1]*10)  # Viterbi transition at bar 10
        labels = {0: "bear", 1: "bull"}

        cps = predictor.detect_changepoints(cp_prob, states, labels)
        early = [cp for cp in cps if cp.bar == 8]
        assert len(early) == 1
        assert early[0].detection_lag < 0  # detected before Viterbi

    def test_no_changepoints_below_threshold(self, predictor):
        cp_prob = np.full(50, 0.1)
        states = np.zeros(50, dtype=int)
        labels = {0: "neutral"}

        cps = predictor.detect_changepoints(cp_prob, states, labels)
        assert len(cps) == 0


# ── Regime Forecast Tests ────────────────────────────────────────────────────

class TestRegimeForecast:

    def test_forecast_shape(self, predictor, two_regime_hmm_params):
        _, _, transmat = two_regime_hmm_params
        current_dist = np.array([0.1, 0.9])

        forecast = predictor.forecast_regime_probabilities(transmat, current_dist)
        assert forecast.shape == (20, 2)  # default horizon=20, 2 states

    def test_forecast_probabilities_sum_to_one(self, predictor, two_regime_hmm_params):
        _, _, transmat = two_regime_hmm_params
        current_dist = np.array([0.1, 0.9])

        forecast = predictor.forecast_regime_probabilities(transmat, current_dist)
        row_sums = forecast.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_forecast_converges_to_stationary(self, predictor, two_regime_hmm_params):
        """Far enough in the future, forecast should approach stationary distribution."""
        _, _, transmat = two_regime_hmm_params
        current_dist = np.array([1.0, 0.0])  # start fully in bear

        forecast = predictor.forecast_regime_probabilities(
            transmat, current_dist, horizon=500
        )

        # Compute stationary distribution
        eigenvalues, eigenvectors = np.linalg.eig(transmat.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()

        # Last row should be close to stationary
        np.testing.assert_allclose(forecast[-1], stationary, atol=0.01)

    def test_forecast_custom_horizon(self, predictor, two_regime_hmm_params):
        _, _, transmat = two_regime_hmm_params
        current_dist = np.array([0.5, 0.5])

        forecast = predictor.forecast_regime_probabilities(
            transmat, current_dist, horizon=5
        )
        assert forecast.shape == (5, 2)

    def test_forecast_from_pure_state(self, predictor, two_regime_hmm_params):
        """Starting from pure bull state, first step should reflect transition row."""
        _, _, transmat = two_regime_hmm_params
        current_dist = np.array([0.0, 1.0])  # pure bull

        forecast = predictor.forecast_regime_probabilities(transmat, current_dist)
        np.testing.assert_allclose(forecast[0], transmat[1], atol=1e-10)


# ── Duration Forecast Tests ──────────────────────────────────────────────────

class TestDurationForecast:

    def test_duration_forecast_keys(self, predictor, two_regime_hmm_params):
        _, _, transmat = two_regime_hmm_params

        result = predictor.expected_regime_duration(transmat, 1, 10)

        expected_keys = {
            "current_run_length", "expected_remaining", "median_remaining",
            "self_transition_prob", "p_transition_5_bars",
            "most_likely_next_state", "next_state_probability",
            "survival_curve",
        }
        assert set(result.keys()) == expected_keys

    def test_expected_remaining_positive(self, predictor, two_regime_hmm_params):
        _, _, transmat = two_regime_hmm_params

        result = predictor.expected_regime_duration(transmat, 0, 5)
        assert result["expected_remaining"] > 0

    def test_self_transition_prob_matches(self, predictor, two_regime_hmm_params):
        _, _, transmat = two_regime_hmm_params

        result = predictor.expected_regime_duration(transmat, 0, 5)
        assert result["self_transition_prob"] == transmat[0, 0]

    def test_expected_duration_formula(self, predictor, two_regime_hmm_params):
        """E[remaining] should be 1/(1 - a_ii) for geometric distribution."""
        _, _, transmat = two_regime_hmm_params

        result = predictor.expected_regime_duration(transmat, 1, 0)
        expected = 1.0 / (1.0 - transmat[1, 1])
        assert abs(result["expected_remaining"] - expected) < 1e-10

    def test_survival_curve_decreasing(self, predictor, two_regime_hmm_params):
        _, _, transmat = two_regime_hmm_params

        result = predictor.expected_regime_duration(transmat, 1, 0)
        curve = result["survival_curve"]

        # Survival probability should be monotonically decreasing
        assert np.all(np.diff(curve) <= 0)

    def test_p_transition_5_bars_in_range(self, predictor, two_regime_hmm_params):
        _, _, transmat = two_regime_hmm_params

        result = predictor.expected_regime_duration(transmat, 0, 5)
        assert 0 <= result["p_transition_5_bars"] <= 1

    def test_most_likely_next_excludes_self(self, predictor, two_regime_hmm_params):
        _, _, transmat = two_regime_hmm_params

        result = predictor.expected_regime_duration(transmat, 0, 5)
        assert result["most_likely_next_state"] != 0  # should be state 1 (bull)


# ── Composite Early Detection Tests ──────────────────────────────────────────

class TestCompositeDetection:

    def test_score_in_range(self, predictor):
        cp_prob = np.array([0.0, 0.3, 0.7, 1.0])
        entropy = np.array([0.0, 0.5, 0.8, 1.0])
        confidence = np.array([1.0, 0.5, 0.2, 0.0])

        score = predictor.composite_early_detection(cp_prob, entropy, confidence)
        assert np.all(score >= 0)
        assert np.all(score <= 1.0)

    def test_high_cp_gives_high_score(self, predictor):
        cp_prob = np.array([0.9])
        entropy = np.array([0.1])
        confidence = np.array([0.9])

        score = predictor.composite_early_detection(cp_prob, entropy, confidence)
        assert score[0] > 0.5

    def test_low_everything_gives_low_score(self, predictor):
        cp_prob = np.array([0.0])
        entropy = np.array([0.0])
        confidence = np.array([1.0])

        score = predictor.composite_early_detection(cp_prob, entropy, confidence)
        assert score[0] < 0.1

    def test_blending_weights(self):
        """Test that custom blend weights are respected."""
        config = {
            "prediction": {
                "blend_weight_bocd": 1.0,
                "blend_weight_entropy": 0.0,
            }
        }
        pred = RegimePredictor(config)

        cp_prob = np.array([0.5])
        entropy = np.array([1.0])
        confidence = np.array([0.0])

        score = pred.composite_early_detection(cp_prob, entropy, confidence)
        assert abs(score[0] - 0.5) < 1e-10  # only BOCD contributes


# ── Full Pipeline Test ───────────────────────────────────────────────────────

class TestFullPipeline:

    def test_generate_forecast_summary(self, predictor, synthetic_regime_data, two_regime_hmm_params):
        X, states = synthetic_regime_data
        means, covars, transmat = two_regime_hmm_params
        n_states = 2

        # Simulate posteriors and entropy/confidence
        posteriors = np.zeros((len(X), n_states))
        for t in range(len(X)):
            posteriors[t, states[t]] = 0.8
            posteriors[t, 1 - states[t]] = 0.2

        labels = {0: "bear", 1: "bull"}
        eps = 1e-12
        p = np.clip(posteriors, eps, 1.0)
        entropy = -np.sum(p * np.log2(p), axis=1)
        confidence = 1.0 - entropy / np.log2(n_states)

        summary = predictor.generate_forecast_summary(
            X, states, posteriors, labels, transmat,
            means, covars, "full", entropy, confidence
        )

        # Check all expected keys
        assert "bocd_results" in summary
        assert "changepoints" in summary
        assert "early_detection_score" in summary
        assert "duration_forecast" in summary
        assert "regime_forecast" in summary
        assert "current_regime" in summary
        assert "n_changepoints_detected" in summary
        assert "stationary_distribution" in summary

    def test_summary_regime_forecast_shape(self, predictor, synthetic_regime_data, two_regime_hmm_params):
        X, states = synthetic_regime_data
        means, covars, transmat = two_regime_hmm_params

        posteriors = np.zeros((len(X), 2))
        for t in range(len(X)):
            posteriors[t, states[t]] = 0.9
            posteriors[t, 1 - states[t]] = 0.1

        labels = {0: "bear", 1: "bull"}
        entropy = np.full(len(X), 0.3)
        confidence = np.full(len(X), 0.7)

        summary = predictor.generate_forecast_summary(
            X, states, posteriors, labels, transmat,
            means, covars, "full", entropy, confidence
        )

        assert summary["regime_forecast"].shape == (20, 2)
        assert summary["current_regime"] in ("bear", "bull")

    def test_summary_early_detection_shape(self, predictor, synthetic_regime_data, two_regime_hmm_params):
        X, states = synthetic_regime_data
        means, covars, transmat = two_regime_hmm_params

        posteriors = np.zeros((len(X), 2))
        for t in range(len(X)):
            posteriors[t, states[t]] = 0.85
            posteriors[t, 1 - states[t]] = 0.15

        labels = {0: "bear", 1: "bull"}
        entropy = np.full(len(X), 0.2)
        confidence = np.full(len(X), 0.8)

        summary = predictor.generate_forecast_summary(
            X, states, posteriors, labels, transmat,
            means, covars, "full", entropy, confidence
        )

        assert summary["early_detection_score"].shape == (len(X),)
        assert np.all(summary["early_detection_score"] >= 0)
        assert np.all(summary["early_detection_score"] <= 1)


# ── Edge Cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_single_state_hmm(self, predictor):
        """Should handle degenerate case of 1 state."""
        means = np.array([[0.01, 0.01]])
        covars = np.array([[[0.001, 0.0], [0.0, 0.001]]])
        transmat = np.array([[1.0]])
        stationary = np.array([1.0])

        X = np.random.randn(30, 2) * 0.03
        result = predictor.run_bocd(X, means, covars, "full", transmat, stationary)
        assert result["changepoint_prob"].shape == (30,)

    def test_very_short_sequence(self, predictor, two_regime_hmm_params):
        means, covars, transmat = two_regime_hmm_params
        stationary = np.array([0.5, 0.5])

        X = np.random.randn(3, 2) * 0.01
        result = predictor.run_bocd(X, means, covars, "full", transmat, stationary)
        assert result["changepoint_prob"].shape == (3,)

    def test_default_config(self):
        """Should work with empty config using defaults."""
        pred = RegimePredictor({})
        assert pred.hazard_rate == 1 / 100
        assert pred.forecast_horizon == 20

    def test_three_state_forecast(self, predictor):
        transmat = np.array([
            [0.90, 0.05, 0.05],
            [0.05, 0.90, 0.05],
            [0.05, 0.05, 0.90],
        ])
        current_dist = np.array([0.0, 1.0, 0.0])

        forecast = predictor.forecast_regime_probabilities(transmat, current_dist)
        assert forecast.shape == (20, 3)
        np.testing.assert_allclose(forecast.sum(axis=1), 1.0, atol=1e-10)
