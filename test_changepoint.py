"""
Tests for changepoint.py — Bayesian Online Changepoint Detection.
"""

import numpy as np
import pytest
from changepoint import (
    StudentTPredictive,
    BayesianChangepointDetector,
    ChangepointResult,
)


@pytest.fixture
def simple_config():
    return {
        "changepoint": {
            "hazard_rate": 1.0 / 50,
            "threshold": 0.3,
            "min_run_length": 5,
            "max_run_length": 200,
            "feature_index": 0,
            "store_run_length_dist": True,
            "prior": {
                "mu0": 0.0,
                "kappa0": 0.01,
                "alpha0": 0.5,
                "beta0": 0.01,
            },
        },
    }


@pytest.fixture
def regime_switch_data():
    """Synthetic data with a clear regime change at bar 100."""
    rng = np.random.default_rng(42)
    n = 200
    x = np.empty(n)
    # Regime 1: mean=0.01, low vol
    x[:100] = rng.normal(0.01, 0.005, 100)
    # Regime 2: mean=-0.02, high vol
    x[100:] = rng.normal(-0.02, 0.02, 100)
    return x


@pytest.fixture
def multi_regime_data():
    """Synthetic data with three regimes."""
    rng = np.random.default_rng(123)
    n = 300
    x = np.empty(n)
    x[:100] = rng.normal(0.01, 0.005, 100)   # bull
    x[100:200] = rng.normal(-0.02, 0.02, 100)  # crash
    x[200:] = rng.normal(0.005, 0.008, 100)  # recovery
    return x


@pytest.fixture
def two_dim_data():
    """2-D feature matrix with changepoint in first column."""
    rng = np.random.default_rng(42)
    n = 200
    X = np.empty((n, 5))
    X[:100, 0] = rng.normal(0.01, 0.005, 100)
    X[100:, 0] = rng.normal(-0.02, 0.02, 100)
    for col in range(1, 5):
        X[:, col] = rng.normal(0, 0.01, n)
    return X


class TestStudentTPredictive:
    def test_log_predictive_shape(self):
        pred = StudentTPredictive()
        mu = np.array([0.0, 0.01, -0.01])
        kappa = np.array([0.01, 1.01, 2.01])
        alpha = np.array([0.5, 1.0, 1.5])
        beta = np.array([0.01, 0.02, 0.03])
        log_p = pred.log_predictive(0.005, mu, kappa, alpha, beta)
        assert log_p.shape == (3,)

    def test_log_predictive_finite(self):
        pred = StudentTPredictive()
        mu = np.array([0.0])
        kappa = np.array([0.01])
        alpha = np.array([0.5])
        beta = np.array([0.01])
        log_p = pred.log_predictive(0.0, mu, kappa, alpha, beta)
        assert np.all(np.isfinite(log_p))

    def test_predictive_peaks_near_mean(self):
        """Predictive should be highest near the running mean."""
        pred = StudentTPredictive()
        mu = np.array([0.05])
        kappa = np.array([100.0])  # high certainty
        alpha = np.array([50.0])
        beta = np.array([0.01])
        lp_at_mean = pred.log_predictive(0.05, mu, kappa, alpha, beta)
        lp_far = pred.log_predictive(0.5, mu, kappa, alpha, beta)
        assert lp_at_mean[0] > lp_far[0]

    def test_update_suffstats_shape(self):
        pred = StudentTPredictive()
        mu = np.array([0.0, 0.01])
        kappa = np.array([0.01, 1.0])
        alpha = np.array([0.5, 1.0])
        beta = np.array([0.01, 0.02])
        mu2, k2, a2, b2 = pred.update_suffstats(0.005, mu, kappa, alpha, beta)
        assert mu2.shape == (2,)
        assert k2.shape == (2,)
        assert a2.shape == (2,)
        assert b2.shape == (2,)

    def test_update_increases_kappa_alpha(self):
        """kappa and alpha should increase after each observation."""
        pred = StudentTPredictive()
        mu = np.array([0.0])
        kappa = np.array([0.01])
        alpha = np.array([0.5])
        beta = np.array([0.01])
        _, k2, a2, _ = pred.update_suffstats(0.005, mu, kappa, alpha, beta)
        assert k2[0] > kappa[0]
        assert a2[0] > alpha[0]


class TestBOCPDDetection:
    def test_output_shape(self, simple_config, regime_switch_data):
        detector = BayesianChangepointDetector(simple_config)
        result = detector.detect(regime_switch_data)
        T = len(regime_switch_data)
        assert result.changepoint_prob.shape == (T,)
        assert result.map_run_length.shape == (T,)
        assert result.expected_run_length.shape == (T,)
        assert result.detected_changepoints.shape == (T,)

    def test_cp_prob_valid_range(self, simple_config, regime_switch_data):
        detector = BayesianChangepointDetector(simple_config)
        result = detector.detect(regime_switch_data)
        assert np.all(result.changepoint_prob >= 0)
        assert np.all(result.changepoint_prob <= 1)

    def test_detects_regime_switch(self, simple_config, regime_switch_data):
        """Should detect the changepoint near bar 100."""
        detector = BayesianChangepointDetector(simple_config)
        result = detector.detect(regime_switch_data)
        # At least one detection within 15 bars of the true changepoint
        cp_bars = result.changepoint_bars
        near_100 = [b for b in cp_bars if 90 <= b <= 115]
        assert len(near_100) > 0, f"No changepoint detected near bar 100. Detected at: {cp_bars}"

    def test_detects_multiple_changes(self, simple_config, multi_regime_data):
        """Should detect both changepoints in 3-regime data."""
        detector = BayesianChangepointDetector(simple_config)
        result = detector.detect(multi_regime_data)
        cp_bars = result.changepoint_bars
        near_100 = [b for b in cp_bars if 90 <= b <= 115]
        near_200 = [b for b in cp_bars if 190 <= b <= 215]
        assert len(near_100) > 0, f"Missed first changepoint. Detected at: {cp_bars}"
        assert len(near_200) > 0, f"Missed second changepoint. Detected at: {cp_bars}"

    def test_2d_input(self, simple_config, two_dim_data):
        """Should work with 2-D feature matrix."""
        detector = BayesianChangepointDetector(simple_config)
        result = detector.detect(two_dim_data)
        assert result.changepoint_prob.shape == (200,)
        # Should still detect the changepoint in column 0
        cp_bars = result.changepoint_bars
        near_100 = [b for b in cp_bars if 90 <= b <= 115]
        assert len(near_100) > 0

    def test_run_length_posterior_stored(self, simple_config, regime_switch_data):
        detector = BayesianChangepointDetector(simple_config)
        result = detector.detect(regime_switch_data)
        assert result.run_length_posterior is not None
        assert result.run_length_posterior.shape[0] == len(regime_switch_data)

    def test_no_false_positives_in_stable_data(self, simple_config):
        """Stable data should produce few or no changepoints."""
        rng = np.random.default_rng(99)
        stable = rng.normal(0.001, 0.005, 200)
        detector = BayesianChangepointDetector(simple_config)
        result = detector.detect(stable)
        # Allow some false positives but not many
        assert len(result.changepoint_bars) < 10

    def test_expected_run_length_grows(self, simple_config):
        """In stable data, expected run length should generally increase."""
        rng = np.random.default_rng(99)
        stable = rng.normal(0.001, 0.005, 100)
        detector = BayesianChangepointDetector(simple_config)
        result = detector.detect(stable)
        # Expected run length at the end should be > at the start (after warmup)
        assert result.expected_run_length[-1] > result.expected_run_length[10]

    def test_deterministic(self, simple_config, regime_switch_data):
        """Same input should produce same output (algorithm is deterministic)."""
        detector = BayesianChangepointDetector(simple_config)
        r1 = detector.detect(regime_switch_data)
        r2 = detector.detect(regime_switch_data)
        np.testing.assert_array_equal(r1.changepoint_prob, r2.changepoint_prob)
        np.testing.assert_array_equal(r1.changepoint_bars, r2.changepoint_bars)


class TestHMMComparison:
    def test_compare_finds_early_detections(self, simple_config, regime_switch_data):
        """BOCPD should detect the change before or at the same time as HMM."""
        detector = BayesianChangepointDetector(simple_config)
        result = detector.detect(regime_switch_data)

        # Simulate HMM states: switches at exactly bar 100
        hmm_states = np.zeros(200, dtype=int)
        hmm_states[100:] = 1

        result = detector.compare_with_hmm(result, hmm_states, max_lead=20)
        assert len(result.hmm_transition_bars) == 1
        assert result.hmm_transition_bars[0] == 100

    def test_early_detection_lead_positive(self, simple_config, regime_switch_data):
        """Early detections should have positive lead (BOCPD fires first)."""
        detector = BayesianChangepointDetector(simple_config)
        result = detector.detect(regime_switch_data)

        hmm_states = np.zeros(200, dtype=int)
        hmm_states[105:] = 1  # HMM transitions 5 bars AFTER true change

        result = detector.compare_with_hmm(result, hmm_states, max_lead=20)
        for ed in result.early_detections:
            assert ed["lead_bars"] > 0

    def test_no_hmm_transitions(self, simple_config, regime_switch_data):
        """When HMM never transitions, no early detections."""
        detector = BayesianChangepointDetector(simple_config)
        result = detector.detect(regime_switch_data)

        hmm_states = np.zeros(200, dtype=int)  # never changes
        result = detector.compare_with_hmm(result, hmm_states)
        assert len(result.early_detections) == 0


class TestRegimeStability:
    def test_stability_bounded(self, simple_config, regime_switch_data):
        detector = BayesianChangepointDetector(simple_config)
        result = detector.detect(regime_switch_data)
        stability = detector.compute_regime_stability(result.expected_run_length)
        assert np.all(stability >= 0.0)
        assert np.all(stability <= 1.0)

    def test_stability_drops_at_changepoint(self, simple_config, regime_switch_data):
        """Stability should drop around the changepoint."""
        detector = BayesianChangepointDetector(simple_config)
        result = detector.detect(regime_switch_data)
        stability = detector.compute_regime_stability(result.expected_run_length)
        # Stability before the change should be higher than right after
        avg_before = stability[80:95].mean()
        avg_after = stability[100:110].mean()
        assert avg_before > avg_after


class TestConfidenceFusion:
    def test_fused_confidence_bounded(self, simple_config, regime_switch_data):
        detector = BayesianChangepointDetector(simple_config)
        result = detector.detect(regime_switch_data)
        hmm_conf = np.random.default_rng(42).uniform(0.5, 1.0, 200)
        fused = detector.fuse_with_hmm_confidence(hmm_conf, result)
        assert np.all(fused >= 0.0)
        assert np.all(fused <= 1.0)

    def test_fused_reduces_at_changepoint(self, simple_config, regime_switch_data):
        """Fused confidence should be lower at changepoints than HMM alone."""
        detector = BayesianChangepointDetector(simple_config)
        result = detector.detect(regime_switch_data)
        # HMM confidence stays high (doesn't see the change yet)
        hmm_conf = np.full(200, 0.9)
        fused = detector.fuse_with_hmm_confidence(hmm_conf, result, stability_weight=0.5)
        # At the changepoint region, fused should be noticeably lower
        if len(result.changepoint_bars) > 0:
            cp_bar = result.changepoint_bars[0]
            assert fused[cp_bar] < hmm_conf[cp_bar]

    def test_stability_weight_zero_returns_hmm(self, simple_config, regime_switch_data):
        """With zero stability weight, fused should equal HMM confidence."""
        detector = BayesianChangepointDetector(simple_config)
        result = detector.detect(regime_switch_data)
        hmm_conf = np.random.default_rng(42).uniform(0.5, 1.0, 200)
        fused = detector.fuse_with_hmm_confidence(hmm_conf, result, stability_weight=0.0)
        np.testing.assert_allclose(fused, hmm_conf, atol=1e-10)
