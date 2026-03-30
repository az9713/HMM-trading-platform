"""
test_regime_predictor.py — Tests for forward-looking regime prediction engine.
"""

import numpy as np
import pandas as pd
import pytest

from regime_predictor import (
    BayesianChangepoint,
    RegimePredictor,
    RegimeForecast,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_transmat():
    """2-state transition matrix with strong persistence."""
    return np.array([
        [0.95, 0.05],
        [0.10, 0.90],
    ])


@pytest.fixture
def three_state_transmat():
    """3-state transition matrix."""
    return np.array([
        [0.90, 0.07, 0.03],
        [0.05, 0.85, 0.10],
        [0.02, 0.08, 0.90],
    ])


@pytest.fixture
def labels_2():
    return {0: "bear", 1: "bull"}


@pytest.fixture
def labels_3():
    return {0: "bear", 1: "neutral", 2: "bull"}


@pytest.fixture
def predictor():
    return RegimePredictor(
        horizons=(1, 5, 10, 20),
        bocpd_hazard=1 / 50,
        momentum_window=5,
    )


@pytest.fixture
def synthetic_data():
    """Generate synthetic regime data for integration tests."""
    np.random.seed(42)
    T = 200
    # Two regimes: 0 (bear) and 1 (bull), switching every ~50 bars
    states = np.zeros(T, dtype=int)
    for t in range(1, T):
        if np.random.rand() < 0.02:
            states[t] = 1 - states[t - 1]
        else:
            states[t] = states[t - 1]

    # Posteriors: peaked at current state
    posteriors = np.zeros((T, 2))
    for t in range(T):
        posteriors[t, states[t]] = 0.85
        posteriors[t, 1 - states[t]] = 0.15

    # Entropy
    eps = 1e-12
    entropy = -np.sum(posteriors * np.log2(posteriors + eps), axis=1)
    confidence = 1.0 - entropy / np.log2(2)

    # Returns
    log_returns = np.where(states == 1, 0.001, -0.001) + np.random.normal(0, 0.01, T)
    volatility = pd.Series(log_returns).rolling(10, min_periods=1).std().values
    volume_change = np.random.normal(0, 0.5, T)

    return {
        "states": states,
        "posteriors": posteriors,
        "entropy": entropy,
        "confidence": confidence,
        "log_returns": log_returns,
        "volatility": volatility,
        "volume_change": volume_change,
    }


# ── Chapman-Kolmogorov Tests ─────────────────────────────────────────────────

class TestChapmanKolmogorov:
    def test_one_step_matches_transmat(self, predictor, simple_transmat, labels_2):
        """1-step forecast should equal posterior @ transmat."""
        posterior = np.array([1.0, 0.0])  # certain bear
        result = predictor.chapman_kolmogorov_forecast(
            simple_transmat, posterior, labels_2, horizons=(1,)
        )
        assert "bear" in result[1]
        assert "bull" in result[1]
        assert abs(result[1]["bear"] - 0.95) < 1e-6
        assert abs(result[1]["bull"] - 0.05) < 1e-6

    def test_probabilities_sum_to_one(self, predictor, three_state_transmat, labels_3):
        posterior = np.array([0.5, 0.3, 0.2])
        result = predictor.chapman_kolmogorov_forecast(
            three_state_transmat, posterior, labels_3
        )
        for k, probs in result.items():
            total = sum(probs.values())
            assert abs(total - 1.0) < 1e-6, f"Horizon {k}: probs sum to {total}"

    def test_long_horizon_converges_to_stationary(self, predictor, simple_transmat, labels_2):
        """Very long horizon should converge to stationary distribution."""
        posterior = np.array([1.0, 0.0])
        result = predictor.chapman_kolmogorov_forecast(
            simple_transmat, posterior, labels_2, horizons=(500,)
        )
        # Stationary: pi_bear = 0.10/(0.05+0.10) = 2/3, pi_bull = 1/3
        assert abs(result[500]["bear"] - 2 / 3) < 0.01
        assert abs(result[500]["bull"] - 1 / 3) < 0.01

    def test_uniform_posterior(self, predictor, simple_transmat, labels_2):
        """Uniform posterior should still produce valid forecasts."""
        posterior = np.array([0.5, 0.5])
        result = predictor.chapman_kolmogorov_forecast(
            simple_transmat, posterior, labels_2
        )
        for k, probs in result.items():
            assert all(0 <= p <= 1 for p in probs.values())

    def test_multiple_horizons_returned(self, predictor, simple_transmat, labels_2):
        posterior = np.array([0.8, 0.2])
        result = predictor.chapman_kolmogorov_forecast(
            simple_transmat, posterior, labels_2
        )
        assert set(result.keys()) == {1, 5, 10, 20}


# ── Transition Probability Tests ─────────────────────────────────────────────

class TestTransitionProbability:
    def test_one_step_transition(self, predictor, simple_transmat):
        posterior = np.array([1.0, 0.0])  # certain bear
        result = predictor.transition_probability(simple_transmat, posterior, 0)
        # P(leave bear in 1 step) = P(go to bull) = 0.05
        assert abs(result[1] - 0.05) < 1e-6

    def test_transition_prob_increases_with_horizon(self, predictor, simple_transmat):
        posterior = np.array([1.0, 0.0])
        result = predictor.transition_probability(simple_transmat, posterior, 0)
        # Transition probability should increase with horizon
        for i in range(len(predictor.horizons) - 1):
            h1 = predictor.horizons[i]
            h2 = predictor.horizons[i + 1]
            assert result[h2] >= result[h1] - 1e-10

    def test_transition_prob_bounded(self, predictor, simple_transmat):
        posterior = np.array([0.6, 0.4])
        result = predictor.transition_probability(simple_transmat, posterior, 0)
        for k, prob in result.items():
            assert 0 <= prob <= 1


# ── BOCPD Tests ──────────────────────────────────────────────────────────────

class TestBayesianChangepoint:
    def test_constant_series_no_changepoints(self):
        """Constant data should have low changepoint probability."""
        bocpd = BayesianChangepoint(hazard_rate=1 / 50)
        data = np.ones(100) * 5.0
        cp_probs = bocpd.run(data)
        assert len(cp_probs) == 100
        # After initial warmup, changepoint prob should be low
        assert np.mean(cp_probs[20:]) < 0.3

    def test_clear_changepoint_detected(self):
        """Abrupt mean shift should produce elevated changepoint probability."""
        bocpd = BayesianChangepoint(hazard_rate=1 / 50)
        data = np.concatenate([
            np.random.normal(0, 0.1, 80),
            np.random.normal(5, 0.1, 80),
        ])
        np.random.seed(42)
        cp_probs = bocpd.run(data)
        # Changepoint prob should spike near bar 80
        window = cp_probs[75:90]
        assert np.max(window) > np.mean(cp_probs[:70])

    def test_empty_data(self):
        bocpd = BayesianChangepoint()
        result = bocpd.run(np.array([]))
        assert len(result) == 0

    def test_single_data_point(self):
        bocpd = BayesianChangepoint()
        result = bocpd.run(np.array([1.0]))
        assert len(result) == 1

    def test_output_probabilities_valid(self):
        """All outputs should be valid probabilities."""
        bocpd = BayesianChangepoint(hazard_rate=1 / 30)
        np.random.seed(123)
        data = np.random.normal(0, 1, 50)
        cp_probs = bocpd.run(data)
        assert all(0 <= p <= 1 for p in cp_probs)


# ── Momentum Score Tests ─────────────────────────────────────────────────────

class TestMomentumScore:
    def test_output_shape(self, predictor):
        T = 100
        entropy = np.random.rand(T)
        vol = np.random.rand(T) * 0.02
        vol_chg = np.random.randn(T)
        scores = predictor.compute_momentum_score(entropy, vol, vol_chg)
        assert len(scores) == T

    def test_output_bounded(self, predictor):
        T = 100
        np.random.seed(42)
        entropy = np.random.rand(T)
        vol = np.random.rand(T) * 0.02
        vol_chg = np.random.randn(T)
        scores = predictor.compute_momentum_score(entropy, vol, vol_chg)
        assert all(0 <= s <= 1 for s in scores)

    def test_short_series(self, predictor):
        """Very short series should not crash."""
        scores = predictor.compute_momentum_score(
            np.array([0.5, 0.6]),
            np.array([0.01, 0.02]),
            np.array([0.1, -0.1]),
        )
        assert len(scores) == 2

    def test_constant_features_low_score(self, predictor):
        """Constant features should produce zero or near-zero momentum."""
        T = 50
        scores = predictor.compute_momentum_score(
            np.ones(T) * 0.5,
            np.ones(T) * 0.01,
            np.zeros(T),
        )
        assert np.mean(scores) < 0.6


# ── Regime Stress Tests ──────────────────────────────────────────────────────

class TestRegimeStress:
    def test_stress_bounded(self, predictor):
        stress = predictor.compute_regime_stress(
            {5: 0.8}, changepoint_prob=0.5, momentum_score=0.7
        )
        assert 0 <= stress <= 1

    def test_low_inputs_low_stress(self, predictor):
        stress = predictor.compute_regime_stress(
            {5: 0.05}, changepoint_prob=0.01, momentum_score=0.1
        )
        assert stress < 0.3

    def test_high_inputs_high_stress(self, predictor):
        stress = predictor.compute_regime_stress(
            {5: 0.9}, changepoint_prob=0.8, momentum_score=0.9
        )
        assert stress > 0.6


# ── Alert Level Tests ────────────────────────────────────────────────────────

class TestAlertLevel:
    def test_low(self, predictor):
        level, color = predictor.stress_alert_level(0.1)
        assert level == "LOW"

    def test_moderate(self, predictor):
        level, _ = predictor.stress_alert_level(0.4)
        assert level == "MODERATE"

    def test_elevated(self, predictor):
        level, _ = predictor.stress_alert_level(0.6)
        assert level == "ELEVATED"

    def test_critical(self, predictor):
        level, _ = predictor.stress_alert_level(0.8)
        assert level == "CRITICAL"


# ── Integration: Full Predict Pipeline ───────────────────────────────────────

class TestFullPrediction:
    def test_predict_returns_forecasts(self, predictor, simple_transmat, labels_2, synthetic_data):
        forecasts = predictor.predict(
            transmat=simple_transmat,
            posteriors=synthetic_data["posteriors"],
            states=synthetic_data["states"],
            labels=labels_2,
            entropy=synthetic_data["entropy"],
            log_returns=synthetic_data["log_returns"],
            volatility=synthetic_data["volatility"],
            volume_change=synthetic_data["volume_change"],
        )
        assert len(forecasts) == len(synthetic_data["states"])
        assert all(isinstance(f, RegimeForecast) for f in forecasts)

    def test_forecast_fields_populated(self, predictor, simple_transmat, labels_2, synthetic_data):
        forecasts = predictor.predict(
            transmat=simple_transmat,
            posteriors=synthetic_data["posteriors"],
            states=synthetic_data["states"],
            labels=labels_2,
            entropy=synthetic_data["entropy"],
            log_returns=synthetic_data["log_returns"],
            volatility=synthetic_data["volatility"],
            volume_change=synthetic_data["volume_change"],
        )
        f = forecasts[50]
        assert f.current_regime in ("bear", "bull")
        assert set(f.forecast_probs.keys()) == {1, 5, 10, 20}
        assert set(f.predicted_regime.keys()) == {1, 5, 10, 20}
        assert 0 <= f.changepoint_prob <= 1
        assert 0 <= f.momentum_score <= 1
        assert 0 <= f.regime_stress <= 1

    def test_forecast_summary_dataframe(self, predictor, simple_transmat, labels_2, synthetic_data):
        forecasts = predictor.predict(
            transmat=simple_transmat,
            posteriors=synthetic_data["posteriors"],
            states=synthetic_data["states"],
            labels=labels_2,
            entropy=synthetic_data["entropy"],
            log_returns=synthetic_data["log_returns"],
            volatility=synthetic_data["volatility"],
            volume_change=synthetic_data["volume_change"],
        )
        summary = predictor.forecast_summary(forecasts)
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == len(forecasts)
        assert "regime_stress" in summary.columns
        assert "changepoint_prob" in summary.columns
        assert "transition_prob_5" in summary.columns

    def test_calibration_backtest(self, predictor, simple_transmat, labels_2, synthetic_data):
        forecasts = predictor.predict(
            transmat=simple_transmat,
            posteriors=synthetic_data["posteriors"],
            states=synthetic_data["states"],
            labels=labels_2,
            entropy=synthetic_data["entropy"],
            log_returns=synthetic_data["log_returns"],
            volatility=synthetic_data["volatility"],
            volume_change=synthetic_data["volume_change"],
        )
        cal = predictor.calibration_backtest(
            forecasts, synthetic_data["states"], labels_2
        )
        assert isinstance(cal, pd.DataFrame)
        assert len(cal) == len(predictor.horizons)
        assert "accuracy" in cal.columns
        # Short-horizon predictions should be more accurate than random
        assert cal.iloc[0]["accuracy"] > 0.3
