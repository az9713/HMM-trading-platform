"""
Tests for regime_predictor.py — Predictive regime forecasting engine.

Tests cover: Chapman-Kolmogorov forecasting, regime shift probabilities,
expected duration, next-regime ranking, macro stress index, Bayesian fusion,
forecast summary generation, and bars-in-regime counting.
"""

import numpy as np
import pandas as pd
import pytest

from regime_predictor import (
    RegimePredictor,
    MacroFeatureCollector,
    count_bars_in_current_regime,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_config(max_horizon=20, macro_weight=0.3, macro_n_states=3):
    return {
        "predictor": {
            "max_horizon": max_horizon,
            "macro_weight": macro_weight,
            "macro_n_states": macro_n_states,
            "macro_n_restarts": 5,
            "stress_thresholds": {"low": -0.5, "high": 0.5},
        }
    }


def make_simple_transmat():
    """Simple 3-state transition matrix with strong persistence."""
    return np.array([
        [0.90, 0.07, 0.03],  # bear: stays bear 90%
        [0.05, 0.85, 0.10],  # neutral: stays neutral 85%
        [0.02, 0.08, 0.90],  # bull: stays bull 90%
    ])


def make_labels():
    return {0: "bear", 1: "neutral", 2: "bull"}


def make_macro_features(n=200, seed=42):
    """Synthetic macro feature DataFrame."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "vix_level": 15 + rng.normal(0, 3, n).cumsum() * 0.1,
        "vix_change": rng.normal(0, 0.05, n),
        "yield_level": 4.0 + rng.normal(0, 0.1, n).cumsum() * 0.01,
        "yield_change": rng.normal(0, 0.02, n),
        "credit_spread": rng.normal(0, 0.01, n),
        "usd_momentum": rng.normal(0, 0.005, n),
        "gold_momentum": rng.normal(0, 0.01, n),
    }, index=dates)


# ---------------------------------------------------------------------------
# Tests: RegimePredictor initialization
# ---------------------------------------------------------------------------

class TestRegimePredictorInit:
    def test_default_config(self):
        rp = RegimePredictor({})
        assert rp.max_horizon == 20
        assert rp.macro_weight == 0.3
        assert rp.macro_n_states == 3

    def test_custom_config(self):
        rp = RegimePredictor(make_config(max_horizon=50, macro_weight=0.5))
        assert rp.max_horizon == 50
        assert rp.macro_weight == 0.5


# ---------------------------------------------------------------------------
# Tests: Chapman-Kolmogorov forecasting
# ---------------------------------------------------------------------------

class TestForecastRegimeProbs:
    def test_1step_matches_transmat(self):
        """1-step forecast should equal posterior @ A."""
        rp = RegimePredictor(make_config())
        A = make_simple_transmat()
        post = np.array([0.1, 0.2, 0.7])

        forecasts = rp.forecast_regime_probs(A, post, horizons=[1])
        expected = post @ A
        np.testing.assert_allclose(forecasts[1], expected, atol=1e-10)

    def test_probabilities_sum_to_one(self):
        """All forecast probability vectors must sum to 1."""
        rp = RegimePredictor(make_config())
        A = make_simple_transmat()
        post = np.array([0.0, 0.0, 1.0])

        forecasts = rp.forecast_regime_probs(A, post, horizons=[1, 5, 10, 20])
        for h, probs in forecasts.items():
            assert abs(probs.sum() - 1.0) < 1e-10, f"Horizon {h} doesn't sum to 1"

    def test_long_horizon_converges_to_stationary(self):
        """Very long horizon should converge to stationary distribution."""
        rp = RegimePredictor(make_config(max_horizon=1000))
        A = make_simple_transmat()

        # Compute stationary distribution
        eigenvalues, eigenvectors = np.linalg.eig(A.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary /= stationary.sum()

        # From any starting point, long horizon should converge
        for start in [np.array([1, 0, 0]), np.array([0, 0, 1]), np.array([0.33, 0.34, 0.33])]:
            forecasts = rp.forecast_regime_probs(A, start, horizons=[1000])
            np.testing.assert_allclose(forecasts[1000], stationary, atol=1e-4)

    def test_identity_transmat_preserves_posterior(self):
        """Identity transition matrix should preserve posteriors at all horizons."""
        rp = RegimePredictor(make_config())
        A = np.eye(3)
        post = np.array([0.2, 0.3, 0.5])

        forecasts = rp.forecast_regime_probs(A, post, horizons=[1, 5, 10])
        for h, probs in forecasts.items():
            np.testing.assert_allclose(probs, post, atol=1e-10)

    def test_horizons_capped_at_max(self):
        """Horizons beyond max_horizon should be excluded."""
        rp = RegimePredictor(make_config(max_horizon=10))
        A = make_simple_transmat()
        post = np.array([0.1, 0.2, 0.7])

        forecasts = rp.forecast_regime_probs(A, post, horizons=[1, 5, 10, 20, 50])
        assert 20 not in forecasts
        assert 50 not in forecasts
        assert 10 in forecasts


# ---------------------------------------------------------------------------
# Tests: Regime shift probability
# ---------------------------------------------------------------------------

class TestRegimeShiftProbability:
    def test_shift_prob_increases_with_horizon(self):
        """Shift probability should generally increase with horizon."""
        rp = RegimePredictor(make_config())
        A = make_simple_transmat()

        shifts = rp.regime_shift_probability(A, current_state=0, horizons=[1, 5, 10, 20])
        assert shifts[1] < shifts[5]
        assert shifts[5] < shifts[10]

    def test_1step_shift_equals_exit_prob(self):
        """1-step shift prob should equal 1 - a_ii."""
        rp = RegimePredictor(make_config())
        A = make_simple_transmat()

        for state in range(3):
            shifts = rp.regime_shift_probability(A, state, horizons=[1])
            expected = 1.0 - A[state, state]
            assert abs(shifts[1] - expected) < 1e-10

    def test_shift_prob_between_0_and_1(self):
        """Shift probabilities must be in [0, 1]."""
        rp = RegimePredictor(make_config())
        A = make_simple_transmat()

        for state in range(3):
            shifts = rp.regime_shift_probability(A, state, horizons=[1, 5, 10, 20])
            for h, p in shifts.items():
                assert 0.0 <= p <= 1.0, f"State {state}, horizon {h}: {p}"

    def test_absorbing_state_zero_shift(self):
        """Absorbing state (a_ii=1) should have 0 shift probability."""
        rp = RegimePredictor(make_config())
        A = np.array([
            [1.0, 0.0, 0.0],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ])
        shifts = rp.regime_shift_probability(A, current_state=0, horizons=[1, 5, 10])
        for h, p in shifts.items():
            assert abs(p) < 1e-10


# ---------------------------------------------------------------------------
# Tests: Expected regime duration
# ---------------------------------------------------------------------------

class TestExpectedRegimeDuration:
    def test_basic_duration(self):
        """E[duration] = 1/(1-a_ii). For a_ii=0.9, expect 10 bars."""
        rp = RegimePredictor(make_config())
        A = make_simple_transmat()

        dur = rp.expected_regime_duration(A, current_state=0, bars_in_regime=5)
        assert abs(dur["expected_total"] - 10.0) < 1e-10  # 1/(1-0.9) = 10
        assert dur["bars_elapsed"] == 5
        assert dur["exit_prob_per_bar"] == pytest.approx(0.1)

    def test_near_absorbing_state(self):
        """Very persistent state should have very long duration."""
        rp = RegimePredictor(make_config())
        A = np.array([
            [0.999, 0.001, 0.0],
            [0.05, 0.9, 0.05],
            [0.0, 0.1, 0.9],
        ])
        dur = rp.expected_regime_duration(A, current_state=0)
        assert dur["expected_total"] == pytest.approx(1000.0, rel=0.01)

    def test_median_less_than_mean(self):
        """For geometric distribution, median < mean (right-skewed)."""
        rp = RegimePredictor(make_config())
        A = make_simple_transmat()

        dur = rp.expected_regime_duration(A, current_state=0)
        assert dur["median_remaining"] < dur["expected_remaining"]

    def test_p90_greater_than_median(self):
        """90th percentile should exceed median."""
        rp = RegimePredictor(make_config())
        A = make_simple_transmat()

        dur = rp.expected_regime_duration(A, current_state=0)
        assert dur["p90_remaining"] > dur["median_remaining"]


# ---------------------------------------------------------------------------
# Tests: Most likely next regime
# ---------------------------------------------------------------------------

class TestMostLikelyNextRegime:
    def test_excludes_current_state(self):
        """Current state should not appear in next-regime list."""
        rp = RegimePredictor(make_config())
        A = make_simple_transmat()
        labels = make_labels()

        results = rp.most_likely_next_regime(A, current_state=0, labels=labels)
        state_ids = [r["state"] for r in results]
        assert 0 not in state_ids

    def test_sorted_by_probability(self):
        """Results should be sorted by descending probability."""
        rp = RegimePredictor(make_config())
        A = make_simple_transmat()
        labels = make_labels()

        results = rp.most_likely_next_regime(A, current_state=0, labels=labels)
        probs = [r["probability"] for r in results]
        assert probs == sorted(probs, reverse=True)

    def test_probabilities_sum_to_one(self):
        """Exit probabilities should sum to ~1 (renormalized)."""
        rp = RegimePredictor(make_config())
        A = make_simple_transmat()
        labels = make_labels()

        results = rp.most_likely_next_regime(A, current_state=0, labels=labels)
        total = sum(r["probability"] for r in results)
        assert abs(total - 1.0) < 1e-6

    def test_labels_correct(self):
        """Labels should match the labels dict."""
        rp = RegimePredictor(make_config())
        A = make_simple_transmat()
        labels = make_labels()

        results = rp.most_likely_next_regime(A, current_state=2, labels=labels)
        for r in results:
            assert r["label"] == labels[r["state"]]


# ---------------------------------------------------------------------------
# Tests: Macro stress index
# ---------------------------------------------------------------------------

class TestMacroStressIndex:
    def test_output_length(self):
        """Stress index should have same length as input."""
        collector = MacroFeatureCollector()
        features = make_macro_features()
        stress = collector.compute_macro_stress_index(features)
        assert len(stress) == len(features)

    def test_mean_near_zero(self):
        """Z-score based index should have mean near 0."""
        collector = MacroFeatureCollector()
        features = make_macro_features()
        stress = collector.compute_macro_stress_index(features)
        assert abs(stress.mean()) < 0.5

    def test_empty_features(self):
        """Should handle features with no matching columns gracefully."""
        collector = MacroFeatureCollector()
        features = pd.DataFrame({"unrelated_col": [1, 2, 3]})
        stress = collector.compute_macro_stress_index(features)
        assert (stress == 0.0).all()


# ---------------------------------------------------------------------------
# Tests: Bayesian fusion
# ---------------------------------------------------------------------------

class TestBayesianFusion:
    def test_output_sums_to_one(self):
        """Fused probabilities must sum to 1."""
        rp = RegimePredictor(make_config(macro_weight=0.3))
        asset_forecast = np.array([0.2, 0.3, 0.5])
        macro_post = np.array([0.7, 0.2, 0.1])  # mostly risk_on
        asset_labels = make_labels()
        macro_labels = {0: "risk_on", 1: "neutral", 2: "risk_off"}

        fused = rp.bayesian_fusion(asset_forecast, macro_post, asset_labels, macro_labels)
        assert abs(fused.sum() - 1.0) < 1e-10

    def test_risk_on_boosts_bull(self):
        """Risk-on macro environment should boost bull probability."""
        rp = RegimePredictor(make_config(macro_weight=0.5))
        asset_forecast = np.array([0.3, 0.3, 0.4])
        asset_labels = make_labels()
        macro_labels = {0: "risk_on", 1: "neutral", 2: "risk_off"}

        # Strong risk_on
        risk_on_post = np.array([0.9, 0.05, 0.05])
        fused_on = rp.bayesian_fusion(asset_forecast, risk_on_post, asset_labels, macro_labels)

        # Strong risk_off
        risk_off_post = np.array([0.05, 0.05, 0.9])
        fused_off = rp.bayesian_fusion(asset_forecast, risk_off_post, asset_labels, macro_labels)

        # Bull prob should be higher in risk-on
        assert fused_on[2] > fused_off[2]
        # Bear prob should be higher in risk-off
        assert fused_off[0] > fused_on[0]

    def test_zero_macro_weight_no_change(self):
        """With macro_weight=0, fusion should return original forecast."""
        rp = RegimePredictor(make_config(macro_weight=0.0))
        asset_forecast = np.array([0.2, 0.3, 0.5])
        macro_post = np.array([0.0, 0.0, 1.0])  # extreme risk_off
        asset_labels = make_labels()
        macro_labels = {0: "risk_on", 1: "neutral", 2: "risk_off"}

        fused = rp.bayesian_fusion(asset_forecast, macro_post, asset_labels, macro_labels)
        np.testing.assert_allclose(fused, asset_forecast, atol=1e-10)

    def test_neutral_macro_minimal_change(self):
        """Neutral macro regime should cause minimal adjustment."""
        rp = RegimePredictor(make_config(macro_weight=0.3))
        asset_forecast = np.array([0.2, 0.3, 0.5])
        macro_post = np.array([0.0, 1.0, 0.0])  # purely neutral
        asset_labels = make_labels()
        macro_labels = {0: "risk_on", 1: "neutral", 2: "risk_off"}

        fused = rp.bayesian_fusion(asset_forecast, macro_post, asset_labels, macro_labels)
        # Should be very close to original
        np.testing.assert_allclose(fused, asset_forecast, atol=1e-8)


# ---------------------------------------------------------------------------
# Tests: Forecast summary
# ---------------------------------------------------------------------------

class TestGenerateForecastSummary:
    def test_summary_structure(self):
        """Summary should have all expected keys."""
        rp = RegimePredictor(make_config())
        A = make_simple_transmat()
        post = np.array([0.1, 0.2, 0.7])
        labels = make_labels()

        summary = rp.generate_forecast_summary(A, post, current_state=2, labels=labels)

        assert "current" in summary
        assert "forecasts" in summary
        assert "shift_probs" in summary
        assert "duration" in summary
        assert "next_regimes" in summary
        assert "macro_adjusted" in summary

    def test_summary_without_macro(self):
        """Without macro data, macro_adjusted should be False."""
        rp = RegimePredictor(make_config())
        A = make_simple_transmat()
        post = np.array([0.1, 0.2, 0.7])
        labels = make_labels()

        summary = rp.generate_forecast_summary(A, post, current_state=2, labels=labels)
        assert summary["macro_adjusted"] is False

    def test_summary_with_macro(self):
        """With macro data, macro_adjusted should be True."""
        rp = RegimePredictor(make_config())
        A = make_simple_transmat()
        post = np.array([0.1, 0.2, 0.7])
        labels = make_labels()
        macro_post = np.array([0.6, 0.3, 0.1])
        macro_labels = {0: "risk_on", 1: "neutral", 2: "risk_off"}

        summary = rp.generate_forecast_summary(
            A, post, current_state=2, labels=labels,
            macro_posteriors=macro_post, macro_labels=macro_labels,
        )
        assert summary["macro_adjusted"] is True

    def test_current_info(self):
        """Current regime info should be correct."""
        rp = RegimePredictor(make_config())
        A = make_simple_transmat()
        post = np.array([0.1, 0.2, 0.7])
        labels = make_labels()

        summary = rp.generate_forecast_summary(
            A, post, current_state=2, labels=labels, bars_in_regime=15
        )
        assert summary["current"]["label"] == "bull"
        assert summary["current"]["bars_in_regime"] == 15
        assert summary["current"]["confidence"] == pytest.approx(0.7)

    def test_forecast_table_horizons(self):
        """Forecast table should have entries for standard horizons."""
        rp = RegimePredictor(make_config(max_horizon=20))
        A = make_simple_transmat()
        post = np.array([0.1, 0.2, 0.7])
        labels = make_labels()

        summary = rp.generate_forecast_summary(A, post, current_state=2, labels=labels)
        horizons_in_table = [f["horizon"] for f in summary["forecasts"]]
        assert 1 in horizons_in_table
        assert 5 in horizons_in_table
        assert 20 in horizons_in_table


# ---------------------------------------------------------------------------
# Tests: count_bars_in_current_regime
# ---------------------------------------------------------------------------

class TestCountBarsInCurrentRegime:
    def test_all_same(self):
        states = np.array([1, 1, 1, 1, 1])
        assert count_bars_in_current_regime(states) == 5

    def test_recent_change(self):
        states = np.array([0, 0, 0, 1, 1])
        assert count_bars_in_current_regime(states) == 2

    def test_single_bar(self):
        states = np.array([0, 1, 0, 1, 2])
        assert count_bars_in_current_regime(states) == 1

    def test_empty(self):
        assert count_bars_in_current_regime(np.array([])) == 0


# ---------------------------------------------------------------------------
# Tests: Macro HMM fitting (integration-style, uses synthetic data)
# ---------------------------------------------------------------------------

class TestMacroHMMFit:
    def test_fit_produces_valid_model(self):
        """Macro HMM should fit successfully on synthetic data."""
        rp = RegimePredictor(make_config(macro_n_states=3))
        features = make_macro_features(n=300, seed=42)

        model, states, posteriors = rp.fit_macro_hmm(features)
        assert model is not None
        assert len(states) == len(features)
        assert posteriors.shape == (len(features), model.n_components)

    def test_posteriors_sum_to_one(self):
        """Macro posteriors should sum to 1 at each time step."""
        rp = RegimePredictor(make_config(macro_n_states=3))
        features = make_macro_features(n=300, seed=42)

        _, _, posteriors = rp.fit_macro_hmm(features)
        row_sums = posteriors.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_label_macro_regimes(self):
        """Should produce valid macro regime labels."""
        rp = RegimePredictor(make_config(macro_n_states=3))
        features = make_macro_features(n=300, seed=42)

        model, _, _ = rp.fit_macro_hmm(features)
        labels = rp.label_macro_regimes(model, features)

        assert len(labels) == model.n_components
        assert "risk_on" in labels.values()
        assert "risk_off" in labels.values()
