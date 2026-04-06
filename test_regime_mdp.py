"""
Tests for regime_mdp.py — analytical regime forecasting and Bellman optimal policy.
"""

import numpy as np
import pytest

from regime_mdp import (
    bellman_optimal_policy,
    evaluate_fixed_policy,
    expected_hitting_times,
    expected_return_path,
    forecast_regime_distribution,
    mixing_time,
    stationary_distribution,
)


# ─── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def two_state_chain():
    # Persistent 2-state chain (bear / bull)
    A = np.array([[0.9, 0.1], [0.2, 0.8]])
    means = np.array([-0.002, 0.003])
    variances = np.array([0.0004, 0.0002])
    return A, means, variances


@pytest.fixture
def three_state_chain():
    A = np.array([
        [0.85, 0.10, 0.05],
        [0.15, 0.70, 0.15],
        [0.05, 0.10, 0.85],
    ])
    means = np.array([-0.004, 0.0, 0.005])
    variances = np.array([0.0009, 0.0003, 0.0004])
    return A, means, variances


# ─── stationary_distribution ───────────────────────────────────────────────


def test_stationary_distribution_sums_to_one(two_state_chain):
    A, _, _ = two_state_chain
    pi = stationary_distribution(A)
    assert pi.shape == (2,)
    assert np.isclose(pi.sum(), 1.0)
    # pi @ A == pi
    assert np.allclose(pi @ A, pi, atol=1e-10)


def test_stationary_known_two_state(two_state_chain):
    """For 2x2 chain stationary = [b/(a+b), a/(a+b)] with a=P(0->1), b=P(1->0)."""
    A, _, _ = two_state_chain
    a = A[0, 1]  # 0.1
    b = A[1, 0]  # 0.2
    expected = np.array([b / (a + b), a / (a + b)])
    pi = stationary_distribution(A)
    assert np.allclose(pi, expected, atol=1e-10)


# ─── forecast_regime_distribution ──────────────────────────────────────────


def test_forecast_zero_step_is_pi0(two_state_chain):
    A, _, _ = two_state_chain
    pi0 = np.array([0.7, 0.3])
    path = forecast_regime_distribution(pi0, A, horizon=5)
    assert path.shape == (6, 2)
    assert np.allclose(path[0], pi0)


def test_forecast_one_step_is_matrix_mul(two_state_chain):
    A, _, _ = two_state_chain
    pi0 = np.array([0.7, 0.3])
    path = forecast_regime_distribution(pi0, A, horizon=1)
    assert np.allclose(path[1], pi0 @ A)


def test_forecast_converges_to_stationary(two_state_chain):
    A, _, _ = two_state_chain
    pi0 = np.array([1.0, 0.0])
    path = forecast_regime_distribution(pi0, A, horizon=200)
    assert np.allclose(path[-1], stationary_distribution(A), atol=1e-6)


def test_forecast_rows_sum_to_one(three_state_chain):
    A, _, _ = three_state_chain
    pi0 = np.array([0.2, 0.5, 0.3])
    path = forecast_regime_distribution(pi0, A, horizon=20)
    assert np.allclose(path.sum(axis=1), 1.0, atol=1e-10)


def test_forecast_validates_horizon(two_state_chain):
    A, _, _ = two_state_chain
    with pytest.raises(ValueError):
        forecast_regime_distribution(np.array([0.5, 0.5]), A, horizon=0)


# ─── expected_return_path ──────────────────────────────────────────────────


def test_expected_return_path_shapes(two_state_chain):
    A, mu, var = two_state_chain
    path = forecast_regime_distribution(np.array([0.5, 0.5]), A, horizon=10)
    out = expected_return_path(path, mu, var)
    assert "mean" in out and "std" in out
    assert out["mean"].shape == (11,)
    assert out["std"].shape == (11,)
    # Quantile order: q05 <= q50 <= q95
    assert np.all(out["q05"] <= out["q50"] + 1e-12)
    assert np.all(out["q50"] <= out["q95"] + 1e-12)


def test_expected_return_matches_dot_product(two_state_chain):
    A, mu, var = two_state_chain
    pi0 = np.array([0.4, 0.6])
    path = forecast_regime_distribution(pi0, A, horizon=3)
    out = expected_return_path(path, mu, var)
    # Step-0 mean must equal pi0 . mu
    assert np.isclose(out["mean"][0], pi0 @ mu)
    # And variance row 0 = pi0 . (mu^2 + var) - mean^2
    expected_v = pi0 @ (mu**2 + var) - (pi0 @ mu) ** 2
    assert np.isclose(out["std"][0] ** 2, expected_v, atol=1e-12)


def test_expected_return_quantiles_use_normal_approx(two_state_chain):
    A, mu, var = two_state_chain
    path = forecast_regime_distribution(np.array([0.5, 0.5]), A, horizon=2)
    out = expected_return_path(path, mu, var, quantiles=(0.5,))
    # Median of a Normal(mean, std) approx equals the mean
    assert np.allclose(out["q50"], out["mean"], atol=1e-8)


# ─── expected_hitting_times ────────────────────────────────────────────────


def test_hitting_times_two_state_known(two_state_chain):
    """
    For a 2-state chain, E[T_{0->1}] = 1 / P(0->1) = 1/0.1 = 10
    and E[T_{1->0}] = 1 / 0.2 = 5. Diagonal = mean recurrence = 1/pi_i.
    """
    A, _, _ = two_state_chain
    H = expected_hitting_times(A)
    assert H.shape == (2, 2)
    assert np.isclose(H[0, 1], 1.0 / 0.1)
    assert np.isclose(H[1, 0], 1.0 / 0.2)
    pi = stationary_distribution(A)
    assert np.isclose(H[0, 0], 1.0 / pi[0])
    assert np.isclose(H[1, 1], 1.0 / pi[1])


def test_hitting_times_three_state_finite(three_state_chain):
    A, _, _ = three_state_chain
    H = expected_hitting_times(A)
    assert H.shape == (3, 3)
    assert np.all(np.isfinite(H))
    # Going farther (0 -> 2) takes longer than one step (0 -> 1) for this chain
    assert H[0, 2] > H[0, 1]


# ─── mixing_time ───────────────────────────────────────────────────────────


def test_mixing_time_keys(two_state_chain):
    A, _, _ = two_state_chain
    info = mixing_time(A)
    assert set(info.keys()) == {"lambda2", "spectral_gap", "half_life", "mixing_time"}
    assert 0.0 < info["lambda2"] < 1.0
    assert info["spectral_gap"] > 0
    assert info["half_life"] > 0


def test_mixing_time_two_state_analytical(two_state_chain):
    """For a 2x2 stochastic matrix, the second eigenvalue equals 1 - a - b."""
    A, _, _ = two_state_chain
    a, b = A[0, 1], A[1, 0]  # 0.1, 0.2
    expected_l2 = abs(1.0 - a - b)
    info = mixing_time(A)
    assert np.isclose(info["lambda2"], expected_l2, atol=1e-12)


def test_mixing_time_more_persistent_chain_has_longer_half_life():
    sticky = np.array([[0.99, 0.01], [0.01, 0.99]])
    fast = np.array([[0.6, 0.4], [0.4, 0.6]])
    assert mixing_time(sticky)["half_life"] > mixing_time(fast)["half_life"]


# ─── bellman_optimal_policy ────────────────────────────────────────────────


def test_bellman_converges(two_state_chain):
    A, mu, var = two_state_chain
    sol = bellman_optimal_policy(A, mu, var, gamma=0.95, cost=0.0)
    assert sol.converged
    assert sol.V.shape == (2, 3)
    assert sol.policy.shape == (2, 3)
    assert sol.Q.shape == (2, 3, 3)


def test_bellman_picks_long_in_bull_short_in_bear(two_state_chain):
    """With cost=0 and persistent regimes, optimal action should be sign(mu)."""
    A, mu, var = two_state_chain  # mu = [-0.002, +0.003]
    sol = bellman_optimal_policy(
        A, mu, var, actions=(-1.0, 0.0, 1.0),
        risk_aversion=0.1, cost=0.0, gamma=0.95,
    )
    # In bear regime (state 0) optimal action should be negative (short)
    # In bull regime (state 1) optimal action should be positive (long)
    # Use stationary_policy as the steady-state action mapping.
    assert sol.stationary_policy[0] < 0
    assert sol.stationary_policy[1] > 0


def test_bellman_high_cost_prefers_no_trading(two_state_chain):
    """With ruinous transaction costs, the optimal policy should hold flat."""
    A, mu, var = two_state_chain
    sol = bellman_optimal_policy(
        A, mu, var, actions=(-1.0, 0.0, 1.0),
        risk_aversion=0.5, cost=10.0, gamma=0.95,
    )
    # Once in flat, staying flat avoids the huge cost of switching
    assert np.all(sol.stationary_policy == 0.0)


def test_bellman_high_risk_aversion_shrinks_position(two_state_chain):
    A, mu, var = two_state_chain
    low = bellman_optimal_policy(A, mu, var, risk_aversion=0.01, cost=0.0)
    high = bellman_optimal_policy(A, mu, var, risk_aversion=1e6, cost=0.0)
    # With huge risk aversion the only safe action in any regime is flat
    assert np.all(high.stationary_policy == 0.0)
    # And the low-risk-aversion case should NOT all be flat
    assert not np.all(low.stationary_policy == 0.0)


def test_bellman_validates_gamma(two_state_chain):
    A, mu, var = two_state_chain
    with pytest.raises(ValueError):
        bellman_optimal_policy(A, mu, var, gamma=1.0)


# ─── evaluate_fixed_policy ─────────────────────────────────────────────────


def test_evaluate_flat_policy_zero(two_state_chain):
    A, mu, var = two_state_chain
    res = evaluate_fixed_policy(A, mu, var, policy=np.array([0.0, 0.0]))
    assert res["expected_step_reward"] == 0.0


def test_evaluate_long_only_matches_pi_dot_mu(two_state_chain):
    A, mu, var = two_state_chain
    res = evaluate_fixed_policy(A, mu, var, policy=np.array([1.0, 1.0]))
    pi = stationary_distribution(A)
    assert np.isclose(res["expected_step_reward"], float(pi @ mu))


def test_optimal_policy_beats_random(two_state_chain):
    """The Bellman-optimal Sharpe should not be worse than a random fixed policy."""
    A, mu, var = two_state_chain
    sol = bellman_optimal_policy(A, mu, var, risk_aversion=0.1, cost=0.0)
    # Compare against an obviously bad policy: long in bear, short in bull
    bad = evaluate_fixed_policy(A, mu, var, policy=np.array([1.0, -1.0]))
    assert sol.theoretical_sharpe > bad["annualized_sharpe"]
