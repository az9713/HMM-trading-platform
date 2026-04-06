"""
regime_mdp.py — Analytical regime forecasting and optimal policy via Bellman MDP.

This module turns the fitted HMM from a *descriptive* tool into a *prescriptive*
decision engine. Given a transition matrix A and per-regime emission statistics
(mean return, variance), it provides:

1.  Closed-form n-step regime probability forecasts:  pi_{t+k} = pi_t · A^k
2.  Expected return / variance / quantile fan charts at each horizon
3.  First-passage / expected hitting times via the fundamental matrix N = (I-Q)^-1
4.  Mixing time / chain half-life from the spectral gap of A
5.  Bellman value iteration on the (regime × last_action) MDP with mean-variance
    utility and proportional transaction costs, yielding the *theoretically
    optimal* long/flat/short policy and a Sharpe ceiling
6.  Comparison helpers to evaluate any user policy against the optimum

Everything here is closed-form or finite-horizon DP — no Monte Carlo needed —
which makes it both fast and complementary to monte_carlo.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ────────────────────────────────────────────────────────────────────────────
# Forecasting
# ────────────────────────────────────────────────────────────────────────────


def forecast_regime_distribution(
    pi0: np.ndarray, transmat: np.ndarray, horizon: int
) -> np.ndarray:
    """
    Analytically forecast the regime probability distribution forward `horizon`
    steps using pi_k = pi_0 @ A^k.

    Parameters
    ----------
    pi0 : (n,) current regime posterior (must sum to 1)
    transmat : (n, n) row-stochastic transition matrix
    horizon : number of bars to project (>= 1)

    Returns
    -------
    (horizon + 1, n) array. Row 0 is pi0, row k is the distribution at step k.
    """
    pi0 = np.asarray(pi0, dtype=float).ravel()
    A = np.asarray(transmat, dtype=float)
    n = A.shape[0]
    if pi0.shape != (n,):
        raise ValueError(f"pi0 shape {pi0.shape} != ({n},)")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    out = np.zeros((horizon + 1, n))
    out[0] = pi0
    cur = pi0.copy()
    for k in range(1, horizon + 1):
        cur = cur @ A
        out[k] = cur
    return out


def expected_return_path(
    pi_path: np.ndarray,
    regime_means: np.ndarray,
    regime_vars: np.ndarray,
    quantiles: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95),
) -> dict[str, np.ndarray]:
    """
    Closed-form per-bar expected return and variance under the forecast
    distribution, plus Gaussian-approx quantiles for fan-chart plotting.

    For each forecast row pi_k:
        E[r]   = sum_i pi_k[i] * mu_i
        E[r^2] = sum_i pi_k[i] * (mu_i^2 + sigma_i^2)
        Var[r] = E[r^2] - E[r]^2          (regime-mixture variance)

    Quantiles are computed as a Normal(E, sqrt(Var)) approximation, which is the
    matched first-two-moments fan; this is standard for regime-mixture forecasts.

    Returns
    -------
    dict with keys "mean", "std", and one entry per requested quantile
    (e.g. "q05", "q50", "q95"), each a 1-D array of length len(pi_path).
    """
    pi_path = np.asarray(pi_path, dtype=float)
    mu = np.asarray(regime_means, dtype=float).ravel()
    var = np.asarray(regime_vars, dtype=float).ravel()
    if pi_path.shape[1] != mu.shape[0] or mu.shape != var.shape:
        raise ValueError("pi_path / means / vars dimension mismatch")

    e_r = pi_path @ mu
    e_r2 = pi_path @ (mu**2 + var)
    v_r = np.clip(e_r2 - e_r**2, 0.0, None)
    s_r = np.sqrt(v_r)

    # Inverse standard normal via erfinv to avoid scipy dep
    from math import erf, sqrt as msqrt

    def _norm_ppf(p: float) -> float:
        # Beasley-Springer-Moro is overkill; use a robust rational approx.
        # Here we lean on numpy via a small Newton refinement on erf.
        # For our small fixed quantile set this is fine and dependency-free.
        if p <= 0.0 or p >= 1.0:
            raise ValueError("quantile must be in (0,1)")
        # initial guess via Acklam's approximation
        a = [
            -3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
            1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00,
        ]
        b = [
            -5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
            6.680131188771972e01, -1.328068155288572e01,
        ]
        c = [
            -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
            -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00,
        ]
        d = [
            7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
            3.754408661907416e00,
        ]
        plow = 0.02425
        phigh = 1 - plow
        if p < plow:
            q = msqrt(-2 * np.log(p))
            x = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / (
                (((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
        elif p <= phigh:
            q = p - 0.5
            r = q * q
            x = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / (
                ((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
        else:
            q = msqrt(-2 * np.log(1 - p))
            x = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / (
                (((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
        # one Newton refinement using erf
        e = 0.5 * (1 + erf(x / msqrt(2))) - p
        pdf = (1.0 / msqrt(2 * np.pi)) * np.exp(-0.5 * x * x)
        x -= e / pdf
        return x

    out = {"mean": e_r, "std": s_r}
    for q in quantiles:
        z = _norm_ppf(q)
        out[f"q{int(round(q*100)):02d}"] = e_r + z * s_r
    return out


# ────────────────────────────────────────────────────────────────────────────
# Hitting times & mixing time
# ────────────────────────────────────────────────────────────────────────────


def expected_hitting_times(transmat: np.ndarray) -> np.ndarray:
    """
    Expected first-passage time E[T_{ij}] = expected number of bars to first
    arrive in state j starting from state i.

    Method: for each target j, make state j absorbing, restrict to transient
    states Q (the (n-1)x(n-1) submatrix obtained by deleting row j and column j),
    and compute N = (I - Q)^-1. The expected absorption time from i is the i-th
    row sum of N, i.e. N @ 1.

    The diagonal entries E[T_{ii}] are the *expected return times* and are
    computed from the stationary distribution: E[T_{ii}] = 1 / pi_i.

    Returns
    -------
    (n, n) matrix H where H[i, j] = E[T_{ij}]. Entries that diverge (no path
    from i to j in the chain) become np.inf.
    """
    A = np.asarray(transmat, dtype=float)
    n = A.shape[0]
    H = np.zeros((n, n))

    pi_stat = stationary_distribution(A)

    for j in range(n):
        idx = [i for i in range(n) if i != j]
        Q = A[np.ix_(idx, idx)]
        try:
            N = np.linalg.inv(np.eye(n - 1) - Q)
            row_sums = N @ np.ones(n - 1)
        except np.linalg.LinAlgError:
            row_sums = np.full(n - 1, np.inf)
        for k, i in enumerate(idx):
            H[i, j] = row_sums[k]
        # mean recurrence time
        H[j, j] = 1.0 / pi_stat[j] if pi_stat[j] > 0 else np.inf

    return H


def stationary_distribution(transmat: np.ndarray) -> np.ndarray:
    """
    Stationary distribution pi satisfying pi @ A = pi via the dominant left
    eigenvector of A (eigenvalue closest to 1), normalized to sum to 1.
    """
    A = np.asarray(transmat, dtype=float)
    eigvals, eigvecs = np.linalg.eig(A.T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    pi = np.real(eigvecs[:, idx])
    s = pi.sum()
    if s == 0:
        n = A.shape[0]
        return np.full(n, 1.0 / n)
    pi = pi / s
    # Numerical safety: clip tiny negatives from eigensolver noise
    pi = np.clip(pi, 0.0, None)
    pi = pi / pi.sum()
    return pi


def mixing_time(transmat: np.ndarray) -> dict[str, float]:
    """
    Spectral-gap-based mixing diagnostics.

    Let lambda_2 be the second-largest eigenvalue of A by magnitude.
        spectral_gap = 1 - |lambda_2|
        half_life    = log(0.5) / log(|lambda_2|)            (bars)
        mixing_time  = 1 / spectral_gap                       (bars to ~63% mix)

    Smaller |lambda_2| -> faster mixing -> regimes are short-lived /
    transition quickly. Closer to 1 -> sticky / persistent regimes.
    """
    A = np.asarray(transmat, dtype=float)
    eigvals = np.linalg.eigvals(A)
    mags = np.sort(np.abs(eigvals))[::-1]
    if len(mags) < 2:
        return {"lambda2": 0.0, "spectral_gap": 1.0, "half_life": 0.0, "mixing_time": 0.0}
    l2 = float(mags[1])
    gap = 1.0 - l2
    if 0.0 < l2 < 1.0:
        half_life = float(np.log(0.5) / np.log(l2))
    else:
        half_life = float("inf")
    mt = float(1.0 / gap) if gap > 0 else float("inf")
    return {
        "lambda2": l2,
        "spectral_gap": gap,
        "half_life": half_life,
        "mixing_time": mt,
    }


# ────────────────────────────────────────────────────────────────────────────
# Bellman MDP optimal policy
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class MDPSolution:
    """Output of `bellman_optimal_policy`."""

    actions: np.ndarray            # (A,) action grid, e.g. [-1, 0, +1]
    V: np.ndarray                  # (n_regimes, A) value at (regime, last_action)
    policy: np.ndarray             # (n_regimes, A) optimal action *index* given (regime, last_action)
    policy_action: np.ndarray      # (n_regimes, A) optimal action *value*
    Q: np.ndarray                  # (n_regimes, A, A) Q[r, a_prev, a_new]
    stationary_policy: np.ndarray  # (n_regimes,) optimal action assuming you are already in that action (steady-state view)
    expected_step_reward: float    # long-run reward per bar under optimal policy
    theoretical_sharpe: float      # annualized Sharpe ceiling estimate
    n_iterations: int
    converged: bool


def bellman_optimal_policy(
    transmat: np.ndarray,
    regime_means: np.ndarray,
    regime_vars: np.ndarray,
    actions: tuple[float, ...] = (-1.0, 0.0, 1.0),
    risk_aversion: float = 1.0,
    cost: float = 0.0005,
    gamma: float = 0.99,
    bars_per_year: float = 252.0,
    tol: float = 1e-9,
    max_iter: int = 5000,
) -> MDPSolution:
    """
    Solve the regime MDP with state s = (regime r, last action a_prev).

    Reward (mean-variance utility, proportional cost on action change):
        R(r, a_prev, a) = a * mu_r - 0.5 * lambda * a^2 * sigma_r^2
                         - cost * |a - a_prev|

    Transition: (r, a_prev) --a--> (r', a) with prob A[r, r'].

    Bellman equation:
        V(r, a_prev) = max_a [ R(r, a_prev, a)
                              + gamma * sum_{r'} A[r, r'] * V(r', a) ]

    Returns the value function, optimal policy, Q-table, the long-run (under
    stationary) expected per-bar reward of the optimal policy, and an
    annualized Sharpe ceiling computed from the optimal action's regime mix.

    Notes
    -----
    `risk_aversion` (lambda) is in mean-variance utility units. cost is the
    proportional friction charged each time the position changes (per unit of
    |delta_position|). gamma should be < 1 for contraction.
    """
    A = np.asarray(transmat, dtype=float)
    mu = np.asarray(regime_means, dtype=float).ravel()
    var = np.asarray(regime_vars, dtype=float).ravel()
    acts = np.asarray(actions, dtype=float).ravel()
    n = A.shape[0]
    nA = len(acts)
    if mu.shape != (n,) or var.shape != (n,):
        raise ValueError("means / vars must match transmat dimension")
    if not (0.0 < gamma < 1.0):
        raise ValueError("gamma must be in (0, 1)")

    # Reward tensor R[r, a_prev_idx, a_new_idx]
    R = np.zeros((n, nA, nA))
    for r in range(n):
        for ap_i, ap in enumerate(acts):
            for a_i, a in enumerate(acts):
                util = a * mu[r] - 0.5 * risk_aversion * (a**2) * var[r]
                friction = cost * abs(a - ap)
                R[r, ap_i, a_i] = util - friction

    # Value iteration on V[r, a_prev]
    V = np.zeros((n, nA))
    converged = False
    iters = 0
    for it in range(max_iter):
        iters = it + 1
        # For each (r, ap), compute Q[r, ap, a] = R + gamma * sum_{r'} A[r, r'] V[r', a]
        # cont[r, a] = sum_{r'} A[r, r'] * V[r', a]   shape (n, nA)
        cont = A @ V                      # (n, nA)
        # Q[r, ap, a] = R[r, ap, a] + gamma * cont[r, a]
        Q = R + gamma * cont[:, None, :]  # broadcast over ap axis
        V_new = Q.max(axis=2)             # (n, nA)
        diff = float(np.max(np.abs(V_new - V)))
        V = V_new
        if diff < tol:
            converged = True
            break

    # Final Q and policy
    cont = A @ V
    Q = R + gamma * cont[:, None, :]
    policy_idx = Q.argmax(axis=2)         # (n, nA)
    policy_action = acts[policy_idx]      # (n, nA)

    # "Stationary policy" view: if last action equals the optimal one, what's chosen?
    # We surface the diagonal-ish view: action chosen when a_prev is *itself* optimal.
    # Approach: iterate policy until consistent (always converges in 1-2 steps for well-posed cases).
    stat_pol = np.zeros(n, dtype=int)
    # Start from "no position" prior
    zero_idx = int(np.argmin(np.abs(acts)))
    cur = np.full(n, zero_idx, dtype=int)
    for _ in range(50):
        nxt = policy_idx[np.arange(n), cur]
        if np.array_equal(nxt, cur):
            stat_pol = nxt
            break
        cur = nxt
    else:
        stat_pol = cur

    # Long-run expected per-bar reward under (regime r, optimal action stat_pol[r])
    pi_stat = stationary_distribution(A)
    optimal_actions = acts[stat_pol]
    bar_means = optimal_actions * mu
    bar_vars = (optimal_actions**2) * var
    e_step = float(np.sum(pi_stat * bar_means))
    # Variance of bar return under random regime + within-regime noise
    e_step2 = float(np.sum(pi_stat * (bar_means**2 + bar_vars)))
    v_step = max(e_step2 - e_step**2, 1e-18)
    sharpe = float((e_step / np.sqrt(v_step)) * np.sqrt(bars_per_year))

    return MDPSolution(
        actions=acts,
        V=V,
        policy=policy_idx,
        policy_action=policy_action,
        Q=Q,
        stationary_policy=optimal_actions,
        expected_step_reward=e_step,
        theoretical_sharpe=sharpe,
        n_iterations=iters,
        converged=converged,
    )


def evaluate_fixed_policy(
    transmat: np.ndarray,
    regime_means: np.ndarray,
    regime_vars: np.ndarray,
    policy: np.ndarray,
    bars_per_year: float = 252.0,
) -> dict[str, float]:
    """
    Evaluate a fixed (regime -> action) policy under the chain's stationary
    distribution. Returns expected per-bar return, variance, and annualized
    Sharpe — directly comparable to `bellman_optimal_policy`'s ceiling.
    """
    A = np.asarray(transmat, dtype=float)
    mu = np.asarray(regime_means, dtype=float).ravel()
    var = np.asarray(regime_vars, dtype=float).ravel()
    pol = np.asarray(policy, dtype=float).ravel()
    if not (mu.shape == var.shape == pol.shape == (A.shape[0],)):
        raise ValueError("dimension mismatch")
    pi = stationary_distribution(A)
    bar_means = pol * mu
    bar_vars = (pol**2) * var
    e = float(np.sum(pi * bar_means))
    e2 = float(np.sum(pi * (bar_means**2 + bar_vars)))
    v = max(e2 - e**2, 1e-18)
    return {
        "expected_step_reward": e,
        "step_variance": v,
        "annualized_sharpe": float((e / np.sqrt(v)) * np.sqrt(bars_per_year)),
    }
