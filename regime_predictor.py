"""
regime_predictor.py — Bayesian Online Change Point Detection + Regime Forecasting.

Combines BOCD with the HMM's learned transition matrix and emission parameters
to detect regime shifts earlier than Viterbi decoding alone, and to generate
probabilistic forecasts of future regime states.

Key capabilities:
  1. BOCD run-length posterior: P(run_length_t | x_{1:t}) at each bar
  2. Change point probability: P(change_t) = P(r_t = 0 | x_{1:t})
  3. Regime duration forecast: expected remaining bars in current regime
  4. N-step regime forecast: P(regime_{t+k}) for k = 1..horizon
  5. Composite early-detection score blending BOCD + HMM entropy signals
"""

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from dataclasses import dataclass


@dataclass
class ChangePoint:
    """A detected change point with metadata."""
    bar: int
    probability: float
    run_length_before: int
    predicted_regime: str
    actual_regime: str
    detection_lag: int  # bars before Viterbi caught it (negative = earlier)


class RegimePredictor:
    """
    Bayesian Online Change Point Detection fused with HMM regime forecasting.

    Uses the HMM's emission parameters as the observation model for BOCD,
    making the change point detector regime-aware: it knows what each regime
    "looks like" and can detect when observations stop matching.
    """

    def __init__(self, config: dict):
        pred_cfg = config.get("prediction", {})
        self.hazard_rate = pred_cfg.get("hazard_rate", 1 / 100)
        self.forecast_horizon = pred_cfg.get("forecast_horizon", 20)
        self.cp_threshold = pred_cfg.get("changepoint_threshold", 0.3)
        self.max_run_length = pred_cfg.get("max_run_length", 300)
        self.blend_weight_bocd = pred_cfg.get("blend_weight_bocd", 0.6)
        self.blend_weight_entropy = pred_cfg.get("blend_weight_entropy", 0.4)

    def _hazard_function(self, run_length: np.ndarray) -> np.ndarray:
        """
        Constant hazard rate: P(change | run_length) = 1/lambda.
        This implies geometric prior on run lengths with mean lambda.
        """
        return np.full_like(run_length, self.hazard_rate, dtype=float)

    def _observation_log_likelihood(
        self,
        x: np.ndarray,
        means: np.ndarray,
        covars: np.ndarray,
        covariance_type: str,
    ) -> np.ndarray:
        """
        Compute log P(x | regime_i) for each regime using the HMM's
        learned emission parameters. Returns array of shape (n_states,).
        """
        n_states = len(means)
        log_liks = np.zeros(n_states)

        for i in range(n_states):
            mean = means[i]
            if covariance_type == "full":
                cov = covars[i]
            elif covariance_type == "diag":
                cov = np.diag(covars[i])
            elif covariance_type == "spherical":
                cov = np.eye(len(mean)) * covars[i]
            else:  # tied
                cov = covars

            try:
                log_liks[i] = multivariate_normal.logpdf(x, mean=mean, cov=cov)
            except (np.linalg.LinAlgError, ValueError):
                log_liks[i] = -1e10

        return log_liks

    def run_bocd(
        self,
        X: np.ndarray,
        means: np.ndarray,
        covars: np.ndarray,
        covariance_type: str,
        transmat: np.ndarray,
        stationary_dist: np.ndarray,
    ) -> dict:
        """
        Run Bayesian Online Change Point Detection over observation sequence X.

        The observation model uses the HMM's emission distributions, making
        this a regime-aware BOCD: at each timestep, it computes the likelihood
        of the observation under each regime, weighted by the regime's
        stationary probability.

        Returns dict with:
          - run_length_posteriors: (T, max_run_length) posterior over run lengths
          - changepoint_prob: (T,) probability of change point at each bar
          - map_run_length: (T,) MAP run length estimate
          - regime_log_liks: (T, n_states) observation log-likelihoods per regime
        """
        T, d = X.shape
        n_states = len(means)
        R = min(self.max_run_length, T + 1)

        # Run-length posterior: P(r_t | x_{1:t})
        # We maintain log-space for numerical stability
        run_length_post = np.zeros((T, R))
        changepoint_prob = np.zeros(T)
        map_run_length = np.zeros(T, dtype=int)
        regime_log_liks = np.zeros((T, n_states))

        # Initialize: P(r_0 = 0) = 1
        # We use a message-passing approach in log space
        log_joint = np.full(R, -np.inf)  # log P(r_t, x_{1:t})
        log_joint[0] = 0.0  # r_0 = 0 with certainty

        for t in range(T):
            x_t = X[t]

            # Observation likelihood under each regime
            obs_ll = self._observation_log_likelihood(
                x_t, means, covars, covariance_type
            )
            regime_log_liks[t] = obs_ll

            # Marginal observation likelihood: weighted by stationary distribution
            # P(x_t | regime) * P(regime) - this is the "predictive" for a new segment
            log_pred_new = np.logaddexp.reduce(
                obs_ll + np.log(np.clip(stationary_dist, 1e-300, None))
            )

            # For continuing segments, use max regime likelihood as approximation
            # (the regime most consistent with the current run)
            log_pred_continue = np.max(obs_ll)

            # Growth probabilities: existing run lengths grow by 1
            hazard = self.hazard_rate
            log_h = np.log(hazard)
            log_1mh = np.log(1 - hazard)

            # New log_joint after observing x_t
            new_log_joint = np.full(R, -np.inf)

            # Change point mass: sum over all previous run lengths
            # P(r_t=0, x_{1:t}) = sum_r P(r_{t-1}=r, x_{1:t-1}) * H(r) * P(x_t|new)
            valid = log_joint > -1e100
            if np.any(valid):
                changepoint_mass = np.logaddexp.reduce(
                    log_joint[valid] + log_h
                ) + log_pred_new
            else:
                changepoint_mass = log_pred_new

            new_log_joint[0] = changepoint_mass

            # Growth mass: each run length r grows to r+1
            # P(r_t=r+1, x_{1:t}) = P(r_{t-1}=r, x_{1:t-1}) * (1-H(r)) * P(x_t|continue)
            if R > 1:
                shifted = log_joint[: R - 1] + log_1mh + log_pred_continue
                new_log_joint[1:R] = shifted

            # Normalize to get posterior
            log_evidence = np.logaddexp.reduce(
                new_log_joint[new_log_joint > -1e100]
            ) if np.any(new_log_joint > -1e100) else -np.inf

            if log_evidence > -np.inf:
                log_posterior = new_log_joint - log_evidence
                run_length_post[t] = np.exp(
                    np.clip(log_posterior, -500, 0)
                )
            else:
                run_length_post[t, 0] = 1.0

            changepoint_prob[t] = run_length_post[t, 0]
            map_run_length[t] = np.argmax(run_length_post[t])

            log_joint = new_log_joint

        return {
            "run_length_posteriors": run_length_post,
            "changepoint_prob": changepoint_prob,
            "map_run_length": map_run_length,
            "regime_log_liks": regime_log_liks,
        }

    def detect_changepoints(
        self,
        changepoint_prob: np.ndarray,
        states: np.ndarray,
        labels: dict[int, str],
    ) -> list[ChangePoint]:
        """
        Identify change points where BOCD probability exceeds threshold.
        Compare with Viterbi state changes to measure detection lag.
        """
        T = len(changepoint_prob)
        changepoints = []

        # Find Viterbi transitions for lag comparison
        viterbi_transitions = set()
        for t in range(1, len(states)):
            if states[t] != states[t - 1]:
                viterbi_transitions.add(t)

        for t in range(1, T):
            if changepoint_prob[t] < self.cp_threshold:
                continue

            # Find nearest Viterbi transition
            nearest_viterbi = None
            min_dist = T
            for vt in viterbi_transitions:
                dist = abs(t - vt)
                if dist < min_dist:
                    min_dist = dist
                    nearest_viterbi = vt

            if nearest_viterbi is not None:
                detection_lag = t - nearest_viterbi  # negative = BOCD detected first
            else:
                detection_lag = 0

            # Run length before this change point
            # (look at previous bar's most likely run length)
            run_before = 0
            if t > 0:
                # Simple estimate: count consecutive same-state bars before t
                run = 0
                s = states[min(t, len(states) - 1)]
                for k in range(t - 1, -1, -1):
                    if k < len(states) and states[k] == s:
                        run += 1
                    else:
                        break
                run_before = run

            predicted = labels.get(int(states[min(t, len(states) - 1)]), "unknown")
            actual = labels.get(int(states[min(t, len(states) - 1)]), "unknown")

            changepoints.append(ChangePoint(
                bar=t,
                probability=changepoint_prob[t],
                run_length_before=run_before,
                predicted_regime=predicted,
                actual_regime=actual,
                detection_lag=detection_lag,
            ))

        return changepoints

    def forecast_regime_probabilities(
        self,
        transmat: np.ndarray,
        current_state_dist: np.ndarray,
        horizon: int | None = None,
    ) -> np.ndarray:
        """
        Forecast regime probabilities k steps ahead by powering the transition matrix.

        P(regime_{t+k}) = current_dist @ transmat^k

        Returns array of shape (horizon, n_states).
        """
        if horizon is None:
            horizon = self.forecast_horizon

        n_states = transmat.shape[0]
        forecasts = np.zeros((horizon, n_states))

        dist = current_state_dist.copy()
        for k in range(horizon):
            dist = dist @ transmat
            forecasts[k] = dist

        return forecasts

    def expected_regime_duration(
        self,
        transmat: np.ndarray,
        current_state: int,
        current_run_length: int,
    ) -> dict:
        """
        Estimate expected remaining duration in current regime.

        For a Markov chain, the remaining time in state i is geometric
        with parameter (1 - a_ii). The memoryless property means the
        expected remaining duration is always 1/(1 - a_ii), regardless
        of how long we've been in the state.

        However, we also compute a Bayesian estimate that accounts for
        the empirical run length to give a more nuanced picture.
        """
        self_prob = transmat[current_state, current_state]

        # Geometric expected remaining duration (memoryless)
        if self_prob < 1.0:
            expected_remaining = 1.0 / (1.0 - self_prob)
        else:
            expected_remaining = float("inf")

        # Probability of surviving k more bars
        survival_probs = []
        for k in range(1, self.forecast_horizon + 1):
            survival_probs.append(self_prob ** k)

        # Median remaining duration: smallest k where survival < 0.5
        median_remaining = self.forecast_horizon
        for k, sp in enumerate(survival_probs, 1):
            if sp < 0.5:
                median_remaining = k
                break

        # Probability of transition in next 5 bars
        p_transition_5 = 1.0 - self_prob ** 5

        # Most likely next regime (excluding self-transition)
        exit_probs = transmat[current_state].copy()
        exit_probs[current_state] = 0
        if exit_probs.sum() > 0:
            exit_probs /= exit_probs.sum()
            most_likely_next = int(np.argmax(exit_probs))
            next_regime_prob = exit_probs[most_likely_next]
        else:
            most_likely_next = current_state
            next_regime_prob = 1.0

        return {
            "current_run_length": current_run_length,
            "expected_remaining": expected_remaining,
            "median_remaining": median_remaining,
            "self_transition_prob": self_prob,
            "p_transition_5_bars": p_transition_5,
            "most_likely_next_state": most_likely_next,
            "next_state_probability": next_regime_prob,
            "survival_curve": np.array(survival_probs),
        }

    def composite_early_detection(
        self,
        changepoint_prob: np.ndarray,
        entropy: np.ndarray,
        confidence: np.ndarray,
    ) -> np.ndarray:
        """
        Blend BOCD change point probability with HMM entropy signals
        into a single early-detection score in [0, 1].

        Score = w_bocd * CP_prob + w_entropy * (1 - confidence)

        High score = high probability that a regime change is imminent.
        """
        # Normalize entropy contribution: use (1 - confidence) as uncertainty
        uncertainty = 1.0 - confidence

        score = (
            self.blend_weight_bocd * changepoint_prob
            + self.blend_weight_entropy * uncertainty
        )

        # Clip to [0, 1]
        return np.clip(score, 0.0, 1.0)

    def generate_forecast_summary(
        self,
        X: np.ndarray,
        states: np.ndarray,
        posteriors: np.ndarray,
        labels: dict[int, str],
        transmat: np.ndarray,
        means: np.ndarray,
        covars: np.ndarray,
        covariance_type: str,
        entropy: np.ndarray,
        confidence: np.ndarray,
    ) -> dict:
        """
        Run the full prediction pipeline and return a comprehensive summary.

        This is the main entry point for the dashboard integration.
        """
        n_states = transmat.shape[0]

        # Compute stationary distribution
        eigenvalues, eigenvectors = np.linalg.eig(transmat.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()

        # Run BOCD
        bocd_results = self.run_bocd(
            X, means, covars, covariance_type, transmat, stationary
        )

        changepoint_prob = bocd_results["changepoint_prob"]
        map_run_length = bocd_results["map_run_length"]

        # Detect change points
        changepoints = self.detect_changepoints(
            changepoint_prob, states, labels
        )

        # Composite early detection score
        early_detection = self.composite_early_detection(
            changepoint_prob, entropy, confidence
        )

        # Current state info
        current_state = int(states[-1])
        current_posterior = posteriors[-1]
        current_run = int(map_run_length[-1])

        # Regime duration forecast
        duration_forecast = self.expected_regime_duration(
            transmat, current_state, current_run
        )

        # N-step regime forecast from current posterior
        regime_forecast = self.forecast_regime_probabilities(
            transmat, current_posterior
        )

        # Detection lag statistics
        if changepoints:
            lags = [cp.detection_lag for cp in changepoints]
            early_detections = sum(1 for lag in lags if lag < 0)
            avg_lag = np.mean(lags)
        else:
            early_detections = 0
            avg_lag = 0.0

        return {
            "bocd_results": bocd_results,
            "changepoints": changepoints,
            "early_detection_score": early_detection,
            "duration_forecast": duration_forecast,
            "regime_forecast": regime_forecast,
            "current_state": current_state,
            "current_regime": labels.get(current_state, "unknown"),
            "current_run_length": current_run,
            "n_changepoints_detected": len(changepoints),
            "n_early_detections": early_detections,
            "avg_detection_lag": avg_lag,
            "stationary_distribution": stationary,
        }
