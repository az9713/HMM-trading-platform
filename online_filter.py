"""
online_filter.py — Online Bayesian Regime Filter.

Transforms the batch HMM into a streaming regime detector by applying
the forward algorithm incrementally, one observation at a time.

Core capabilities:
  1. Streaming posterior updates via α-recursion (forward algorithm)
  2. Regime change-point detection via CUSUM on posterior entropy
  3. N-step-ahead regime forecasting via transition matrix powers
  4. Predictive observation distributions (Gaussian mixture weighted by forecast)
  5. Regime persistence tracking and transition event logging

Mathematical foundation:
  α_t(j) = p(o_1,...,o_t, s_t=j)
  α_t(j) ∝ b_j(o_t) * Σ_i α_{t-1}(i) * a_{i,j}

  where a_{i,j} = transition prob, b_j(o_t) = emission density.

  The filtered posterior is: p(s_t=j | o_1:t) = α_t(j) / Σ_k α_t(k)

References:
  - Rabiner (1989) "A Tutorial on HMMs and Selected Applications"
  - Page (1954) "Continuous Inspection Schemes" (CUSUM)
  - Hamilton (1989) "A New Approach to the Economic Analysis of
    Nonstationary Time Series and the Business Cycle"
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.stats import multivariate_normal


@dataclass
class RegimeChangeEvent:
    """A detected regime transition."""
    bar_index: int
    timestamp: object  # datetime or int
    from_regime: int
    to_regime: int
    from_label: str
    to_label: str
    confidence: float  # posterior probability of new regime
    cusum_value: float  # CUSUM statistic at detection


@dataclass
class RegimeForecast:
    """N-step-ahead regime probability forecast."""
    current_posteriors: np.ndarray
    forecast_horizons: list[int]
    forecast_probs: dict[int, np.ndarray]  # horizon -> (n_states,)
    stationary_dist: np.ndarray
    convergence_horizon: int  # bars until within 5% of stationary


@dataclass
class FilterState:
    """Complete snapshot of the online filter's internal state."""
    bar_count: int = 0
    log_alpha: np.ndarray = field(default_factory=lambda: np.array([]))
    posteriors: np.ndarray = field(default_factory=lambda: np.array([]))
    current_regime: int = -1
    regime_duration: int = 0
    entropy: float = 0.0
    confidence: float = 0.0
    cusum_pos: float = 0.0
    cusum_neg: float = 0.0
    running_entropy_mean: float = 0.0
    running_entropy_var: float = 0.0
    log_likelihood: float = 0.0  # cumulative


class OnlineBayesianFilter:
    """
    Streaming regime filter using the HMM forward algorithm.

    Takes a fitted RegimeDetector and processes observations one at a time,
    maintaining filtered regime posteriors, detecting change points, and
    forecasting future regime probabilities.
    """

    def __init__(
        self,
        detector,  # RegimeDetector with fitted model
        cusum_threshold: float = 3.0,
        cusum_drift: float = 0.0,
        forecast_horizons: list[int] | None = None,
        entropy_warmup: int = 20,
    ):
        """
        Args:
            detector: Fitted RegimeDetector instance.
            cusum_threshold: CUSUM detection threshold (in std devs of entropy).
            cusum_drift: Allowance parameter for CUSUM (0 = most sensitive).
            forecast_horizons: Steps ahead to forecast (default [1,5,10,25,50]).
            entropy_warmup: Bars before CUSUM activates (need entropy baseline).
        """
        if detector.model is None:
            raise ValueError("RegimeDetector must be fitted before filtering")

        self.detector = detector
        self.model = detector.model
        self.n_states = detector.n_states
        self.labels = detector.labels or {}

        # Extract HMM parameters
        self.log_transmat = np.log(self.model.transmat_ + 1e-300)
        self.means = self.model.means_
        self.covars = self._extract_covars()
        self.log_startprob = np.log(self.model.startprob_ + 1e-300)

        # CUSUM parameters
        self.cusum_threshold = cusum_threshold
        self.cusum_drift = cusum_drift
        self.entropy_warmup = entropy_warmup

        # Forecast horizons
        self.forecast_horizons = forecast_horizons or [1, 5, 10, 25, 50]

        # Precompute stationary distribution
        self.stationary_dist = self._compute_stationary()

        # Internal state
        self.state = FilterState()
        self._posterior_history: list[np.ndarray] = []
        self._entropy_history: list[float] = []
        self._regime_history: list[int] = []
        self._change_events: list[RegimeChangeEvent] = []
        self._log_lik_history: list[float] = []

    def _extract_covars(self) -> list[np.ndarray]:
        """Extract per-state covariance matrices regardless of covariance_type."""
        cov_type = self.model.covariance_type
        d = self.means.shape[1]
        raw = self.model.covars_
        covars = []
        for i in range(self.n_states):
            if cov_type == "full":
                covars.append(raw[i])
            elif cov_type == "diag":
                covars.append(np.diag(raw[i]))
            elif cov_type == "spherical":
                covars.append(np.eye(d) * raw[i])
            elif cov_type == "tied":
                # Tied: single (d, d) matrix shared across all states
                if raw.ndim == 2:
                    covars.append(raw.copy())
                else:
                    covars.append(raw[0])
        return covars

    def _compute_stationary(self) -> np.ndarray:
        """Compute stationary distribution from transition matrix."""
        eigenvalues, eigenvectors = np.linalg.eig(self.model.transmat_.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi = np.real(eigenvectors[:, idx])
        pi = pi / pi.sum()
        return np.clip(pi, 0, 1)

    def _log_emission(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute log emission probabilities log b_j(o_t) for each state.
        Uses multivariate Gaussian density.
        """
        log_probs = np.zeros(self.n_states)
        for j in range(self.n_states):
            try:
                log_probs[j] = multivariate_normal.logpdf(
                    obs, mean=self.means[j], cov=self.covars[j]
                )
            except np.linalg.LinAlgError:
                # Singular covariance — use diagonal fallback
                diag_cov = np.diag(np.diag(self.covars[j]) + 1e-6)
                log_probs[j] = multivariate_normal.logpdf(
                    obs, mean=self.means[j], cov=diag_cov
                )
        return log_probs

    def _shannon_entropy(self, probs: np.ndarray) -> float:
        """Normalized Shannon entropy in [0, 1]."""
        eps = 1e-12
        p = np.clip(probs, eps, 1.0)
        h = -np.sum(p * np.log2(p))
        max_h = np.log2(self.n_states)
        return float(h / max_h) if max_h > 0 else 0.0

    def reset(self):
        """Reset filter to initial state."""
        self.state = FilterState()
        self._posterior_history.clear()
        self._entropy_history.clear()
        self._regime_history.clear()
        self._change_events.clear()
        self._log_lik_history.clear()

    def update(self, obs: np.ndarray, timestamp=None) -> dict:
        """
        Process a single observation through the forward filter.

        Args:
            obs: Feature vector (d,) — same features used for HMM training.
            timestamp: Optional timestamp for this bar.

        Returns:
            dict with keys: posteriors, regime, regime_label, confidence,
                           entropy, change_detected, cusum_pos, cusum_neg,
                           log_likelihood_bar
        """
        obs = np.asarray(obs, dtype=np.float64).ravel()
        log_b = self._log_emission(obs)

        if self.state.bar_count == 0:
            # First observation: α_0(j) = π_j * b_j(o_0)
            log_alpha = self.log_startprob + log_b
        else:
            # α_t(j) = b_j(o_t) * Σ_i α_{t-1}(i) * a_{i,j}
            # In log space: log α_t(j) = log b_j(o_t) + logsumexp_i(log α_{t-1}(i) + log a_{i,j})
            prev_log_alpha = self.state.log_alpha
            log_alpha = np.zeros(self.n_states)
            for j in range(self.n_states):
                log_alpha[j] = log_b[j] + _logsumexp(
                    prev_log_alpha + self.log_transmat[:, j]
                )

        # Normalize to get posterior: p(s_t | o_{1:t})
        log_norm = _logsumexp(log_alpha)
        log_posterior = log_alpha - log_norm
        posteriors = np.exp(log_posterior)
        posteriors = posteriors / posteriors.sum()  # ensure exact normalization

        # Current regime (MAP estimate)
        regime = int(np.argmax(posteriors))
        regime_label = self.labels.get(regime, f"state_{regime}")
        confidence = float(posteriors[regime])

        # Entropy
        entropy = self._shannon_entropy(posteriors)

        # Regime duration tracking
        if regime == self.state.current_regime:
            regime_duration = self.state.regime_duration + 1
        else:
            regime_duration = 1

        # CUSUM change-point detection on entropy
        change_detected = False
        cusum_pos = self.state.cusum_pos
        cusum_neg = self.state.cusum_neg

        if self.state.bar_count >= self.entropy_warmup:
            # Update running stats (Welford's online algorithm)
            n = self.state.bar_count - self.entropy_warmup + 1
            if n == 1:
                new_mean = entropy
                new_var = 0.0
            else:
                old_mean = self.state.running_entropy_mean
                old_var = self.state.running_entropy_var
                new_mean = old_mean + (entropy - old_mean) / n
                new_var = old_var + (entropy - old_mean) * (entropy - new_mean)

            std_entropy = np.sqrt(new_var / max(n, 2)) if n >= 2 else 1.0

            if std_entropy > 1e-8:
                z = (entropy - new_mean) / std_entropy
            else:
                z = 0.0

            # CUSUM update
            cusum_pos = max(0.0, cusum_pos + z - self.cusum_drift)
            cusum_neg = max(0.0, cusum_neg - z - self.cusum_drift)

            if cusum_pos > self.cusum_threshold or cusum_neg > self.cusum_threshold:
                change_detected = True
                cusum_pos = 0.0
                cusum_neg = 0.0

            self.state.running_entropy_mean = new_mean
            self.state.running_entropy_var = new_var

        # Log regime change event
        if (self.state.bar_count > 0
                and regime != self.state.current_regime):
            event = RegimeChangeEvent(
                bar_index=self.state.bar_count,
                timestamp=timestamp,
                from_regime=self.state.current_regime,
                to_regime=regime,
                from_label=self.labels.get(self.state.current_regime, f"state_{self.state.current_regime}"),
                to_label=regime_label,
                confidence=confidence,
                cusum_value=max(cusum_pos, cusum_neg),
            )
            self._change_events.append(event)

        # Update internal state
        self.state.bar_count += 1
        self.state.log_alpha = log_alpha
        self.state.posteriors = posteriors
        self.state.current_regime = regime
        self.state.regime_duration = regime_duration
        self.state.entropy = entropy
        self.state.confidence = confidence
        self.state.cusum_pos = cusum_pos
        self.state.cusum_neg = cusum_neg
        self.state.log_likelihood += log_norm

        # Record history
        self._posterior_history.append(posteriors.copy())
        self._entropy_history.append(entropy)
        self._regime_history.append(regime)
        self._log_lik_history.append(log_norm)

        return {
            "posteriors": posteriors,
            "regime": regime,
            "regime_label": regime_label,
            "confidence": confidence,
            "entropy": entropy,
            "regime_duration": regime_duration,
            "change_detected": change_detected,
            "cusum_pos": cusum_pos,
            "cusum_neg": cusum_neg,
            "log_likelihood_bar": float(log_norm),
        }

    def process_batch(
        self, X: np.ndarray, timestamps=None
    ) -> pd.DataFrame:
        """
        Process a sequence of observations, returning full history as DataFrame.

        Args:
            X: (T, d) feature matrix.
            timestamps: Optional sequence of timestamps.

        Returns:
            DataFrame with columns: regime, regime_label, confidence, entropy,
            change_detected, cusum_pos, cusum_neg, + posterior columns per state.
        """
        results = []
        for t in range(len(X)):
            ts = timestamps[t] if timestamps is not None else t
            res = self.update(X[t], timestamp=ts)
            results.append(res)

        df = pd.DataFrame(results)
        if timestamps is not None:
            df.index = timestamps[:len(df)]

        # Add per-state posterior columns
        post_matrix = np.array(self._posterior_history[-len(X):])
        for j in range(self.n_states):
            label = self.labels.get(j, f"state_{j}")
            df[f"posterior_{label}"] = post_matrix[:, j]

        return df

    def forecast(self, horizons: list[int] | None = None) -> RegimeForecast:
        """
        Forecast regime probabilities N steps ahead using transition matrix powers.

        p(s_{t+h} | o_{1:t}) = p(s_t | o_{1:t}) @ A^h

        where A is the transition matrix.

        Also computes convergence horizon: minimum h such that
        ||forecast_h - π|| < 0.05 (L1 distance to stationary).
        """
        if self.state.bar_count == 0:
            raise ValueError("No observations processed yet")

        horizons = horizons or self.forecast_horizons
        current = self.state.posteriors.copy()
        A = self.model.transmat_

        forecast_probs = {}
        convergence_horizon = max(horizons)

        for h in sorted(horizons):
            A_h = np.linalg.matrix_power(A, h)
            forecast_probs[h] = current @ A_h
            # Ensure valid probability distribution
            forecast_probs[h] = np.clip(forecast_probs[h], 0, 1)
            forecast_probs[h] /= forecast_probs[h].sum()

        # Find convergence horizon
        for h in range(1, max(horizons) + 1):
            A_h = np.linalg.matrix_power(A, h)
            fcast = current @ A_h
            fcast = np.clip(fcast, 0, 1)
            fcast /= fcast.sum()
            if np.sum(np.abs(fcast - self.stationary_dist)) < 0.05:
                convergence_horizon = h
                break

        return RegimeForecast(
            current_posteriors=current,
            forecast_horizons=sorted(horizons),
            forecast_probs=forecast_probs,
            stationary_dist=self.stationary_dist,
            convergence_horizon=convergence_horizon,
        )

    def predictive_distribution(self, horizon: int = 1) -> dict:
        """
        Compute the predictive observation distribution at horizon h.

        p(o_{t+h} | o_{1:t}) = Σ_j p(s_{t+h}=j | o_{1:t}) * N(o; μ_j, Σ_j)

        Returns dict with mixture weights, means, covariances for the
        Gaussian mixture predictive distribution.
        """
        if self.state.bar_count == 0:
            raise ValueError("No observations processed yet")

        A_h = np.linalg.matrix_power(self.model.transmat_, horizon)
        weights = self.state.posteriors @ A_h
        weights = np.clip(weights, 0, 1)
        weights /= weights.sum()

        # Mixture mean and covariance
        d = self.means.shape[1]
        mix_mean = np.zeros(d)
        mix_cov = np.zeros((d, d))

        for j in range(self.n_states):
            mix_mean += weights[j] * self.means[j]

        for j in range(self.n_states):
            diff = self.means[j] - mix_mean
            mix_cov += weights[j] * (self.covars[j] + np.outer(diff, diff))

        return {
            "weights": weights,
            "means": self.means.copy(),
            "covariances": [c.copy() for c in self.covars],
            "mixture_mean": mix_mean,
            "mixture_covariance": mix_cov,
        }

    def regime_persistence_prob(self, horizon: int = 1) -> float:
        """
        Probability that the current regime persists for the next `horizon` bars.

        P(stay) = a_{ii}^h where i is the current regime.
        """
        if self.state.bar_count == 0:
            return 0.0
        i = self.state.current_regime
        a_ii = self.model.transmat_[i, i]
        return float(a_ii ** horizon)

    def expected_regime_duration(self) -> float:
        """Expected remaining duration of current regime: E[d] = 1/(1 - a_{ii})."""
        if self.state.bar_count == 0:
            return 0.0
        i = self.state.current_regime
        a_ii = self.model.transmat_[i, i]
        return float(1.0 / (1.0 - a_ii)) if a_ii < 1.0 else float("inf")

    def most_likely_next_regime(self) -> tuple[int, str, float]:
        """
        Most likely next regime if a transition occurs.

        Finds argmax_{j≠i} a_{i,j} and returns (state, label, probability).
        """
        if self.state.bar_count == 0:
            return (0, "unknown", 0.0)

        i = self.state.current_regime
        trans_row = self.model.transmat_[i].copy()
        trans_row[i] = 0.0  # exclude self-transition
        if trans_row.sum() > 0:
            trans_row /= trans_row.sum()
        j = int(np.argmax(trans_row))
        label = self.labels.get(j, f"state_{j}")
        prob = float(self.model.transmat_[i, j])
        return (j, label, prob)

    @property
    def change_events(self) -> list[RegimeChangeEvent]:
        """All detected regime change events."""
        return list(self._change_events)

    @property
    def posterior_history(self) -> np.ndarray:
        """(T, n_states) array of all filtered posteriors."""
        if not self._posterior_history:
            return np.array([]).reshape(0, self.n_states)
        return np.array(self._posterior_history)

    @property
    def entropy_history(self) -> np.ndarray:
        """(T,) array of all entropy values."""
        return np.array(self._entropy_history)

    @property
    def regime_history(self) -> np.ndarray:
        """(T,) array of MAP regime assignments."""
        return np.array(self._regime_history)

    def summary(self) -> dict:
        """Compact summary of current filter state."""
        if self.state.bar_count == 0:
            return {"status": "no observations processed"}

        next_regime, next_label, next_prob = self.most_likely_next_regime()
        return {
            "bars_processed": self.state.bar_count,
            "current_regime": self.state.current_regime,
            "current_label": self.labels.get(self.state.current_regime, "unknown"),
            "confidence": round(self.state.confidence, 4),
            "entropy": round(self.state.entropy, 4),
            "regime_duration": self.state.regime_duration,
            "expected_remaining_duration": round(self.expected_regime_duration(), 1),
            "persistence_prob_5bar": round(self.regime_persistence_prob(5), 4),
            "next_likely_regime": next_label,
            "next_regime_prob": round(next_prob, 4),
            "total_change_events": len(self._change_events),
            "cumulative_log_likelihood": round(self.state.log_likelihood, 2),
        }


def _logsumexp(x: np.ndarray) -> float:
    """Numerically stable log-sum-exp."""
    c = np.max(x)
    if np.isinf(c):
        return float("-inf")
    return float(c + np.log(np.sum(np.exp(x - c))))
