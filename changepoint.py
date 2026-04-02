"""
changepoint.py — Bayesian Online Changepoint Detection (BOCPD).

Implements the Adams & MacKay (2007) algorithm for real-time regime
change detection. Runs bar-by-bar maintaining a probability distribution
over run lengths (time since last changepoint). When P(run_length=0)
spikes, a changepoint is detected — often BEFORE the batch HMM transitions.

This creates a dual-detection architecture:
  - HMM: identifies WHICH regime we're in (bull, bear, crash, etc.)
  - BOCPD: detects WHEN transitions happen, with lower latency

Key components:
  - StudentTPredictive: conjugate Bayesian model for streaming data
    with unknown mean and variance (robust to outliers)
  - BayesianChangepointDetector: core BOCPD algorithm
  - HMM-BOCPD fusion: combines both detectors for enhanced signals
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.special import gammaln, logsumexp


class StudentTPredictive:
    """
    Bayesian conjugate predictive model for univariate Gaussian
    observations with unknown mean and variance.

    Prior: Normal-Inverse-Gamma(mu0, kappa0, alpha0, beta0)

    The predictive distribution at each step is a Student-t, which
    naturally accounts for parameter uncertainty and is robust to
    outliers — critical for financial data.

    Sufficient statistics are updated incrementally, making this
    O(1) per observation per run length.
    """

    def __init__(self, mu0: float = 0.0, kappa0: float = 0.01,
                 alpha0: float = 0.5, beta0: float = 0.01):
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0

    def log_predictive(
        self,
        x: float,
        mu: np.ndarray,
        kappa: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
    ) -> np.ndarray:
        """
        Compute log P(x | sufficient_stats) for each run length.

        The predictive distribution is Student-t with:
          df = 2 * alpha
          loc = mu
          scale = sqrt(beta * (kappa + 1) / (alpha * kappa))

        Returns log-probability array of shape (n_run_lengths,).
        """
        df = 2.0 * alpha
        scale_sq = beta * (kappa + 1.0) / (alpha * kappa)
        scale = np.sqrt(scale_sq)

        # Student-t log PDF
        z = (x - mu) / scale
        log_prob = (
            gammaln((df + 1.0) / 2.0)
            - gammaln(df / 2.0)
            - 0.5 * np.log(df * np.pi)
            - np.log(scale)
            - ((df + 1.0) / 2.0) * np.log(1.0 + z**2 / df)
        )
        return log_prob

    def update_suffstats(
        self,
        x: float,
        mu: np.ndarray,
        kappa: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Update sufficient statistics after observing x.

        Normal-Inverse-Gamma conjugate update:
          kappa' = kappa + 1
          mu' = (kappa * mu + x) / kappa'
          alpha' = alpha + 0.5
          beta' = beta + 0.5 * kappa * (x - mu)^2 / kappa'
        """
        kappa_new = kappa + 1.0
        mu_new = (kappa * mu + x) / kappa_new
        alpha_new = alpha + 0.5
        beta_new = beta + 0.5 * kappa * (x - mu) ** 2 / kappa_new
        return mu_new, kappa_new, alpha_new, beta_new


@dataclass
class ChangepointResult:
    """Container for BOCPD outputs."""
    # Per-bar arrays (length T)
    changepoint_prob: np.ndarray      # P(run_length=0) at each bar
    map_run_length: np.ndarray        # most probable run length per bar
    expected_run_length: np.ndarray   # E[run_length] per bar
    detected_changepoints: np.ndarray  # boolean: True where CP detected
    changepoint_bars: list[int]       # bar indices of detected changepoints

    # Full run-length distribution (optional, for visualization)
    run_length_posterior: np.ndarray | None = None  # (T, max_run_len)

    # Comparison with HMM transitions
    hmm_transition_bars: list[int] = field(default_factory=list)
    early_detections: list[dict] = field(default_factory=list)
    detection_lead_bars: list[int] = field(default_factory=list)


class BayesianChangepointDetector:
    """
    Bayesian Online Changepoint Detection (Adams & MacKay 2007).

    At each timestep t, maintains the full posterior over run lengths:
      P(r_t | x_{1:t})

    The algorithm:
      1. Evaluate predictive probability P(x_t | r_{t-1}, x) for each run length
      2. Growth: P(r_t = r_{t-1}+1) ∝ P(x_t | r_{t-1}) * P(r_{t-1}) * (1-H)
      3. Changepoint: P(r_t = 0) ∝ sum over all r_{t-1} of P(x_t | r_{t-1}) * P(r_{t-1}) * H
      4. Normalize to get P(r_t | x_{1:t})

    where H is the hazard rate (prior probability of changepoint at any step).
    """

    def __init__(self, config: dict):
        cp_cfg = config.get("changepoint", {})
        self.hazard_rate = cp_cfg.get("hazard_rate", 1.0 / 100)
        self.threshold = cp_cfg.get("threshold", 0.5)
        self.min_run_length = cp_cfg.get("min_run_length", 10)
        self.max_run_length = cp_cfg.get("max_run_length", 500)
        self.feature_index = cp_cfg.get("feature_index", 0)  # log_return by default
        self.store_run_length_dist = cp_cfg.get("store_run_length_dist", True)

        prior = cp_cfg.get("prior", {})
        self.predictive = StudentTPredictive(
            mu0=prior.get("mu0", 0.0),
            kappa0=prior.get("kappa0", 0.01),
            alpha0=prior.get("alpha0", 0.5),
            beta0=prior.get("beta0", 0.01),
        )

    def detect(self, X: np.ndarray) -> ChangepointResult:
        """
        Run BOCPD on a 1-D or 2-D observation sequence.

        Internally normalizes data to z-scores so that the Student-t
        prior is well-calibrated regardless of the data's native scale.
        This is critical for financial data where returns are O(1e-3).

        Parameters
        ----------
        X : array of shape (T,) or (T, D)
            If 2-D, uses column self.feature_index for detection.

        Returns
        -------
        ChangepointResult with all detection outputs.
        """
        if X.ndim == 2:
            x = X[:, self.feature_index].copy()
        else:
            x = X.copy()

        # Normalize to z-scores for prior calibration
        x_mean = np.mean(x)
        x_std = np.std(x)
        if x_std > 0:
            x = (x - x_mean) / x_std

        T = len(x)
        max_rl = min(self.max_run_length, T)

        # Sufficient statistics arrays — index i corresponds to run length i
        # We maintain arrays of size (max_rl + 1,) and shift them each step
        mu = np.full(max_rl + 1, self.predictive.mu0)
        kappa = np.full(max_rl + 1, self.predictive.kappa0)
        alpha = np.full(max_rl + 1, self.predictive.alpha0)
        beta = np.full(max_rl + 1, self.predictive.beta0)

        # Run-length log probabilities
        log_R = np.full(max_rl + 1, -np.inf)
        log_R[0] = 0.0  # start with run length 0, probability 1

        log_hazard = np.log(self.hazard_rate)
        log_1m_hazard = np.log(1.0 - self.hazard_rate)

        # Output arrays
        cp_prob = np.zeros(T)
        map_rl = np.zeros(T, dtype=int)
        expected_rl = np.zeros(T)

        if self.store_run_length_dist:
            rl_posterior = np.zeros((T, min(max_rl + 1, T + 1)))
        else:
            rl_posterior = None

        for t in range(T):
            # Current run lengths in play: 0 to min(t, max_rl)
            active = min(t + 1, max_rl + 1)

            # Step 1: Predictive probabilities for each active run length
            log_pred = self.predictive.log_predictive(
                x[t],
                mu[:active],
                kappa[:active],
                alpha[:active],
                beta[:active],
            )

            # Step 2: Growth probabilities (run continues)
            log_growth = log_R[:active] + log_pred + log_1m_hazard

            # Step 3: Changepoint probability (new run starts)
            log_cp = logsumexp(log_R[:active] + log_pred + log_hazard)

            # Step 4: Construct new run-length distribution
            new_log_R = np.full(max_rl + 1, -np.inf)
            new_log_R[0] = log_cp
            # Shift growth probs: run length i was i-1, now becomes i
            limit = min(active, max_rl)
            new_log_R[1:limit + 1] = log_growth[:limit]

            # Normalize
            active_new = min(t + 2, max_rl + 1)
            log_evidence = logsumexp(new_log_R[:active_new])
            new_log_R[:active_new] -= log_evidence

            # Store outputs
            R_probs = np.exp(new_log_R[:active_new])
            cp_prob[t] = R_probs[0]
            map_rl[t] = np.argmax(R_probs)
            expected_rl[t] = np.sum(np.arange(active_new) * R_probs)

            if rl_posterior is not None:
                store_len = min(active_new, rl_posterior.shape[1])
                rl_posterior[t, :store_len] = R_probs[:store_len]

            # Step 5: Update sufficient statistics
            mu_new, kappa_new, alpha_new, beta_new = (
                self.predictive.update_suffstats(
                    x[t], mu[:active], kappa[:active],
                    alpha[:active], beta[:active],
                )
            )

            # Shift: run length i's stats move to position i+1
            new_mu = np.full(max_rl + 1, self.predictive.mu0)
            new_kappa = np.full(max_rl + 1, self.predictive.kappa0)
            new_alpha = np.full(max_rl + 1, self.predictive.alpha0)
            new_beta = np.full(max_rl + 1, self.predictive.beta0)

            limit = min(active, max_rl)
            new_mu[1:limit + 1] = mu_new[:limit]
            new_kappa[1:limit + 1] = kappa_new[:limit]
            new_alpha[1:limit + 1] = alpha_new[:limit]
            new_beta[1:limit + 1] = beta_new[:limit]

            mu, kappa, alpha, beta = new_mu, new_kappa, new_alpha, new_beta
            log_R = new_log_R

        # Detect changepoints using multiple signals:
        # 1. MAP run-length reset: MAP(r_t) drops to near zero
        # 2. Expected run-length collapse: E[r_t] drops sharply
        # 3. Short run-length mass: P(r_t < k) exceeds threshold
        detected = np.zeros(T, dtype=bool)

        for t in range(self.min_run_length, T):
            # Signal 1: MAP run length resets (drops to < min_run_length)
            # while previous MAP was substantial
            map_reset = (
                map_rl[t] < self.min_run_length
                and map_rl[t - 1] >= self.min_run_length
            )

            # Signal 2: Expected run length drops by > 50%
            if expected_rl[t - 1] > 0:
                rl_drop = (expected_rl[t] / expected_rl[t - 1]) < 0.5
            else:
                rl_drop = False

            # Signal 3: Mass concentration on short run lengths
            if rl_posterior is not None:
                short_mass = rl_posterior[t, :self.min_run_length].sum()
                short_mass_high = short_mass > self.threshold
            else:
                short_mass_high = False

            if map_reset or (rl_drop and short_mass_high):
                detected[t] = True

        cp_bars = list(np.where(detected)[0])

        return ChangepointResult(
            changepoint_prob=cp_prob,
            map_run_length=map_rl,
            expected_run_length=expected_rl,
            detected_changepoints=detected,
            changepoint_bars=cp_bars,
            run_length_posterior=rl_posterior,
        )

    def compare_with_hmm(
        self,
        cp_result: ChangepointResult,
        hmm_states: np.ndarray,
        max_lead: int = 20,
    ) -> ChangepointResult:
        """
        Compare BOCPD detections against HMM regime transitions.

        Finds cases where BOCPD detected a changepoint BEFORE the HMM
        switched states, measuring the detection lead in bars.

        Parameters
        ----------
        cp_result : ChangepointResult
            Output from detect().
        hmm_states : array of shape (T,)
            HMM Viterbi-decoded state sequence.
        max_lead : int
            Maximum bars to look ahead for matching HMM transition.

        Returns
        -------
        Updated ChangepointResult with HMM comparison fields populated.
        """
        T = len(hmm_states)

        # Find HMM transition bars
        hmm_transitions = []
        for t in range(1, T):
            if hmm_states[t] != hmm_states[t - 1]:
                hmm_transitions.append(t)

        cp_result.hmm_transition_bars = hmm_transitions

        # For each BOCPD changepoint, find nearest future HMM transition
        early_detections = []
        lead_bars = []

        for cp_bar in cp_result.changepoint_bars:
            best_match = None
            best_lead = None

            for hmm_bar in hmm_transitions:
                lead = hmm_bar - cp_bar
                if 0 < lead <= max_lead:
                    if best_lead is None or lead < best_lead:
                        best_match = hmm_bar
                        best_lead = lead

            if best_match is not None:
                early_detections.append({
                    "bocpd_bar": cp_bar,
                    "hmm_bar": best_match,
                    "lead_bars": best_lead,
                    "cp_probability": float(cp_result.changepoint_prob[cp_bar]),
                })
                lead_bars.append(best_lead)

        cp_result.early_detections = early_detections
        cp_result.detection_lead_bars = lead_bars

        return cp_result

    def compute_regime_stability(
        self,
        expected_run_length: np.ndarray,
        window: int = 20,
    ) -> np.ndarray:
        """
        Compute a rolling regime stability score from expected run length.

        Stability = sigmoid(E[run_length] / window), normalized to [0, 1].
        High stability means the current regime has been consistent for many bars.
        Low stability (short expected run length) suggests recent or imminent transition.

        Useful as an independent confidence signal alongside HMM entropy.
        """
        # Normalize expected run length by window to get a 0-1 score
        stability = 1.0 - np.exp(-expected_run_length / window)
        return np.clip(stability, 0.0, 1.0)

    def fuse_with_hmm_confidence(
        self,
        hmm_confidence: np.ndarray,
        cp_result: ChangepointResult,
        stability_weight: float = 0.3,
    ) -> np.ndarray:
        """
        Fuse HMM entropy-based confidence with BOCPD regime stability
        to produce an enhanced confidence score.

        When both HMM and BOCPD agree the regime is stable, confidence
        amplifies. When BOCPD detects instability (short run length or
        high changepoint probability), confidence is reduced even if
        HMM posteriors look confident — providing early de-risking.

        fused = (1 - w) * hmm_confidence + w * bocpd_stability
               * penalty_for_high_cp_prob

        Parameters
        ----------
        hmm_confidence : array (T,)
            Entropy-based confidence from HMM.
        cp_result : ChangepointResult
            BOCPD outputs.
        stability_weight : float
            Weight given to BOCPD stability signal.

        Returns
        -------
        Enhanced confidence array (T,).
        """
        stability = self.compute_regime_stability(cp_result.expected_run_length)

        # Penalty: reduce confidence when changepoint probability is high
        cp_penalty = 1.0 - cp_result.changepoint_prob

        bocpd_signal = stability * cp_penalty

        fused = (
            (1.0 - stability_weight) * hmm_confidence
            + stability_weight * bocpd_signal
        )
        return np.clip(fused, 0.0, 1.0)
