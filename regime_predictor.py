"""
regime_predictor.py — Forward-looking regime transition forecasting.

Fuses three complementary mathematical frameworks to predict upcoming
regime changes BEFORE they happen:

1. Chapman-Kolmogorov k-step forecasting from the HMM transition matrix
2. Bayesian Online Changepoint Detection (BOCPD) for structural break detection
3. Feature momentum scoring (leading indicators that precede regime shifts)

The ensemble produces a unified "Regime Stress Index" (0-1) that quantifies
how likely the current regime is to change in the near future.
"""

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm
from dataclasses import dataclass


@dataclass
class RegimeForecast:
    """Forward-looking regime prediction at a single point in time."""
    bar: int
    current_regime: str
    # k-step ahead probabilities from Chapman-Kolmogorov
    forecast_probs: dict[int, dict[str, float]]  # {horizon: {regime: prob}}
    # Most likely regime at each horizon
    predicted_regime: dict[int, str]  # {horizon: regime_label}
    # Probability of leaving current regime within k steps
    transition_prob: dict[int, float]  # {horizon: P(leave current)}
    # BOCPD changepoint probability
    changepoint_prob: float
    # Feature momentum score (0-1, higher = more stress)
    momentum_score: float
    # Ensemble regime stress index (0-1)
    regime_stress: float


class BayesianChangepoint:
    """
    Bayesian Online Changepoint Detection (Adams & MacKay, 2007).

    Maintains a posterior over run lengths (how long since the last
    changepoint) and updates it online as each new observation arrives.
    When P(run_length=0) spikes, a changepoint is likely.
    """

    def __init__(self, hazard_rate: float = 1 / 100, mu_0: float = 0.0,
                 kappa_0: float = 1.0, alpha_0: float = 1.0, beta_0: float = 1.0):
        """
        Args:
            hazard_rate: Prior probability of a changepoint at any step (1/expected_run_length).
            mu_0, kappa_0, alpha_0, beta_0: Normal-Inverse-Gamma prior hyperparameters.
        """
        self.hazard = hazard_rate
        self.mu_0 = mu_0
        self.kappa_0 = kappa_0
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0

    def run(self, data: np.ndarray) -> np.ndarray:
        """
        Run BOCPD on a 1-D time series.

        Returns array of shape (T,) with P(changepoint at t) for each time step.
        This is the posterior probability that a changepoint occurred at each bar.
        """
        T = len(data)
        if T == 0:
            return np.array([])

        # Run-length posterior: R[t, r] = P(run_length=r at time t)
        # We only need the current and previous time step
        max_rl = T + 1

        # Sufficient statistics for each run length
        mu = np.full(max_rl, self.mu_0)
        kappa = np.full(max_rl, self.kappa_0)
        alpha = np.full(max_rl, self.alpha_0)
        beta = np.full(max_rl, self.beta_0)

        # Log run-length probabilities
        log_R = np.full(max_rl, -np.inf)
        log_R[0] = 0.0  # Start with run length 0

        changepoint_prob = np.zeros(T)
        log_H = np.log(self.hazard)
        log_1mH = np.log(1 - self.hazard)

        for t in range(T):
            x = data[t]

            # Predictive probability for each run length (Student-t)
            # Using the conjugate Normal-Inverse-Gamma update
            active = t + 1  # number of active run lengths
            pred_log_prob = np.full(active, -np.inf)

            for r in range(active):
                # Student-t predictive: p(x | mu_r, kappa_r, alpha_r, beta_r)
                scale = np.sqrt(beta[r] * (kappa[r] + 1) / (alpha[r] * kappa[r]))
                if scale <= 0 or np.isnan(scale):
                    scale = 1.0
                dof = 2 * alpha[r]
                if dof <= 0:
                    dof = 1.0
                # Approximate Student-t with normal for speed (good for dof > ~5)
                pred_log_prob[r] = norm.logpdf(x, loc=mu[r], scale=scale)

            # Growth probabilities (no changepoint)
            log_growth = log_R[:active] + pred_log_prob + log_1mH

            # Changepoint probability (run length resets to 0)
            log_cp = logsumexp(log_R[:active] + pred_log_prob + log_H)

            # Update run-length posterior
            new_log_R = np.full(active + 1, -np.inf)
            new_log_R[0] = log_cp
            new_log_R[1:active + 1] = log_growth

            # Normalize
            log_evidence = logsumexp(new_log_R[:active + 1])
            new_log_R[:active + 1] -= log_evidence

            log_R[:active + 1] = new_log_R[:active + 1]

            # P(changepoint at t) = P(run_length = 0 at t)
            changepoint_prob[t] = np.exp(new_log_R[0])

            # Update sufficient statistics for each run length
            # Shift: new run lengths get updated stats
            new_mu = np.full(active + 1, self.mu_0)
            new_kappa = np.full(active + 1, self.kappa_0)
            new_alpha = np.full(active + 1, self.alpha_0)
            new_beta = np.full(active + 1, self.beta_0)

            for r in range(active):
                # Bayesian update for Normal-Inverse-Gamma
                new_kappa[r + 1] = kappa[r] + 1
                new_mu[r + 1] = (kappa[r] * mu[r] + x) / new_kappa[r + 1]
                new_alpha[r + 1] = alpha[r] + 0.5
                new_beta[r + 1] = beta[r] + 0.5 * kappa[r] * (x - mu[r]) ** 2 / new_kappa[r + 1]

            mu[:active + 1] = new_mu[:active + 1]
            kappa[:active + 1] = new_kappa[:active + 1]
            alpha[:active + 1] = new_alpha[:active + 1]
            beta[:active + 1] = new_beta[:active + 1]

        return changepoint_prob


class RegimePredictor:
    """
    Forward-looking regime predictor combining HMM dynamics,
    Bayesian changepoint detection, and feature momentum.
    """

    def __init__(self, horizons: tuple[int, ...] = (1, 5, 10, 20),
                 bocpd_hazard: float = 1 / 100,
                 momentum_window: int = 10,
                 stress_weights: tuple[float, float, float] = (0.4, 0.35, 0.25)):
        """
        Args:
            horizons: Forecast horizons (bars ahead).
            bocpd_hazard: BOCPD hazard rate (1/expected_run_length).
            momentum_window: Window for feature momentum computation.
            stress_weights: (chapman_kolmogorov_weight, bocpd_weight, momentum_weight)
                for the ensemble stress index.
        """
        self.horizons = horizons
        self.bocpd_hazard = bocpd_hazard
        self.momentum_window = momentum_window
        w = np.array(stress_weights)
        self.stress_weights = w / w.sum()  # normalize to sum to 1

    def chapman_kolmogorov_forecast(
        self,
        transmat: np.ndarray,
        current_posterior: np.ndarray,
        labels: dict[int, str],
        horizons: tuple[int, ...] | None = None,
    ) -> dict[int, dict[str, float]]:
        """
        Compute k-step-ahead regime probabilities using Chapman-Kolmogorov.

        P(state_{t+k} | observations_{1:t}) = current_posterior @ transmat^k

        Args:
            transmat: (n_states, n_states) transition matrix.
            current_posterior: (n_states,) posterior at current time.
            labels: {state_id: label_string} mapping.
            horizons: Override horizons if provided.

        Returns:
            {horizon: {regime_label: probability}} for each horizon.
        """
        if horizons is None:
            horizons = self.horizons

        forecasts = {}
        for k in horizons:
            # Matrix power: transmat^k
            mat_k = np.linalg.matrix_power(transmat, k)
            # Forward projection
            probs_k = current_posterior @ mat_k
            # Normalize (numerical safety)
            probs_k = np.clip(probs_k, 0, None)
            total = probs_k.sum()
            if total > 0:
                probs_k /= total

            forecasts[k] = {
                labels.get(i, f"state_{i}"): float(probs_k[i])
                for i in range(len(probs_k))
            }

        return forecasts

    def transition_probability(
        self,
        transmat: np.ndarray,
        current_posterior: np.ndarray,
        current_state: int,
        horizons: tuple[int, ...] | None = None,
    ) -> dict[int, float]:
        """
        Compute P(leaving current regime within k steps).

        This answers: "How likely is it that we'll be in a DIFFERENT
        regime k bars from now?"
        """
        if horizons is None:
            horizons = self.horizons

        result = {}
        for k in horizons:
            mat_k = np.linalg.matrix_power(transmat, k)
            probs_k = current_posterior @ mat_k
            probs_k = np.clip(probs_k, 0, None)
            total = probs_k.sum()
            if total > 0:
                probs_k /= total
            # P(leave) = 1 - P(stay in current state)
            result[k] = float(1.0 - probs_k[current_state])

        return result

    def run_bocpd(self, log_returns: np.ndarray) -> np.ndarray:
        """
        Run Bayesian Online Changepoint Detection on log returns.

        Returns array of P(changepoint) at each bar.
        """
        bocpd = BayesianChangepoint(hazard_rate=self.bocpd_hazard)
        return bocpd.run(log_returns)

    def compute_momentum_score(
        self,
        entropy: np.ndarray,
        volatility: np.ndarray,
        volume_change: np.ndarray,
        window: int | None = None,
    ) -> np.ndarray:
        """
        Compute feature momentum score (0-1) from leading indicators.

        Combines three signals that empirically precede regime transitions:
        1. Entropy acceleration: d²(entropy)/dt² — rising uncertainty
        2. Volatility surge: rate of change in rolling vol
        3. Volume spike: abnormal volume activity

        Each component is normalized to [0, 1] and averaged.
        """
        if window is None:
            window = self.momentum_window
        T = len(entropy)
        scores = np.zeros(T)

        if T < window + 2:
            return scores

        # 1. Entropy acceleration (second derivative)
        ent_diff1 = np.diff(entropy, prepend=entropy[0])
        ent_diff2 = np.diff(ent_diff1, prepend=ent_diff1[0])
        # Smooth with rolling mean
        ent_accel = pd.Series(np.abs(ent_diff2)).rolling(window, min_periods=1).mean().values

        # 2. Volatility rate of change
        vol_roc = np.zeros(T)
        for t in range(window, T):
            prev_vol = np.mean(volatility[t - window:t])
            if prev_vol > 0:
                vol_roc[t] = (volatility[t] - prev_vol) / prev_vol

        # 3. Volume surge (absolute value of volume_change, smoothed)
        vol_surge = pd.Series(np.abs(volume_change)).rolling(window, min_periods=1).mean().values

        # Normalize each to [0, 1] using percentile ranks
        def percentile_normalize(arr: np.ndarray) -> np.ndarray:
            valid = arr[~np.isnan(arr)]
            if len(valid) == 0 or np.std(valid) == 0:
                return np.zeros_like(arr)
            ranks = np.zeros_like(arr)
            for i, val in enumerate(arr):
                if np.isnan(val):
                    ranks[i] = 0.0
                else:
                    ranks[i] = np.searchsorted(np.sort(valid), val) / len(valid)
            return np.clip(ranks, 0, 1)

        norm_ent = percentile_normalize(ent_accel)
        norm_vol = percentile_normalize(np.abs(vol_roc))
        norm_surge = percentile_normalize(vol_surge)

        # Equal-weighted combination
        scores = (norm_ent + norm_vol + norm_surge) / 3.0
        return scores

    def compute_regime_stress(
        self,
        transition_probs: dict[int, float],
        changepoint_prob: float,
        momentum_score: float,
        reference_horizon: int = 5,
    ) -> float:
        """
        Compute ensemble Regime Stress Index (0-1).

        Combines:
        - Chapman-Kolmogorov transition probability at reference horizon
        - BOCPD changepoint probability
        - Feature momentum score

        Higher stress = higher probability of imminent regime change.
        """
        ck_signal = transition_probs.get(reference_horizon, 0.0)

        # Weighted combination
        stress = (
            self.stress_weights[0] * ck_signal
            + self.stress_weights[1] * min(changepoint_prob * 3.0, 1.0)  # scale up BOCPD
            + self.stress_weights[2] * momentum_score
        )
        return float(np.clip(stress, 0, 1))

    def predict(
        self,
        transmat: np.ndarray,
        posteriors: np.ndarray,
        states: np.ndarray,
        labels: dict[int, str],
        entropy: np.ndarray,
        log_returns: np.ndarray,
        volatility: np.ndarray,
        volume_change: np.ndarray,
    ) -> list[RegimeForecast]:
        """
        Generate forward-looking regime predictions for every bar.

        This is the main entry point. Returns a RegimeForecast for each
        time step with k-step ahead probabilities, transition likelihoods,
        changepoint detection, and the ensemble stress index.
        """
        T = len(states)

        # Run BOCPD on log returns
        cp_probs = self.run_bocpd(log_returns)

        # Compute momentum scores
        momentum = self.compute_momentum_score(entropy, volatility, volume_change)

        forecasts = []
        for t in range(T):
            current_state = int(states[t])
            current_label = labels.get(current_state, f"state_{current_state}")
            current_post = posteriors[t]

            # Chapman-Kolmogorov forecasts
            forecast_probs = self.chapman_kolmogorov_forecast(
                transmat, current_post, labels
            )

            # Predicted regime at each horizon (argmax)
            predicted_regime = {}
            for k, probs in forecast_probs.items():
                predicted_regime[k] = max(probs, key=probs.get)

            # Transition probability
            trans_prob = self.transition_probability(
                transmat, current_post, current_state
            )

            # Ensemble stress
            stress = self.compute_regime_stress(
                trans_prob, cp_probs[t], momentum[t]
            )

            forecasts.append(RegimeForecast(
                bar=t,
                current_regime=current_label,
                forecast_probs=forecast_probs,
                predicted_regime=predicted_regime,
                transition_prob=trans_prob,
                changepoint_prob=float(cp_probs[t]),
                momentum_score=float(momentum[t]),
                regime_stress=float(stress),
            ))

        return forecasts

    def forecast_summary(self, forecasts: list[RegimeForecast]) -> pd.DataFrame:
        """
        Convert forecasts to a summary DataFrame for visualization.
        """
        rows = []
        for f in forecasts:
            row = {
                "bar": f.bar,
                "current_regime": f.current_regime,
                "changepoint_prob": f.changepoint_prob,
                "momentum_score": f.momentum_score,
                "regime_stress": f.regime_stress,
            }
            for k in self.horizons:
                row[f"transition_prob_{k}"] = f.transition_prob.get(k, np.nan)
                row[f"predicted_regime_{k}"] = f.predicted_regime.get(k, "")
            rows.append(row)

        return pd.DataFrame(rows)

    def stress_alert_level(self, stress: float) -> tuple[str, str]:
        """
        Convert stress index to human-readable alert level and color.

        Returns (level_name, hex_color).
        """
        if stress >= 0.75:
            return "CRITICAL", "#ef4444"
        elif stress >= 0.55:
            return "ELEVATED", "#f59e0b"
        elif stress >= 0.35:
            return "MODERATE", "#3b82f6"
        else:
            return "LOW", "#00e599"

    def calibration_backtest(
        self,
        forecasts: list[RegimeForecast],
        actual_states: np.ndarray,
        labels: dict[int, str],
    ) -> pd.DataFrame:
        """
        Backtest prediction accuracy: for each horizon, compute what fraction
        of predictions at each confidence level actually matched the realized regime.

        Returns DataFrame with calibration statistics per horizon.
        """
        rows = []
        for k in self.horizons:
            correct = 0
            total = 0
            stress_when_correct = []
            stress_when_wrong = []

            for f in forecasts:
                future_bar = f.bar + k
                if future_bar >= len(actual_states):
                    continue

                actual_label = labels.get(int(actual_states[future_bar]), "unknown")
                predicted = f.predicted_regime.get(k, "")
                total += 1

                if actual_label == predicted:
                    correct += 1
                    stress_when_correct.append(f.regime_stress)
                else:
                    stress_when_wrong.append(f.regime_stress)

            accuracy = correct / total if total > 0 else 0.0
            rows.append({
                "horizon": k,
                "accuracy": accuracy,
                "n_predictions": total,
                "avg_stress_correct": np.mean(stress_when_correct) if stress_when_correct else np.nan,
                "avg_stress_wrong": np.mean(stress_when_wrong) if stress_when_wrong else np.nan,
            })

        return pd.DataFrame(rows)
