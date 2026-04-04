"""
regime_predictor.py — Predictive regime forecasting with cross-asset intelligence.

Uses the HMM transition matrix for N-step-ahead regime probability forecasts
(Chapman-Kolmogorov), enhanced with cross-asset macro leading indicators
(VIX, yield curve, credit spreads, USD, gold) via Bayesian fusion.

Core math:
  P(s_{t+n}) = posterior_t @ A^n        (Chapman-Kolmogorov)
  P(shift)   = 1 - (A^n)[s_t, s_t]     (regime shift probability)
  E[dur]     = 1 / (1 - a_{ii})         (expected remaining duration, geometric)

Cross-asset fusion:
  Macro features -> separate HMM -> macro regime posteriors
  -> Bayesian update on single-asset forecast using likelihood weighting.
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from hmmlearn.hmm import GaussianHMM


# ── Cross-Asset Macro Data ──────────────────────────────────────────────────

# Default macro tickers: VIX, 10Y yield, high-yield spread proxy, USD, gold
MACRO_TICKERS = {
    "vix": "^VIX",
    "tnx": "^TNX",        # 10-year treasury yield
    "hyg": "HYG",         # high-yield corporate bond ETF
    "lqd": "LQD",         # investment-grade bond ETF
    "usd": "DX-Y.NYB",    # US dollar index
    "gold": "GC=F",       # gold futures
}

# Fallback tickers if primary ones fail (e.g. forex issues)
MACRO_FALLBACKS = {
    "usd": "UUP",         # USD bull ETF
    "gold": "GLD",        # gold ETF
}


class MacroFeatureCollector:
    """Fetch cross-asset macro data and engineer leading indicator features."""

    def __init__(self, tickers: dict[str, str] | None = None):
        self.tickers = tickers or MACRO_TICKERS

    def fetch(
        self,
        interval: str = "1d",
        lookback_days: int = 365,
    ) -> pd.DataFrame:
        """
        Fetch macro data and compute features:
          - vix_level: VIX close (fear gauge)
          - vix_change: 5-bar VIX change (fear acceleration)
          - yield_level: 10Y yield
          - yield_change: 5-bar yield change (rate momentum)
          - credit_spread: HYG/LQD ratio change (credit stress)
          - usd_momentum: 10-bar USD change (dollar strength)
          - gold_momentum: 10-bar gold change (safe-haven demand)

        Returns DataFrame aligned to a common date index.
        """
        end = datetime.now()
        start = end - timedelta(days=lookback_days)

        raw = {}
        for name, tick in self.tickers.items():
            try:
                df = yf.download(tick, start=start, end=end, interval=interval, progress=False)
                if df.empty and name in MACRO_FALLBACKS:
                    df = yf.download(MACRO_FALLBACKS[name], start=start, end=end,
                                     interval=interval, progress=False)
                if not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    if df.columns.duplicated().any():
                        df = df.loc[:, ~df.columns.duplicated()]
                    raw[name] = df["Close"]
            except Exception:
                if name in MACRO_FALLBACKS:
                    try:
                        df = yf.download(MACRO_FALLBACKS[name], start=start, end=end,
                                         interval=interval, progress=False)
                        if not df.empty:
                            if isinstance(df.columns, pd.MultiIndex):
                                df.columns = df.columns.get_level_values(0)
                            if df.columns.duplicated().any():
                                df = df.loc[:, ~df.columns.duplicated()]
                            raw[name] = df["Close"]
                    except Exception:
                        continue

        if not raw:
            raise ValueError("Failed to fetch any macro data")

        macro = pd.DataFrame(raw)
        macro.dropna(how="all", inplace=True)
        macro.ffill(inplace=True)

        features = pd.DataFrame(index=macro.index)

        if "vix" in macro.columns:
            features["vix_level"] = macro["vix"]
            features["vix_change"] = macro["vix"].pct_change(5)

        if "tnx" in macro.columns:
            features["yield_level"] = macro["tnx"]
            features["yield_change"] = macro["tnx"].diff(5)

        if "hyg" in macro.columns and "lqd" in macro.columns:
            spread_ratio = macro["hyg"] / macro["lqd"]
            features["credit_spread"] = spread_ratio.pct_change(5)

        if "usd" in macro.columns:
            features["usd_momentum"] = macro["usd"].pct_change(10)

        if "gold" in macro.columns:
            features["gold_momentum"] = macro["gold"].pct_change(10)

        features.dropna(inplace=True)
        return features

    def compute_macro_stress_index(self, features: pd.DataFrame) -> pd.Series:
        """
        Composite macro stress index: z-score average of stress indicators.
        Higher = more stress (risk-off environment).

        Components (sign-adjusted so positive = stress):
          +VIX level, +VIX change, -credit spread (HYG/LQD falling = stress),
          -yield change (falling yields = flight to safety), +gold momentum.
        """
        stress_components = []

        for col, sign in [
            ("vix_level", 1),
            ("vix_change", 1),
            ("credit_spread", -1),
            ("yield_change", -1),
            ("gold_momentum", 1),
        ]:
            if col in features.columns:
                series = features[col] * sign
                z = (series - series.mean()) / (series.std() + 1e-10)
                stress_components.append(z)

        if not stress_components:
            return pd.Series(0.0, index=features.index, name="macro_stress")

        stress = pd.concat(stress_components, axis=1).mean(axis=1)
        stress.name = "macro_stress"
        return stress


# ── Regime Forecasting Engine ───────────────────────────────────────────────

class RegimePredictor:
    """
    N-step-ahead regime probability forecasting using Chapman-Kolmogorov,
    with optional cross-asset Bayesian fusion.
    """

    def __init__(self, config: dict | None = None):
        pred_cfg = (config or {}).get("predictor", {})
        self.max_horizon = pred_cfg.get("max_horizon", 20)
        self.macro_weight = pred_cfg.get("macro_weight", 0.3)
        self.macro_n_states = pred_cfg.get("macro_n_states", 3)
        self.macro_n_restarts = pred_cfg.get("macro_n_restarts", 10)
        self.stress_thresholds = pred_cfg.get(
            "stress_thresholds", {"low": -0.5, "high": 0.5}
        )

    def forecast_regime_probs(
        self,
        transmat: np.ndarray,
        current_posteriors: np.ndarray,
        horizons: list[int] | None = None,
    ) -> dict[int, np.ndarray]:
        """
        N-step-ahead regime probability forecast via Chapman-Kolmogorov.

        P(s_{t+n}) = posterior_t @ A^n

        Args:
            transmat: (K, K) transition matrix from fitted HMM.
            current_posteriors: (K,) posterior probabilities at current time.
            horizons: list of forecast horizons [1, 2, 5, 10, 20].

        Returns:
            Dict mapping horizon -> (K,) probability vector.
        """
        if horizons is None:
            horizons = [1, 2, 5, 10, self.max_horizon]
        horizons = [h for h in horizons if h <= self.max_horizon]

        forecasts = {}
        for n in horizons:
            A_n = np.linalg.matrix_power(transmat, n)
            forecasts[n] = current_posteriors @ A_n

        return forecasts

    def regime_shift_probability(
        self,
        transmat: np.ndarray,
        current_state: int,
        horizons: list[int] | None = None,
    ) -> dict[int, float]:
        """
        Probability of leaving the current regime within n steps.

        P(shift by n) = 1 - (A^n)[current_state, current_state]

        This uses the full transition dynamics, not just (1-a_ii)^n,
        because re-entry is possible (leave and come back).
        """
        if horizons is None:
            horizons = [1, 2, 5, 10, self.max_horizon]

        shifts = {}
        for n in horizons:
            A_n = np.linalg.matrix_power(transmat, n)
            shifts[n] = 1.0 - A_n[current_state, current_state]

        return shifts

    def expected_regime_duration(
        self,
        transmat: np.ndarray,
        current_state: int,
        bars_in_regime: int = 0,
    ) -> dict:
        """
        Expected remaining duration in current regime.

        For a geometric distribution (memoryless):
          E[duration] = 1 / (1 - a_ii)
          E[remaining] = E[duration]  (memoryless property)

        Also computes median remaining and 90th percentile.
        """
        a_ii = transmat[current_state, current_state]
        exit_prob = 1.0 - a_ii

        if exit_prob < 1e-10:
            return {
                "expected_total": float("inf"),
                "expected_remaining": float("inf"),
                "median_remaining": float("inf"),
                "p90_remaining": float("inf"),
                "exit_prob_per_bar": 0.0,
                "bars_elapsed": bars_in_regime,
            }

        expected_total = 1.0 / exit_prob
        # Memoryless: expected remaining = expected total
        expected_remaining = expected_total
        # Median: P(T > m) = a_ii^m = 0.5 => m = log(0.5) / log(a_ii)
        median_remaining = np.log(0.5) / np.log(a_ii) if a_ii > 0 else 0.0
        # 90th percentile
        p90_remaining = np.log(0.1) / np.log(a_ii) if a_ii > 0 else 0.0

        return {
            "expected_total": expected_total,
            "expected_remaining": expected_remaining,
            "median_remaining": median_remaining,
            "p90_remaining": p90_remaining,
            "exit_prob_per_bar": exit_prob,
            "bars_elapsed": bars_in_regime,
        }

    def most_likely_next_regime(
        self,
        transmat: np.ndarray,
        current_state: int,
        labels: dict[int, str],
    ) -> list[dict]:
        """
        Rank the most likely next regimes upon exit from current state.

        Returns list of {state, label, probability} sorted by probability desc,
        excluding current state (self-transition).
        """
        row = transmat[current_state].copy()
        # Zero out self-transition and renormalize
        row[current_state] = 0.0
        total = row.sum()
        if total > 0:
            row /= total

        results = []
        for s in np.argsort(row)[::-1]:
            if s == current_state:
                continue
            prob = row[s]
            if prob < 1e-6:
                continue
            results.append({
                "state": int(s),
                "label": labels.get(int(s), f"state_{s}"),
                "probability": float(prob),
            })

        return results

    def fit_macro_hmm(
        self,
        macro_features: pd.DataFrame,
    ) -> tuple[GaussianHMM, np.ndarray, np.ndarray]:
        """
        Fit a small HMM (2-4 states) on macro features to detect
        macro regimes (risk-on, neutral, risk-off).

        Returns (model, states, posteriors).
        """
        X = macro_features.values
        # Z-score standardize
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma[sigma == 0] = 1.0
        X_z = (X - mu) / sigma

        best_bic = np.inf
        best_model = None

        for n in range(2, self.macro_n_states + 1):
            for seed in range(self.macro_n_restarts):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = GaussianHMM(
                            n_components=n,
                            covariance_type="diag",
                            n_iter=150,
                            tol=1e-4,
                            random_state=seed,
                        )
                        model.fit(X_z)
                        ll = model.score(X_z)
                except Exception:
                    continue

                T, d = X_z.shape
                k = (n - 1) + n * (n - 1) + n * d + n * d  # diag cov
                bic = -2 * ll + k * np.log(T)

                if bic < best_bic:
                    best_bic = bic
                    best_model = model

        if best_model is None:
            raise RuntimeError("Macro HMM fitting failed")

        states = best_model.predict(X_z)
        posteriors = best_model.predict_proba(X_z)
        return best_model, states, posteriors

    def label_macro_regimes(
        self,
        model: GaussianHMM,
        macro_features: pd.DataFrame,
    ) -> dict[int, str]:
        """
        Label macro regimes by stress level.
        Uses mean VIX feature (or first feature) to order states.
        """
        n = model.n_components
        # Use first feature mean to sort (VIX level if available)
        means = model.means_[:, 0]
        sorted_idx = np.argsort(means)

        if n == 2:
            names = ["risk_on", "risk_off"]
        elif n == 3:
            names = ["risk_on", "neutral", "risk_off"]
        else:
            names = [f"macro_{i}" for i in range(n)]
            names[0] = "risk_on"
            names[-1] = "risk_off"

        labels = {}
        for rank, state_id in enumerate(sorted_idx):
            labels[int(state_id)] = names[rank]
        return labels

    def bayesian_fusion(
        self,
        asset_forecast: np.ndarray,
        macro_posteriors: np.ndarray,
        asset_labels: dict[int, str],
        macro_labels: dict[int, str],
    ) -> np.ndarray:
        """
        Bayesian fusion of single-asset regime forecast with macro regime signal.

        Strategy:
          - If macro says risk_off, downweight bull/bull_run probabilities
          - If macro says risk_on, upweight bull/bull_run probabilities
          - Weight of macro influence controlled by self.macro_weight

        Returns adjusted (K,) probability vector over asset regimes.
        """
        # Determine macro sentiment: weighted average of risk scores
        macro_sentiment = 0.0
        for state_id, label in macro_labels.items():
            if state_id < len(macro_posteriors):
                if "risk_off" in label:
                    macro_sentiment += macro_posteriors[state_id] * (-1.0)
                elif "risk_on" in label:
                    macro_sentiment += macro_posteriors[state_id] * 1.0
                # neutral contributes 0

        # Build adjustment vector: boost/penalize based on regime type + macro
        K = len(asset_forecast)
        adjustment = np.ones(K)
        w = self.macro_weight

        for state_id, label in asset_labels.items():
            if state_id >= K:
                continue
            if label in ("bull", "bull_run"):
                # Positive sentiment boosts bull, negative penalizes
                adjustment[state_id] = 1.0 + w * macro_sentiment
            elif label in ("bear", "crash"):
                # Positive sentiment penalizes bear, negative boosts
                adjustment[state_id] = 1.0 - w * macro_sentiment

        # Apply and renormalize
        adjusted = asset_forecast * adjustment
        adjusted = np.clip(adjusted, 1e-10, None)
        adjusted /= adjusted.sum()
        return adjusted

    def generate_forecast_summary(
        self,
        transmat: np.ndarray,
        current_posteriors: np.ndarray,
        current_state: int,
        labels: dict[int, str],
        bars_in_regime: int = 0,
        macro_posteriors: np.ndarray | None = None,
        macro_labels: dict[int, str] | None = None,
    ) -> dict:
        """
        Generate a complete forecast summary combining all components.

        Returns dict with:
          - current: current regime info
          - forecasts: N-step ahead probabilities (raw + macro-adjusted)
          - shift_probs: regime shift probabilities at each horizon
          - duration: expected remaining duration stats
          - next_regimes: ranked likely next regimes
          - macro_adjustment: whether macro fusion was applied
        """
        horizons = [1, 2, 5, 10, self.max_horizon]
        current_label = labels.get(current_state, f"state_{current_state}")

        # Raw forecasts
        raw_forecasts = self.forecast_regime_probs(transmat, current_posteriors, horizons)

        # Macro-adjusted forecasts
        adjusted_forecasts = {}
        macro_applied = False
        if macro_posteriors is not None and macro_labels is not None:
            macro_applied = True
            for h, raw_probs in raw_forecasts.items():
                adjusted_forecasts[h] = self.bayesian_fusion(
                    raw_probs, macro_posteriors, labels, macro_labels
                )
        else:
            adjusted_forecasts = raw_forecasts

        # Shift probabilities
        shift_probs = self.regime_shift_probability(transmat, current_state, horizons)

        # Duration analysis
        duration = self.expected_regime_duration(transmat, current_state, bars_in_regime)

        # Next regime ranking
        next_regimes = self.most_likely_next_regime(transmat, current_state, labels)

        # Build forecast table: for each horizon, dominant regime + confidence
        forecast_table = []
        for h in horizons:
            probs = adjusted_forecasts[h]
            dominant_state = int(np.argmax(probs))
            dominant_label = labels.get(dominant_state, f"state_{dominant_state}")
            forecast_table.append({
                "horizon": h,
                "dominant_regime": dominant_label,
                "dominant_prob": float(probs[dominant_state]),
                "shift_prob": shift_probs[h],
                "probs": {labels.get(i, f"state_{i}"): float(p) for i, p in enumerate(probs)},
            })

        return {
            "current": {
                "state": current_state,
                "label": current_label,
                "confidence": float(current_posteriors[current_state]),
                "bars_in_regime": bars_in_regime,
            },
            "forecasts": forecast_table,
            "raw_forecasts": {h: probs.tolist() for h, probs in raw_forecasts.items()},
            "adjusted_forecasts": {h: probs.tolist() for h, probs in adjusted_forecasts.items()},
            "shift_probs": shift_probs,
            "duration": duration,
            "next_regimes": next_regimes,
            "macro_adjusted": macro_applied,
        }


def count_bars_in_current_regime(states: np.ndarray) -> int:
    """Count how many consecutive bars the current (last) state has persisted."""
    if len(states) == 0:
        return 0
    current = states[-1]
    count = 0
    for i in range(len(states) - 1, -1, -1):
        if states[i] == current:
            count += 1
        else:
            break
    return count
