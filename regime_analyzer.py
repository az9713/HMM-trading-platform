"""
regime_analyzer.py — Regime transition analysis and alpha attribution.

Extracts alpha from regime transitions: identifies which transitions
generate returns, computes early-warning signals from entropy gradients
and posterior shifts, and stratifies backtest performance by regime.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class TransitionEvent:
    """A single regime transition."""
    bar: int
    from_regime: str
    to_regime: str
    forward_return_5: float  # 5-bar forward return
    forward_return_10: float  # 10-bar forward return
    forward_return_20: float  # 20-bar forward return
    entropy_before: float  # avg entropy 5 bars before
    entropy_after: float  # avg entropy 5 bars after
    confidence_at: float  # confidence at transition bar


class RegimeTransitionAnalyzer:
    """Analyze regime transitions for alpha signals and performance attribution."""

    def __init__(self, lookforward_windows: tuple[int, ...] = (5, 10, 20)):
        self.lookforward_windows = lookforward_windows

    def detect_transitions(
        self,
        states: np.ndarray,
        labels: dict[int, str],
        prices: np.ndarray,
        entropy: np.ndarray,
        confidence: np.ndarray,
    ) -> list[TransitionEvent]:
        """
        Detect all regime transitions and compute forward returns at each.

        A transition occurs when states[t] != states[t-1].
        Forward returns are computed at 5, 10, 20 bars out.
        """
        T = len(states)
        transitions = []

        for t in range(1, T):
            if states[t] == states[t - 1]:
                continue

            from_label = labels.get(int(states[t - 1]), f"state_{states[t-1]}")
            to_label = labels.get(int(states[t]), f"state_{states[t]}")

            # Forward returns
            fwd = {}
            for w in self.lookforward_windows:
                if t + w < T:
                    fwd[w] = (prices[t + w] / prices[t]) - 1.0
                else:
                    fwd[w] = np.nan

            # Entropy context
            start_before = max(0, t - 5)
            ent_before = np.mean(entropy[start_before:t]) if t > 0 else entropy[t]
            end_after = min(T, t + 6)
            ent_after = np.mean(entropy[t:end_after])

            transitions.append(TransitionEvent(
                bar=t,
                from_regime=from_label,
                to_regime=to_label,
                forward_return_5=fwd.get(5, np.nan),
                forward_return_10=fwd.get(10, np.nan),
                forward_return_20=fwd.get(20, np.nan),
                entropy_before=ent_before,
                entropy_after=ent_after,
                confidence_at=confidence[t],
            ))

        return transitions

    def transition_matrix_empirical(
        self,
        transitions: list[TransitionEvent],
        regime_labels: list[str],
    ) -> pd.DataFrame:
        """
        Count observed transitions to build an empirical transition frequency matrix.
        Returns DataFrame with from-regime as rows, to-regime as columns.
        """
        matrix = pd.DataFrame(
            0, index=regime_labels, columns=regime_labels, dtype=int
        )
        for tr in transitions:
            if tr.from_regime in regime_labels and tr.to_regime in regime_labels:
                matrix.loc[tr.from_regime, tr.to_regime] += 1
        return matrix

    def transition_forward_returns(
        self,
        transitions: list[TransitionEvent],
    ) -> pd.DataFrame:
        """
        Compute average forward returns grouped by transition type (from -> to).
        Returns DataFrame with columns: from, to, count, mean_5, mean_10, mean_20,
        std_5, hit_rate_5 (% positive at 5-bar).
        """
        if not transitions:
            return pd.DataFrame()

        records = []
        for tr in transitions:
            records.append({
                "from": tr.from_regime,
                "to": tr.to_regime,
                "fwd_5": tr.forward_return_5,
                "fwd_10": tr.forward_return_10,
                "fwd_20": tr.forward_return_20,
            })

        df = pd.DataFrame(records)
        grouped = df.groupby(["from", "to"])

        rows = []
        for (fr, to), grp in grouped:
            fwd5 = grp["fwd_5"].dropna()
            fwd10 = grp["fwd_10"].dropna()
            fwd20 = grp["fwd_20"].dropna()
            rows.append({
                "from_regime": fr,
                "to_regime": to,
                "count": len(grp),
                "mean_fwd_5": fwd5.mean() if len(fwd5) > 0 else np.nan,
                "mean_fwd_10": fwd10.mean() if len(fwd10) > 0 else np.nan,
                "mean_fwd_20": fwd20.mean() if len(fwd20) > 0 else np.nan,
                "std_fwd_5": fwd5.std() if len(fwd5) > 1 else np.nan,
                "hit_rate_5": (fwd5 > 0).mean() if len(fwd5) > 0 else np.nan,
            })

        return pd.DataFrame(rows)

    def early_warning_signals(
        self,
        posteriors: np.ndarray,
        entropy: np.ndarray,
        states: np.ndarray,
        labels: dict[int, str],
        gradient_window: int = 5,
        entropy_spike_threshold: float = 0.3,
        posterior_shift_threshold: float = 0.15,
    ) -> pd.DataFrame:
        """
        Generate early-warning signals for upcoming regime transitions.

        Two signals:
        1. Entropy gradient: rapid increase in entropy (uncertainty rising)
           suggests the model is losing confidence in the current regime.
        2. Posterior divergence: the gap between the dominant posterior
           and the runner-up is shrinking, suggesting a transition is brewing.

        Returns DataFrame with columns:
          bar, entropy_gradient, posterior_gap, dominant_regime,
          runner_up_regime, warning_level (0-3).
        """
        T = len(entropy)
        results = []

        for t in range(gradient_window, T):
            # Entropy gradient: change over window
            ent_grad = entropy[t] - entropy[t - gradient_window]

            # Posterior gap: difference between top-2 posteriors
            post = posteriors[t]
            sorted_post = np.sort(post)[::-1]
            posterior_gap = sorted_post[0] - sorted_post[1] if len(sorted_post) > 1 else 1.0

            # Identify dominant and runner-up regimes
            top_2_idx = np.argsort(post)[::-1][:2]
            dominant = labels.get(int(top_2_idx[0]), f"state_{top_2_idx[0]}")
            runner_up = labels.get(int(top_2_idx[1]), f"state_{top_2_idx[1]}") if len(top_2_idx) > 1 else "none"

            # Warning level (0-3)
            warning = 0
            if ent_grad > entropy_spike_threshold:
                warning += 1
            if ent_grad > entropy_spike_threshold * 2:
                warning += 1
            if posterior_gap < posterior_shift_threshold:
                warning += 1

            results.append({
                "bar": t,
                "entropy_gradient": ent_grad,
                "posterior_gap": posterior_gap,
                "dominant_regime": dominant,
                "runner_up_regime": runner_up,
                "warning_level": min(warning, 3),
            })

        return pd.DataFrame(results)

    def regime_attribution(
        self,
        returns: np.ndarray,
        states: np.ndarray,
        labels: dict[int, str],
        signals: np.ndarray,
    ) -> pd.DataFrame:
        """
        Stratify P&L by regime: compute return, Sharpe, win rate, and
        contribution per regime.

        This reveals which regimes the strategy actually profits from
        vs. which destroy alpha.
        """
        T = min(len(returns), len(states), len(signals))
        regime_labels_arr = np.array([
            labels.get(int(states[t]), f"state_{states[t]}") for t in range(T)
        ])

        # Strategy returns = market returns * signal * position direction
        strat_returns = returns[:T] * signals[:T]

        unique_regimes = sorted(set(regime_labels_arr))
        rows = []

        total_strat_return = np.nansum(strat_returns)

        for regime in unique_regimes:
            mask = regime_labels_arr == regime
            n_bars = mask.sum()
            if n_bars == 0:
                continue

            regime_strat_rets = strat_returns[mask]
            regime_mkt_rets = returns[:T][mask]

            # Metrics
            cumulative = np.nansum(regime_strat_rets)
            mean_ret = np.nanmean(regime_strat_rets)
            std_ret = np.nanstd(regime_strat_rets)
            sharpe = mean_ret / std_ret * np.sqrt(8760) if std_ret > 0 else 0.0

            active_bars = np.sum(signals[:T][mask] != 0)
            if active_bars > 0:
                active_rets = regime_strat_rets[signals[:T][mask] != 0]
                win_rate = np.mean(active_rets > 0) if len(active_rets) > 0 else 0.0
            else:
                win_rate = 0.0

            contribution = cumulative / total_strat_return if total_strat_return != 0 else 0.0

            rows.append({
                "regime": regime,
                "n_bars": int(n_bars),
                "pct_time": n_bars / T,
                "cumulative_return": cumulative,
                "mean_return": mean_ret,
                "volatility": std_ret,
                "sharpe": sharpe,
                "win_rate": win_rate,
                "pnl_contribution": contribution,
                "market_return": np.nansum(regime_mkt_rets),
                "alpha": cumulative - np.nansum(regime_mkt_rets),
            })

        return pd.DataFrame(rows)

    def transition_timing_analysis(
        self,
        transitions: list[TransitionEvent],
    ) -> dict:
        """
        Analyze timing patterns: average bars between transitions,
        which transitions are preceded by entropy spikes, etc.
        """
        if len(transitions) < 2:
            return {
                "avg_bars_between": np.nan,
                "median_bars_between": np.nan,
                "entropy_precedes_transition": np.nan,
                "avg_confidence_at_transition": np.nan,
                "n_transitions": len(transitions),
            }

        gaps = [
            transitions[i].bar - transitions[i - 1].bar
            for i in range(1, len(transitions))
        ]

        entropy_spikes = sum(
            1 for tr in transitions
            if tr.entropy_before > tr.entropy_after
        )

        return {
            "avg_bars_between": np.mean(gaps),
            "median_bars_between": np.median(gaps),
            "entropy_precedes_transition": entropy_spikes / len(transitions),
            "avg_confidence_at_transition": np.mean([
                tr.confidence_at for tr in transitions
            ]),
            "n_transitions": len(transitions),
        }
