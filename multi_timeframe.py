"""
multi_timeframe.py — Multi-Timeframe Regime Fusion Engine.

Fits independent HMMs at multiple timeframes (e.g., 1h + 1d), aligns
their regime outputs to a common timeline, and computes a confluence
score measuring cross-timeframe regime agreement. When regimes agree
across timeframes, signal confidence amplifies; when they disagree,
confidence is penalized.

Key concepts:
  - Each timeframe gets its own HMM fit (independent BIC selection)
  - Higher timeframes are forward-filled to align with the base (fastest) timeframe
  - Confluence score: weighted agreement across timeframes
  - Regime conflict detection: identifies divergence between fast/slow regimes
  - Enhanced confidence: base_confidence * confluence_score
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from data_loader import fetch_ohlcv, compute_features, standardize, get_feature_matrix
from hmm_engine import RegimeDetector


# Timeframe ordering for hierarchy (fastest to slowest)
TIMEFRAME_ORDER = ["1m", "5m", "15m", "1h", "1d", "1wk"]

# Default weights: higher timeframes carry more weight in confluence
DEFAULT_WEIGHTS = {
    "1m": 0.3,
    "5m": 0.4,
    "15m": 0.5,
    "1h": 0.6,
    "1d": 1.0,
    "1wk": 1.2,
}

# Regime sentiment mapping for agreement computation
REGIME_SENTIMENT = {
    "crash": -2,
    "bear": -1,
    "neutral": 0,
    "bull": 1,
    "bull_run": 2,
}


@dataclass
class TimeframeResult:
    """Result of HMM fitting on a single timeframe."""
    interval: str
    df: pd.DataFrame
    states: np.ndarray
    posteriors: np.ndarray
    labels: dict[int, str]
    confidence: np.ndarray
    entropy: np.ndarray
    n_states: int
    bic_scores: dict
    detector: RegimeDetector


@dataclass
class FusionResult:
    """Result of multi-timeframe regime fusion."""
    base_interval: str
    timeframe_results: dict[str, TimeframeResult]
    aligned_regimes: pd.DataFrame  # columns: regime_{interval} for each tf
    aligned_sentiments: pd.DataFrame  # numeric sentiment per tf
    confluence_score: np.ndarray  # 0-1 agreement score per bar
    regime_conflicts: pd.DataFrame  # bars where timeframes disagree
    enhanced_confidence: np.ndarray  # base confidence * confluence
    dominant_regime: pd.Series  # weighted-vote regime label per bar


def _regime_to_sentiment(label: str) -> int:
    """Map regime label to numeric sentiment score."""
    lower = label.lower()
    # Check exact matches and longer keys first to avoid "bull" matching "bull_run"
    for key in sorted(REGIME_SENTIMENT.keys(), key=len, reverse=True):
        if key in lower:
            return REGIME_SENTIMENT[key]
    return 0  # default neutral for unknown labels


def _sentiment_to_regime(sentiment: float) -> str:
    """Map average sentiment back to nearest regime label."""
    if sentiment <= -1.5:
        return "crash"
    elif sentiment <= -0.5:
        return "bear"
    elif sentiment <= 0.5:
        return "neutral"
    elif sentiment <= 1.5:
        return "bull"
    else:
        return "bull_run"


def fit_timeframe(
    ticker: str,
    interval: str,
    lookback_days: int,
    config: dict,
) -> TimeframeResult:
    """
    Fetch data and fit an HMM for a single timeframe.

    Parameters
    ----------
    ticker : str
        Asset ticker symbol.
    interval : str
        yfinance interval string (e.g., '1h', '1d').
    lookback_days : int
        Number of days of history to fetch.
    config : dict
        Full configuration dict (uses hmm and data sections).

    Returns
    -------
    TimeframeResult with all HMM outputs for this timeframe.
    """
    df = fetch_ohlcv(ticker, interval, lookback_days)
    df = compute_features(df, config.get("data", {}))

    feature_cols = config.get("data", {}).get(
        "features",
        ["log_return", "rolling_vol", "volume_change", "intraday_range", "rsi"],
    )

    train_z, _, stats = standardize(df, cols=feature_cols)
    X = get_feature_matrix(train_z, feature_cols)

    detector = RegimeDetector(config)
    bic_scores = detector.fit_and_select(X)
    states, posteriors = detector.decode(X)
    labels = detector.label_regimes(X)
    entropy, confidence = detector.shannon_entropy(posteriors)

    return TimeframeResult(
        interval=interval,
        df=df,
        states=states,
        posteriors=posteriors,
        labels=labels,
        confidence=confidence,
        entropy=entropy,
        n_states=detector.n_states,
        bic_scores=bic_scores,
        detector=detector,
    )


def align_timeframes(
    base_result: TimeframeResult,
    other_results: list[TimeframeResult],
) -> pd.DataFrame:
    """
    Align regime labels from multiple timeframes to the base (fastest) timeline.

    Higher timeframes are reindexed to the base timeline using forward-fill,
    which is the correct causal approach: at any base-timeframe bar, the
    higher-timeframe regime is whatever was last observed (no lookahead).

    Parameters
    ----------
    base_result : TimeframeResult
        The fastest timeframe (determines output index).
    other_results : list[TimeframeResult]
        Higher timeframe results to align.

    Returns
    -------
    DataFrame with datetime index and columns regime_{interval} for each tf.
    """
    base_df = base_result.df.copy()

    # Use the datetime column if available, otherwise the index
    if "Datetime" in base_df.columns:
        base_index = pd.DatetimeIndex(base_df["Datetime"])
    elif "Date" in base_df.columns:
        base_index = pd.DatetimeIndex(base_df["Date"])
    else:
        base_index = base_df.index

    # Build aligned frame
    aligned = pd.DataFrame(index=base_index)
    base_regimes = [base_result.labels.get(s, "unknown") for s in base_result.states]
    aligned[f"regime_{base_result.interval}"] = base_regimes

    for result in other_results:
        other_df = result.df.copy()
        if "Datetime" in other_df.columns:
            other_index = pd.DatetimeIndex(other_df["Datetime"])
        elif "Date" in other_df.columns:
            other_index = pd.DatetimeIndex(other_df["Date"])
        else:
            other_index = other_df.index

        other_regimes = pd.Series(
            [result.labels.get(s, "unknown") for s in result.states],
            index=other_index,
        )

        # Reindex to base timeline with forward-fill (causal)
        aligned_regime = other_regimes.reindex(base_index, method="ffill")
        aligned[f"regime_{result.interval}"] = aligned_regime

    return aligned


def compute_confluence(
    aligned_regimes: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, pd.Series]:
    """
    Compute confluence score measuring cross-timeframe regime agreement.

    The confluence score is a weighted cosine-similarity-inspired metric:
    for each bar, we compute how aligned the regime sentiments are across
    timeframes. Perfect agreement = 1.0, complete disagreement = 0.0.

    Parameters
    ----------
    aligned_regimes : DataFrame
        Output of align_timeframes(), columns are regime_{interval}.
    weights : dict, optional
        Weight per interval. Defaults to DEFAULT_WEIGHTS.

    Returns
    -------
    tuple of:
        - confluence_score: array of shape (T,), values in [0, 1]
        - regime_conflicts: DataFrame flagging bars with disagreement
        - aligned_sentiments: array of shape (T, n_timeframes)
        - dominant_regime: Series with weighted-vote regime label
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    intervals = []
    for col in aligned_regimes.columns:
        if col.startswith("regime_"):
            intervals.append(col.replace("regime_", ""))

    T = len(aligned_regimes)
    n_tf = len(intervals)

    # Convert regimes to numeric sentiments
    sentiment_cols = {}
    w_arr = np.zeros(n_tf)
    for i, interval in enumerate(intervals):
        col = f"regime_{interval}"
        sentiments = aligned_regimes[col].apply(_regime_to_sentiment).values.astype(float)
        sentiment_cols[f"sentiment_{interval}"] = sentiments
        w_arr[i] = weights.get(interval, 0.5)

    sentiment_df = pd.DataFrame(sentiment_cols, index=aligned_regimes.index)
    sentiment_matrix = sentiment_df.values  # (T, n_tf)

    # Normalize weights
    w_arr = w_arr / w_arr.sum()

    # Weighted mean sentiment per bar
    weighted_sentiment = (sentiment_matrix * w_arr[np.newaxis, :]).sum(axis=1)

    # Confluence = 1 - normalized weighted standard deviation of sentiments
    # When all timeframes agree, std=0 → confluence=1
    # Max possible std is ~2 (crash vs bull_run), normalize by 2
    weighted_mean = weighted_sentiment  # already weighted
    deviations = sentiment_matrix - weighted_mean[:, np.newaxis]
    weighted_var = (w_arr[np.newaxis, :] * deviations ** 2).sum(axis=1)
    weighted_std = np.sqrt(weighted_var)

    # Normalize: max std when one tf says crash (-2) and another says bull_run (2)
    max_possible_std = 2.0
    confluence_score = np.clip(1.0 - weighted_std / max_possible_std, 0.0, 1.0)

    # Detect conflicts: bars where any two timeframes differ by > 1 sentiment level
    conflicts = []
    for t in range(T):
        sentiments_t = sentiment_matrix[t]
        # Skip rows with NaN (before higher TF data available)
        if np.any(np.isnan(sentiments_t)):
            conflicts.append(False)
            continue
        spread = np.max(sentiments_t) - np.min(sentiments_t)
        conflicts.append(spread > 1)

    conflict_df = pd.DataFrame({
        "conflict": conflicts,
        "confluence": confluence_score,
        "weighted_sentiment": weighted_sentiment,
    }, index=aligned_regimes.index)

    # Dominant regime: map weighted sentiment back to label
    dominant = pd.Series(
        [_sentiment_to_regime(s) for s in weighted_sentiment],
        index=aligned_regimes.index,
        name="dominant_regime",
    )

    return confluence_score, conflict_df, sentiment_matrix, dominant


def run_multi_timeframe_analysis(
    ticker: str,
    intervals: list[str],
    lookback_days: int,
    config: dict,
    weights: dict[str, float] | None = None,
) -> FusionResult:
    """
    Full multi-timeframe analysis pipeline.

    Fits independent HMMs at each timeframe, aligns them to the fastest
    timeframe's timeline, and computes confluence scores.

    Parameters
    ----------
    ticker : str
        Asset ticker symbol.
    intervals : list[str]
        List of intervals to analyze, e.g. ['1h', '1d'].
        First element is treated as the base (fastest) timeframe.
    lookback_days : int
        Days of history to fetch.
    config : dict
        Full configuration dict.
    weights : dict, optional
        Confluence weights per interval.

    Returns
    -------
    FusionResult with all analysis outputs.
    """
    # Sort intervals from fastest to slowest
    sorted_intervals = sorted(
        intervals,
        key=lambda x: TIMEFRAME_ORDER.index(x) if x in TIMEFRAME_ORDER else 99,
    )

    # Fit HMM per timeframe
    tf_results: dict[str, TimeframeResult] = {}
    for interval in sorted_intervals:
        tf_results[interval] = fit_timeframe(ticker, interval, lookback_days, config)

    base_interval = sorted_intervals[0]
    base_result = tf_results[base_interval]
    other_results = [tf_results[iv] for iv in sorted_intervals[1:]]

    # Align to base timeline
    aligned_regimes = align_timeframes(base_result, other_results)

    # Compute confluence
    confluence_score, conflict_df, sentiment_matrix, dominant = compute_confluence(
        aligned_regimes, weights
    )

    # Enhanced confidence = base confidence * confluence
    enhanced_confidence = base_result.confidence * confluence_score

    # Build sentiment DataFrame
    sentiment_cols = {}
    for i, iv in enumerate(sorted_intervals):
        sentiment_cols[f"sentiment_{iv}"] = sentiment_matrix[:, i]
    aligned_sentiments = pd.DataFrame(sentiment_cols, index=aligned_regimes.index)

    return FusionResult(
        base_interval=base_interval,
        timeframe_results=tf_results,
        aligned_regimes=aligned_regimes,
        aligned_sentiments=aligned_sentiments,
        confluence_score=confluence_score,
        regime_conflicts=conflict_df,
        enhanced_confidence=enhanced_confidence,
        dominant_regime=dominant,
    )
