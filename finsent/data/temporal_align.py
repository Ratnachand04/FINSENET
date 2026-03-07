"""
Temporal alignment and data integrity checks.
=============================================

Ensures no look-ahead bias in the data pipeline.
This module is the gatekeeper of temporal correctness.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


def validate_temporal_ordering(
    train_dates: pd.DatetimeIndex,
    val_dates: pd.DatetimeIndex,
    test_dates: pd.DatetimeIndex,
) -> bool:
    """Assert strict chronological ordering across splits.
    
    Rules:
    1. max(train_dates) < min(val_dates)
    2. max(val_dates) < min(test_dates)
    3. No overlap between any pair
    """
    checks = []
    
    # Check 1: Train before Val
    if train_dates.max() >= val_dates.min():
        raise ValueError(
            f"LOOK-AHEAD DETECTED: Train ends {train_dates.max()}, "
            f"but Val starts {val_dates.min()}"
        )
    checks.append(True)
    
    # Check 2: Val before Test
    if val_dates.max() >= test_dates.min():
        raise ValueError(
            f"LOOK-AHEAD DETECTED: Val ends {val_dates.max()}, "
            f"but Test starts {test_dates.min()}"
        )
    checks.append(True)
    
    # Check 3: No overlap
    train_set = set(train_dates)
    val_set = set(val_dates)
    test_set = set(test_dates)
    
    overlap_tv = train_set & val_set
    overlap_vt = val_set & test_set
    overlap_tt = train_set & test_set
    
    if overlap_tv:
        raise ValueError(f"Train/Val overlap: {len(overlap_tv)} dates")
    if overlap_vt:
        raise ValueError(f"Val/Test overlap: {len(overlap_vt)} dates")
    if overlap_tt:
        raise ValueError(f"Train/Test overlap: {len(overlap_tt)} dates")
    
    print("[TemporalCheck] ✓ All temporal ordering checks passed")
    return True


def validate_news_alignment(
    news_timestamps: pd.Series,
    price_dates: pd.DatetimeIndex,
    min_lag_hours: float = 24.0,
) -> Dict[str, float]:
    """Validate that news data respects temporal lag requirements.
    
    For each (news_time, associated_price_date) pair:
        price_date - news_time >= min_lag_hours
    
    Returns metrics about the alignment quality.
    """
    violations = 0
    total = 0
    lag_hours = []
    
    for news_time in news_timestamps:
        # Find closest future price date
        future_dates = price_dates[price_dates > news_time]
        if len(future_dates) == 0:
            continue
        
        closest_price_date = future_dates[0]
        lag = (closest_price_date - news_time).total_seconds() / 3600
        lag_hours.append(lag)
        total += 1
        
        if lag < min_lag_hours:
            violations += 1
    
    lag_arr = np.array(lag_hours)
    metrics = {
        "total_pairs": total,
        "violations": violations,
        "violation_rate": violations / max(total, 1),
        "mean_lag_hours": float(np.mean(lag_arr)) if len(lag_arr) > 0 else 0,
        "min_lag_hours": float(np.min(lag_arr)) if len(lag_arr) > 0 else 0,
        "max_lag_hours": float(np.max(lag_arr)) if len(lag_arr) > 0 else 0,
    }
    
    if violations > 0:
        print(f"[TemporalCheck] ⚠ {violations}/{total} news-price pairs violate "
              f"minimum lag of {min_lag_hours}h")
    else:
        print(f"[TemporalCheck] ✓ All {total} news-price pairs respect "
              f"minimum lag of {min_lag_hours}h")
    
    return metrics


def create_labels(
    close_prices: pd.Series,
    forward_period: int = 1,
    neutral_threshold: float = 0.001,
) -> Tuple[pd.Series, pd.Series]:
    """Create direction labels from forward returns.
    
    CRITICAL: Labels are based on FUTURE returns.
    label[t] = direction of return from t to t+forward_period
    
    This is the prediction target, NOT a feature.
    
    Labels:
        0 = Down  (return < -threshold)
        1 = Neutral  (|return| <= threshold)
        2 = Up  (return > threshold)
    
    Returns:
        (labels: pd.Series[int], forward_returns: pd.Series[float])
    """
    # Forward returns (shifted by -forward_period so return at t predicts t→t+1)
    forward_returns = close_prices.pct_change(forward_period).shift(-forward_period)
    
    # Direction labels
    labels = pd.Series(1, index=close_prices.index, dtype=np.int64)  # default neutral
    labels[forward_returns > neutral_threshold] = 2   # Up
    labels[forward_returns < -neutral_threshold] = 0  # Down
    
    # Mark last forward_period rows as NaN (no future data available)
    labels.iloc[-forward_period:] = -1  # sentinel for "no label"
    forward_returns.iloc[-forward_period:] = np.nan
    
    # Print class distribution
    valid = labels[labels >= 0]
    dist = valid.value_counts().sort_index()
    total = len(valid)
    print(f"[Labels] Distribution — Down: {dist.get(0, 0)/total:.1%}, "
          f"Neutral: {dist.get(1, 0)/total:.1%}, "
          f"Up: {dist.get(2, 0)/total:.1%}")
    
    return labels, forward_returns


def compute_class_weights(labels: pd.Series) -> np.ndarray:
    """Compute inverse-frequency class weights for imbalanced data.
    
    weight_c = N / (n_classes * count_c)
    
    This is preferred over oversampling for financial data because
    oversampling creates artificial autocorrelation.
    """
    valid = labels[labels >= 0]
    n_classes = valid.nunique()
    total = len(valid)
    
    weights = np.zeros(n_classes)
    for c in range(n_classes):
        count = (valid == c).sum()
        if count > 0:
            weights[c] = total / (n_classes * count)
        else:
            weights[c] = 1.0
    
    print(f"[ClassWeights] {weights}")
    return weights
