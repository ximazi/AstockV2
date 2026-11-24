import pandas as pd

from research.mean_reversion_threshold_study import (
    crossing_indices,
    forward_outcome,
    select_thresholds,
)


def test_crossing_indices_only_emits_first_cross_and_honors_cooldown():
    values = pd.Series([-2, -9, -11, -7, -12, -13, -6, -11, -7, -12], dtype=float)
    eligible = pd.Series(True, index=values.index)

    assert crossing_indices(values, -10, eligible, cooldown=3) == [2, 7]


def test_forward_outcome_enters_at_supplied_next_open_and_uses_exact_horizon():
    frame = pd.DataFrame({
        "adj_open": [100, 90, 94, 96, 500],
        "adj_high": [101, 95, 100, 97, 600],
        "adj_low": [99, 85, 92, 89, 1],
        "adj_close": [100, 94, 96, 90, 500],
        "ema20": [100, 96, 95, 94, 100],
    })

    result = forward_outcome(frame, entry_index=1, horizon=3)

    assert result is not None
    assert result["entry_price_index"] == 90
    assert round(result["mfe_pct"], 4) == round((100 / 90 - 1) * 100, 4)
    assert round(result["mae_pct"], 4) == round((85 / 90 - 1) * 100, 4)
    assert result["close_return_pct"] == 0
    assert result["recovered_ema20"]
    assert round(result["managed_return_pct"], 4) == 5
    assert result["managed_exit_reason"] == "TARGET"


def test_managed_exit_assumes_stop_first_when_daily_bar_hits_both():
    frame = pd.DataFrame({
        "adj_open": [100, 100], "adj_high": [101, 106],
        "adj_low": [99, 91], "adj_close": [100, 104], "ema20": [100, 100],
    })

    result = forward_outcome(frame, entry_index=1, horizon=1)

    assert result is not None
    assert round(result["managed_return_pct"], 4) == -8
    assert result["managed_exit_reason"] == "BOTH_STOP_FIRST"


def test_threshold_selection_is_frozen_before_validation_and_oos():
    rows = []
    for threshold, dev_lift, oos_lift in [(-10.0, 3.0, -20.0), (-20.0, 1.0, 20.0)]:
        for split in ("DEV_2022_2024", "VALID_2025", "OOS_2026"):
            lift = dev_lift if split == "DEV_2022_2024" else (
                oos_lift if split == "OOS_2026" else 1.0
            )
            rows.append({
                "sector": "sample", "scope": "STOCK", "entry_mode": "NEXT_OPEN",
                "threshold_pct": threshold, "split": split, "horizon": 10,
                "n": 20, "event_days": 10, "median_mfe_pct": 8.0,
                "median_mae_pct": -5.0, "median_close_return_pct": 2.0,
                "close_win_rate": 0.6, "hit_5pct_rate": 0.6,
                "loss_10pct_rate": 0.2,
                "ema20_recovery_rate": 0.7,
                "median_close_return_pct_lift": lift,
                "hit_5pct_rate_lift": 0.1, "median_mfe_pct_lift": 1.0,
            })

    selected = select_thresholds(pd.DataFrame(rows))
    preferred = selected[selected["preferred_for_mode"]].iloc[0]

    assert preferred["threshold_pct"] == -10.0
