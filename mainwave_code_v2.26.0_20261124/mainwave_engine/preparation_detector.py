"""Low-extension preparation detectors for the discovery layer (research candidates, pre-registered).

The recall audit of 2026-09-23 found the only early, low-extension signal (route_evidence.prepared) covers
9% of TREND episode starts, that most starts are preceded by FAILED / REBUILDING / DISTRIBUTION_RISK states,
and that the lifecycle's early stages never fire from bars alone. These variants try to see "repaired and
coiling before the move" from bars only. Thresholds reuse the ones already frozen elsewhere (0.85 contraction,
-1..2 ATR band); they are fixed here before any evaluation and are not tuned on results. Every column is causal
(rolling/shift only), so a reading at a date equals what could have been known that day."""
import pandas as pd

VERSION = "preparation_detector_v1_20260923"
BAND = (-1.0, 2.0)
CONTRACT = 0.85
VARIANTS = ("P0_prepared", "P1_repair", "P2_repair_rs", "P3_retained_contract", "P4_reclaim", "P4b_reclaim_held", "P5_union")
LABELS = {"P0_prepared": "现有准备（冲击保留+供给收缩+带内）", "P1_repair": "修复：MA20 转上且站上、20 日内曾在下方、换手与振幅不扩张",
          "P2_repair_rs": "修复 + 20 日相对强度不弱于沪深300", "P3_retained_contract": "冲击保留 + 换手振幅收缩（不要求结构底线）",
          "P4_reclaim": "近 5 日曾收复 MA20（此前 20 日至少 10 日在下方）且无量能高潮；今日可能已回落", "P4b_reclaim_held": "P4 且今日仍在 MA20 上方（待验证变体）",
          "P5_union": "P2 或 P4"}
# Pre-registered choice rule: on the design year, among P1–P5 keep those whose precision is at least the
# base rate and whose median extension at hit is at most 1.5 ATR; pick the highest recall; confirm on holdout.
# Set only from the study's choice rule (research/preparation_detector_study.py); None means no watch list is written.
# 2026-09-23 run 20260923_065117: design 2026-01..04 base rate 11.35%, P5 recall 58.6% / precision 12.0% / lead -6 / 0.51 ATR;
# holdout 2025 base rate 13.72%, P5 recall 55.9% / precision 13.9% / lead -6 / 0.50 ATR. A recall detector, not a discriminator.
CHOSEN = "P5_union"
CHOICE_RULE = {"precision_at_least_base_rate": True, "max_median_bias_at_hit": 1.5, "pick": "max_recall", "design": "2026-01..04", "holdout": "2025"}


def benchmark_ret20(path="data/000300_SH_index.csv"):
    b = pd.read_csv(path, usecols=["trade_date", "close"])
    b["date"] = pd.to_datetime(b.trade_date.astype(str)).dt.strftime("%Y-%m-%d")
    b = b.sort_values("date").drop_duplicates("date")
    ret = pd.to_numeric(b.close, errors="coerce")
    return pd.Series((ret / ret.shift(20) - 1).values * 100, index=b.date.values)


def annotate(frame, bench_ret20=None):
    """Add detector columns to a route_evidence frame; returns a new DataFrame with the same index."""
    f = frame
    close = f.return_index
    ma20 = close.rolling(20).mean()
    out = pd.DataFrame(index=f.index)
    gate = f.eligible.fillna(False).astype(bool) & f.bias_atr.between(*BAND)
    ma_up = ma20.ge(ma20.shift(5))
    # "Above" and "below" are only defined once the mean exists; the warm-up must not read as a stay below.
    above = close.ge(ma20) & ma20.notna()
    below = close.lt(ma20) & ma20.notna()
    was_below20 = below.shift(1).rolling(20, min_periods=1).max().eq(1)
    calm = f.turnover_contract.le(1.0) & f.range_contract.le(1.0)
    dates = f.date.dt.strftime("%Y-%m-%d") if hasattr(f.date, "dt") else f.date.astype(str)
    ret20 = (close / close.shift(20) - 1) * 100
    if bench_ret20 is not None:
        excess20 = ret20 - pd.Series(dates.map(bench_ret20).values, index=f.index)
    else:
        excess20 = pd.Series(float("nan"), index=f.index)
    out["excess20"] = excess20
    out["P0_prepared"] = f.prepared.fillna(False).astype(bool)
    out["P1_repair"] = gate & ma_up & above & was_below20 & calm
    out["P2_repair_rs"] = out.P1_repair & excess20.ge(0)
    out["P3_retained_contract"] = gate & f.retained.fillna(False).astype(bool) & f.turnover_contract.le(CONTRACT) & f.range_contract.le(CONTRACT)
    crossed = above & below.shift(1, fill_value=False)
    recent_cross = crossed.rolling(5, min_periods=1).max().eq(1)
    long_below = below.shift(5).rolling(20, min_periods=15).sum().ge(10)
    # The relative-strength floor is a sanity filter here, so an unknown benchmark does not veto the reclaim.
    out["P4_reclaim"] = gate & recent_cross & long_below & f.amount_ratio.le(2.5) & (excess20.ge(-5) | excess20.isna())
    out["P4b_reclaim_held"] = out.P4_reclaim & above
    out["above_ma20_now"] = above
    out["reclaim_failed"] = out.P4_reclaim & ~above
    out["P5_union"] = out.P2_repair_rs | out.P4_reclaim
    return out


def latest(frame, bench_ret20=None):
    """Flags of the last session, for a daily scan record."""
    if frame is None or frame.empty:
        return {v: False for v in VARIANTS}
    ann = annotate(frame, bench_ret20)
    row = ann.iloc[-1]
    return {**{v: bool(row[v]) for v in VARIANTS}, "above_ma20_now": bool(row.above_ma20_now), "reclaim_failed": bool(row.reclaim_failed),
            "excess20": (None if pd.isna(row.excess20) else round(float(row.excess20), 2)), "version": VERSION}
