# -*- coding: utf-8 -*-
"""Point-in-time drawdown threshold study for three mainstream A-share themes.

The study is deliberately separate from the main-wave engine. A drawdown first
creates an oversold watch event; an optional repair confirmation creates a later
entry signal. All simulated entries use the next trading day's open.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "mean_reversion_threshold_study_v1_20260903"
DEFAULT_OUTPUT = Path("results/mean_reversion_thresholds")
HORIZONS = (5, 10, 20)
THRESHOLDS = (-8.0, -10.0, -12.0, -15.0, -18.0, -20.0, -25.0, -30.0, -35.0, -40.0)
COOLDOWN = 10
MIN_STOCK_AMOUNT = 50_000_000

SPLITS = {
    "DEV_2022_2024": (pd.Timestamp("2022-01-01"), pd.Timestamp("2024-12-31")),
    "VALID_2025": (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")),
    "OOS_2026": (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-12-31")),
}

BASKETS = {
    "semi_manufacturing_equipment": {
        "label": "半导体制造/设备",
        "note": "华虹是晶圆制造代表，不是设备公司；本篮子覆盖制造与核心设备。",
        "members": {
            "sh_688347": "华虹公司",
            "sz_002371": "北方华创",
            "sh_688012": "中微公司",
            "sh_688072": "拓荆科技",
            "sh_688120": "华海清科",
            "sh_688082": "盛美上海",
        },
    },
    "optical_modules": {
        "label": "光模块/CPO",
        "note": "以易中天为核心，并加入光迅科技、太辰光作为扩展样本。",
        "members": {
            "sz_300502": "新易盛",
            "sz_300308": "中际旭创",
            "sz_300394": "天孚通信",
            "sz_002281": "光迅科技",
            "sz_300570": "太辰光",
        },
    },
    "storage_chain": {
        "label": "存储链",
        "note": "长鑫存储只作产业锚；回测使用本地具备行情的A股存储映射标的。",
        "members": {
            "sh_603986": "兆易创新",
            "sz_300223": "北京君正",
            "sh_688110": "东芯股份",
            "sz_001309": "德明利",
            "sh_688525": "佰维存储",
            "sh_688008": "澜起科技",
            "sz_301308": "江波龙",
        },
    },
}

INDEX_PATHS = {
    "SHCOMP": Path("data/000001_SH_index.csv"),
    "STAR50": Path("data/000688_SH_index.csv"),
    "CHINEXT": Path("data/399006_SZ_index.csv"),
    "CSI2000": Path("data/399303_SZ_index.csv"),
}


def split_name(value: pd.Timestamp) -> str:
    for name, (start, end) in SPLITS.items():
        if start <= value <= end:
            return name
    return ""


def crossing_indices(values: pd.Series, threshold: float, eligible: pd.Series,
                     cooldown: int = COOLDOWN) -> list[int]:
    """Return first-cross indices, suppressing repeated events during one selloff."""
    crossed = values.le(threshold) & values.shift(1).gt(threshold) & eligible
    selected: list[int] = []
    last = -10**9
    for index in np.flatnonzero(crossed.fillna(False).to_numpy()):
        if index - last >= cooldown:
            selected.append(int(index))
            last = int(index)
    return selected


def _numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add point-in-time features to adjusted OHLC columns."""
    result = frame.sort_values("date").reset_index(drop=True).copy()
    close = result["adj_close"]
    previous = close.shift(1)
    true_range = pd.concat([
        result["adj_high"] - result["adj_low"],
        (result["adj_high"] - previous).abs(),
        (result["adj_low"] - previous).abs(),
    ], axis=1).max(axis=1)
    result["ema20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
    result["peak20"] = result["adj_high"].rolling(20, min_periods=20).max()
    result["peak60"] = result["adj_high"].rolling(60, min_periods=60).max()
    result["atr20"] = true_range.rolling(20, min_periods=15).mean()
    result["drawdown20_pct"] = (close / result["peak20"] - 1.0) * 100
    result["drawdown60_pct"] = (close / result["peak60"] - 1.0) * 100
    result["ema20_bias_pct"] = (close / result["ema20"] - 1.0) * 100
    result["atr20_pct"] = result["atr20"] / close * 100
    result["atr_deviation"] = (close - result["ema20"]) / result["atr20"]
    result["ret1_pct"] = close.pct_change() * 100
    result["ret5_pct"] = close.pct_change(5) * 100
    result["ret20_pct"] = close.pct_change(20) * 100
    result["positive_bar"] = (close > result["adj_open"]) & (close > previous)
    result["amount_med20"] = result["amount"].rolling(20, min_periods=10).median()
    result["split"] = result["date"].map(split_name)
    return result


def load_stock(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8-sig")
    required = {"date", "open", "high", "low", "close", "amount"}
    if not required.issubset(raw.columns):
        raise ValueError(f"{path} missing columns: {sorted(required - set(raw.columns))}")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = _numeric(raw, ("open", "high", "low", "close", "amount"))
    raw = raw.dropna(subset=["date", "open", "high", "low", "close", "amount"])
    raw = raw.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

    raw_return = raw["close"].pct_change()
    if "pct_chg" in raw.columns:
        reported = pd.to_numeric(raw["pct_chg"], errors="coerce") / 100
        reported = reported.fillna(raw_return)
    else:
        reported = raw_return
    reported = reported.fillna(0).clip(lower=-0.95, upper=3.0)
    adjusted_close = 100 * np.exp(np.log1p(reported).cumsum())
    scale = adjusted_close / raw["close"].replace(0, np.nan)
    frame = pd.DataFrame({
        "date": raw["date"],
        "adj_open": raw["open"] * scale,
        "adj_high": raw["high"] * scale,
        "adj_low": raw["low"] * scale,
        "adj_close": adjusted_close,
        "raw_close": raw["close"],
        "amount": raw["amount"],
    })
    return add_features(frame)


def build_basket(frames: dict[str, pd.DataFrame], minimum_members: int) -> pd.DataFrame:
    """Build a daily rebalanced equal-weight OHLC proxy from member returns."""
    parts = []
    for symbol, frame in frames.items():
        previous = frame["adj_close"].shift(1)
        part = pd.DataFrame({
            "date": frame["date"],
            "symbol": symbol,
            "open_rel": frame["adj_open"] / previous,
            "high_rel": frame["adj_high"] / previous,
            "low_rel": frame["adj_low"] / previous,
            "close_rel": frame["adj_close"] / previous,
            "amount": frame["amount"],
        }).dropna()
        parts.append(part)
    combined = pd.concat(parts, ignore_index=True)
    daily = combined.groupby("date", as_index=False).agg(
        open_rel=("open_rel", "mean"),
        high_rel=("high_rel", "mean"),
        low_rel=("low_rel", "mean"),
        close_rel=("close_rel", "mean"),
        amount=("amount", "sum"),
        member_count=("symbol", "nunique"),
    )
    daily = daily[daily["member_count"] >= minimum_members].reset_index(drop=True)
    daily["adj_close"] = 100 * daily["close_rel"].cumprod()
    previous_basket = daily["adj_close"] / daily["close_rel"]
    daily["adj_open"] = previous_basket * daily["open_rel"]
    daily["adj_high"] = previous_basket * daily["high_rel"]
    daily["adj_low"] = previous_basket * daily["low_rel"]
    daily["raw_close"] = daily["adj_close"]
    return add_features(daily)


def build_market_frame() -> pd.DataFrame:
    pieces = []
    vote_columns = []
    for name, path in INDEX_PATHS.items():
        raw = pd.read_csv(path)
        date_column = "trade_date" if "trade_date" in raw.columns else "date"
        raw["date"] = pd.to_datetime(raw[date_column], errors="coerce")
        raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
        raw = raw.dropna(subset=["date", "close"]).sort_values("date")
        raw[f"{name}_ret1"] = raw["close"].pct_change()
        raw[f"{name}_ma60"] = raw["close"].rolling(60, min_periods=60).mean()
        raw[f"{name}_slope20"] = raw[f"{name}_ma60"].pct_change(20)
        raw[f"{name}_bull"] = (
            (raw["close"] > raw[f"{name}_ma60"]) & (raw[f"{name}_slope20"] > 0)
        )
        raw[f"{name}_bear"] = (
            (raw["close"] < raw[f"{name}_ma60"]) & (raw[f"{name}_slope20"] < 0)
        )
        pieces.append(raw[["date", f"{name}_ret1", f"{name}_bull", f"{name}_bear"]])
        vote_columns.append(name)

    market = pieces[0]
    for piece in pieces[1:]:
        market = market.merge(piece, on="date", how="outer")
    ret_columns = [f"{name}_ret1" for name in vote_columns]
    bull_columns = [f"{name}_bull" for name in vote_columns]
    bear_columns = [f"{name}_bear" for name in vote_columns]
    market = market.sort_values("date").reset_index(drop=True)
    market["market_ret1"] = market[ret_columns].mean(axis=1, skipna=True)
    market["market_ret5"] = (1 + market["market_ret1"]).rolling(5).apply(np.prod, raw=True) - 1
    market["bull_votes"] = market[bull_columns].sum(axis=1)
    market["bear_votes"] = market[bear_columns].sum(axis=1)
    market["structure"] = np.select(
        [market["bull_votes"] >= 3, market["bear_votes"] >= 3],
        ["BULL", "BEAR"], default="RANGE",
    )

    prior_ret1 = market["market_ret1"].shift(1).rolling(250, min_periods=100)
    prior_ret5 = market["market_ret5"].shift(1).rolling(250, min_periods=100)
    market["ret1_p05"] = prior_ret1.quantile(0.05)
    market["ret1_p95"] = prior_ret1.quantile(0.95)
    market["ret5_p05"] = prior_ret5.quantile(0.05)
    market["ret5_p95"] = prior_ret5.quantile(0.95)

    breadth_path = Path("market_daily_stats.csv")
    if breadth_path.exists():
        breadth = pd.read_csv(breadth_path)
        breadth["date"] = pd.to_datetime(breadth["date"], errors="coerce")
        for column in ("avg_pct", "up_count", "down_count", "limit_up", "limit_down"):
            breadth[column] = pd.to_numeric(breadth[column], errors="coerce")
        denominator = (breadth["up_count"] + breadth["down_count"]).replace(0, np.nan)
        breadth["up_ratio"] = breadth["up_count"] / denominator
        market = market.merge(
            breadth[["date", "avg_pct", "up_ratio", "limit_up", "limit_down"]],
            on="date", how="left",
        )
    else:
        for column in ("avg_pct", "up_ratio", "limit_up", "limit_down"):
            market[column] = np.nan

    relative_panic = (
        ((market["market_ret1"] <= market["ret1_p05"]) & (market["market_ret1"] <= -0.012))
        | ((market["market_ret5"] <= market["ret5_p05"]) & (market["market_ret5"] <= -0.04))
    )
    breadth_panic = (
        ((market["avg_pct"] <= -2.0) & (market["up_ratio"] <= 0.22))
        | (market["limit_down"] >= 100)
    ).fillna(False)
    relative_euphoria = (
        ((market["market_ret1"] >= market["ret1_p95"]) & (market["market_ret1"] >= 0.012))
        | ((market["market_ret5"] >= market["ret5_p95"]) & (market["market_ret5"] >= 0.04))
    )
    breadth_euphoria = (
        ((market["avg_pct"] >= 2.0) & (market["up_ratio"] >= 0.78))
        | (market["limit_up"] >= 150)
    ).fillna(False)
    market["temperature"] = np.select(
        [relative_panic | breadth_panic, relative_euphoria | breadth_euphoria],
        ["PANIC", "EUPHORIA"], default="NORMAL",
    )
    recent_panic = market["temperature"].eq("PANIC").shift(1).rolling(3, min_periods=1).max().fillna(0)
    market["market_context"] = np.select(
        [
            market["temperature"].eq("PANIC"),
            recent_panic.astype(bool),
            market["structure"].eq("BEAR"),
            market["structure"].eq("BULL"),
        ],
        ["PANIC", "POST_PANIC", "BEAR", "BULL"],
        default="RANGE",
    )
    market["environment_bucket"] = np.where(
        market["market_context"].isin(["PANIC", "POST_PANIC", "BEAR"]),
        "WEAK_OR_PANIC", "NORMAL_OR_STRONG",
    )
    return market


def attach_market(frame: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "date", "market_ret1", "market_ret5", "structure", "temperature",
        "market_context", "environment_bucket", "avg_pct", "up_ratio",
    ]
    return frame.merge(market[columns], on="date", how="left")


def find_entry_index(frame: pd.DataFrame, anchor: int, mode: str) -> tuple[int | None, int | None]:
    if mode == "NEXT_OPEN":
        return (anchor + 1, anchor) if anchor + 1 < len(frame) else (None, None)
    if mode != "REPAIR_CONFIRM":
        raise ValueError(f"unknown entry mode: {mode}")
    end = min(len(frame) - 1, anchor + 5)
    for confirm in range(anchor, end + 1):
        row = frame.iloc[confirm]
        if (
            bool(row["positive_bar"])
            and row.get("temperature", "NORMAL") != "PANIC"
            and pd.notna(row.get("market_ret1"))
            and float(row["market_ret1"]) > 0
        ):
            return confirm + 1, confirm
    return None, None


def forward_outcome(frame: pd.DataFrame, entry_index: int, horizon: int) -> dict | None:
    """Measure a next-open trade over exactly ``horizon`` trading sessions."""
    end = entry_index + horizon
    if entry_index >= len(frame) or end > len(frame):
        return None
    entry = float(frame.at[entry_index, "adj_open"])
    if not np.isfinite(entry) or entry <= 0:
        return None
    future = frame.iloc[entry_index:end]
    recovery_mask = future["adj_close"] >= future["ema20"]
    recovery_positions = np.flatnonzero(recovery_mask.fillna(False).to_numpy())
    target = entry * 1.05
    stop = entry * 0.92
    managed_exit = float(future.iloc[-1]["adj_close"])
    managed_reason = "TIME_EXIT"
    managed_days = horizon
    for day_number, (_, day) in enumerate(future.iterrows(), start=1):
        day_open = float(day["adj_open"])
        if day_open <= stop:
            managed_exit, managed_reason, managed_days = day_open, "GAP_STOP", day_number
            break
        if day_open >= target:
            managed_exit, managed_reason, managed_days = day_open, "GAP_TARGET", day_number
            break
        stop_hit = float(day["adj_low"]) <= stop
        target_hit = float(day["adj_high"]) >= target
        if stop_hit:
            managed_exit = stop
            managed_reason = "BOTH_STOP_FIRST" if target_hit else "STOP"
            managed_days = day_number
            break
        if target_hit:
            managed_exit, managed_reason, managed_days = target, "TARGET", day_number
            break
    return {
        "entry_price_index": entry,
        "mfe_pct": (float(future["adj_high"].max()) / entry - 1) * 100,
        "mae_pct": (float(future["adj_low"].min()) / entry - 1) * 100,
        "close_return_pct": (float(future.iloc[-1]["adj_close"]) / entry - 1) * 100,
        "hit_5pct": float(future["adj_high"].max()) >= entry * 1.05,
        "hit_8pct": float(future["adj_high"].max()) >= entry * 1.08,
        "loss_10pct": float(future["adj_low"].min()) <= entry * 0.90,
        "recovered_ema20": bool(len(recovery_positions)),
        "recovery_days": int(recovery_positions[0] + 1) if len(recovery_positions) else np.nan,
        "managed_return_pct": (managed_exit / entry - 1) * 100,
        "managed_exit_reason": managed_reason,
        "managed_days": managed_days,
    }


def collect_events(frame: pd.DataFrame, sector: str, scope: str, symbol: str,
                   name: str) -> list[dict]:
    eligible = (
        frame["split"].ne("")
        & frame["drawdown60_pct"].notna()
        & frame["amount_med20"].ge(MIN_STOCK_AMOUNT if scope == "STOCK" else 0)
    )
    rows: list[dict] = []
    for threshold in THRESHOLDS:
        for anchor in crossing_indices(frame["drawdown60_pct"], threshold, eligible):
            for mode in ("NEXT_OPEN", "REPAIR_CONFIRM"):
                entry_index, confirm_index = find_entry_index(frame, anchor, mode)
                if entry_index is None or confirm_index is None:
                    continue
                for horizon in HORIZONS:
                    outcome = forward_outcome(frame, entry_index, horizon)
                    if outcome is None:
                        continue
                    rows.append({
                        "sector": sector,
                        "scope": scope,
                        "symbol": symbol,
                        "name": name,
                        "threshold_pct": threshold,
                        "entry_mode": mode,
                        "split": str(frame.at[anchor, "split"]),
                        "signal_date": frame.at[anchor, "date"].date().isoformat(),
                        "confirm_date": frame.at[confirm_index, "date"].date().isoformat(),
                        "entry_date": frame.at[entry_index, "date"].date().isoformat(),
                        "wait_days": int(confirm_index - anchor),
                        "signal_drawdown60_pct": float(frame.at[anchor, "drawdown60_pct"]),
                        "signal_ema20_bias_pct": float(frame.at[anchor, "ema20_bias_pct"]),
                        "signal_atr_deviation": float(frame.at[anchor, "atr_deviation"]),
                        "signal_market_context": str(frame.at[anchor, "market_context"]),
                        "signal_environment_bucket": str(frame.at[anchor, "environment_bucket"]),
                        "entry_market_context": str(frame.at[confirm_index, "market_context"]),
                        "horizon": horizon,
                        **outcome,
                    })
    return rows


def collect_baselines(frame: pd.DataFrame, sector: str, scope: str, symbol: str,
                      name: str) -> list[dict]:
    rows = []
    eligible = frame.index[
        frame["split"].ne("")
        & frame["drawdown60_pct"].notna()
        & frame["amount_med20"].ge(MIN_STOCK_AMOUNT if scope == "STOCK" else 0)
    ].tolist()
    for anchor in eligible[::COOLDOWN]:
        entry_index, _ = find_entry_index(frame, int(anchor), "NEXT_OPEN")
        if entry_index is None:
            continue
        for horizon in HORIZONS:
            outcome = forward_outcome(frame, entry_index, horizon)
            if outcome is None:
                continue
            rows.append({
                "sector": sector,
                "scope": scope,
                "symbol": symbol,
                "name": name,
                "split": str(frame.at[anchor, "split"]),
                "signal_date": frame.at[anchor, "date"].date().isoformat(),
                "signal_market_context": str(frame.at[anchor, "market_context"]),
                "signal_environment_bucket": str(frame.at[anchor, "environment_bucket"]),
                "horizon": horizon,
                **outcome,
            })
    return rows


def _aggregate(grouped) -> pd.DataFrame:
    return grouped.agg(
        n=("symbol", "size"),
        event_days=("signal_date", "nunique"),
        median_mfe_pct=("mfe_pct", "median"),
        median_mae_pct=("mae_pct", "median"),
        median_close_return_pct=("close_return_pct", "median"),
        close_win_rate=("close_return_pct", lambda value: float((value > 0).mean())),
        hit_5pct_rate=("hit_5pct", "mean"),
        hit_8pct_rate=("hit_8pct", "mean"),
        loss_10pct_rate=("loss_10pct", "mean"),
        ema20_recovery_rate=("recovered_ema20", "mean"),
        median_recovery_days=("recovery_days", "median"),
        mean_managed_return_pct=("managed_return_pct", "mean"),
        median_managed_return_pct=("managed_return_pct", "median"),
        managed_win_rate=("managed_return_pct", lambda value: float((value > 0).mean())),
        target_exit_rate=("managed_exit_reason", lambda value: float(value.isin(["TARGET", "GAP_TARGET"]).mean())),
        stop_exit_rate=("managed_exit_reason", lambda value: float(value.isin(["STOP", "GAP_STOP", "BOTH_STOP_FIRST"]).mean())),
    ).reset_index()


def summarize(events: pd.DataFrame, baselines: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_summary = _aggregate(events.groupby([
        "sector", "scope", "entry_mode", "threshold_pct", "split", "horizon"
    ], dropna=False))
    baseline_summary = _aggregate(baselines.groupby([
        "sector", "scope", "split", "horizon"
    ], dropna=False))
    baseline_summary = baseline_summary.rename(columns={
        column: f"baseline_{column}" for column in baseline_summary.columns
        if column not in {"sector", "scope", "split", "horizon"}
    })
    merged = event_summary.merge(
        baseline_summary, on=["sector", "scope", "split", "horizon"], how="left"
    )
    for metric in ("median_mfe_pct", "median_mae_pct", "median_close_return_pct", "hit_5pct_rate"):
        merged[f"{metric}_lift"] = merged[metric] - merged[f"baseline_{metric}"]
    return merged, baseline_summary


def select_thresholds(summary: pd.DataFrame) -> pd.DataFrame:
    source = summary[(summary["scope"] == "STOCK") & (summary["horizon"] == 10)]
    rows = []
    for (sector, mode, threshold), block in source.groupby(
        ["sector", "entry_mode", "threshold_pct"]
    ):
        dev = block[block["split"] == "DEV_2022_2024"]
        valid = block[block["split"] == "VALID_2025"]
        oos = block[block["split"] == "OOS_2026"]
        if dev.empty:
            dev_eligible = False
            score = -999.0
        else:
            d = dev.iloc[0]
            dev_eligible = bool(
                d["n"] >= 12
                and d["event_days"] >= 6
                and d["median_close_return_pct"] > 0
                and d["median_close_return_pct_lift"] > 0
                and d["hit_5pct_rate"] >= 0.50
                and d["median_mae_pct"] > -12
            )
            score = (
                d["median_close_return_pct_lift"]
                + 0.20 * d["hit_5pct_rate_lift"] * 100
                + 0.10 * d["median_mfe_pct_lift"]
            )
        if valid.empty or oos.empty:
            status = "INSUFFICIENT"
        else:
            v = valid.iloc[0]
            o = oos.iloc[0]
            sample_ok = (
                v["n"] >= 8 and o["n"] >= 4
                and v["event_days"] >= 4 and o["event_days"] >= 2
            )
            directional = (
                v["median_close_return_pct_lift"] > 0
                and o["median_close_return_pct_lift"] > 0
            )
            quality = (
                v["median_close_return_pct"] > 0
                and o["median_close_return_pct"] > 0
                and v["hit_5pct_rate"] >= 0.50
                and o["hit_5pct_rate"] >= 0.50
                and v["loss_10pct_rate"] <= 0.40
                and o["loss_10pct_rate"] <= 0.40
                and v["median_mae_pct"] > -12
                and o["median_mae_pct"] > -12
            )
            status = "PASS" if sample_ok and directional and quality else (
                "WATCH" if directional else "REJECT"
            )
        row = {
            "sector": sector,
            "entry_mode": mode,
            "threshold_pct": threshold,
            "status": status,
            "dev_eligible": dev_eligible,
            "dev_score": score,
        }
        for split, prefix in ((dev, "dev"), (valid, "valid"), (oos, "oos")):
            if split.empty:
                continue
            item = split.iloc[0]
            for column in (
                "n", "event_days", "median_mfe_pct", "median_mae_pct",
                "median_close_return_pct", "close_win_rate", "hit_5pct_rate",
                "loss_10pct_rate", "ema20_recovery_rate", "median_close_return_pct_lift",
                "hit_5pct_rate_lift",
            ):
                row[f"{prefix}_{column}"] = item[column]
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    priority = {"PASS": 3, "WATCH": 2, "REJECT": 1, "INSUFFICIENT": 0}
    result["status_priority"] = result["status"].map(priority)
    result["preferred_for_mode"] = False
    for _, indices in result.groupby(["sector", "entry_mode"]).groups.items():
        candidates = result.loc[list(indices)].sort_values(
            ["dev_eligible", "dev_score", "threshold_pct"], ascending=[False, False, False]
        )
        result.loc[candidates.index[0], "preferred_for_mode"] = True
    return result.sort_values(
        ["sector", "entry_mode", "preferred_for_mode", "dev_score"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)


def prospective_candidates(selections: pd.DataFrame) -> pd.DataFrame:
    """Choose post-study candidates that must be validated prospectively."""
    passed = selections[selections["status"] == "PASS"].copy()
    if passed.empty:
        return passed
    passed["evidence_class"] = np.where(
        (passed["dev_median_close_return_pct"] > 0)
        & (passed["dev_median_close_return_pct_lift"] > 0)
        & (passed["dev_loss_10pct_rate"] <= 0.40),
        "FULL_HISTORY_DIRECTIONAL", "RECENT_REGIME_ONLY",
    )
    passed["forward_score"] = (
        passed[["valid_median_close_return_pct", "oos_median_close_return_pct"]].min(axis=1)
        + 5 * passed[["valid_hit_5pct_rate", "oos_hit_5pct_rate"]].min(axis=1)
        - 10 * passed[["valid_loss_10pct_rate", "oos_loss_10pct_rate"]].max(axis=1)
    )
    passed["candidate_for_sector"] = False
    evidence_priority = {"FULL_HISTORY_DIRECTIONAL": 2, "RECENT_REGIME_ONLY": 1}
    passed["_evidence_priority"] = passed["evidence_class"].map(evidence_priority)
    for _, indices in passed.groupby("sector").groups.items():
        ranked = passed.loc[list(indices)].sort_values(
            ["_evidence_priority", "forward_score", "threshold_pct"],
            ascending=[False, False, False],
        )
        passed.loc[ranked.index[0], "candidate_for_sector"] = True
    return passed.drop(columns="_evidence_priority").sort_values(
        ["sector", "candidate_for_sector", "forward_score"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def context_summary(events: pd.DataFrame, selections: pd.DataFrame) -> pd.DataFrame:
    if selections.empty or events.empty:
        return pd.DataFrame()
    preferred = selections[selections["preferred_for_mode"]][
        ["sector", "entry_mode", "threshold_pct", "status"]
    ]
    selected = events.merge(
        preferred, on=["sector", "entry_mode", "threshold_pct"], how="inner"
    )
    selected = selected[(selected["scope"] == "STOCK") & (selected["horizon"] == 10)]
    if selected.empty:
        return pd.DataFrame()
    return _aggregate(selected.groupby([
        "sector", "entry_mode", "threshold_pct", "status", "split",
        "signal_environment_bucket"
    ], dropna=False))


def choose_sector_trigger(selections: pd.DataFrame, sector: str) -> pd.Series | None:
    block = selections[(selections["sector"] == sector) & selections["preferred_for_mode"]]
    if block.empty:
        return None
    priority = {"PASS": 3, "WATCH": 2, "REJECT": 1, "INSUFFICIENT": 0}
    ranked = block.assign(_priority=block["status"].map(priority)).sort_values(
        ["_priority", "dev_score", "entry_mode"], ascending=[False, False, False]
    )
    return ranked.iloc[0]


def current_snapshot(frames: dict[str, dict[str, pd.DataFrame]], baskets: dict[str, pd.DataFrame],
                     selections: pd.DataFrame, candidates: pd.DataFrame,
                     as_of: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for sector, definition in BASKETS.items():
        forward = candidates[
            (candidates["sector"] == sector) & candidates["candidate_for_sector"]
        ] if not candidates.empty else pd.DataFrame()
        selected = forward.iloc[0] if not forward.empty else choose_sector_trigger(selections, sector)
        threshold = float(selected["threshold_pct"]) if selected is not None else np.nan
        mode = str(selected["entry_mode"]) if selected is not None else ""
        is_forward_shadow = not forward.empty
        status = (
            f"FORWARD_SHADOW_{selected['evidence_class']}" if is_forward_shadow
            else (str(selected["status"]) if selected is not None else "INSUFFICIENT")
        )
        items = [("BASKET", sector, definition["label"], baskets[sector])]
        items.extend(
            ("STOCK", symbol, definition["members"][symbol], frame)
            for symbol, frame in frames[sector].items()
        )
        for scope, symbol, name, frame in items:
            available = frame[frame["date"] <= as_of].reset_index(drop=True)
            if available.empty:
                continue
            row = available.iloc[-1]
            oversold = pd.notna(threshold) and row["drawdown60_pct"] <= threshold
            repaired = (
                bool(row["positive_bar"])
                and row["temperature"] != "PANIC"
                and pd.notna(row["market_ret1"])
                and row["market_ret1"] > 0
            )
            eligible = (
                available["split"].ne("")
                & available["drawdown60_pct"].notna()
                & available["amount_med20"].ge(MIN_STOCK_AMOUNT if scope == "STOCK" else 0)
            )
            crossings = crossing_indices(
                available["drawdown60_pct"], threshold, eligible
            ) if pd.notna(threshold) else []
            latest_cross = crossings[-1] if crossings else None
            signal_age = len(available) - 1 - latest_cross if latest_cross is not None else np.nan
            latest_cross_date = (
                available.at[latest_cross, "date"].date().isoformat()
                if latest_cross is not None else ""
            )
            if not is_forward_shadow:
                action = "RESEARCH_ONLY"
            elif mode == "NEXT_OPEN" and signal_age == 0:
                action = "SIGNAL_NEXT_OPEN"
            elif mode == "REPAIR_CONFIRM" and latest_cross is not None and signal_age <= 5:
                _, confirm_index = find_entry_index(available, latest_cross, mode)
                if confirm_index == len(available) - 1:
                    action = "REPAIR_SIGNAL_NEXT_OPEN"
                elif confirm_index is None:
                    action = "WATCH_REPAIR"
                else:
                    action = "SIGNAL_EXPIRED"
            elif oversold:
                action = "OVERSOLD_STALE"
            elif row["drawdown60_pct"] <= threshold + 3:
                action = "NEAR_TRIGGER"
            else:
                action = "NO_SIGNAL"
            if is_forward_shadow:
                action = f"SHADOW_{action}"
            rows.append({
                "date": row["date"].date().isoformat(),
                "sector": sector,
                "scope": scope,
                "symbol": symbol,
                "name": name,
                "close": float(row["raw_close"]),
                "drawdown20_pct": float(row["drawdown20_pct"]),
                "drawdown60_pct": float(row["drawdown60_pct"]),
                "ema20_bias_pct": float(row["ema20_bias_pct"]),
                "atr_deviation": float(row["atr_deviation"]),
                "ret5_pct": float(row["ret5_pct"]),
                "ret20_pct": float(row["ret20_pct"]),
                "market_structure": str(row["structure"]),
                "market_temperature": str(row["temperature"]),
                "market_context": str(row["market_context"]),
                "research_threshold_pct": threshold,
                "research_entry_mode": mode,
                "validation_status": status,
                "latest_cross_date": latest_cross_date,
                "signal_age_days": signal_age,
                "action": action,
            })
    return pd.DataFrame(rows)


def _pct(value) -> str:
    return "-" if pd.isna(value) else f"{float(value):.1f}%"


def _rate(value) -> str:
    return "-" if pd.isna(value) else f"{float(value) * 100:.0f}%"


def _integer(value) -> int:
    return 0 if pd.isna(value) else int(value)


def write_report(output: Path, as_of: pd.Timestamp, market: pd.DataFrame,
                 summary: pd.DataFrame, selections: pd.DataFrame, candidates: pd.DataFrame,
                 contexts: pd.DataFrame,
                 snapshot: pd.DataFrame) -> None:
    latest_market = market[market["date"] <= as_of].iloc[-1]
    lines = [
        f"# 主流板块下跌阈值与均值回归研究（{as_of.date()}）",
        "",
        f"- 研究版本：`{VERSION}`",
        "- 信号口径：距60日最高价的首次跌破；同股票、同阈值10个交易日冷却。",
        "- 成交口径：信号只能在收盘后确认，统一按下一交易日开盘进入。",
        "- 验证切分：阈值只由2022-2024发现期选择；2025冻结验证、2026 OOS均不参与选参。",
        "- 收益口径：同时看MFE、MAE、期末收盘收益和回到EMA20概率；MFE不等于可实现收益。",
        "- 固定主动风控：另测 +5%止盈、-8%止损、最多10日；同日双触发按先止损，参数不参与寻优。",
        "- 结论边界：固定核心篮子的事件研究，只决定是否值得进入影子验证，不直接生成实盘订单。",
        "",
        "## 当前市场环境",
        "",
        f"- 结构：`{latest_market['structure']}`；温度：`{latest_market['temperature']}`；组合状态：`{latest_market['market_context']}`。",
        f"- 四指数等权当日收益：{latest_market['market_ret1'] * 100:.2f}%；5日收益：{latest_market['market_ret5'] * 100:.2f}%。",
        "- 规则解释：弱市或恐慌只激活均值回归的观察池；是否入场由修复确认决定，不与主升浪趋势买点混用。",
        "",
        "## 阈值裁定",
        "",
        "| 板块 | 入场方式 | 60日回撤阈值 | 状态 | 2025样本/事件日 | 2025 10日收盘/MFE/MAE | 2026样本/事件日 | 2026 10日收盘/MFE/MAE | OOS触及+5% |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    labels = {key: value["label"] for key, value in BASKETS.items()}
    preferred = selections[selections["preferred_for_mode"]].copy()
    for _, row in preferred.sort_values(["sector", "entry_mode"]).iterrows():
        mode = "跌破后次日" if row["entry_mode"] == "NEXT_OPEN" else "等待修复确认"
        lines.append(
            f"| {labels[row['sector']]} | {mode} | {row['threshold_pct']:.0f}% | {row['status']} | "
            f"{_integer(row.get('valid_n', 0))}/{_integer(row.get('valid_event_days', 0))} | "
            f"{_pct(row.get('valid_median_close_return_pct'))} / {_pct(row.get('valid_median_mfe_pct'))} / {_pct(row.get('valid_median_mae_pct'))} | "
            f"{_integer(row.get('oos_n', 0))}/{_integer(row.get('oos_event_days', 0))} | "
            f"{_pct(row.get('oos_median_close_return_pct'))} / {_pct(row.get('oos_median_mfe_pct'))} / {_pct(row.get('oos_median_mae_pct'))} | "
            f"{_rate(row.get('oos_hit_5pct_rate'))} |"
        )
    lines.extend([
        "",
        "`PASS` 要求2025和2026均跑赢各自常态基线、期末中位收益为正、触及+5%概率不低于50%、10日内再跌10%的比例不高于40%、中位MAE优于-12%，且样本量达标。`WATCH` 仅表示方向一致但未完全过门，`REJECT` 不应转生产。",
        "",
        "## 板块篮子交叉核对",
        "",
        "| 板块 | 入场方式 | 阈值 | 2025篮子样本/10日收盘 | 2026篮子样本/10日收盘 |",
        "|---|---|---:|---:|---:|",
    ])
    basket_summary = summary[(summary["scope"] == "BASKET") & (summary["horizon"] == 10)]
    for _, selected in preferred.sort_values(["sector", "entry_mode"]).iterrows():
        matched = basket_summary[
            (basket_summary["sector"] == selected["sector"])
            & (basket_summary["entry_mode"] == selected["entry_mode"])
            & (basket_summary["threshold_pct"] == selected["threshold_pct"])
        ]
        valid = matched[matched["split"] == "VALID_2025"]
        oos = matched[matched["split"] == "OOS_2026"]
        valid_text = "-" if valid.empty else f"{int(valid.iloc[0]['n'])} / {_pct(valid.iloc[0]['median_close_return_pct'])}"
        oos_text = "-" if oos.empty else f"{int(oos.iloc[0]['n'])} / {_pct(oos.iloc[0]['median_close_return_pct'])}"
        mode = "跌破后次日" if selected["entry_mode"] == "NEXT_OPEN" else "等待修复确认"
        lines.append(
            f"| {labels[selected['sector']]} | {mode} | {selected['threshold_pct']:.0f}% | {valid_text} | {oos_text} |"
        )
    lines.extend([
        "",
        "板块篮子事件少，只作方向交叉核对；若个股池PASS但篮子方向相反，应继续影子观察，不能直接放大仓位。",
        "",
        "## 从本日冻结的前瞻影子候选",
        "",
        "这些阈值是在看过2025和2026结果后形成，只能从下一交易日起影子验证，不能再称2026 OOS。",
        "",
        "| 板块 | 阈值与入场 | 证据类别 | 2025收盘/尾损率 | 2026收盘/尾损率 |",
        "|---|---|---|---:|---:|",
    ])
    candidate_rows = candidates[candidates["candidate_for_sector"]] if not candidates.empty else candidates
    for sector in BASKETS:
        rowset = candidate_rows[candidate_rows["sector"] == sector]
        if rowset.empty:
            lines.append(f"| {labels[sector]} | 无 | NO_CANDIDATE | - | - |")
            continue
        row = rowset.iloc[0]
        mode = "跌破后次日" if row["entry_mode"] == "NEXT_OPEN" else "等待修复确认"
        lines.append(
            f"| {labels[sector]} | {row['threshold_pct']:.0f}% / {mode} | {row['evidence_class']} | "
            f"{_pct(row['valid_median_close_return_pct'])} / {_rate(row['valid_loss_10pct_rate'])} | "
            f"{_pct(row['oos_median_close_return_pct'])} / {_rate(row['oos_loss_10pct_rate'])} |"
        )
    lines.extend([
        "",
        "## 固定止盈止损核对",
        "",
        "| 类型 | 板块 | 阈值与入场 | 2026固定策略均值/中位 | 止盈率 | 止损率 |",
        "|---|---|---|---:|---:|---:|",
    ])
    risk_rows = []
    for _, selected in preferred.iterrows():
        risk_rows.append(("原始冻结", selected))
    for _, selected in candidate_rows.iterrows():
        risk_rows.append(("前瞻影子", selected))
    for kind, selected in risk_rows:
        matched = summary[
            (summary["scope"] == "STOCK")
            & (summary["horizon"] == 10)
            & (summary["split"] == "OOS_2026")
            & (summary["sector"] == selected["sector"])
            & (summary["entry_mode"] == selected["entry_mode"])
            & (summary["threshold_pct"] == selected["threshold_pct"])
        ]
        if matched.empty:
            continue
        row = matched.iloc[0]
        mode = "跌破后次日" if selected["entry_mode"] == "NEXT_OPEN" else "等待修复确认"
        lines.append(
            f"| {kind} | {labels[selected['sector']]} | {selected['threshold_pct']:.0f}% / {mode} | "
            f"{_pct(row['mean_managed_return_pct'])} / {_pct(row['median_managed_return_pct'])} | "
            f"{_rate(row['target_exit_rate'])} | {_rate(row['stop_exit_rate'])} |"
        )
    lines.extend([
        "",
        "固定策略只检验“反弹是否能主动兑现”。它没有滑点和手续费，仍需前瞻影子记录；若均值为负或止损率过高，不能因MFE好看而放行。",
        "",
        "## 坏环境是否更适合均值回归",
        "",
    ])
    if contexts.empty:
        lines.append("当前没有足够的已选阈值事件可分层。")
    else:
        oos_context = contexts[contexts["split"] == "OOS_2026"]
        lines.extend([
            "| 板块 | 入场方式 | 环境 | 样本 | 10日收盘中位 | MFE中位 | MAE中位 | 触及+5% |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ])
        for _, row in oos_context.sort_values(
            ["sector", "entry_mode", "signal_environment_bucket"]
        ).iterrows():
            mode = "跌破后次日" if row["entry_mode"] == "NEXT_OPEN" else "等待修复确认"
            lines.append(
                f"| {labels[row['sector']]} | {mode} | {row['signal_environment_bucket']} | {int(row['n'])} | "
                f"{_pct(row['median_close_return_pct'])} | {_pct(row['median_mfe_pct'])} | "
                f"{_pct(row['median_mae_pct'])} | {_rate(row['hit_5pct_rate'])} |"
            )
    lines.extend([
        "",
        "坏环境不是“无机会”，但也不是自动抄底许可。正确状态流转是：`弱市/恐慌 -> 超跌观察 -> 市场与个股修复确认 -> 次日执行 -> EMA20/时间止盈止损`。",
        "",
        "## 当前板块状态",
        "",
        "| 板块 | 60日回撤 | EMA20偏离 | 5日涨幅 | 研究阈值 | 最近跌破/龄期 | 验证状态 | 当前动作 |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ])
    basket_rows = snapshot[snapshot["scope"] == "BASKET"]
    for _, row in basket_rows.iterrows():
        lines.append(
            f"| {labels[row['sector']]} | {_pct(row['drawdown60_pct'])} | "
            f"{_pct(row['ema20_bias_pct'])} | {_pct(row['ret5_pct'])} | "
            f"{_pct(row['research_threshold_pct'])} | {row['latest_cross_date'] or '-'} / "
            f"{_integer(row['signal_age_days']) if pd.notna(row['signal_age_days']) else '-'} | "
            f"{row['validation_status']} | {row['action']} |"
        )
    lines.extend([
        "",
        "## 当前个股状态",
        "",
        "| 板块 | 股票 | 60日回撤 | EMA20偏离 | ATR偏离 | 5日/20日 | 当前动作 |",
        "|---|---|---:|---:|---:|---:|---|",
    ])
    stock_rows = snapshot[snapshot["scope"] == "STOCK"].sort_values("drawdown60_pct")
    for _, row in stock_rows.iterrows():
        lines.append(
            f"| {labels[row['sector']]} | {row['name']} `{row['symbol']}` | "
            f"{_pct(row['drawdown60_pct'])} | {_pct(row['ema20_bias_pct'])} | "
            f"{row['atr_deviation']:.2f} ATR | {_pct(row['ret5_pct'])}/{_pct(row['ret20_pct'])} | {row['action']} |"
        )
    lines.extend([
        "",
        "## 使用边界",
        "",
        "1. 静态核心篮子解决了历史成分漂移，但不能代表整个题材所有股票。",
        "2. 同一板块成员会同步下跌，股票池事件并非完全独立；因此同时保留板块篮子结果和独立事件日数量。",
        "3. 坏环境分层是当前研究代理定义，待市场环境两维模型冻结后再替换，不据此自动交易。",
        "4. 只有PASS阈值可进入影子观察；实盘还需流动性、跌停可成交性、基本面突发风险和组合仓位检查。",
    ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(as_of: str | None = None, output_root: Path = DEFAULT_OUTPUT) -> Path:
    market = build_market_frame()
    requested = pd.Timestamp(as_of) if as_of else market["date"].max()
    latest = min(requested, market["date"].max())
    output = output_root / latest.strftime("%Y%m%d")
    output.mkdir(parents=True, exist_ok=True)

    frames: dict[str, dict[str, pd.DataFrame]] = {}
    baskets: dict[str, pd.DataFrame] = {}
    event_rows: list[dict] = []
    baseline_rows: list[dict] = []
    coverage_rows = []

    for sector, definition in BASKETS.items():
        frames[sector] = {}
        for symbol, name in definition["members"].items():
            path = Path("data") / f"{symbol}_daily.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            frame = attach_market(load_stock(path), market)
            frames[sector][symbol] = frame
            coverage_rows.append({
                "sector": sector, "symbol": symbol, "name": name,
                "start": frame["date"].min().date().isoformat(),
                "end": frame["date"].max().date().isoformat(), "rows": len(frame),
            })
            event_rows.extend(collect_events(frame, sector, "STOCK", symbol, name))
            baseline_rows.extend(collect_baselines(frame, sector, "STOCK", symbol, name))

        minimum = max(2, math.ceil(len(frames[sector]) / 2))
        basket = attach_market(build_basket(frames[sector], minimum), market)
        baskets[sector] = basket
        event_rows.extend(collect_events(
            basket, sector, "BASKET", sector, definition["label"]
        ))
        baseline_rows.extend(collect_baselines(
            basket, sector, "BASKET", sector, definition["label"]
        ))

    events = pd.DataFrame(event_rows)
    baselines = pd.DataFrame(baseline_rows)
    summary, baseline_summary = summarize(events, baselines)
    selections = select_thresholds(summary)
    candidates = prospective_candidates(selections)
    contexts = context_summary(events, selections)
    snapshot = current_snapshot(frames, baskets, selections, candidates, latest)

    pd.DataFrame(coverage_rows).to_csv(output / "data_coverage.csv", index=False, encoding="utf-8-sig")
    events.to_csv(output / "event_samples.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "threshold_summary.csv", index=False, encoding="utf-8-sig")
    baseline_summary.to_csv(output / "baseline_summary.csv", index=False, encoding="utf-8-sig")
    selections.to_csv(output / "threshold_selection.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(output / "prospective_candidates.csv", index=False, encoding="utf-8-sig")
    contexts.to_csv(output / "market_context_summary.csv", index=False, encoding="utf-8-sig")
    snapshot.to_csv(output / "current_snapshot.csv", index=False, encoding="utf-8-sig")
    write_report(
        output / "study_report.md", latest, market, summary, selections,
        candidates, contexts, snapshot,
    )

    metadata = {
        "version": VERSION,
        "as_of": latest.date().isoformat(),
        "baskets": BASKETS,
        "thresholds": THRESHOLDS,
        "horizons": HORIZONS,
        "cooldown": COOLDOWN,
        "entry_modes": ["NEXT_OPEN", "REPAIR_CONFIRM"],
        "production_wiring": False,
    }
    (output / "study_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="As-of date, YYYY-MM-DD; defaults to latest local data")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = run(args.date, args.output_root)
    print(f"[{VERSION}] output -> {output}")
    print(f"Read first: {output / 'study_report.md'}")


if __name__ == "__main__":
    main()
