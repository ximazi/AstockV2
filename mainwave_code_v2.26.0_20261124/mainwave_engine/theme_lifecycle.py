# -*- coding: utf-8 -*-
"""Persistent, cycle-aware lifecycle for registered tight chains.

All thresholds are observation-layer v0 placeholders. They do not feed the
READY/ARMED production state machine and may only be calibrated on 2022-2024.
Historical reconstruction uses current registered members and is therefore
marked as backfilled context, not point-in-time discovery evidence.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


OUT_ROOT = Path("results/mainwave_engine")
STORE = OUT_ROOT / "theme_state_store.json"
HISTORY = OUT_ROOT / "theme_lifecycle_history.csv"

PULSE_PCT = 7.0
PULSE_WINDOW = 60
BATCH_LIMIT_UP = 3
BATCH_COLD_WINDOW = 10
IGNITED_DAYS = 15
NEW_CYCLE_GAP = 40
MIN_STAGE_DAYS = 3

STAGES = {"DORMANT", "SEED", "FERMENTING", "IGNITED", "DIFFUSING", "CROWDED", "RETREAT"}


def limit_threshold(symbol: str) -> float:
    return 19.7 if symbol.startswith(("sh_68", "sz_30")) else 9.85


def market_calendar(as_of: str) -> list[str]:
    df = pd.read_csv("data/000001_SH_index.csv", usecols=["trade_date"])

    def norm(value) -> str:
        text = str(value)
        return text if "-" in text else f"{text[:4]}-{text[4:6]}-{text[6:8]}"

    return [d for d in sorted({norm(v) for v in df["trade_date"]}) if d <= as_of]


def _member_panel(theme: dict, calendar: list[str]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    panel: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for symbol in theme["members"]:
        try:
            df = pd.read_csv(
                Path("data") / f"{symbol}_daily.csv",
                usecols=["date", "close", "pct_chg", "volume", "amount"],
            )
        except Exception:
            missing.append(symbol)
            continue
        df["date"] = df["date"].astype(str)
        df = df[df["date"].isin(calendar)].set_index("date").reindex(calendar)
        if df["close"].notna().sum() < 120:
            missing.append(symbol)
            continue
        panel[symbol] = df
    return panel, missing


def _sessions_since(index: int, prior: int | None) -> int:
    return 10**6 if prior is None else index - prior


def _stage_timeline(metrics: pd.DataFrame, theme: dict) -> pd.DataFrame:
    stage = "DORMANT"
    stage_since = 0
    cycle = 0
    cycle_ignition_idx: int | None = None
    last_batch_idx: int | None = None
    retreated_in_cycle = False
    known_anchors = set(theme.get("historical_cycle_anchors", []))
    active_from = theme["active_from"]
    rows = []

    for i, row in metrics.iterrows():
        event_date = row["date"]
        batch_event = bool(row["batch_event"] or event_date in known_anchors)
        days_since_cycle_ignition = _sessions_since(i, cycle_ignition_idx)
        ordinary = stage

        crowded = bool(row["ret20"] > 0.40 or row["breadth_full_5d"])
        retreat = bool(
            cycle_ignition_idx is not None
            and (row["drawdown_120"] <= -0.25 or row["breadth"] < 0.30)
        )
        fermenting = bool(
            row["pulse_members"] >= 3
            and days_since_cycle_ignition >= NEW_CYCLE_GAP
            and not crowded
        )

        if batch_event:
            starts_new_cycle = event_date in known_anchors or (
                event_date >= active_from
                and (cycle_ignition_idx is None or (
                    days_since_cycle_ignition >= NEW_CYCLE_GAP
                    and stage == "FERMENTING"
                ))
            )
            if starts_new_cycle:
                cycle += 1
                cycle_ignition_idx = i
                retreated_in_cycle = False
            last_batch_idx = i
            target = "IGNITED"
            hard_change = True
        elif fermenting and retreated_in_cycle and stage in {"RETREAT", "FERMENTING"}:
            target = "FERMENTING"
            hard_change = False
        elif retreat:
            target = "RETREAT"
            hard_change = True
            retreated_in_cycle = True
        elif crowded:
            target = "CROWDED"
            hard_change = True
        elif cycle_ignition_idx is not None and _sessions_since(i, last_batch_idx) <= IGNITED_DAYS:
            target = "IGNITED"
            hard_change = False
        elif fermenting:
            target = "FERMENTING"
            hard_change = False
        elif cycle_ignition_idx is not None and row["breadth"] >= 0.60 and row["drawdown_120"] > -0.20:
            target = "DIFFUSING"
            hard_change = False
        elif 1 <= row["pulse_members"] <= 2:
            target = "SEED"
            hard_change = False
        elif stage == "RETREAT" and row["no_pulse_days"] < NEW_CYCLE_GAP:
            target = "RETREAT"
            hard_change = False
        else:
            target = "DORMANT"
            hard_change = False

        age = i - stage_since
        if target != stage and (hard_change or age >= MIN_STAGE_DAYS):
            stage = target
            stage_since = i

        display_cycle = max(1, cycle)
        if stage in {"SEED", "FERMENTING"} and cycle_ignition_idx is not None:
            display_cycle = cycle + 1
        rows.append(
            {
                "stage": stage,
                "stage_since": metrics.at[stage_since, "date"],
                "cycle": display_cycle,
                "last_ignition": metrics.at[cycle_ignition_idx, "date"] if cycle_ignition_idx is not None else "",
                "last_batch_event": metrics.at[last_batch_idx, "date"] if last_batch_idx is not None else "",
                "days_since_ignition": days_since_cycle_ignition if cycle_ignition_idx is not None else "",
                "previous_stage": ordinary,
            }
        )
    return pd.concat([metrics.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def analyze_theme(theme: dict, as_of: str) -> dict:
    calendar = market_calendar(as_of)
    panel, missing = _member_panel(theme, calendar)
    if len(panel) < 3:
        raise RuntimeError(f"题材 {theme['id']} 有效成员不足3只")

    dates = calendar
    n = len(dates)
    ret_matrix = np.vstack([df["pct_chg"].to_numpy(float) for df in panel.values()])
    close_matrix = np.vstack([df["close"].to_numpy(float) for df in panel.values()])
    volume_matrix = np.vstack([df["volume"].to_numpy(float) for df in panel.values()])
    chain_ret = np.nanmean(ret_matrix, axis=0) / 100.0
    chain_index = np.cumprod(1 + np.nan_to_num(chain_ret, nan=0.0))
    chain_series = pd.Series(chain_index)
    peak120 = chain_series.rolling(120, min_periods=20).max().to_numpy()
    peak_idx = np.zeros(n, dtype=int)
    for i in range(n):
        lo = max(0, i - 119)
        peak_idx[i] = lo + int(np.nanargmax(chain_index[lo : i + 1]))

    ma20 = np.vstack([pd.Series(row).rolling(20).mean().to_numpy() for row in close_matrix])
    breadth = np.nanmean(close_matrix > ma20, axis=0)
    ratio_matrix = np.vstack(
        [
            pd.Series(row).rolling(10).mean().to_numpy()
            / pd.Series(row).rolling(60).mean().to_numpy()
            for row in volume_matrix
        ]
    )
    # Avoid numpy's all-NaN warning during the rolling-window warm-up rows.
    volume_ratio = pd.DataFrame(ratio_matrix).median(axis=0, skipna=True).to_numpy()
    lu_n = np.zeros(n, dtype=int)
    pulse_members = np.zeros(n, dtype=int)
    member_pulses: dict[str, list[str]] = {}
    for member_i, symbol in enumerate(panel):
        pct = ret_matrix[member_i]
        close = close_matrix[member_i]
        lu_n += np.where(np.isfinite(pct) & (pct >= limit_threshold(symbol)), 1, 0)
        pulses = np.where(np.isfinite(pct) & (pct >= PULSE_PCT))[0]
        member_pulses[symbol] = [dates[j] for j in pulses]
        for i in range(n):
            recent = pulses[(pulses >= max(0, i - PULSE_WINDOW + 1)) & (pulses <= i)]
            if len(recent) and np.isfinite(close[i]) and np.any(close[i] >= close[recent]):
                pulse_members[i] += 1

    batch_event = np.zeros(n, dtype=bool)
    for i in range(BATCH_COLD_WINDOW, n):
        batch_event[i] = bool(
            lu_n[i] >= BATCH_LIMIT_UP
            and np.max(lu_n[i - BATCH_COLD_WINDOW : i]) < BATCH_LIMIT_UP
        )

    no_pulse_days = np.zeros(n, dtype=int)
    streak = 0
    for i, count in enumerate(pulse_members):
        streak = streak + 1 if count == 0 else 0
        no_pulse_days[i] = streak

    breadth_full = pd.Series(breadth >= 0.999).rolling(5, min_periods=5).sum().eq(5).to_numpy()
    metrics = pd.DataFrame(
        {
            "date": dates,
            "chain_index": chain_index,
            "ret20": chain_series.pct_change(20).to_numpy(),
            "breadth": breadth,
            "volume_ratio": volume_ratio,
            "breadth_full_5d": breadth_full,
            "pulse_members": pulse_members,
            "lu_n": lu_n,
            "batch_event": batch_event,
            "drawdown_120": chain_index / peak120 - 1,
            "peak_120_date": [dates[j] for j in peak_idx],
            "no_pulse_days": no_pulse_days,
        }
    )
    timeline = _stage_timeline(metrics, theme)
    last = timeline.iloc[-1]
    snapshot = {
        "id": theme["id"],
        "label": theme["label"],
        "as_of": as_of,
        "active_from": theme["active_from"],
        "stage": last["stage"],
        "stage_since": last["stage_since"],
        "cycle": int(last["cycle"]),
        "last_ignition": last["last_ignition"],
        "last_batch_event": last["last_batch_event"],
        "days_since_ignition": last["days_since_ignition"],
        "peak_120_date": last["peak_120_date"],
        "drawdown_from_120d_peak_pct": round(float(last["drawdown_120"]) * 100, 1),
        "pulse_members": int(last["pulse_members"]),
        "breadth": round(float(last["breadth"]), 3),
        "volume_ratio": round(float(last["volume_ratio"]), 2) if np.isfinite(last["volume_ratio"]) else None,
        "lu_n": int(last["lu_n"]),
        "n_members": len(panel),
        "missing_members": missing,
        "backfilled_from_current_members": True,
        "historical_cycle_anchors": theme.get("historical_cycle_anchors", []),
        "threshold_status": "placeholder_observation_only",
    }
    active_pulses = {}
    current_i = len(dates) - 1
    for symbol, pulse_dates in member_pulses.items():
        keep = [d for d in pulse_dates if current_i - dates.index(d) < PULSE_WINDOW]
        if keep:
            active_pulses[symbol] = keep
    return {"snapshot": snapshot, "timeline": timeline, "active_pulses": active_pulses}


def save_snapshots(as_of: str, snapshots: list[dict]) -> None:
    previous = ""
    if STORE.exists():
        try:
            previous = json.loads(STORE.read_text(encoding="utf-8")).get("_meta", {}).get("as_of", "")
        except Exception as exc:
            raise RuntimeError(f"题材生命周期状态库损坏: {STORE} ({exc})") from exc
    if previous and as_of < previous:
        raise RuntimeError(f"拒绝用历史日期 {as_of} 覆盖题材状态库 {previous}")

    payload = {
        "_meta": {
            "as_of": as_of,
            "threshold_status": "placeholder_observation_only",
            "backfill_warning": "Historical stages use current registered members and are context only.",
        },
        "themes": {row["id"]: row for row in snapshots},
    }
    STORE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STORE.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, STORE)

    new_rows = pd.DataFrame(snapshots)
    if HISTORY.exists():
        old = pd.read_csv(HISTORY, encoding="utf-8-sig")
        old = old[~((old["as_of"] == as_of) & old["id"].isin(new_rows["id"]))]
        new_rows = pd.concat([old, new_rows], ignore_index=True)
    new_rows.sort_values(["as_of", "id"]).to_csv(HISTORY, index=False, encoding="utf-8-sig")
