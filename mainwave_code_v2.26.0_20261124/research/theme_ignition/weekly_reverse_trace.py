# -*- coding: utf-8 -*-
"""Detect weekly theme ignitions, then replay pre-ignition carrier evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from mainwave_engine import capital_episode as CE
from mainwave_engine import artifacts as ART
from mainwave_engine import carrier_quality as CQ
from mainwave_engine import data as D
from mainwave_engine import state_machine as SM
from mainwave_engine.daily_run import market_calendar
from mainwave_engine import theme_semantics as TS
from research.theme_ignition import audit_replay as AUDIT


VERSION = "weekly_reverse_trace_v1_4_20260904"
REPLAY_CACHE_VERSION = "weekly_replay_v1_20260829"
PANEL_CACHE_VERSION = "weekly_limit_panel_v1_20260901"
EVENT_CACHE_VERSION = "weekly_event_cache_v1_20260904"
LEGACY_REPLAY_VERSIONS = {
    "weekly_reverse_trace_v1_1_20260829",
    "weekly_reverse_trace_v1_2_20260901",
    "weekly_reverse_trace_v1_3_20260903",
    VERSION,
}
REPLAY_COMPATIBLE_CARRIER_VERSIONS = {
    "carrier_quality_v1_1_20260829",
    "carrier_quality_v1_2_20260903",
}
OUT_ROOT = Path("results/mainwave_engine")
PANEL_COLUMNS = [
    "date", "theme", "member_count", "ignition_threshold", "limit_count", "limit_stocks",
]
LOOKBACK_SESSIONS = 30
REPLAY_SESSIONS = 60
IGNITION_COOLDOWN = 10
EVENT_TYPES = {"NEW_IGNITION_30D", "REIGNITION"}


def _member_tables(mapping: dict, universe: list[str]) -> tuple[Counter, dict[str, list[str]]]:
    member_counts: Counter[str] = Counter()
    members_by_theme: dict[str, list[str]] = defaultdict(list)
    for symbol in universe:
        for concept in set(mapping.get(symbol, {}).get("concepts", [])):
            if concept:
                member_counts[concept] += 1
                members_by_theme[concept].append(symbol)
    return member_counts, {key: sorted(value) for key, value in members_by_theme.items()}


def _panel_rows_for_dates(scan_dates: set[str], mapping: dict, universe: list[str],
                          member_counts: Counter) -> pd.DataFrame:
    daily_stocks: dict[tuple[str, str], list[str]] = defaultdict(list)
    for symbol in universe:
        concepts = set(mapping.get(symbol, {}).get("concepts", []))
        if not concepts:
            continue
        frame = D.load_tail(symbol, max(70, len(scan_dates) + 15), as_of=max(scan_dates))
        if frame is None:
            continue
        recent = frame[frame["date"].astype(str).isin(scan_dates)]
        hits = recent[recent["pct_chg"] >= D.limit_pct(symbol) * 0.985]
        for date in hits["date"].astype(str):
            for concept in concepts:
                daily_stocks[(date, concept)].append(symbol)
    rows = []
    for date in sorted(scan_dates):
        for concept, count in member_counts.items():
            stocks = sorted(set(daily_stocks.get((date, concept), [])))
            if stocks:
                rows.append({
                    "date": date, "theme": concept, "member_count": count,
                    "ignition_threshold": max(3, math.ceil(count * 0.03)),
                    "limit_count": len(stocks), "limit_stocks": "|".join(stocks),
                })
    return pd.DataFrame(rows, columns=PANEL_COLUMNS)


def _panel_from_snapshot(date: str, mapping: dict, member_counts: Counter,
                         output_root: Path = OUT_ROOT) -> pd.DataFrame | None:
    path = output_root / date.replace("-", "") / "limit_up_snapshot.csv"
    if not path.exists():
        return None
    snapshot = pd.read_csv(path, encoding="utf-8-sig")
    daily_stocks: dict[str, list[str]] = defaultdict(list)
    for symbol in snapshot.get("symbol", pd.Series(dtype=str)).astype(str):
        for concept in set(mapping.get(symbol, {}).get("concepts", [])):
            if concept:
                daily_stocks[concept].append(symbol)
    rows = []
    for concept, stocks in daily_stocks.items():
        count = int(member_counts.get(concept, 0))
        unique = sorted(set(stocks))
        if count and unique:
            rows.append({
                "date": date, "theme": concept, "member_count": count,
                "ignition_threshold": max(3, math.ceil(count * 0.03)),
                "limit_count": len(unique), "limit_stocks": "|".join(unique),
            })
    return pd.DataFrame(rows, columns=PANEL_COLUMNS)


def _mapping_hash(mapping_meta: dict) -> str:
    return ART.sha256(Path(mapping_meta.get("path", "concept_mapping.json")))


def _weekly_output(date: str, output_root: Path = OUT_ROOT) -> Path:
    return output_root / date.replace("-", "") / "weekly_reverse_trace"


def _prior_weekly_outputs(calendar: list[str], week_start: str, end_date: str,
                          output_root: Path = OUT_ROOT) -> list[Path]:
    _ = week_start
    dates = [date for date in calendar if date <= end_date][-45:]
    return [_weekly_output(date, output_root) for date in reversed(dates)]


def _prior_panel_outputs(calendar: list[str], end_date: str,
                         output_root: Path = OUT_ROOT) -> list[Path]:
    dates = [date for date in calendar if date <= end_date][-45:]
    return [_weekly_output(date, output_root) for date in reversed(dates)]


def load_or_build_limit_panel(calendar: list[str], week_start: str, end_date: str,
                              mapping: dict, mapping_meta: dict, output: Path,
                              force_rebuild: bool = False,
                              output_root: Path = OUT_ROOT
                              ) -> tuple[pd.DataFrame, dict[str, list[str]], dict]:
    start_index = max(0, calendar.index(week_start) - LOOKBACK_SESSIONS)
    required_dates = calendar[start_index:calendar.index(end_date) + 1]
    required = set(required_dates)
    universe = D.list_universe()
    member_counts, members = _member_tables(mapping, universe)
    mapping_hash = _mapping_hash(mapping_meta)
    panel = pd.DataFrame(columns=PANEL_COLUMNS)
    processed: set[str] = set()
    source = ""
    source_mapping_hash = ""

    if not force_rebuild:
        for candidate in _prior_panel_outputs(calendar, end_date, output_root):
            manifest_path = candidate / "limit_panel_manifest.json"
            panel_path = candidate / "limit_panel.csv"
            if not manifest_path.exists() or not panel_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("version") != PANEL_CACHE_VERSION:
                    continue
                cached_dates = set(manifest.get("processed_dates", []))
                if required_dates[0] not in cached_dates:
                    continue
                panel = pd.read_csv(panel_path, encoding="utf-8-sig")
                processed = cached_dates & required
                source = str(candidate)
                source_mapping_hash = str(manifest.get("mapping_sha256", ""))
                break
            except (OSError, ValueError, KeyError, TypeError):
                continue

    missing = required - processed
    mode = "CACHE_HIT"
    scanned_dates: list[str] = []
    snapshot_dates: list[str] = []
    if panel.empty and not processed:
        panel = _panel_rows_for_dates(required, mapping, universe, member_counts)
        processed = set(required)
        scanned_dates = sorted(required)
        mode = "FULL_REBUILD"
    elif missing:
        additions = []
        fallback = set()
        for date in sorted(missing):
            daily = _panel_from_snapshot(date, mapping, member_counts, output_root)
            if daily is None:
                fallback.add(date)
            else:
                additions.append(daily)
                snapshot_dates.append(date)
        if fallback:
            additions.append(_panel_rows_for_dates(fallback, mapping, universe, member_counts))
            scanned_dates.extend(sorted(fallback))
        if additions:
            panel = pd.concat([panel, *additions], ignore_index=True)
        processed.update(missing)
        mode = "INCREMENTAL"

    if len(panel):
        panel = panel[panel["date"].astype(str).isin(required)].copy()
        panel = panel.sort_values(["date", "theme"]).reset_index(drop=True)
    output.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output / "limit_panel.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "version": PANEL_CACHE_VERSION, "mapping_sha256": mapping_hash,
        "mapping_generated": mapping_meta.get("generated", ""),
        "processed_dates": required_dates, "source": source,
        "source_mapping_sha256": source_mapping_hash,
        "mode": mode, "snapshot_dates": snapshot_dates,
        "scanned_dates": scanned_dates,
    }
    (output / "limit_panel_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[reverse_trace] limit panel {mode}: cached_from={source or '-'} "
          f"snapshots={len(snapshot_dates)} scanned={len(scanned_dates)}", flush=True)
    return panel, members, manifest


def default_week_start(calendar: list[str], end_date: str) -> str:
    end = pd.Timestamp(end_date)
    same_week = [date for date in calendar
                 if pd.Timestamp(date).isocalendar()[:2] == end.isocalendar()[:2]]
    return same_week[0] if same_week else end_date


def _slice_calendar(calendar: list[str], start: str, end: str) -> list[str]:
    return [date for date in calendar if start <= date <= end]


def build_limit_panel(calendar: list[str], week_start: str, end_date: str,
                      mapping: dict) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Reconstruct point-in-time limit-up breadth from local daily bars."""
    start_index = max(0, calendar.index(week_start) - LOOKBACK_SESSIONS)
    scan_dates = set(calendar[start_index:calendar.index(end_date) + 1])
    universe = D.list_universe()
    member_counts, members = _member_tables(mapping, universe)
    return _panel_rows_for_dates(scan_dates, mapping, universe, member_counts), members


def detect_events(panel: pd.DataFrame, calendar: list[str], week_start: str,
                  end_date: str) -> pd.DataFrame:
    counts = {(row.date, row.theme): int(row.limit_count)
              for row in panel.itertuples(index=False)}
    thresholds = {(row.date, row.theme): int(row.ignition_threshold)
                  for row in panel.itertuples(index=False)}
    member_counts = {(row.date, row.theme): int(row.member_count)
                     for row in panel.itertuples(index=False)}
    stocks = {(row.date, row.theme): row.limit_stocks
              for row in panel.itertuples(index=False)}
    events = []
    for theme in sorted(panel["theme"].unique()):
        batch_dates = [date for date in calendar
                       if counts.get((date, theme), 0)
                       >= thresholds.get((date, theme), math.inf)]
        for date in _slice_calendar(calendar, week_start, end_date):
            threshold = thresholds.get((date, theme))
            if threshold is None:
                continue
            count = counts.get((date, theme), 0)
            if count < threshold:
                continue
            index = calendar.index(date)
            prior10 = calendar[max(0, index - IGNITION_COOLDOWN):index]
            prior10_counts = [counts.get((value, theme), 0) for value in prior10]
            prior_batch = any(
                counts.get((value, theme), 0)
                >= thresholds.get((value, theme), math.inf)
                for value in prior10
            )
            older_batch = any(value < date and value not in prior10 for value in batch_dates)
            if not prior_batch:
                event_type = "REIGNITION" if older_batch else "NEW_IGNITION_30D"
            else:
                event_type = ("EXPANSION" if count > max(prior10_counts, default=0)
                              else "CONTINUATION")
            later = [counts.get((value, theme), 0)
                     for value in _slice_calendar(calendar, date, end_date) if value > date]
            post_max = max(later, default=0)
            if not later:
                follow_through = "PENDING"
            elif post_max > count:
                follow_through = "EXPANDED"
            elif post_max >= threshold:
                follow_through = "HELD"
            else:
                follow_through = "FADED"
            events.append({
                "date": date, "theme": theme, "event_type": event_type,
                "member_count": member_counts[(date, theme)],
                "ignition_threshold": threshold, "limit_count": count,
                "prior10_max": max(prior10_counts, default=0),
                "post_week_max": post_max, "follow_through": follow_through,
                "limit_stocks": stocks.get((date, theme), ""),
            })
    order = {"NEW_IGNITION_30D": 0, "REIGNITION": 1, "EXPANSION": 2, "CONTINUATION": 3}
    frame = pd.DataFrame(events)
    if len(frame):
        frame = frame.assign(_order=frame["event_type"].map(order)).sort_values(
            ["date", "_order", "limit_count", "theme"], ascending=[True, True, False, True]
        ).drop(columns="_order").reset_index(drop=True)
    return frame


def _lead_sessions(first: str, t0: str, calendar: list[str]) -> int | None:
    if not first or first not in calendar:
        return None
    return calendar.index(t0) - calendar.index(first)


def _event_pct(frame: pd.DataFrame, date: str) -> float | None:
    hit = frame[frame["date"].astype(str) == date]
    return round(float(hit["pct_chg"].iloc[-1]), 2) if len(hit) else None


def replay_event(event: dict, members: list[str], calendar: list[str], mapping: dict,
                 current_quality: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    t0 = str(event["date"])
    t0_index = calendar.index(t0)
    dates = calendar[max(0, t0_index - REPLAY_SESSIONS):t0_index]
    end_date = t0
    histories = {symbol: AUDIT._load_full(symbol, end_date) for symbol in members}
    pieces = []
    for symbol, frame in histories.items():
        if frame is not None:
            pieces.append(frame[["date", "pct_chg"]].rename(columns={"pct_chg": "ret"}))
    industry_panel = (pd.concat(pieces, ignore_index=True).groupby("date", as_index=False)["ret"].median()
                      if pieces else pd.DataFrame(columns=["date", "ret"]))
    ignition_stocks = set(str(event.get("limit_stocks", "")).split("|"))
    names = D.name_map()
    summaries = []
    traces = []
    for symbol in members:
        full = histories.get(symbol)
        record = None
        episode = None
        sequence = 1
        first_probe = ""
        first_retained = ""
        first_qualified = ""
        latest = {}
        if full is not None:
            for today in dates:
                frame = full[full["date"] <= today].tail(320).reset_index(drop=True)
                if len(frame) < 70 or str(frame["date"].iloc[-1]) != today:
                    continue
                try:
                    evidence, raw = AUDIT.evaluate_day(
                        symbol, frame, today, industry_panel, record
                    )
                    next_record = SM.step(symbol, record, evidence, today,
                                          [date for date in calendar if date <= today])
                except (ValueError, KeyError, TypeError):
                    continue
                state = next_record.get("state", "OUT")
                episode_row = {"symbol": symbol, "state": state, "data_fresh": True, **raw}
                active, ended = CE.update_episode(
                    episode, episode_row, today, calendar, sequence
                )
                observed_episode = active or ended
                if ended:
                    sequence += 1
                episode = active
                metrics = CQ.metrics_from_frame(symbol, frame.tail(80), next_record)
                quality = CQ.evaluate(
                    observed_episode, metrics, next_record.get("route", raw.get("route", "")), state
                )
                stage = (observed_episode or {}).get("shadow_stage", "NONE")
                retention = (observed_episode or {}).get("retention_state", "")
                if not first_probe and stage in {"PROBED", "ABSORBING", "COILED", "ARMED", "READY"}:
                    first_probe = today
                if not first_retained and retention == "RETAINED":
                    first_retained = today
                if not first_qualified and quality["hard_gate_pass"]:
                    first_qualified = today
                latest = {
                    "pre_state": state, "pre_shadow_stage": stage,
                    "pre_retention_state": retention,
                    "pre_qualification": quality["carrier_qualification"],
                    "pre_carrier_tier": quality["carrier_tier"],
                    "pre_gate_failures": quality["gate_failures"],
                    "pre_close": metrics.get("close"), "pre_r5": metrics.get("r5"),
                    "pre_r20": metrics.get("r20"), "pre_ma20_bias": metrics.get("ma20_bias"),
                    "pre_drawdown20": metrics.get("drawdown20"),
                    "pre_repair5": metrics.get("repair5"),
                    "pre_trigger_distance_pct": metrics.get("trigger_distance_pct"),
                    "pre_stop_distance_pct": metrics.get("stop_distance_pct"),
                }
                if stage != "NONE" or raw.get("prefilter"):
                    traces.append({
                        "raw_theme": event.get("raw_theme", event["theme"]),
                        "theme": event["theme"], "t0": t0, "date": today,
                        "symbol": symbol, "state": state, "shadow_stage": stage,
                        "retention_state": retention,
                        "carrier_qualification": quality["carrier_qualification"],
                        "prefilter": raw.get("prefilter"), "imp": raw.get("imp"),
                        "fresh": raw.get("fresh"), "pit": raw.get("pit"),
                        "supply_ready": raw.get("supply_ready"),
                        "structure_ready": raw.get("structure_ready"),
                    })
                record = (next_record if next_record.get("state") != "OUT"
                          or next_record.get("out_cooldown_from") else None)
        current = current_quality.get(symbol, {})
        eve_qualified = latest.get("pre_qualification") == "QUALIFIED_LURK"
        role = "IGNITION_CARRIER" if symbol in ignition_stocks else "SAME_THEME_MEMBER"
        if role != "IGNITION_CARRIER" and eve_qualified:
            role = "EVE_QUALIFIED_LURK"
        elif role != "IGNITION_CARRIER" and first_qualified:
            role = "HISTORICAL_QUALIFIED_LOST"
        if eve_qualified:
            bottleneck = "QUALIFIED_ON_EVE"
        elif first_qualified:
            bottleneck = "QUALIFICATION_LOST_BEFORE_T0"
        elif not first_probe:
            bottleneck = "NO_CAPITAL_PROBE"
        elif not first_retained:
            bottleneck = "RETENTION_UNPROVEN"
        else:
            bottleneck = latest.get("pre_qualification", "QUALIFICATION_NOT_REACHED")
        summaries.append({
            "raw_theme": event.get("raw_theme", event["theme"]),
            "theme": event["theme"], "t0": t0, "symbol": symbol,
            "name": names.get(symbol, mapping.get(symbol, {}).get("name", "")),
            "semantic_status": event.get("semantic_status", "RAW_THEME"),
            "membership_source": event.get("membership_source", ""),
            "role": role, "event_day_pct": _event_pct(full, t0) if full is not None else None,
            "first_probe": first_probe, "probe_lead_sessions": _lead_sessions(first_probe, t0, calendar),
            "first_retained": first_retained,
            "retained_lead_sessions": _lead_sessions(first_retained, t0, calendar),
            "first_qualified": first_qualified,
            "qualified_lead_sessions": _lead_sessions(first_qualified, t0, calendar),
            "bottleneck": bottleneck, **latest,
            "current_qualification": current.get("carrier_qualification", "NO_ACTIVE_EPISODE"),
            "current_carrier_tier": current.get("carrier_tier", ""),
            "current_gate_failures": current.get("gate_failures", ""),
        })
    summary = pd.DataFrame(summaries)
    if len(summary):
        qualified = summary["first_qualified"].astype(str).ne("")
        summary["pre_relative_drawdown20"] = (
            summary["pre_drawdown20"] - summary["pre_drawdown20"].median()
        )
        summary["pre_relative_repair5"] = (
            summary["pre_repair5"] - summary["pre_repair5"].median()
        )
        summary = summary.assign(
            _qualified=np.where(qualified, 0, 1),
            _role=np.where(summary["role"] == "IGNITION_CARRIER", 0, 1),
        ).sort_values(
            ["_qualified", "pre_relative_drawdown20", "pre_relative_repair5",
             "pre_r20", "_role", "pre_trigger_distance_pct", "symbol"],
            ascending=[True, False, False, False, True, True, True], na_position="last",
        ).drop(columns=["_qualified", "_role"])
    return summary, pd.DataFrame(traces)


def _refresh_current_quality(summary: pd.DataFrame,
                             current_quality: dict[str, dict]) -> pd.DataFrame:
    summary = summary.copy()
    fields = {
        "current_qualification": "carrier_qualification",
        "current_carrier_tier": "carrier_tier",
        "current_gate_failures": "gate_failures",
    }
    for target, source in fields.items():
        summary[target] = summary["symbol"].map(
            lambda symbol: current_quality.get(symbol, {}).get(
                source, "NO_ACTIVE_EPISODE" if target == "current_qualification" else ""
            )
        )
    return summary


def _load_replay_source(calendar: list[str], week_start: str, end_date: str,
                        mapping_meta: dict, aux_meta: dict,
                        output_root: Path = OUT_ROOT) -> dict | None:
    # Event-level validation below compares T0, resolved theme and the exact member set.
    # Daily mapping/auxiliary metadata changes must not invalidate every historical replay.
    _ = mapping_meta, aux_meta
    for candidate in _prior_weekly_outputs(calendar, week_start, end_date, output_root):
        summary_path = candidate / "weekly_reverse_trace_summary.json"
        stocks_path = candidate / "pre_ignition_carriers.csv"
        traces_path = candidate / "pre_ignition_daily_trace.csv"
        if not summary_path.exists() or not stocks_path.exists() or not traces_path.exists():
            continue
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            replay_version = payload.get("replay_cache_version")
            legacy_ok = payload.get("version") in LEGACY_REPLAY_VERSIONS
            if replay_version not in (None, REPLAY_CACHE_VERSION) or not legacy_ok:
                continue
            if payload.get("carrier_version") not in REPLAY_COMPATIBLE_CARRIER_VERSIONS:
                continue
            return {
                "path": str(candidate), "payload": payload,
                "stocks": pd.read_csv(stocks_path, encoding="utf-8-sig"),
                "traces": pd.read_csv(traces_path, encoding="utf-8-sig"),
            }
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def _cached_event_frames(source: dict | None, event: dict, members: list[str],
                         current_quality: dict[str, dict]
                         ) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    if not source:
        return None
    stocks = source["stocks"]
    raw_theme = str(event.get("raw_theme", event["theme"]))
    t0 = str(event["date"])
    group = stocks[
        stocks["raw_theme"].astype(str).eq(raw_theme)
        & stocks["t0"].astype(str).eq(t0)
    ].copy()
    if set(group.get("symbol", pd.Series(dtype=str)).astype(str)) != set(members):
        return None
    if len(group) and not group["theme"].astype(str).eq(str(event["theme"])).all():
        return None
    for field in ("raw_theme", "theme", "semantic_status", "membership_source"):
        if field in event:
            group[field] = event[field]
    traces = source["traces"]
    if traces.empty:
        trace = traces.copy()
    else:
        trace = traces[
            traces["raw_theme"].astype(str).eq(raw_theme)
            & traces["t0"].astype(str).eq(t0)
        ].copy()
    return _refresh_current_quality(group, current_quality), trace


def _member_digest(members: list[str]) -> str:
    return hashlib.sha256("|".join(sorted(members)).encode("utf-8")).hexdigest()[:10]


def _event_cache_identity(event: dict, members: list[str]) -> dict:
    ignition_stocks = sorted({
        value for value in str(event.get("limit_stocks", "")).split("|") if value
    })
    return {
        "version": EVENT_CACHE_VERSION,
        "replay_version": REPLAY_CACHE_VERSION,
        "raw_theme": str(event.get("raw_theme", event["theme"])),
        "theme": str(event["theme"]),
        "t0": str(event["date"]),
        "members": sorted(set(members)),
        "ignition_stocks": ignition_stocks,
    }


def _event_cache_path(event: dict, members: list[str], output_root: Path) -> Path:
    identity = _event_cache_identity(event, members)
    encoded = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return (output_root / "_cache" / "weekly_reverse_trace"
            / EVENT_CACHE_VERSION / digest[:2] / digest)


def _read_event_cache_frame(path: Path, columns: list[str], rows: int) -> pd.DataFrame:
    if rows == 0:
        return pd.DataFrame(columns=columns)
    return pd.read_csv(path, encoding="utf-8-sig")


def _write_event_cache(event: dict, members: list[str], summary: pd.DataFrame,
                       trace: pd.DataFrame, output_root: Path = OUT_ROOT) -> Path:
    cache = _event_cache_path(event, members, output_root)
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = cache / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()
    summary.to_csv(cache / "summary.csv", index=False, encoding="utf-8-sig")
    trace.to_csv(cache / "trace.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "identity": _event_cache_identity(event, members),
        "summary_rows": len(summary), "summary_columns": list(summary.columns),
        "trace_rows": len(trace), "trace_columns": list(trace.columns),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return cache


def _load_event_cache(event: dict, members: list[str], current_quality: dict[str, dict],
                      output_root: Path = OUT_ROOT
                      ) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    cache = _event_cache_path(event, members, output_root)
    manifest_path = cache / "manifest.json"
    summary_path = cache / "summary.csv"
    trace_path = cache / "trace.csv"
    if not manifest_path.exists() or not summary_path.exists() or not trace_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("identity") != _event_cache_identity(event, members):
            return None
        source = {
            "stocks": _read_event_cache_frame(
                summary_path, manifest.get("summary_columns", []),
                int(manifest.get("summary_rows", 0)),
            ),
            "traces": _read_event_cache_frame(
                trace_path, manifest.get("trace_columns", []),
                int(manifest.get("trace_rows", 0)),
            ),
        }
        return _cached_event_frames(source, event, members, current_quality)
    except (OSError, ValueError, KeyError, TypeError, pd.errors.ParserError):
        return None


def _coverage(events: pd.DataFrame, stocks: pd.DataFrame) -> pd.DataFrame:
    def present(series: pd.Series) -> pd.Series:
        return series.notna() & series.astype(str).str.strip().ne("")

    rows = []
    for event in events[events["event_type"].isin(EVENT_TYPES)].to_dict("records"):
        group = stocks[(stocks["raw_theme"] == event["theme"])
                       & (stocks["t0"] == event["date"])]
        carriers = group[group["role"] == "IGNITION_CARRIER"]
        rows.append({
            **event,
            "event_carriers": len(carriers),
            "carriers_pre_probed": int(present(carriers["first_probe"]).sum()),
            "carriers_pre_retained": int(present(carriers["first_retained"]).sum()),
            "carriers_pre_qualified": int(present(carriers["first_qualified"]).sum()),
            "carriers_eve_qualified": int(
                carriers["pre_qualification"].astype(str).eq("QUALIFIED_LURK").sum()),
            "same_theme_eve_lurks": int((group["role"] == "EVE_QUALIFIED_LURK").sum()),
            "historical_qualified_lost": int(
                (group["role"] == "HISTORICAL_QUALIFIED_LOST").sum()),
        })
    return pd.DataFrame(rows)


def write_report(as_of: str, week_start: str, events: pd.DataFrame,
                 stocks: pd.DataFrame, coverage: pd.DataFrame, output: Path,
                 mapping_meta: dict, aux_meta: dict, run_meta: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    events.to_csv(output / "weekly_theme_events.csv", index=False, encoding="utf-8-sig")
    stocks.to_csv(output / "pre_ignition_carriers.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(output / "ignition_coverage.csv", index=False, encoding="utf-8-sig")
    event_columns = ["date", "theme", "resolved_theme", "semantic_status", "event_type",
                     "member_count", "ignition_threshold",
                     "limit_count", "prior10_max", "post_week_max", "follow_through",
                     "limit_stocks"]
    coverage_columns = ["date", "theme", "resolved_theme", "event_type", "event_carriers",
                        "carriers_pre_probed", "carriers_pre_retained",
                        "carriers_pre_qualified", "carriers_eve_qualified",
                        "same_theme_eve_lurks", "historical_qualified_lost"]
    candidate_columns = [
        "raw_theme", "theme", "t0", "symbol", "name", "role", "first_probe", "probe_lead_sessions",
        "first_retained", "retained_lead_sessions", "first_qualified",
        "qualified_lead_sessions", "pre_shadow_stage", "pre_retention_state",
        "pre_qualification", "pre_r20", "pre_trigger_distance_pct", "pre_stop_distance_pct",
        "current_qualification", "bottleneck",
    ]
    selected = stocks[
        stocks["role"].isin(["IGNITION_CARRIER", "EVE_QUALIFIED_LURK"])
    ] if len(stocks) else stocks
    lines = [
        f"# Weekly Theme Ignition Reverse Trace  {week_start} to {as_of}", "",
        f"- Version: `{VERSION}` / carrier `{CQ.VERSION}`.",
        f"- Cache: panel `{run_meta.get('panel_mode', '')}`; replay hits "
        f"`{run_meta.get('replay_hits', 0)}` / misses `{run_meta.get('replay_misses', 0)}`; "
        f"elapsed `{run_meta.get('elapsed_seconds', 0):.1f}s`.",
        f"- Point-in-time mapping: `{mapping_meta.get('generated', '')}` / `{mapping_meta.get('status', '')}` / `{mapping_meta.get('path', '')}`.",
        f"- Semantic auxiliary relationship: `{aux_meta.get('status', '')}` / updated `{aux_meta.get('updated', '')}`; it can relabel a unanimous leader chain but is not current membership truth.",
        "- Ignition: limit-up breadth >= max(3 stocks, 3% of mapped members) after 10 sessions without a batch event.",
        "- NEW_IGNITION_30D means no earlier batch in the available 30-session reconstruction; it is not an all-history claim.",
        "- Reverse trace uses only bars before T0. QUALIFIED_LURK is carrier eligibility, not an order.", "",
        "## Weekly event classification", "",
        events[event_columns].to_markdown(index=False) if len(events) else "No batch events.", "",
        "## Pre-ignition discovery coverage", "",
        coverage[coverage_columns].to_markdown(index=False) if len(coverage) else "No new/re-ignition events.", "",
        "## Ignition carriers and pre-ignition lurks", "",
        selected[candidate_columns].to_markdown(index=False) if len(selected) else "No pre-ignition candidates.", "",
        "## Interpretation boundary", "",
        "- EXPANSION/CONTINUATION is not treated as a fresh ignition; do not relabel an old cycle as a new low-position entry.",
        "- A stock-level qualified carrier still needs tight-chain membership, propagation and attention lifecycle confirmation.",
        "- A broad industry label is relabelled only when all ignition leaders share one specific concept; otherwise the raw theme is retained.",
        "- Auxiliary concept membership is explicitly marked stale and cannot by itself authorize an order.",
    ]
    (output / "weekly_reverse_trace.md").write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "as_of": as_of, "week_start": week_start, "version": VERSION,
        "carrier_version": CQ.VERSION, "mapping": mapping_meta,
        "semantic_auxiliary": aux_meta,
        "replay_cache_version": REPLAY_CACHE_VERSION,
        "cache": run_meta,
        "event_counts": events["event_type"].value_counts().to_dict() if len(events) else {},
        "strict_events": len(coverage), "order_generation": False,
    }
    (output / "weekly_reverse_trace_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run(end_date: str, week_start: str = "", force_rebuild: bool = False) -> int:
    started = time.perf_counter()
    calendar = market_calendar(end_date)
    week_start = week_start or default_week_start(calendar, end_date)
    if end_date not in calendar or week_start not in calendar:
        raise RuntimeError("week start/end must be trading dates in the local market calendar")
    mapping, mapping_meta = TS.load_mapping_as_of(end_date)
    if not mapping:
        raise RuntimeError(f"no concept mapping available as of {end_date}")
    aux_by_symbol, aux_members, aux_meta = TS.load_aux_relationships()
    output = _weekly_output(end_date)
    panel, members, panel_meta = load_or_build_limit_panel(
        calendar, week_start, end_date, mapping, mapping_meta, output,
        force_rebuild=force_rebuild,
    )
    events = detect_events(panel, calendar, week_start, end_date)
    event_mappings: dict[str, dict] = {}
    point_members_by_date: dict[str, dict[str, set[str]]] = {}
    if len(events):
        for event_date in events["date"].astype(str).unique():
            historical_mapping, _ = TS.load_mapping_as_of(event_date)
            event_mappings[event_date] = historical_mapping or mapping
            point_members_by_date[event_date] = TS.members_by_concept(
                event_mappings[event_date]
            )
        semantic_rows = []
        for row in events.to_dict("records"):
            leaders = [value for value in str(row.get("limit_stocks", "")).split("|") if value]
            semantic_rows.append(TS.resolve_event_theme(
                row["theme"], leaders, event_mappings[str(row["date"])], aux_by_symbol
            ))
        for column in ("resolved_theme", "semantic_status", "semantic_confidence",
                       "semantic_evidence"):
            events[column] = [row[column] for row in semantic_rows]
    strict = events[events["event_type"].isin(EVENT_TYPES)] if len(events) else events
    quality_path = output.parent / "carrier_qualification.csv"
    current_frame = (pd.read_csv(quality_path, encoding="utf-8-sig")
                     if quality_path.exists() else pd.DataFrame())
    current_quality = {
        row["symbol"]: row for row in current_frame.to_dict("records")
    } if len(current_frame) else {}
    replay_source = None if force_rebuild else _load_replay_source(
        calendar, week_start, end_date, mapping_meta, aux_meta
    )
    print(f"[reverse_trace] strict_events={len(strict)} replay_cache="
          f"{replay_source['path'] if replay_source else '-'}", flush=True)
    stock_frames = []
    trace_frames = []
    replay_hits = 0
    event_cache_hits = 0
    report_cache_hits = 0
    replay_misses = 0
    for index, event in enumerate(strict.to_dict("records"), 1):
        event_started = time.perf_counter()
        leaders = [value for value in str(event.get("limit_stocks", "")).split("|") if value]
        event_date = str(event["date"])
        event_mapping = event_mappings[event_date]
        theme_members, membership_source = TS.resolved_members(
            event, leaders, point_members_by_date[event_date], aux_members
        )
        replay_spec = {
            **event, "raw_theme": event["theme"], "theme": event["resolved_theme"],
            "membership_source": membership_source,
        }
        cached = None if force_rebuild else _load_event_cache(
            replay_spec, theme_members, current_quality
        )
        if cached is not None:
            summary, trace = cached
            replay_hits += 1
            event_cache_hits += 1
            cache_state = "EVENT_HIT"
        else:
            cached = _cached_event_frames(
                replay_source, replay_spec, theme_members, current_quality
            )
            if cached is not None:
                summary, trace = cached
                replay_hits += 1
                report_cache_hits += 1
                cache_state = "REPORT_HIT"
            else:
                print(f"[reverse_trace] [{index}/{len(strict)}] replay start "
                      f"{event['date']} {event['theme']} -> {event['resolved_theme']} "
                      f"members={len(theme_members)} hash={_member_digest(theme_members)}",
                      flush=True)
                summary, trace = replay_event(
                    replay_spec, theme_members, calendar, event_mapping, current_quality
                )
                replay_misses += 1
                cache_state = "MISS"
            _write_event_cache(replay_spec, theme_members, summary, trace)
        stock_frames.append(summary)
        trace_frames.append(trace)
        print(f"[reverse_trace] [{index}/{len(strict)}] {cache_state} "
              f"{event['date']} {event['theme']} -> {event['resolved_theme']} "
              f"{event['event_type']} "
              f"members={len(theme_members)} elapsed={time.perf_counter() - event_started:.1f}s",
              flush=True)
    stocks = pd.concat(stock_frames, ignore_index=True) if stock_frames else pd.DataFrame()
    traces = pd.concat(trace_frames, ignore_index=True) if trace_frames else pd.DataFrame()
    coverage = _coverage(strict, stocks) if len(stocks) else pd.DataFrame()
    output.mkdir(parents=True, exist_ok=True)
    traces.to_csv(output / "pre_ignition_daily_trace.csv", index=False, encoding="utf-8-sig")
    run_meta = {
        "panel_mode": panel_meta.get("mode", ""),
        "panel_source": panel_meta.get("source", ""),
        "replay_source": replay_source["path"] if replay_source else "",
        "replay_hits": replay_hits, "replay_misses": replay_misses,
        "event_cache_hits": event_cache_hits,
        "report_cache_hits": report_cache_hits,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "force_rebuild": force_rebuild,
    }
    write_report(end_date, week_start, events, stocks, coverage, output,
                 mapping_meta, aux_meta, run_meta)
    print(f"[reverse_trace] events={len(events)} strict={len(strict)} stocks={len(stocks)} -> "
          f"{output / 'weekly_reverse_trace.md'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--week-start", default="")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="ignore incremental panel and event replay caches")
    args = parser.parse_args()
    end_date = args.end_date if "-" in args.end_date else (
        f"{args.end_date[:4]}-{args.end_date[4:6]}-{args.end_date[6:]}"
    )
    week_start = ""
    if args.week_start:
        week_start = args.week_start if "-" in args.week_start else (
            f"{args.week_start[:4]}-{args.week_start[4:6]}-{args.week_start[6:]}"
        )
    return run(end_date, week_start, force_rebuild=args.force_rebuild)


if __name__ == "__main__":
    raise SystemExit(main())
