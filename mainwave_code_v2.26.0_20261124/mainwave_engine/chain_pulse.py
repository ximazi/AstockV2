# -*- coding: utf-8 -*-
"""Cycle-aware tight-chain observation report (P0, 2026-08-28).

This module is report-only. Lifecycle thresholds are placeholders and never
feed READY/ARMED/TRIGGERED production transitions.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import pandas as pd

from . import theme_lifecycle as TL
from .theme_registry import latest_themes, themes_as_of


OUT_ROOT = Path("results/mainwave_engine")
LEGACY_LOG = OUT_ROOT / "chain_pulse_log.csv"


def _load_stock_store() -> dict:
    path = OUT_ROOT / "state_store.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))["symbols"]
    except Exception:
        return {}


def _write_compat_log(as_of: str, snapshots: list[dict]) -> None:
    rows = pd.DataFrame(
        [
            {
                "date": as_of,
                "chain": row["id"],
                "status": row["stage"],
                "cycle": row["cycle"],
                "n_pulse": row["pulse_members"],
                "batch_days": int(bool(row["last_batch_event"] == as_of)),
                "vol_ratio": row["volume_ratio"],
                "above_ma20": round(row["breadth"] * row["n_members"]),
                "n": row["n_members"],
            }
            for row in snapshots
        ]
    )
    if LEGACY_LOG.exists():
        old = pd.read_csv(LEGACY_LOG, encoding="utf-8-sig")
        old = old[~((old["date"] == as_of) & old["chain"].isin(rows["chain"]))]
        rows = pd.concat([old, rows], ignore_index=True)
    rows.sort_values(["date", "chain"]).to_csv(LEGACY_LOG, index=False, encoding="utf-8-sig")


def _self_check() -> None:
    storage = next(row for row in latest_themes() if row["id"] == "storage")
    result = TL.analyze_theme(storage, "2026-08-27")["snapshot"]
    if not result["last_ignition"] or result["cycle"] < 2:
        raise RuntimeError(f"存储周期回填失败: {result}")
    print(
        "[self-check] storage 2026-08-27 "
        f"stage={result['stage']} cycle={result['cycle']} "
        f"last_ignition={result['last_ignition']} drawdown={result['drawdown_from_120d_peak_pct']}%"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    dash = args.date if "-" in args.date else f"{args.date[:4]}-{args.date[4:6]}-{args.date[6:]}"
    out = OUT_ROOT / dash.replace("-", "")
    out.mkdir(parents=True, exist_ok=True)

    themes = themes_as_of(dash)
    if not themes:
        raise RuntimeError(f"{dash} 没有已生效的紧链注册版本")
    store = _load_stock_store()
    lines = [
        f"# 紧链生命周期监控  {dash}",
        "",
        "- 七态: DORMANT → SEED → FERMENTING → IGNITED → DIFFUSING → CROWDED → RETREAT;",
        "- 阈值状态: 占位待校准，仅观察，不进入READY/ARMED/TRIGGERED生产转移;",
        "- 周期上下文由当前注册成员回填，仅用于识别第N轮，不代表历史时点已知该题材。",
        "",
    ]
    snapshots = []
    for theme in themes:
        result = TL.analyze_theme(theme, dash)
        snap = result["snapshot"]
        snapshots.append(snap)
        pulse_names = [
            f"{theme['members'][symbol]}({dates[-1][5:]})"
            for symbol, dates in result["active_pulses"].items()
        ]
        pool = [
            f"{name}({store.get(symbol, {}).get('state')})"
            for symbol, name in theme["members"].items()
            if store.get(symbol, {}).get("state") not in (None, "OUT")
        ]
        age = snap["days_since_ignition"] if snap["days_since_ignition"] != "" else "无历史点火"
        lines += [
            f"## {snap['label']} · **{snap['stage']}** · 第{snap['cycle']}轮候选周期",
            "",
            f"- 注册生效: {snap['active_from']} | 有效成员 {snap['n_members']} | 回填上下文: 是",
            f"- 上轮点火: {snap['last_ignition'] or '无'} | 距上轮 {age} 个交易日 | 120日峰值回撤 {snap['drawdown_from_120d_peak_pct']:+.1f}%",
            f"- L钟: 近60日保留脉冲 {snap['pulse_members']}只: {', '.join(pulse_names) if pulse_names else '无'}",
            f"- I钟: 今日涨停 {snap['lu_n']}只 | 最近批量事件 {snap['last_batch_event'] or '无'}",
            f"- A/B钟: 量比中位 {snap['volume_ratio']} | >MA20 {snap['breadth']:.0%}",
            f"- 引擎在池: {', '.join(pool) if pool else '无'}",
            "",
        ]

    TL.save_snapshots(dash, snapshots)
    _write_compat_log(dash, snapshots)
    (out / "chain_pulse.md").write_text("\n".join(lines), encoding="utf-8")
    for snap in snapshots:
        print(
            f"[chain_lifecycle] {snap['label']:<12} {snap['stage']:<11} "
            f"cycle={snap['cycle']} pulses={snap['pulse_members']} drawdown={snap['drawdown_from_120d_peak_pct']:+.1f}%"
        )
    print(f"[chain_lifecycle] -> {out / 'chain_pulse.md'}")
    if args.self_check:
        _self_check()
    return 0


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.exit(main())
