"""Describe every registered case's post-event path, not a strategy backtest."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mainwave_engine import historical_cases as HC
from mainwave_engine import stock_analyzer as SA


def path_metrics(frame: pd.DataFrame, date: str) -> dict:
    dates = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
    matches = np.flatnonzero(dates.eq(date).to_numpy())
    if len(matches) != 1:
        raise ValueError(f"Missing or duplicated event date: {date}")
    i = int(matches[0])
    close = frame.return_index.to_numpy(float)
    scale = close / frame.close.to_numpy(float)
    high, low = frame.high.to_numpy(float) * scale, frame.low.to_numpy(float) * scale
    base = close[i]
    result = {"event_date": date, "following_sessions": len(frame) - i - 1,
              "prior20_pct": (base / close[i - 20] - 1) * 100 if i >= 20 else None}
    for n in (5, 10, 20):
        full = i + n < len(frame)
        result[f"r{n}_pct"] = (close[i + n] / base - 1) * 100 if full else None
        result[f"mfe{n}_pct"] = (max(high[i + 1:i + n + 1]) / base - 1) * 100 if full else None
        result[f"mae{n}_pct"] = (min(low[i + 1:i + n + 1]) / base - 1) * 100 if full else None
    return result


def build(output: Path, as_of: str) -> dict:
    library = HC.case_list()
    rows, pictures = [], []
    for case in library["cases"]:
        payload = HC.case_detail(case["id"])
        review = payload["case_review"]
        event_date = review["route"]["target_ignition_at"]
        _, frame = SA._data_frame(case["symbol"], as_of)
        dates = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
        i = int(np.flatnonzero(dates.eq(event_date).to_numpy())[0])
        history = payload["lifecycle"]["history"]
        at = next(r for r in history if r["date"] == event_date)
        end_of_cycle = [r for r in history if r["date"] >= event_date and r["cycle_id"] == at["cycle_id"]]
        confirmed = [r for r in history if r.get("setup_status") == "CONFIRMED"]
        metrics = path_metrics(frame, event_date)
        row = {"id": case["id"], "symbol": case["symbol"], "name": case["name"],
               "category": case["category"], "category_label": case["category_label"],
               "event_kind": review["route"]["target_ignition_kind"],
               "window_end": case["window_end"], "stage_at_event": at["cycle_stage"],
               "breakout_at_event": at.get("breakout_status", ""),
               "prior_cycle_start": at["cycle_started_at"],
               "same_cycle_failed_at": next((r["date"] for r in end_of_cycle if r["cycle_stage"] == "FAILED"), ""),
               "confirmations": [{"date": r["date"], "linked": bool(r.get("setup_breakout_id"))} for r in confirmed],
               "trade_status": review["trade"]["status"], **metrics}
        row["original_window_metrics"] = path_metrics(frame[dates.le(case["window_end"])], event_date)
        if case.get("t0"):
            row["theme_event_metrics"] = path_metrics(frame, case["t0"])
        rows.append(row)
        start, stop = max(0, i - 20), min(len(frame), i + 21)
        section = frame.iloc[start:stop]
        base = frame.iloc[i].return_index
        scale = section.return_index / section.close / base * 100
        pictures.append((row, np.arange(start - i, stop - i),
                         section.open.to_numpy() * scale.to_numpy(),
                         section.high.to_numpy() * scale.to_numpy(),
                         section.low.to_numpy() * scale.to_numpy(), section.return_index.to_numpy() / base * 100))
    groups = {}
    for category in ("ALL", "COMPLETE", "FAILED_LAUNCH", "AUDIT"):
        subset = [r for r in rows if category == "ALL" or r["category"] == category
                  or (category == "AUDIT" and r["category"].startswith("DISCOVERY_"))]
        full = [r for r in subset if r["r20_pct"] is not None]
        groups[category] = {"cases": len(subset), "full20": len(full),
            "positive20": sum(r["r20_pct"] > 0 for r in full),
            "negative20": sum(r["r20_pct"] < 0 for r in full),
            "up10pct20": sum(r["r20_pct"] >= 10 for r in full),
            "down10pct20": sum(r["r20_pct"] <= -10 for r in full),
            "mfe20_ge10": sum(r["mfe20_pct"] >= 10 for r in full),
            "mfe20_ge20": sum(r["mfe20_pct"] >= 20 for r in full),
            "mfe20_le5": sum(r["mfe20_pct"] <= 5 for r in full),
            **{f"median_{k}": float(np.median([r[k] for r in full])) if full else None
               for k in ("r20_pct", "mfe20_pct", "mae20_pct", "prior20_pct")}}
    result = {"scope": f"All registered cases; price outcomes checked through {as_of}; original case IDs/windows/labels unchanged",
              "analysis_as_of": as_of,
              "basis": "Adjusted event-day close baseline; event-day high/low excluded; these are price paths, not fills or strategy returns",
              "groups": groups, "rows": rows}
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=lambda v: v.item()), encoding="utf-8")
    columns = ["name", "symbol", "category_label", "event_date", "prior20_pct", "r5_pct", "r20_pct", "mfe20_pct", "mae20_pct", "stage_at_event", "breakout_at_event"]
    text = ["# 历史案例逐例走势审计", "", result["scope"], "", result["basis"], "",
            "点火后20日为描述性观察窗口，不是经优化的成功阈值；不满20日不纳入20日统计。MFE为摸到的高点，不是可兑现收益。", "",
            pd.DataFrame(groups).T.to_markdown(floatfmt=".2f"), "",
            pd.DataFrame(rows)[columns].to_markdown(index=False, floatfmt=".2f")]
    (output / "audit.md").write_text("\n".join(text), encoding="utf-8")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    for group, subset in (("cycle", pictures[:11]), ("audit", pictures[11:])):
        cols = 3
        fig, axes = plt.subplots(math.ceil(len(subset) / cols), cols,
                                 figsize=(15, 3 * math.ceil(len(subset) / cols)), squeeze=False)
        for ax, (row, x, op, hi, lo, cl) in zip(axes.flat, subset):
            colors = np.where(cl >= op, "#b53640", "#16734a")
            ax.vlines(x, lo, hi, colors=colors, linewidth=.8)
            ax.bar(x, np.maximum(np.abs(cl - op), .05), bottom=np.minimum(op, cl), color=colors, width=.65)
            ax.axvline(0, color="#a06c0a", linestyle="--", linewidth=1.2)
            ax.axhline(100, color="#8b939b", linewidth=.6)
            r20 = f"{row['r20_pct']:+.1f}%" if row["r20_pct"] is not None else "未满20日"
            ax.set_title(f"{row['name']} {row['symbol']}\n{row['event_date']}  后20日 {r20}", fontsize=10)
            ax.set_xlim(-21, 21)
            ax.grid(alpha=.15)
            ax.tick_params(labelsize=8)
        for ax in list(axes.flat)[len(subset):]:
            ax.set_visible(False)
        fig.suptitle("事件日前后20个活跃交易日；0=标注点火日，价格归一为100；非交易收益", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .98))
        fig.savefig(output / f"{group}_candles.png", dpi=130)
        plt.close(fig)
    print(json.dumps(groups, ensure_ascii=False, indent=2, default=lambda v: v.item()))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/mainwave_engine/case_quality_audit_20260910"))
    parser.add_argument("--as-of", default=SA.latest_report_date())
    args = parser.parse_args()
    build(args.output, args.as_of)
