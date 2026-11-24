"""Challenge existing gates on known cases; not a backtest or validation set."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

import x_ai_main_up_scaner as legacy
from mainwave_engine import carrier_quality as CQ


def snapshot(raw: pd.DataFrame, symbol: str, date: str) -> dict:
    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame[frame.date.le(pd.Timestamp(date)) & frame.amount.gt(0)]
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if frame.empty or frame.iloc[-1].date != pd.Timestamp(date):
        raise ValueError(f"Missing active event session: {symbol} {date}")
    if len(frame) < 61:
        raise ValueError("Need at least 61 active sessions")
    prepared = legacy.prepare_signals(frame)
    i = len(prepared) - 1
    count, yang = legacy.check_continuous_yang(prepared, i)
    signals = {
        "legacy_limit_proxy": bool(prepared.iloc[i - 20:i].is_limit_up.any()),
        "legacy_yang": bool(yang),
        "legacy_gap": bool(prepared.iloc[i].is_gap_up),
        "legacy_volume": bool(prepared.iloc[i].is_double_vol),
    }
    recent_four = [str(prepared.iloc[j].date.date()) for j in range(max(25, i - 19), i + 1)
                   if prepared.iloc[j - 20:j].is_limit_up.any()
                   and legacy.check_continuous_yang(prepared, j)[1]
                   and prepared.iloc[j].is_gap_up and prepared.iloc[j].is_double_vol]
    metrics = CQ.metrics_from_frame(symbol, frame)
    # Keep raw-code reproduction separate from a corporate-action sensitivity check.
    adjusted = frame.copy()
    index = (1 + pd.to_numeric(frame.pct_chg) / 100).cumprod() * 100
    scale = index / frame.close
    for column in ("open", "high", "low", "close"):
        adjusted[column] = frame[column] * scale
    adjusted_metrics = CQ.metrics_from_frame(symbol, adjusted)
    last = frame.iloc[-1]
    rng = float(last.high - last.low)
    return {
        **signals, "legacy_four": all(signals.values()),
        "legacy_four_seen20": bool(recent_four), "legacy_four_dates20": recent_four,
        "legacy_signal_count": sum(signals.values()),
        "legacy_prior_bullish_bodies": count,
        "raw_extension": metrics["extension_state"] == "EXTENDED",
        "raw_distribution": bool(metrics["distribution_risk"]),
        "adjusted_extension": adjusted_metrics["extension_state"] == "EXTENDED",
        "adjusted_distribution": bool(adjusted_metrics["distribution_risk"]),
        "adjusted_r20": adjusted_metrics["r20"],
        "adjusted_ma20_bias": adjusted_metrics["ma20_bias"],
        "event_close_location": float((last.close - last.low) / rng) if rng > 0 else None,
        "event_amount_vs_prior20": float(last.amount / frame.amount.iloc[-21:-1].median()),
    }


def summarize(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "positive20": sum(r["r20_pct"] > 0 for r in rows),
        "down10pct20": sum(r["r20_pct"] <= -10 for r in rows),
        "up10pct20": sum(r["r20_pct"] >= 10 for r in rows),
        "mfe20_ge20": sum(r["mfe20_pct"] >= 20 for r in rows),
    }


def build(source: Path, output: Path) -> dict:
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows, data_hashes = [], {}
    for case in payload["rows"]:
        path = Path("data") / f"{case['symbol']}_daily.csv"
        raw = pd.read_csv(path, encoding="utf-8-sig")
        data_hashes[case["symbol"]] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({**{k: case[k] for k in (
            "id", "symbol", "name", "event_date", "category", "stage_at_event",
            "r20_pct", "mfe20_pct", "mae20_pct")},
            **snapshot(raw, case["symbol"], case["event_date"])})
    groups = {"ALL": summarize(rows)}
    for key in ("legacy_limit_proxy", "legacy_yang", "legacy_gap", "legacy_volume",
                "legacy_four", "legacy_four_seen20", "raw_extension", "raw_distribution",
                "adjusted_extension", "adjusted_distribution"):
        for value in (True, False):
            groups[f"{key}={value}"] = summarize([r for r in rows if r[key] == value])
    sources = [source, Path(__file__), Path(legacy.__file__), Path(CQ.__file__)]
    result = {
        "analysis_as_of": payload["analysis_as_of"],
        "scope": "All 37 previously inspected registered cases; biased diagnostic cohort; no OOS inference",
        "outcomes": "Prior audit adjusted event-close price paths, not entry/exit profits; no parameter fitting",
        "limitations": [
            "Legacy uses raw close changes >9.5% across boards: limit-up proxy, not exchange-verified limit events",
            "Legacy bullish bodies and limit history exclude event session; gap and double volume use event close data",
            "Legacy volume baseline is five prior sessions; no sustained-volume confirmation in this implementation",
            "legacy_four_seen20 searches the same full conjunction in 20 sessions ending on the target date; outcomes remain target-anchored, not signal-anchored",
            "Raw vs adjusted carrier gates are individual feature probes, not complete route qualifications",
            "Distribution gate is a three-session alarm, not proof of institutional selling",
            "Adjusted prices use reported daily pct_chg; no independent corporate-action data verification",
            "No historical point-in-time stock universe or theme membership verification in this diagnostic",
        ],
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "data_sha256": data_hashes,
        "groups": groups, "rows": rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "challenge.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    body = ["# Known-case gate challenge", "", result["scope"], "", result["outcomes"], "",
            *[f"- {s}" for s in result["limitations"]], "",
            pd.DataFrame(groups).T.to_markdown(), "", pd.DataFrame(rows).to_markdown(index=False)]
    (output / "challenge.md").write_text("\n".join(body), encoding="utf-8")
    print(json.dumps(groups, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(
        "results/mainwave_engine/case_quality_audit_20260910/audit.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "results/mainwave_engine/route_evidence_challenge_20260910"))
    args = parser.parse_args()
    build(args.source, args.output)
