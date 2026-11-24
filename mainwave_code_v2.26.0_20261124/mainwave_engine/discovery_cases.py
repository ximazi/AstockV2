"""Import frozen discovery audits without calling them production successes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from . import historical_cases as HC
from . import stock_analyzer as SA
from . import stock_lifecycle as SL
from . import price_cycle_setups as PCS
from . import phase_events as PE
from . import case_review as CR

VERSION = "discovery_cases_v2_20260909"
AUDIT = Path("research/theme_ignition/audit_results")
COMPARISON = Path("research/theme_ignition/p1_audit_results/p1_audit_comparison.csv")
ROOT = HC.ROOT / "discovery_audit"
COVERED = {"ARMED", "READY", "TRIGGERED"}
DISCLAIMER = ("旧P0/P1特征层重放，不是当时生产命中；当前K线阶段为现引擎独立重建。"
              "题材点火不等于个股首次启动，曾提前命中不证明同周期持续覆盖或交易成功。"
              "题材成员含事后回填，阈值未校准；独立结构模拟价差不代表旧审计的策略收益。")


def audit_case(row: dict, trace: pd.DataFrame, comparison: dict) -> dict:
    trace = trace.sort_values("date").reset_index(drop=True)
    t0 = row["t0"]
    if trace["date"].duplicated().any() or t0 not in set(trace["date"]):
        raise ValueError("audit trace requires unique dates and the ignition date")
    pre = trace[trace["date"] < t0]
    if pre.empty:
        raise ValueError("audit case has no pre-ignition observations")
    signals = pre[pre["state"].isin(COVERED)]
    first = str(signals.iloc[0]["date"]) if len(signals) else ""
    if first != row["first_armed"]:
        raise ValueError(f"{row['symbol']}: summary and daily trace disagree")
    ever = str(row["covered_before_t0"]).lower() == "true"
    if ever != bool(first):
        raise ValueError("covered flag disagrees with dated evidence")
    last = str(signals.iloc[-1]["date"]) if len(signals) else ""
    eve = pre.iloc[-1]
    category = "DISCOVERY_HIT" if first else "DISCOVERY_MISS"
    milestones = []
    for date, label in ((row["first_capital_stage"], "P0首次资金代理迹象"),
                        (first, "P0首次覆盖"), (last, "P0点火前最近覆盖"),
                        (comparison.get("p1_first_seen", ""), "P1重放首次候选"),
                        (t0, "题材点火T0（非个股启动定义）"),
                        (row["first_limit_date"], "题材T0前后3交易日内首次涨停")):
        if date and trace.iloc[0]["date"] <= date <= trace.iloc[-1]["date"]:
            milestones.append({"date": date, "label": label})
    evidence = {
        "source_kind": "FROZEN_FEATURE_REPLAY", "production_hit_proven": False,
        "same_cycle_success_proven": False, "membership_source": row["membership_source"],
        "first_signal": first, "last_signal_before_t0": last,
        "lead_sessions": int(((trace["date"] > first) & (trace["date"] <= t0)).sum()) if first else None,
        "gap_sessions_to_t0": int(((trace["date"] > last) & (trace["date"] <= t0)).sum()) if last else None,
        "eve_date": str(eve["date"]), "eve_state": str(eve["state"]),
        "covered_on_eve": eve["state"] in COVERED,
        "continuous_from_first_to_eve": bool(first and pre[pre["date"] >= first]["state"].isin(COVERED).all()),
        "t0_state": str(trace.loc[trace["date"] == t0, "state"].iloc[0]),
        "bottleneck": row["bottleneck"], "p1_first_seen": comparison.get("p1_first_seen", ""),
        "p1_first_class": comparison.get("p1_first_class", ""),
        "p1_validation": "INCONCLUSIVE_FOR_B_CLASS_LABEL_GAP",
        "trace": trace[["date", "state", "capital_stage", "prefilter", "supply_ready", "structure_ready"]].to_dict("records"),
    }
    return {
        "id": f"audit_{row['audit_case']}_{row['symbol']}", "symbol": row["symbol"],
        "name": row["name"], "industry": row["industry"], "theme": row["theme_label"],
        "category": category, "category_label": "旧审计曾命中" if first else "旧审计漏检对照",
        "start": first or row["first_capital_stage"] or str(trace.iloc[0]["date"]), "t0": t0,
        "target_ignition_at": row["first_limit_date"],
        "window_start": str(trace.iloc[0]["date"]), "window_end": str(trace.iloc[-1]["date"]),
        "verification_dates": sorted({first or str(pre.iloc[0]["date"]), t0, str(trace.iloc[-1]["date"])}),
        "milestones": sorted(milestones, key=lambda item: item["date"]),
        "coverage_note": DISCLAIMER, "discovery_audit": evidence,
    }


def build(as_of: str, output: Path = ROOT, audit_root: Path = AUDIT,
          comparison_path: Path = COMPARISON) -> dict:
    as_of = str(pd.Timestamp(as_of).date())
    sources = [audit_root / "audit_stock_summary.csv", audit_root / "audit_daily_trace.csv", comparison_path]
    summary, traces, comparison = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in sources]
    lookup = {(r["audit_case"], r["symbol"]): r for r in comparison.to_dict("records")}
    selected = summary[(summary["role"] == "ignition_carrier") & (summary["t0"] <= as_of)]
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for row in selected.to_dict("records"):
        trace = traces[(traces["audit_case"] == row["audit_case"]) &
                       (traces["symbol"] == row["symbol"]) & (traces["date"] <= as_of)]
        case = audit_case(row, trace, lookup.get((row["audit_case"], row["symbol"]), {}))
        raw, _ = SA._data_frame(case["symbol"], case["window_end"])
        dates = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d")
        sessions = int((dates >= case["window_start"]).sum())
        payload = SA.analyze(case["symbol"], case["window_end"], historical=True, chart_sessions=sessions)
        visible = [r for r in payload["lifecycle"]["history"] if r["date"] >= case["window_start"]]
        stage_by_date = {r["date"]: r["cycle_stage_label"] for r in visible}
        evidence = case["discovery_audit"]
        evidence["current_stage_at_first_signal"] = stage_by_date.get(evidence["first_signal"], "无首次覆盖")
        evidence["current_stage_at_t0"] = stage_by_date.get(case["t0"], "数据不足")
        case["sessions"] = len(payload["chart"]["bars"])
        payload["lifecycle"]["history"] = visible
        payload["case"] = case
        _, classified = SA._data_frame(case["symbol"], case["window_end"])
        CR.attach(payload, classified)
        temp = output / f"{case['id']}.tmp"
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(output / f"{case['id']}.json")
        cases.append(case)
        print(f"[discovery_cases] {len(cases)}/{len(selected)} {case['symbol']} {case['category']}", flush=True)
    # Keep per-day evidence in details, not in the sidebar's manifest response.
    manifest_cases = [{**c, "discovery_audit": {k: v for k, v in c["discovery_audit"].items() if k != "trace"}}
                      for c in cases]
    manifest = {"version": VERSION, "review_version": CR.VERSION, "cycle_version": SL.VERSION, "setup_version": PCS.VERSION,
                "phase_version": PE.VERSION, "analyzer_version": SA.VERSION, "as_of": as_of,
                "disclaimer": DISCLAIMER, "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                "selection": "all ignition_carrier rows in the frozen audit, including misses",
                "cases": manifest_cases}
    temp = output / "manifest.tmp"
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(output / "manifest.json")
    verified = HC.verify(output)
    table = pd.DataFrame([{**{k: c[k] for k in ("name", "symbol", "theme", "window_start", "window_end", "t0", "category_label")},
                           **{k: v for k, v in c["discovery_audit"].items() if k != "trace"}} for c in cases])
    table.to_csv(output / "discovery_case_summary.csv", index=False, encoding="utf-8-sig")
    columns = ["name", "symbol", "theme", "first_signal", "last_signal_before_t0", "t0", "eve_state", "category_label"]
    lines = ["# 发现审计案例库", "", DISCLAIMER, "", f"历史截面校验：{verified['prefix_checks']} 次通过。", "",
             table[columns].to_markdown(index=False) if len(table) else "没有符合日期条件的审计案例。"]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="")
    args = parser.parse_args()
    result = build(args.date or SA.latest_report_date())
    print(f"[discovery_cases] saved={len(result['cases'])} -> {ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
