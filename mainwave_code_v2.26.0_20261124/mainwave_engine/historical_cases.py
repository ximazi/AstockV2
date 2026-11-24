"""On-demand, retrospective case library using the current causal cycle engine.

Case selection deliberately sees the outcome. Daily labels do not. This is an
explanatory library, NOT a detection-rate estimate or an out-of-sample backtest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import stock_analyzer as SA
from . import stock_lifecycle as SL
from . import price_cycle_setups as PCS
from . import phase_events as PE
from . import case_review as CR

ROOT = Path("results/mainwave_engine/historical_cases")
VERSION = "historical_cases_v2_20260909"
# Include the stocks discussed by the user; broaden with a fixed universe sample.
REVIEW_SYMBOLS = ["sh_600869", "sh_603118", "sz_301071", "sz_002721", "sz_001309",
                  "sz_300502", "sz_300308", "sh_603986", "sz_300394", "sh_688347",
                  "sz_002384"]
DISCLAIMER = ("现引擎历史重建，非当时生产命中。案例按事后闭合结果选择，包含失败对照；"
              "不是发现率、胜率或收益验证。阶段与形态阈值均为占位待校准。")


def discover(symbol: str, timeline: pd.DataFrame, cutoff: str) -> list[dict]:
    """Find observed launches that failed; distinguish a full arc from launch failure."""
    cases = []
    for cid, rows in timeline[timeline["cycle_id"].ne("")].groupby("cycle_id", sort=False):
        start = str(rows.iloc[0]["cycle_started_at"])
        if start < cutoff or rows.iloc[0]["cycle_stage"] != "IGNITION":
            continue
        failed = rows[rows["cycle_stage"].eq("FAILED")]
        if failed.empty:
            continue
        failure = str(failed.iloc[0]["date"])
        before_failure = rows[rows["date"].le(failure)]
        markup_days = int(before_failure["cycle_stage"].eq("MARKUP").sum())
        adjusted = bool(before_failure["cycle_stage"].eq("PULLBACK").any()
                        or before_failure.get("setup_id", pd.Series(dtype=str)).ne("").any())
        repaired = rows[rows["cycle_stage"].eq("REBUILDING")]
        complete = markup_days >= 5 and adjusted and not repaired.empty
        control = markup_days < 5 and len(before_failure) <= 35
        if not complete and not control:
            continue
        first_index = int(rows.index[0])
        fail_index = int(failed.index[0])
        end_index = min(int(rows.index[-1]), fail_index + 25)
        if complete:
            end_index = min(int(rows.index[-1]), max(end_index, int(repaired.index[0]) + 5))
        start_index = max(0, first_index - 25)
        if first_index - start_index < 20:
            continue
        category = "COMPLETE" if complete else "FAILED_LAUNCH"
        visible = timeline.loc[first_index:end_index]
        milestones = [{"date": start, "label": "启动观察"}]
        for state in ("PULLBACK", "OVERHEATED", "DISTRIBUTION_RISK", "FAILED", "REBUILDING"):
            match = visible[visible["cycle_stage"].eq(state)]
            if len(match):
                milestones.append({"date": str(match.iloc[0]["date"]), "label": SL.LABELS[state]})
        stages = list(dict.fromkeys(visible["cycle_stage"]))
        cases.append({
            "id": f"{symbol}_{start.replace('-', '')}", "symbol": symbol, "cycle_id": cid,
            "category": category, "category_label": "闭合周期（含修复）" if complete else "启动失败对照",
            "start": start, "failed_at": failure,
            "window_start": str(timeline.loc[start_index, "date"]),
            "window_end": str(timeline.loc[end_index, "date"]),
            "sessions": end_index - start_index + 1, "markup_sessions": markup_days,
            "stages": stages, "milestones": sorted(milestones, key=lambda r: r["date"]),
            "coverage_note": "包含启动前25根K线；无当时准备期生产证据时，不补画吸筹。修复不等于底部确认。",
            "warning_observed": bool(set(stages) & {"OVERHEATED", "DISTRIBUTION_RISK"}),
        })
    return cases


def build(as_of: str, scan_limit: int = 120, output: Path = ROOT) -> dict:
    as_of = str(pd.Timestamp(as_of).date())
    cutoff = str((pd.Timestamp(as_of) - pd.DateOffset(years=1)).date())
    paths = sorted(Path("data").glob("*_daily.csv"))
    sample = np.linspace(0, len(paths) - 1, min(scan_limit, len(paths)), dtype=int) if paths else []
    symbols = list(dict.fromkeys(REVIEW_SYMBOLS + [paths[i].name.removesuffix("_daily.csv")
                         for i in sample if paths[i].name.startswith(("sh_", "sz_"))]))
    mapping = SA._read_json(Path("concept_mapping.json"), {}).get("mapping", {})
    matches, skipped = [], []
    for i, symbol in enumerate(symbols, 1):
        try:
            _, timeline = SL.analyze(symbol, as_of)
            if timeline.empty:
                skipped.append({"symbol": symbol, "reason": "insufficient_data"})
                continue
            found = discover(symbol, timeline, cutoff)
            for row in found:
                info = mapping.get(symbol, {})
                row.update(name=info.get("name", symbol), industry=info.get("industry", "未分类"))
            matches.extend(found)
        except (ValueError, KeyError, pd.errors.ParserError) as exc:
            skipped.append({"symbol": symbol, "reason": str(exc)})
        if i % 10 == 0 or i == len(symbols):
            print(f"[history_cases] {i}/{len(symbols)} eligible={len(matches)}", flush=True)
    # Diversify examples by industry first, without ranking by profit.
    selected, seen_symbols, industries = [], set(), set()
    for category, count in (("COMPLETE", 8), ("FAILED_LAUNCH", 3)):
        candidates = [r for r in matches if r["category"] == category]
        candidates.sort(key=lambda r: (not r["warning_observed"], -int(r["start"].replace("-", ""))))
        for diverse in (True, False):
            for row in candidates:
                if sum(r["category"] == category for r in selected) >= count:
                    break
                if row["symbol"] in seen_symbols or (diverse and row["industry"] in industries):
                    continue
                selected.append(row)
                seen_symbols.add(row["symbol"])
                industries.add(row["industry"])
    output.mkdir(parents=True, exist_ok=True)
    for case in selected:
        payload = SA.analyze(case["symbol"], case["window_end"], historical=True,
                             chart_sessions=case["sessions"])
        payload["case"] = case
        payload["lifecycle"]["history"] = [r for r in payload["lifecycle"]["history"]
                                            if r["date"] >= case["window_start"]]
        payload["setups"]["history"] = [r for r in payload["setups"]["history"]
                                         if r["cycle_id"] == case["cycle_id"]]
        payload["case"]["setup_count"] = len(payload["setups"]["history"])
        _, classified = SA._data_frame(case["symbol"], case["window_end"])
        CR.attach(payload, classified)
        path = output / f"{case['id']}.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)
    manifest = {"version": VERSION, "review_version": CR.VERSION, "cycle_version": SL.VERSION, "setup_version": PCS.VERSION,
                "phase_version": PE.VERSION, "analyzer_version": SA.VERSION,
                "as_of": as_of, "cutoff": cutoff, "disclaimer": DISCLAIMER,
                "universe": "用户讨论标的 + 按文件名排序的全域等距样本，非全市场穷举",
                "scanned_symbols": symbols, "eligible_count": len(matches), "skipped": skipped,
                "cases": selected}
    temp = output / "manifest.tmp"
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(output / "manifest.json")
    audit = verify(output)
    columns = ["name", "symbol", "industry", "category_label", "start", "failed_at", "window_end", "setup_count"]
    table = pd.DataFrame(selected)
    lines = [f"# 历史生命周期案例 / {as_of}", "", DISCLAIMER, "",
             f"扫描 {len(symbols)} 只，符合规则 {len(matches)} 个周期，展示 {len(selected)} 个。", "",
             f"逐例复核：{audit['cases_checked']} 个案例 / {audit['prefix_checks']} 个历史截面通过；"
             "追加未来数据未改写已知阶段、速度或形态。", "",
             table[columns].to_markdown(index=False) if len(table) else "没有符合完整规则的案例，不填造。",
             "", "闭合周期要求：观测启动、至少5日上涨延续、整理、结构失效、后续修复，并有至少20日启动前背景。",
             "启动失败对照：35个交易日内失效，且上涨延续不足5日。分类用于解释，不是盈利标签。"]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return manifest


def case_list(root: Path = ROOT) -> dict:
    manifests = [manifest for _, manifest in _libraries(root)]
    return {"as_of": max((m.get("as_of", "") for m in manifests), default=""),
            "cases": [case for m in manifests for case in m.get("cases", [])],
            "disclaimer": "\n".join(dict.fromkeys(m.get("disclaimer", DISCLAIMER) for m in manifests)) or DISCLAIMER,
            "stale": any(_stale(m) for m in manifests), "routes": CR.ROUTES}


def _libraries(root: Path):
    for base in (root, root / "discovery_audit"):
        manifest = SA._read_json(base / "manifest.json", {})
        if manifest:
            yield base, manifest


def _stale(manifest: dict) -> bool:
    return (manifest.get("cycle_version") != SL.VERSION
            or manifest.get("setup_version") != PCS.VERSION
            or manifest.get("phase_version") != PE.VERSION
            or manifest.get("analyzer_version") != SA.VERSION
            or bool(manifest.get("version") and manifest.get("review_version") != CR.VERSION))


def refresh(root: Path = ROOT) -> dict:
    """Refresh only registered cases; preserve windows and frozen audit evidence."""
    for base, manifest in list(_libraries(root)):
        selected = []
        for summary in manifest["cases"]:
            old = SA._read_json(base / f"{summary['id']}.json", {})
            case = old["case"]
            payload = SA.analyze(case["symbol"], case["window_end"], historical=True,
                                 chart_sessions=case["sessions"])
            payload["case"] = case
            payload["lifecycle"]["history"] = [r for r in payload["lifecycle"]["history"]
                                                if r["date"] >= case["window_start"]]
            if case.get("cycle_id"):
                payload["setups"]["history"] = [r for r in payload["setups"]["history"]
                                                 if r["cycle_id"] == case["cycle_id"]]
            if case.get("discovery_audit"):
                audit = case["discovery_audit"]
                stages = {r["date"]: r["cycle_stage_label"] for r in payload["lifecycle"]["history"]}
                audit["current_stage_at_first_signal"] = stages.get(audit["first_signal"], "无首次覆盖")
                audit["current_stage_at_t0"] = stages.get(case["t0"], "数据不足")
            case["setup_count"] = len(payload["setups"]["history"])
            _, classified = SA._data_frame(case["symbol"], case["window_end"])
            CR.attach(payload, classified)
            temp = base / f"{case['id']}.tmp"
            temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            temp.replace(base / f"{case['id']}.json")
            item = dict(case)
            if case.get("discovery_audit"):
                item["discovery_audit"] = {k: v for k, v in case["discovery_audit"].items() if k != "trace"}
            selected.append(item)
            print(f"[history_refresh] {len(selected)}/{len(manifest['cases'])} {case['id']} {case['review_summary']['status']}", flush=True)
        manifest.update(cases=selected, review_version=CR.VERSION, cycle_version=SL.VERSION,
                        setup_version=PCS.VERSION, phase_version=PE.VERSION, analyzer_version=SA.VERSION)
        temp = base / "manifest.tmp"
        temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(base / "manifest.json")
    return verify(root)


def case_detail(case_id: str, root: Path = ROOT) -> dict:
    for base, manifest in _libraries(root):
        if case_id not in {row["id"] for row in manifest.get("cases", [])}:
            continue
        if _stale(manifest):
            raise ValueError("案例引擎版本已变化，请重新构建历史案例库")
        payload = SA._read_json(base / f"{case_id}.json", {})
        if not payload:
            raise FileNotFoundError("案例文件缺失，请重新构建")
        return payload
    raise ValueError("历史案例不存在")


def verify(root: Path = ROOT) -> dict:
    """Recompute selected historical prefixes; fail rather than publish a false audit."""
    checked = 0
    outcomes = {}
    cases = case_list(root)["cases"]
    for case in cases:
        payload = case_detail(case["id"], root)
        history = pd.DataFrame(payload["lifecycle"]["history"])
        dates = [r["date"] for r in payload["chart"]["bars"]]
        spans = payload["chart"]["stage_segments"]
        if sum(s["sessions"] for s in spans) != len(dates):
            raise AssertionError(f"{case['id']}: incomplete chart spans")
        if dates[0] != case["window_start"] or dates[-1] != case["window_end"]:
            raise AssertionError(f"{case['id']}: incorrect visible dates")
        for setup in payload["setups"]["history"]:
            outcome = setup["setup_outcome"] or "FORMING"
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            if setup["setup_confirmed_at"] and setup["setup_confirmed_at"] <= setup["setup_started_at"]:
                raise AssertionError("confirmation cannot predate formation")
        cuts = set(case.get("verification_dates") or
                   [case["start"], case["failed_at"], case["window_end"]])
        for date in sorted(cuts):
            _, classified = SA._data_frame(case["symbol"], date)
            prefix = SL.replay(case["symbol"], classified)
            prefix = prefix[prefix["date"].ge(case["window_start"])].reset_index(drop=True)
            expected = history[history["date"].le(date)].reset_index(drop=True)
            # JSON normalizes floating NaN/dtypes; compare semantic records.
            actual = pd.DataFrame(json.loads(prefix.to_json(orient="records")))
            actual = actual[expected.columns].astype(object).where(actual[expected.columns].notna(), None)
            expected = expected.astype(object).where(expected.notna(), None)
            pd.testing.assert_frame_equal(actual, expected,
                                          check_dtype=False, atol=1e-8, rtol=1e-8)
            checked += 1
    result = {"cases_checked": len(cases), "prefix_checks": checked, "status": "PASS",
              "setup_outcomes": outcomes, "validation_scope": "causality_and_bookkeeping_only",
              "profitability_validated": False}
    (root / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="")
    parser.add_argument("--scan-limit", type=int, default=120)
    parser.add_argument("--refresh", action="store_true", help="Refresh registered cases only, without selecting new cases")
    args = parser.parse_args()
    if args.refresh:
        print(json.dumps(refresh(), ensure_ascii=False))
        return 0
    if args.scan_limit < 1:
        parser.error("--scan-limit must be positive")
    result = build(args.date or SA.latest_report_date(), args.scan_limit)
    print(f"[history_cases] saved={len(result['cases'])} -> {ROOT / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
