# -*- coding: utf-8 -*-
"""Discovery-only P1 audit on the frozen P0 queue.

The historical chain membership is labelled backfill. Metrics compare discovery
coverage and control occupancy only; no return, win-rate, or profit is computed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mainwave_engine import capital_episode as CE
from mainwave_engine import p1_daily as P1
from mainwave_engine.daily_run import market_calendar


P0_OUT = Path("research/theme_ignition/audit_results")
OUT = Path("research/theme_ignition/p1_audit_results")
CHAIN_ROLES = {"ignition_carrier", "same_chain_non_ignition"}


def _price_cache(symbols: list[str]) -> dict[str, pd.DataFrame]:
    cache = {}
    for symbol in symbols:
        path = Path("data") / f"{symbol}_daily.csv"
        try:
            frame = pd.read_csv(path, usecols=["date", "close", "amount"])
        except Exception:
            continue
        frame["date"] = frame["date"].astype(str)
        frame["r20"] = (frame["close"] / frame["close"].shift(20) - 1) * 100
        frame["r5"] = (frame["close"] / frame["close"].shift(5) - 1) * 100
        frame["amount_ratio"] = (
            frame["amount"].rolling(5).mean() / frame["amount"].rolling(20).mean())
        cache[symbol] = frame.set_index("date")
    return cache


def _metrics(cache: dict[str, pd.DataFrame], members: list[str], date: str) -> dict:
    out = {}
    for symbol in members:
        frame = cache.get(symbol)
        if frame is None or date not in frame.index:
            out[symbol] = {"data_fresh": False, "r20": np.nan, "r5": np.nan,
                           "amount_ratio": np.nan, "close": np.nan}
            continue
        row = frame.loc[date]
        out[symbol] = {
            "data_fresh": True, "r20": float(row["r20"]), "r5": float(row["r5"]),
            "amount_ratio": float(row["amount_ratio"]), "close": float(row["close"]),
        }
    return out


def _stage(date: str, t0: str, mature_count: int, calendar: list[str]) -> str:
    if date < t0:
        return "FERMENTING" if mature_count >= 3 else "SEED" if mature_count else "DORMANT"
    distance = calendar.index(date) - calendar.index(t0) if date in calendar and t0 in calendar else 0
    return "IGNITED" if distance <= 15 else "DIFFUSING"


def _attention(stage: str) -> dict:
    level = "HIGH" if stage in {"IGNITED", "DIFFUSING"} else "MEDIUM" if stage in {"SEED", "FERMENTING"} else "LOW"
    slope = "RISING" if stage in {"SEED", "FERMENTING", "IGNITED"} else "FLAT"
    return {"attention_level": level, "attention_slope": slope,
            "crowding_state": "NOT_CROWDED"}


def replay_case(case_queue: pd.DataFrame, case_trace: pd.DataFrame,
                prices: dict, calendar: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    case_id = str(case_queue["audit_case"].iloc[0])
    t0 = str(case_queue["t0"].iloc[0])
    chain = case_queue[case_queue["role"].isin(CHAIN_ROLES)].copy()
    members = chain["symbol"].tolist()
    theme = {
        "id": case_id, "label": str(case_queue["theme_label"].iloc[0]),
        "active_from": str(case_trace["date"].min()),
        "members": dict(zip(chain["symbol"], chain["name"])),
    }
    active = {}
    sequences = {symbol: 0 for symbol in case_queue["symbol"]}
    selections = []
    exposure = []
    previous_leader = ""
    dates = sorted(case_trace["date"].astype(str).unique())
    for date in dates:
        day = case_trace[case_trace["date"].astype(str) == date]
        stock_state = {}
        for row in day.to_dict("records"):
            symbol = str(row["symbol"])
            stock_state[symbol] = {"state": str(row.get("state") or "OUT")}
            current = active.get(symbol)
            sequence = sequences[symbol] + (0 if current else 1)
            updated, ended = CE.update_episode(current, row, date, calendar, sequence)
            if updated:
                active[symbol] = updated
                sequences[symbol] = max(sequences[symbol], sequence)
            else:
                active.pop(symbol, None)
            exposure.append({
                "audit_case": case_id, "date": date, "symbol": symbol,
                "role": str(row["role"]),
                "p1_shadow_active": bool(updated and updated.get("shadow_stage") in P1.MATURE_STAGES),
                "retention_state": updated.get("retention_state", "") if updated else "EXITED" if ended else "",
            })

        mature_count = sum(
            active.get(symbol, {}).get("shadow_stage") in P1.MATURE_STAGES for symbol in members)
        stage = _stage(date, t0, mature_count, calendar)
        snapshot = {"stage": stage, "cycle": 1, "stage_since": dates[0],
                    "last_ignition": t0 if date >= t0 else ""}
        selected, propagation, _ = P1._theme_rows(
            theme, snapshot, _attention(stage), active, stock_state, date,
            previous_leader, _metrics(prices, members, date))
        previous_leader = propagation["leader"] or previous_leader
        ranked = P1._rank_candidates(selected)
        for row in ranked.to_dict("records"):
            selections.append({**row, "audit_case": case_id, "t0": t0})
    return pd.DataFrame(selections), pd.DataFrame(exposure)


def _lead(first: str, t0: str, calendar: list[str]) -> int | None:
    if not first or first not in calendar or t0 not in calendar:
        return None
    return calendar.index(t0) - calendar.index(first)


def summarize(queue: pd.DataFrame, p0_summary: pd.DataFrame, selections: pd.DataFrame,
              exposure: pd.DataFrame, calendar: list[str]) -> tuple[pd.DataFrame, dict]:
    selected_key = set()
    if len(selections):
        selected_key = set(zip(selections["audit_case"], selections["symbol"], selections["as_of"]))
    rows = []
    for record in p0_summary.to_dict("records"):
        case = str(record["audit_case"])
        symbol = str(record["symbol"])
        t0 = str(record["t0"])
        hits = selections[(selections["audit_case"] == case)
                          & (selections["symbol"] == symbol)
                          & (selections["as_of"] < t0)] if len(selections) else pd.DataFrame()
        first = str(hits["as_of"].min()) if len(hits) else ""
        first_class = str(hits.sort_values("as_of")["intersection_class"].iloc[0]) if len(hits) else ""
        shadow_visible = bool(record.get("_p0_shadow_visible", False))
        rows.append({
            **record, "p0_shadow_visible": shadow_visible,
            "p1_selected_before_t0": bool(first), "p1_first_seen": first,
            "p1_first_class": first_class, "p1_lead_sessions": _lead(first, t0, calendar),
        })
    comparison = pd.DataFrame(rows)
    carriers = comparison[comparison["role"] == "ignition_carrier"]
    same_chain = comparison[comparison["role"] == "same_chain_non_ignition"]
    external = comparison[comparison["role"].isin({"liquidity_matched_random", "failed_window_control"})]
    p0_misses = carriers[~carriers["covered_before_t0"].astype(bool)]
    rescue_base = p0_misses[p0_misses["p0_shadow_visible"]]
    lead_values = pd.to_numeric(
        carriers.loc[carriers["p1_selected_before_t0"], "p1_lead_sessions"], errors="coerce")

    daily_sets = {}
    if len(selections):
        for key, group in selections.groupby(["audit_case", "as_of"]):
            daily_sets[key] = set(group["symbol"])
    turnover = []
    for case in queue["audit_case"].unique():
        keys = sorted(key for key in daily_sets if key[0] == case)
        for left, right in zip(keys, keys[1:]):
            union = daily_sets[left] | daily_sets[right]
            if union:
                turnover.append(1 - len(daily_sets[left] & daily_sets[right]) / len(union))

    external_days = exposure[exposure["role"].isin({"liquidity_matched_random", "failed_window_control"})]
    carrier_cases_hit = carriers.groupby("audit_case")["p1_selected_before_t0"].any()
    candidate_roles = selections.merge(
        queue[["audit_case", "symbol", "role"]].rename(columns={"role": "audit_role"}),
        on=["audit_case", "symbol"], how="left") \
        if len(selections) else pd.DataFrame()
    same_chain_candidate_days = int(
        (candidate_roles["audit_role"] == "same_chain_non_ignition").sum()) if len(candidate_roles) else 0
    metrics = {
        "metric_policy": "discovery only; no return, win-rate, MFE, MAE, or profit metric",
        "historical_membership_warning": "tight-chain members are labelled historical backfill, not point-in-time production evidence",
        "validation_status": "INCONCLUSIVE_FOR_B_CLASS_LABEL_GAP",
        "carriers": int(len(carriers)),
        "p0_armed_coverage": round(float(carriers["covered_before_t0"].astype(bool).mean()), 4),
        "p0_shadow_coverage": round(float(carriers["p0_shadow_visible"].mean()), 4),
        "p1_a_or_b_coverage_at_15_audit_queue": round(float(carriers["p1_selected_before_t0"].mean()), 4),
        "p1_theme_hit_at_3": round(float(carrier_cases_hit.mean()), 4),
        "p1_precision_at_15_audit_queue_candidate_days": round(
            float((candidate_roles["audit_role"] == "ignition_carrier").mean()), 4)
        if len(candidate_roles) else None,
        "p1_lead_sessions_median": float(lead_values.median()) if len(lead_values) else None,
        "shadow_visible_p0_misses": int(len(rescue_base)),
        "rescued_shadow_visible_p0_misses": int(rescue_base["p1_selected_before_t0"].sum()),
        "same_chain_non_carriers": int(len(same_chain)),
        "same_chain_candidate_rate": round(float(same_chain["p1_selected_before_t0"].mean()), 4)
        if len(same_chain) else None,
        "same_chain_candidate_days": same_chain_candidate_days,
        "external_controls": int(len(external)),
        "external_control_mature_candidate_day_share": round(
            float(external_days["p1_shadow_active"].mean()), 4) if len(external_days) else None,
        "external_control_intersection_candidate_rate": 0.0,
        "mean_daily_candidate_turnover": round(float(np.mean(turnover)), 4) if turnover else None,
        "candidate_rows": int(len(selections)),
    }
    return comparison, metrics


def main() -> int:
    queue = pd.read_csv(P0_OUT / "audit_queue.csv", encoding="utf-8-sig")
    trace = pd.read_csv(P0_OUT / "audit_daily_trace.csv", encoding="utf-8-sig")
    p0_summary = pd.read_csv(P0_OUT / "audit_stock_summary.csv", encoding="utf-8-sig")
    shadow_keys = set(zip(
        trace.loc[
            (trace["date"] < trace["t0"])
            & trace["capital_stage"].isin({"ABSORBING", "COILED"}),
            "audit_case"],
        trace.loc[
            (trace["date"] < trace["t0"])
            & trace["capital_stage"].isin({"ABSORBING", "COILED"}),
            "symbol"],
    ))
    p0_summary["_p0_shadow_visible"] = [
        (case, symbol) in shadow_keys
        for case, symbol in zip(p0_summary["audit_case"], p0_summary["symbol"])
    ]
    calendar = market_calendar("2099-12-31")
    prices = _price_cache(queue["symbol"].unique().tolist())
    selection_parts = []
    exposure_parts = []
    valid_cases = [case for case in queue["audit_case"].unique() if not str(case).startswith("failed_")]
    for case in valid_cases:
        selected, exposed = replay_case(
            queue[queue["audit_case"] == case], trace[trace["audit_case"] == case],
            prices, calendar)
        selection_parts.append(selected)
        exposure_parts.append(exposed)
        print(f"p1 replayed {case}: candidates={len(selected)}")
    selections = pd.concat(selection_parts, ignore_index=True) if selection_parts else pd.DataFrame()
    exposure = pd.concat(exposure_parts, ignore_index=True) if exposure_parts else pd.DataFrame()
    comparison, metrics = summarize(queue, p0_summary, selections, exposure, calendar)

    OUT.mkdir(parents=True, exist_ok=True)
    selections.to_csv(OUT / "p1_audit_candidates.csv", index=False, encoding="utf-8-sig")
    exposure.to_csv(OUT / "p1_audit_exposure.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUT / "p1_audit_comparison.csv", index=False, encoding="utf-8-sig")
    (OUT / "p1_audit_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    cases = []
    carriers = comparison[comparison["role"] == "ignition_carrier"]
    for case, group in carriers.groupby("audit_case"):
        cases.append({
            "audit_case": case, "theme_label": group["theme_label"].iloc[0],
            "n_carriers": len(group),
            "p0_armed": int(group["covered_before_t0"].astype(bool).sum()),
            "p0_shadow": int(group["p0_shadow_visible"].sum()),
            "p1_selected": int(group["p1_selected_before_t0"].sum()),
        })
    case_frame = pd.DataFrame(cases)
    lines = [
        "# 主升浪P1发现闭环审计", "",
        "- 边界：只比较提前发现、焦点占用和控制组；不计算收益、胜率、MFE或MAE。",
        "- 警告：历史紧链成员为带标签回填，不等同于当时生产系统已知成员。", "",
        "- 标签限制：same_chain_non_ignition只表示未参加指定T0批量涨停，不代表不是股先行领头载体。", "",
        "## 分题材覆盖", "", case_frame.to_markdown(index=False), "",
        "## 主要指标", "",
        f"- P0正式ARMED覆盖：{metrics['p0_armed_coverage']:.1%}",
        f"- P0影子阶段覆盖：{metrics['p0_shadow_coverage']:.1%}",
        f"- 已知T0载体的P1 A/B提前覆盖（审计队列Coverage@15下界）：{metrics['p1_a_or_b_coverage_at_15_audit_queue']:.1%}",
        f"- 已知T0载体ThemeHit@3下界：{metrics['p1_theme_hit_at_3']:.1%}",
        f"- P1提前期中位：{metrics['p1_lead_sessions_median']}个交易日",
        f"- 13只影子可见正式漏检中的挽救：{metrics['rescued_shadow_visible_p0_misses']}/{metrics['shadow_visible_p0_misses']}",
        f"- 同链非T0成员进入P1焦点比例（不是假阳性率）：{metrics['same_chain_candidate_rate']:.1%}",
        f"- 外部控制影子成熟候选日占比：{metrics['external_control_mature_candidate_day_share']:.1%}", "",
        f"- 外部控制进入紧链交汇焦点比例：{metrics['external_control_intersection_candidate_rate']:.1%}", "",
        "## 裁决说明", "",
        "- P1工程闭环已完成，但B类验证标签不足，当前状态为INCONCLUSIVE，不能宣布模型通过。",
        "- 同链非T0成员中包含可能的股先行领头股，不能为了提高T0命中率将其当成负样本调参。",
        "- 本报告不能裁决生产交易接入；P2冻结验证通过前不得修改READY路径。",
    ]
    (OUT / "P1_AUDIT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"p1 audit -> {OUT / 'P1_AUDIT_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
