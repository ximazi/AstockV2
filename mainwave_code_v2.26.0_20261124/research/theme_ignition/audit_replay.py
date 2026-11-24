# -*- coding: utf-8 -*-
"""P0 feature-level replay for pre-ignition main-wave discovery.

This module is deliberately isolated from the production state store. It reuses
the production feature functions and state-machine transitions, but writes only
audit artifacts under research/theme_ignition/audit_results.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from mainwave_engine import data as D
from mainwave_engine import features as F
from mainwave_engine import observation as O
from mainwave_engine import state_machine as SM
from mainwave_engine.daily_run import check_trigger, market_calendar, prefilter_channels
from mainwave_engine.theme_registry import themes_as_of


OUT = Path("research/theme_ignition/audit_results")
CONTROL_EPISODES = Path("research/theme_ignition/control_episodes.csv")
PRE_SESSIONS = 90
POST_SESSIONS = 20
RANDOM_POOL = 240
RANDOM_SEED = 20260828

# Innovation-drug carriers observed around the 2026-08-04 ignition. This is an
# audit-only case definition, not a point-in-time production membership list.
INNOVATION_CARRIER_SEED = [
    "sh_603259", "sh_603127", "sz_000739", "sz_002821", "sz_000566",
    "sz_002437", "sz_002589", "sz_300363", "sz_301230", "sh_600267",
    "sh_600664", "sh_603222", "sh_603567", "sh_600721", "sz_002793",
    "sh_600513", "sz_002900",
]

CASE_SPECS = {
    "storage_20251024": {"theme": "storage", "label": "存储", "t0": "2025-10-24"},
    "optical_20250901": {"theme": "optical_cpo", "label": "光模块/CPO", "t0": "2025-09-01"},
    "innovation_20260804": {"theme": "innovation_drug", "label": "创新药", "t0": "2026-08-04"},
}

APPROXIMATIONS = [
    "题材成员和行业映射使用当前映射回填，不代表历史时点可得成员。",
    "L1 题材/行业 EMERGING 与 ACTIVE 锚、relay 和 sector_sync 在重放中关闭；READY 为保守下界。",
    "行业相对收益按当前行业成员逐日中位数重建；个股特征、基本面时点和状态机转移使用生产函数。",
    "创新药载体由 2026-08-04 点火窗口人工冻结，仅用于命中审计，不写入生产题材注册库。",
]


def _date_window(t0: str, cal: list[str], before: int, after: int) -> list[str]:
    i = cal.index(t0)
    return cal[max(0, i - before):min(len(cal), i + after + 1)]


def _load_full(symbol: str, end_date: str) -> pd.DataFrame | None:
    df = D.load_tail(symbol, rows=10000, as_of=end_date)
    if df is None:
        return None
    out = df.copy()
    out["date"] = out["date"].astype(str)
    return out


def _amount20(symbol: str, as_of: str) -> float | None:
    df = D.load_tail(symbol, rows=30, as_of=as_of)
    if df is None or "amount" not in df:
        return None
    values = pd.to_numeric(df["amount"].tail(20), errors="coerce").dropna()
    if values.empty:
        return None
    v = float(values.median())
    return v if np.isfinite(v) and v > 0 else None


def _first_limit_date(symbol: str, t0: str, cal: list[str]) -> str | None:
    dates = set(_date_window(t0, cal, 3, 3))
    df = D.load_tail(symbol, rows=30, as_of=max(dates))
    if df is None:
        return None
    x = df[df["date"].astype(str).isin(dates)].copy()
    hit = x[(x["pct_chg"] >= D.limit_pct(symbol) * 0.985) & (x["low"] < x["high"])]
    return str(hit["date"].iloc[0]) if len(hit) else None


def _registry_members(theme_id: str) -> list[str]:
    versions = themes_as_of("2026-08-27")
    theme = next((x for x in versions if x["id"] == theme_id), None)
    if not theme:
        raise RuntimeError(f"theme missing from registry: {theme_id}")
    return sorted(theme["members"])


def _innovation_noncarriers(t0: str, excluded: set[str], cap: int = 5) -> list[str]:
    ind_map = D.industry_map()
    carrier_inds = {ind_map.get(s) for s in INNOVATION_CARRIER_SEED if ind_map.get(s)}
    candidates = [s for s, ind in ind_map.items()
                  if ind in carrier_inds and s not in excluded and s not in D.st_set()]
    scored = [(s, _amount20(s, t0)) for s in candidates]
    valid = [(s, v) for s, v in scored if v is not None]
    return [s for s, _ in sorted(valid, key=lambda x: x[1], reverse=True)[:cap]]


def _liquidity_controls(case_id: str, t0: str, targets: list[str], excluded: set[str],
                        universe: list[str], cap: int = 3) -> list[str]:
    target_amounts = [v for v in (_amount20(s, t0) for s in targets) if v is not None]
    if not target_amounts:
        return []
    target = float(np.median(target_amounts))
    pool = [s for s in universe if s not in excluded]
    rng = np.random.default_rng(RANDOM_SEED + sum(ord(c) for c in case_id))
    if len(pool) > RANDOM_POOL:
        pool = list(rng.choice(pool, size=RANDOM_POOL, replace=False))
    scored = []
    for symbol in pool:
        amount = _amount20(symbol, t0)
        if amount is not None:
            scored.append((abs(math.log(amount / target)), symbol))
    return [s for _, s in sorted(scored)[:cap]]


def _failed_industry_controls(excluded: set[str], cap: int = 6) -> list[dict]:
    if not CONTROL_EPISODES.exists():
        return []
    ep = pd.read_csv(CONTROL_EPISODES,
                     usecols=["industry", "set", "date", "ignited"])
    ign = ep["ignited"].astype(str).str.lower().isin({"true", "1"})
    ep = ep[(ep["set"] == "S3") & ~ign & ep["date"].astype(str).str.startswith("2024-")]
    ep = ep.sort_values(["date", "industry"]).drop_duplicates("industry").head(cap)
    ind_map = D.industry_map()
    rows = []
    for _, event in ep.iterrows():
        t0 = str(event["date"])
        members = [s for s, ind in ind_map.items()
                   if ind == event["industry"] and s not in excluded and s not in D.st_set()]
        scored = [(s, _amount20(s, t0)) for s in members]
        valid = [(s, v) for s, v in scored if v is not None]
        if not valid:
            continue
        symbol = max(valid, key=lambda x: x[1])[0]
        excluded.add(symbol)
        rows.append({
            "audit_case": f"failed_{len(rows) + 1:02d}", "theme": str(event["industry"]),
            "theme_label": f"S3未点火:{event['industry']}", "t0": t0,
            "symbol": symbol, "role": "failed_window_control", "carrier_id": "",
            "first_limit_date": "", "membership_source": "current_industry_backfill",
        })
    return rows


def build_audit_queue(cal: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    excluded: set[str] = set()
    universe = D.list_universe()
    names = D.name_map()
    for case_id, spec in CASE_SPECS.items():
        t0 = spec["t0"]
        if spec["theme"] == "innovation_drug":
            members = [s for s in INNOVATION_CARRIER_SEED if s in universe]
            source = "audit_case_carriers_frozen_20260828"
        else:
            members = [s for s in _registry_members(spec["theme"]) if s in universe]
            source = "theme_registry_current_members_backfill"

        first_limits = {s: _first_limit_date(s, t0, cal) for s in members}
        carriers = sorted([s for s in members if first_limits[s]],
                          key=lambda s: (first_limits[s], s))
        if spec["theme"] == "innovation_drug":
            carriers = sorted(members, key=lambda s: (first_limits[s] or "9999", s))
        carrier_ids = {s: f"A{i:03d}" for i, s in enumerate(carriers, 1)}
        for symbol in members:
            role = "ignition_carrier" if symbol in carrier_ids else "same_chain_non_ignition"
            rows.append({
                "audit_case": case_id, "theme": spec["theme"], "theme_label": spec["label"],
                "t0": t0, "symbol": symbol, "role": role,
                "carrier_id": carrier_ids.get(symbol, ""),
                "first_limit_date": first_limits[symbol] or "", "membership_source": source,
            })
        excluded.update(members)

        if spec["theme"] == "innovation_drug":
            for symbol in _innovation_noncarriers(t0, excluded):
                rows.append({
                    "audit_case": case_id, "theme": spec["theme"], "theme_label": spec["label"],
                    "t0": t0, "symbol": symbol, "role": "same_chain_non_ignition",
                    "carrier_id": "", "first_limit_date": "",
                    "membership_source": "current_pharma_industry_backfill",
                })
                excluded.add(symbol)

        controls = _liquidity_controls(case_id, t0, carriers or members, excluded, universe)
        for symbol in controls:
            rows.append({
                "audit_case": case_id, "theme": spec["theme"], "theme_label": spec["label"],
                "t0": t0, "symbol": symbol, "role": "liquidity_matched_random",
                "carrier_id": "", "first_limit_date": "",
                "membership_source": f"seeded_random_pool_{RANDOM_POOL}",
            })
            excluded.add(symbol)

    rows.extend(_failed_industry_controls(excluded))
    queue = pd.DataFrame(rows)
    queue["name"] = queue["symbol"].map(names).fillna("")
    queue["industry"] = queue["symbol"].map(D.industry_map()).fillna("")
    return queue.sort_values(["audit_case", "role", "carrier_id", "symbol"]).reset_index(drop=True)


def build_industry_panel(queue: pd.DataFrame, end_date: str) -> dict[str, pd.DataFrame]:
    ind_map = D.industry_map()
    wanted = {x for x in queue["industry"] if x}
    out: dict[str, pd.DataFrame] = {}
    for industry in sorted(wanted):
        pieces = []
        for symbol, ind in ind_map.items():
            if ind != industry or symbol in D.st_set():
                continue
            df = _load_full(symbol, end_date)
            if df is not None:
                pieces.append(df[["date", "pct_chg"]].rename(columns={"pct_chg": "ret"}))
        if pieces:
            panel = pd.concat(pieces, ignore_index=True)
            out[industry] = panel.groupby("date", as_index=False)["ret"].median()
    return out


def evaluate_day(symbol: str, df: pd.DataFrame, run_date: str,
                 ind_df: pd.DataFrame | None, rec: dict | None) -> tuple[dict, dict]:
    ctx = F.Ctx(symbol, df)
    fund = F.fundamental_anchor(D.fundamental_view(D.code6(symbol), run_date))
    prefilter, channels = prefilter_channels(ctx, bool(fund["ok"]))
    imp = F.impulse_retention(ctx)
    fresh = F.fresh_impulse(ctx)
    rc = F.resistance_cluster(ctx)
    pit = F.pit_reclaim(ctx)
    ip = F.inst_platform(ctx)
    rel20 = F.align_rel(ctx, ind_df, 20)
    va = F.volume_asymmetry(ctx, rel20)
    ss = F.silent_supply(ctx)
    ib = F.institutional_behavior(ctx, rel20)
    health = F.common_health(ctx, run_date)
    struct = F.structure_family(ctx, rc, pit, ip)
    if rec and rec.get("stored_resistance") is not None:
        structure_ready = F.structure_vs_frozen(
            ctx, rec["stored_resistance"], rec["resistance_atr"], rec.get("structure_type"))
    else:
        structure_ready = bool(struct["ok"])
    supply_kind, supply = max(
        [("pit", pit), ("asym", va), ("silent", ss)],
        key=lambda kv: (kv[1]["ok"], kv[1]["score"]),
    )
    supply_ready = bool(supply["ok"])
    hot_anchor = bool(imp["ok"] or fresh["ok"])
    inst_anchor = bool(fund["ok"] and ib["ok"])
    hot_sig = bool(imp["ok"] or (pit["ok"] and (pit.get("age") or 99) <= 20) or fresh["ok"])
    route = ("HYBRID" if hot_sig and inst_anchor else "HOT" if hot_sig
             else "INST" if inst_anchor else "UNCLASSIFIED")
    armed_core = bool(health["ok"] and structure_ready and supply_ready)
    ready_core = bool(armed_core and (hot_anchor or inst_anchor))
    ev_dates = [d for d in (imp.get("event_date"), pit.get("event_date"),
                            fresh.get("event_date")) if d]
    evidence_date = max(ev_dates) if ev_dates else None
    ev = {
        "close": float(ctx.C[-1]), "low": float(ctx.L[-1]),
        "atr_now": float(ctx.atr20[-1]) if np.isfinite(ctx.atr20[-1]) else None,
        "prefilter": bool(prefilter), "armed_core": armed_core, "ready_core": ready_core,
        "structure_ready": bool(structure_ready), "supply_ready": supply_ready,
        "route_anchor": bool(hot_anchor or inst_anchor), "health_ok": bool(health["ok"]),
        "supply_kind": supply_kind if supply_ready else "",
        "supply_score": float(supply["score"]) if supply_ready else 0.0,
        "resistance": struct.get("anchor"), "resistance_atr": struct.get("atr"),
        "invalid_price_candidate": struct.get("invalid"),
        "structure_type": struct.get("type"),
        "structure_started_at": struct.get("started_at"),
        "max_evidence_date": evidence_date, "route": route,
        "trigger_event": False, "trigger_kind": "", "false_breakout": False,
        "armed_resonance": False, "rebuild_ok": False,
    }
    if rec and rec.get("state") == "READY":
        trig, kind, false_bo = check_trigger(rec, ctx, set(), [])
        ev.update(trigger_event=trig, trigger_kind=kind, false_breakout=false_bo)
    if rec and rec.get("state") == "FAILED":
        idx = ctx.dates.index(rec["failed_at"]) if rec.get("failed_at") in ctx.dates else -1
        ev["rebuild_ok"] = idx > 0 and SM.rebuild_ok(
            ctx, idx, rec.get("failure_low", 0), rec.get("stored_resistance", 1e9),
            rec.get("atr_at_failure"))
    raw = {
        "prefilter": bool(prefilter), "prefilter_ch": channels,
        "health": bool(health["ok"]), "structure_ready": bool(structure_ready),
        "supply_ready": supply_ready, "supply_kind": ev["supply_kind"],
        "imp": bool(imp["ok"]), "imp_age": imp.get("age"),
        "fresh": bool(fresh["ok"]), "fresh_age": fresh.get("age"),
        "pit": bool(pit["ok"]), "pit_age": pit.get("age"),
        "armed_core": armed_core, "ready_core": ready_core, "route": route,
        "max_evidence_date": evidence_date,
    }
    return ev, raw


def replay_queue(queue: pd.DataFrame, cal: list[str], panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    for case_id, group in queue.groupby("audit_case", sort=True):
        t0 = str(group["t0"].iloc[0])
        dates = _date_window(t0, cal, PRE_SESSIONS, POST_SESSIONS)
        histories = {s: _load_full(s, dates[-1]) for s in group["symbol"]}
        records: dict[str, dict | None] = {s: None for s in group["symbol"]}
        meta = group.set_index("symbol").to_dict("index")
        for today in dates:
            mdates = [d for d in cal if d <= today]
            for symbol in group["symbol"]:
                rec = records[symbol]
                full = histories[symbol]
                base = {"audit_case": case_id, "theme": meta[symbol]["theme"],
                        "theme_label": meta[symbol]["theme_label"], "t0": t0,
                        "symbol": symbol, "role": meta[symbol]["role"],
                        "carrier_id": meta[symbol]["carrier_id"], "date": today}
                if full is None:
                    rows.append({**base, "state": "NO_DATA", "data_fresh": False,
                                 "capital_stage": "UNKNOWN"})
                    continue
                df = full[full["date"] <= today].tail(320).reset_index(drop=True)
                fresh = len(df) >= 70 and str(df["date"].iloc[-1]) == today
                if not fresh:
                    nrec = SM.time_step(rec, today, mdates) if rec else None
                    records[symbol] = nrec
                    rows.append({**base, "state": (nrec or {}).get("state", "OUT"),
                                 "data_fresh": False, "capital_stage": "UNKNOWN"})
                    continue
                try:
                    ev, raw = evaluate_day(symbol, df, today,
                                           panels.get(meta[symbol]["industry"]), rec)
                    nrec = SM.step(symbol, rec, ev, today, mdates)
                except (ValueError, KeyError, TypeError):
                    rows.append({**base, "state": "EVAL_ERROR", "data_fresh": True,
                                 "capital_stage": "UNKNOWN"})
                    continue
                records[symbol] = (nrec if nrec.get("state") != "OUT"
                                   or nrec.get("out_cooldown_from") else None)
                evidence = {"data_fresh": True, **raw}
                rows.append({
                    **base, **raw, "data_fresh": True, "state": nrec.get("state", "OUT"),
                    "capital_stage": O.derive_capital_stage(evidence, nrec.get("state", "OUT")),
                    "capital_stage_version": O.CAPITAL_STAGE_VERSION,
                })
        print(f"replayed {case_id}: {len(group)} stocks x {len(dates)} sessions")
    return pd.DataFrame(rows)


def _first_date(df: pd.DataFrame, column: str, values: set) -> str:
    hit = df[df[column].isin(values)] if column in df else pd.DataFrame()
    return str(hit["date"].iloc[0]) if len(hit) else ""


def _lead_sessions(first: str, t0: str, cal: list[str]) -> int | None:
    if not first or first not in cal or t0 not in cal:
        return None
    return cal.index(t0) - cal.index(first)


def summarize(queue: pd.DataFrame, trace: pd.DataFrame, cal: list[str]) -> pd.DataFrame:
    summaries = []
    for _, q in queue.iterrows():
        x = trace[(trace["audit_case"] == q["audit_case"]) & (trace["symbol"] == q["symbol"])]
        pre = x[x["date"] < q["t0"]]
        first_prefilter = _first_date(pre, "prefilter", {True})
        first_stage = _first_date(pre, "capital_stage", {"PROBED", "ABSORBING", "COILED",
                                                                 "ARMED", "READY", "TRIGGERED"})
        pulse_mask = pd.Series(False, index=pre.index)
        for column in ("imp", "fresh", "pit"):
            if column in pre:
                pulse_mask |= pre[column].eq(True)
        first_pulse = str(pre.loc[pulse_mask, "date"].iloc[0]) if pulse_mask.any() else ""
        first_supply = _first_date(pre, "supply_ready", {True})
        first_structure = _first_date(pre, "structure_ready", {True})
        first_armed = _first_date(pre, "state", {"ARMED", "READY", "TRIGGERED"})
        first_ready = _first_date(pre, "state", {"READY", "TRIGGERED"})
        covered = bool(first_armed)
        bottleneck = ""
        if q["role"] == "ignition_carrier" and not covered:
            if not first_prefilter:
                bottleneck = "未进预筛"
            elif not first_pulse:
                bottleneck = "脉冲未识别"
            elif not first_supply:
                bottleneck = "供给不成立"
            elif not first_structure:
                bottleneck = "结构过严"
            elif len(pre.tail(5)) and not pre.tail(5)["prefilter"].eq(True).any():
                bottleneck = "状态过期"
            else:
                bottleneck = "锚太晚"
        summaries.append({
            **q.to_dict(), "first_prefilter": first_prefilter,
            "first_pulse": first_pulse, "first_capital_stage": first_stage,
            "first_supply": first_supply, "first_structure": first_structure,
            "first_armed": first_armed, "first_ready": first_ready,
            "covered_before_t0": covered, "lead_sessions": _lead_sessions(first_armed, q["t0"], cal),
            "lead_left_censored": bool(first_armed and len(x) and
                                       cal.index(first_armed) - cal.index(str(x["date"].iloc[0])) <= 5),
            "bottleneck": bottleneck,
            "alert_in_window": bool(x["state"].isin({"ARMED", "READY", "TRIGGERED"}).any()),
        })
    return pd.DataFrame(summaries)


def write_outputs(queue: pd.DataFrame, trace: pd.DataFrame, summary: pd.DataFrame) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    queue.to_csv(OUT / "audit_queue.csv", index=False, encoding="utf-8-sig")
    trace.to_csv(OUT / "audit_daily_trace.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "audit_stock_summary.csv", index=False, encoding="utf-8-sig")
    carriers = summary[summary["role"] == "ignition_carrier"]
    misses = carriers[~carriers["covered_before_t0"]]
    bottlenecks = (misses.groupby(["theme_label", "bottleneck"]).size().rename("count")
                   .reset_index().sort_values(["theme_label", "count"], ascending=[True, False]))
    bottlenecks.to_csv(OUT / "audit_bottlenecks.csv", index=False, encoding="utf-8-sig")

    cases = []
    for case_id, group in carriers.groupby("audit_case"):
        covered = group["covered_before_t0"].sum()
        leads = pd.to_numeric(group.loc[group["covered_before_t0"], "lead_sessions"], errors="coerce")
        cases.append({
            "audit_case": case_id, "theme_label": group["theme_label"].iloc[0],
            "t0": group["t0"].iloc[0], "n_carriers": int(len(group)),
            "covered_before_t0": int(covered),
            "coverage_rate": round(float(covered / len(group)), 4) if len(group) else None,
            "lead_sessions_median": float(leads.median()) if len(leads) else None,
            "lead_sessions_min": int(leads.min()) if len(leads) else None,
            "lead_sessions_max": int(leads.max()) if len(leads) else None,
            "lead_left_censored": int(group["lead_left_censored"].sum()),
        })
    controls = summary[summary["role"].isin({"liquidity_matched_random", "failed_window_control"})]
    metrics = {
        "metric_policy": "discovery-only; no return, win-rate, or profit metric",
        "threshold_policy": "all provisional capital-stage thresholds are placeholders pending P1 calibration",
        "queue_size": int(len(queue)), "trace_rows": int(len(trace)), "cases": cases,
        "controls": {
            "n": int(len(controls)), "alerts": int(controls["alert_in_window"].sum()),
            "alert_rate": round(float(controls["alert_in_window"].mean()), 4) if len(controls) else None,
            "covered_before_t0": int(controls["covered_before_t0"].sum()),
            "covered_before_t0_rate": round(float(controls["covered_before_t0"].mean()), 4)
            if len(controls) else None,
        },
        "approximations": APPROXIMATIONS,
    }
    (OUT / "audit_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    case_df = pd.DataFrame(cases)
    report = [
        "# 主升浪闭环 P0 命中审计", "",
        "## 审计边界", "",
        "- 目标：只检验点火载体能否在 T0 前被发现，以及提前多少个交易日。",
        "- 禁用：收益、胜率、盈亏和任何以结果收益反调阈值的指标。",
        f"- 影子阶段：`{O.CAPITAL_STAGE_VERSION}`，仅观察，不写生产状态库。",
        "- READY 结果为保守下界；ARMED 前置覆盖是本轮主判据。", "",
        "## 三题材结果", "",
        case_df.to_markdown(index=False) if len(case_df) else "(无)", "",
        "## 未覆盖卡点", "",
        bottlenecks.to_markdown(index=False) if len(bottlenecks) else "(无未覆盖载体)", "",
        "## 对照告警", "",
        f"- 对照 {metrics['controls']['n']} 只，T0 前进入 ARMED/READY/TRIGGERED "
        f"{metrics['controls']['covered_before_t0']} 只"
        f"（{metrics['controls']['covered_before_t0_rate']}）；完整审计窗内告警 "
        f"{metrics['controls']['alerts']} 只（{metrics['controls']['alert_rate']}）。",
        "- 该数字只衡量发现端噪声，不代表交易亏损或胜率。", "",
        "## 近似项", "",
    ] + [f"- {x}" for x in APPROXIMATIONS] + [
        "", "## 评审门", "",
        "- 本报告先提交人工裁决；裁决前不实施 P1，不修改 READY/ARMED/TRIGGERED 生产转移。",
        "- P1 若获准，才校准 PROBED/ABSORBING/COILED 定义并接入紧链交汇日报。",
    ]
    (OUT / "AUDIT_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue-only", action="store_true", help="only build the audit queue")
    args = ap.parse_args()
    cal = market_calendar("2099-12-31")
    queue = build_audit_queue(cal)
    OUT.mkdir(parents=True, exist_ok=True)
    queue.to_csv(OUT / "audit_queue.csv", index=False, encoding="utf-8-sig")
    print(f"audit queue: {len(queue)} rows")
    if args.queue_only:
        return 0
    max_end = max(_date_window(t0, cal, PRE_SESSIONS, POST_SESSIONS)[-1]
                  for t0 in queue["t0"].unique())
    panels = build_industry_panel(queue, max_end)
    print(f"industry panels: {len(panels)}")
    trace = replay_queue(queue, cal, panels)
    summary = summarize(queue, trace, cal)
    metrics = write_outputs(queue, trace, summary)
    print(f"done: {OUT / 'AUDIT_REPORT.md'} | cases={len(metrics['cases'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
