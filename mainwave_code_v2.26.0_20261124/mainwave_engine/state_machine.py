# -*- coding: utf-8 -*-
"""READY 状态机 (方案 §4/Q1) — 2026-08-13 按 Codex 审核重写。

关键修订(P1):
  1 阻力 episode 冻结: FORMING 首次获得有效阻力即存储; FORMING/DEGRADED 期只允许
    ±0.5×冻结ATR 的有界更新(轮2细则), READY 后完全冻结; 大偏移=只能等新 episode。
  2 READY 检查顺序: 寿命/公共健康/支柱失效 → DEGRADED 优先于 点火/假突破。
  3 寿命按【市场交易日历】推进(停牌不暂停 READY/FAILED/INVALID 时钟)。
  4 路径切换次日生效(pending_route)。
  5 FAILED 记录失败日 ATR20; 超时→OUT 附带 6 日再入池冷却(防绕过重建纪律)。
  6 状态库原子写 + 损坏 fail-closed + 日期单调守卫(防回放污染生产)。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

STORE = Path("results/mainwave_engine/state_store.json")

LIFE = {"FORMING": 30, "ARMED": 15, "READY": 10, "DEGRADED": 10, "FAILED": 15,
        "INVALID_COOLDOWN": 10, "OUT_COOLDOWN": 6}


class StoreError(RuntimeError):
    pass


def load_store() -> dict:
    if not STORE.exists():
        return {"_meta": {"engine_as_of": ""}, "symbols": {}}
    try:
        raw = json.loads(STORE.read_text(encoding="utf-8"))
    except Exception as e:  # fail closed: 损坏不得当空库 (审核修订)
        raise StoreError(f"状态库损坏, 拒绝启动(防冻结阻力/寿命全丢): {STORE} ({e})")
    if "symbols" not in raw:  # 旧格式迁移
        raw = {"_meta": {"engine_as_of": ""}, "symbols": raw}
    return raw


def save_store(store: dict, as_of: str) -> None:
    store.setdefault("_meta", {})["engine_as_of"] = as_of
    STORE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STORE.with_suffix(".tmp")
    tmp.write_text(json.dumps(store, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, STORE)  # 原子替换 (审核修订)


def guard_date(store: dict, run_date: str) -> None:
    prev = store.get("_meta", {}).get("engine_as_of", "")
    if prev and run_date <= prev:
        raise StoreError(
            f"日期非递增: 状态库已推进到 {prev}, 拒绝回放 {run_date} 污染生产状态。"
            f"(回放请先备份并删除 {STORE})")


def mkt_sessions_since(d: str | None, market_dates: list[str]) -> int:
    """按市场日历计数(停牌不暂停时钟, 审核修订)。d不在日历=大数。"""
    if not d:
        return 10 ** 6
    try:
        return len(market_dates) - 1 - market_dates.index(d)
    except ValueError:
        return 10 ** 6


def _freeze_or_update(rec: dict, f: dict) -> dict:
    """FORMING/ARMED/DEGRADED 期的锚有界更新: 无冻结→存(含结构类型); 同类型且偏移≤0.5×冻结ATR→更新; 否则保持。
    v1.1: 锚来自结构族(A收敛/B挖坑/C平台), 换结构类型不允许原地改锚(须新episode)。"""
    new_res, new_atr = f.get("resistance"), f.get("resistance_atr")
    if new_res is None:
        return rec
    old = rec.get("stored_resistance")
    if old is None:
        rec.update(stored_resistance=new_res, resistance_atr=new_atr,
                   invalid_price=f.get("invalid_price_candidate"),
                   structure_type=f.get("structure_type"),
                   structure_started_at=f.get("structure_started_at"))
    elif (f.get("structure_type") == rec.get("structure_type")
          and abs(new_res - old) <= 0.5 * (rec.get("resistance_atr") or new_atr)):
        rec.update(stored_resistance=new_res)  # 有界微调, ATR/invalid/类型 不重置
    return rec


def _enter_ready(rec: dict, f: dict, today: str, C: float) -> dict | None:
    """READY 入场公共守卫(冻结锚存在/C>失效/C<取整触发) + v1.1 复活指纹检查。
    返回 None = 守卫未过。"""
    inv2 = rec.get("invalid_price")
    if rec.get("stored_resistance") is None:
        return None
    trig = round(rec["stored_resistance"] + 0.2 * rec["resistance_atr"], 2)
    if not ((inv2 is None or C > inv2) and C < trig):
        return None
    # 复活指纹(仅二次READY): 至少一项证据新于上次 ready_at, 同一证据不得无限续命
    last_ready = rec.get("last_ready_at")
    if last_ready:
        ev_date = f.get("max_evidence_date") or ""
        if not ev_date or ev_date <= last_ready:
            return None
    rec.pop("pending_route", None)
    return {**rec, "state": "READY", "ready_at": today, "last_ready_at": today,
            "route": f.get("route") or rec.get("route"), "trigger_price": trig}


def step(sym: str, rec: dict | None, f: dict, today: str, market_dates: list[str]) -> dict:
    rec = dict(rec or {"state": "OUT"})
    st = rec.get("state", "OUT")
    C = f["close"]

    # 路径切换次日生效 (审核修订)
    pend = rec.pop("pending_route", None)
    if pend and pend.get("set_on") != today:
        rec["route"] = pend["route"]
    elif pend:
        rec["pending_route"] = pend
    cur_route = f.get("route")
    if cur_route and cur_route != rec.get("route") and st not in ("OUT",):
        rec["pending_route"] = {"route": cur_route, "set_on": today}
    elif st == "OUT":
        rec["route"] = cur_route

    # 最高优先级: 结构失效
    inv = rec.get("invalid_price")
    if inv is not None and C < inv and st not in ("OUT", "INVALID"):
        return {"state": "INVALID", "invalid_at": today, "route": rec.get("route")}

    if st == "INVALID":
        if (mkt_sessions_since(rec.get("invalid_at"), market_dates) >= LIFE["INVALID_COOLDOWN"]
                and f["prefilter"]):
            nr = {"state": "FORMING", "forming_since": today, "last_refresh": today,
                  "base_started_at": today, "route": cur_route}
            return _freeze_or_update(nr, f)
        return rec

    if st == "OUT":
        cf = rec.get("out_cooldown_from")  # 复审#6: 冷却按"起点+交易日差"算, 不能预取未来日历
        if cf and mkt_sessions_since(cf, market_dates) < LIFE["OUT_COOLDOWN"]:
            return rec
        if f["prefilter"]:
            nr = {"state": "FORMING", "forming_since": today, "last_refresh": today,
                  "base_started_at": today, "route": cur_route}
            return _freeze_or_update(nr, f)
        return rec

    if st == "FORMING":
        rec = _freeze_or_update(rec, f)
        if f["ready_core"]:                       # 三支柱同日齐备: 直接 READY
            nr = _enter_ready(rec, f, today, C)
            if nr:
                return nr
        if f.get("armed_core") and rec.get("stored_resistance") is not None:
            inv2 = rec.get("invalid_price")       # 结构+供给完成、方向未定 → ARMED (v1.1)
            if inv2 is None or C > inv2:
                return {**rec, "state": "ARMED",
                        "armed_at": rec.get("armed_at") or today}
        # 复审#7: 只有【新证据】(事件日期晚于上次刷新)才续命, 条件持续为真不算刷新
        ev_d = f.get("max_evidence_date")
        if ev_d and ev_d > (rec.get("last_refresh") or ""):
            rec["last_refresh"] = today
        elif mkt_sessions_since(rec.get("last_refresh"), market_dates) > LIFE["FORMING"]:
            return {"state": "OUT"}
        return rec

    if st == "ARMED":
        rec = _freeze_or_update(rec, f)
        # 共振通道(P0-1): 前一日已ARMED × 板块首萌 × 个股确认 → 直接TRIGGERED
        # 复审#4修复: 共振日的 armed_core+公共健康 必须仍然成立(结构/供给当日已崩的不许触发);
        # 路径兜底 HOT(题材共振本质是HOT), 禁止 UNCLASSIFIED 流入 TRIGGERED
        if (f.get("armed_resonance") and rec.get("armed_at") != today
                and f.get("armed_core") and f.get("health_ok")
                and (rec.get("invalid_price") is None or C > rec["invalid_price"])):
            rt = f.get("route")
            if not rt or rt == "UNCLASSIFIED":
                rt = rec.get("route") if rec.get("route") not in (None, "UNCLASSIFIED") else "HOT"
            return {**rec, "state": "TRIGGERED", "triggered_at": today,
                    "trigger_kind": "armed_sector_sync", "route": rt,
                    "trigger_price": rec.get("trigger_price") or round(
                        rec["stored_resistance"] + 0.2 * rec["resistance_atr"], 2)}
        if f["ready_core"]:                       # 方向锚确认 → READY
            nr = _enter_ready(rec, f, today, C)
            if nr:
                return nr
        if not f.get("armed_core"):               # 结构/供给失效 → 回 FORMING
            return {**{k: rec[k] for k in ("stored_resistance", "resistance_atr", "invalid_price",
                                           "structure_type", "structure_started_at",
                                           "base_started_at", "route", "last_ready_at") if k in rec},
                    "state": "FORMING", "forming_since": today, "last_refresh": today}
        if mkt_sessions_since(rec.get("armed_at"), market_dates) >= LIFE["ARMED"]:
            rec.pop("armed_at", None)             # 过期: 回 FORMING 重走(证据须刷新)
            return {**{k: rec[k] for k in ("stored_resistance", "resistance_atr", "invalid_price",
                                           "structure_type", "structure_started_at",
                                           "base_started_at", "route", "last_ready_at") if k in rec},
                    "state": "FORMING", "forming_since": today, "last_refresh": today}
        return rec

    if st == "READY":
        # 顺序(复审#3修正): ①硬失效(寿命/健康/结构/供给)→DEGRADED ②假突破 ③点火
        # ④仅方向锚失守→ARMED —— 锚检查放在点火之后: 点火事件本身(如INST股当日涨停)
        # 会让"近20日无强冲击"类锚条件当日失效, 若先查锚就永远错过该触发
        hard_stale = (mkt_sessions_since(rec.get("ready_at"), market_dates) >= LIFE["READY"]
                      or not f.get("health_ok")
                      or not f.get("structure_ready") or not f.get("supply_ready"))
        if hard_stale:
            return {**rec, "state": "DEGRADED", "degraded_at": today}
        if f.get("false_breakout"):
            return {**rec, "state": "FAILED", "failed_at": today,
                    "failure_low": f["low"], "atr_at_failure": f.get("atr_now")}
        if f.get("trigger_event") and rec.get("ready_at") != today:
            return {**rec, "state": "TRIGGERED", "triggered_at": today,
                    "trigger_kind": f.get("trigger_kind")}
        if not f.get("route_anchor"):
            return {**rec, "state": "ARMED", "armed_at": rec.get("armed_at") or today}
        return rec

    if st == "DEGRADED":
        rec = _freeze_or_update(rec, f)
        if f["ready_core"]:                       # 复活: _enter_ready 内含指纹检查(证据须新于上次ready)
            nr = _enter_ready(rec, f, today, C)
            if nr:
                return nr
        if mkt_sessions_since(rec.get("degraded_at"), market_dates) >= LIFE["DEGRADED"]:
            if f["prefilter"]:
                return {**{k: rec[k] for k in ("stored_resistance", "resistance_atr", "invalid_price",
                                               "structure_type", "base_started_at", "route",
                                               "last_ready_at") if k in rec},
                        "state": "FORMING", "forming_since": today, "last_refresh": today}
            return {"state": "OUT"}
        return rec

    if st == "FAILED":
        if f.get("rebuild_ok"):
            return {**{k: rec[k] for k in ("stored_resistance", "resistance_atr", "invalid_price",
                                           "structure_type", "structure_started_at",
                                           "base_started_at", "route", "last_ready_at") if k in rec},
                    "state": "FORMING", "forming_since": today, "last_refresh": today}
        if mkt_sessions_since(rec.get("failed_at"), market_dates) > LIFE["FAILED"]:
            # 超时出局带冷却(复审#6修复: 存起点日, OUT分支按交易日差裁决)
            return {"state": "OUT", "out_cooldown_from": today}
        return rec

    if st == "TRIGGERED":
        if mkt_sessions_since(rec.get("triggered_at"), market_dates) >= 1:
            return {**rec, "state": "DEGRADED", "degraded_at": today}
        return rec

    return rec


def time_step(rec: dict, today: str, market_dates: list[str]) -> dict:
    """停牌/无当日数据的在池记录: 只推进【时间性】转移(复审#5: 停牌不暂停时钟)。
    价格性判定(触发/失效/结构)留待复牌评估。"""
    st = rec.get("state")
    ss = lambda d: mkt_sessions_since(d, market_dates)
    if st == "READY" and ss(rec.get("ready_at")) >= LIFE["READY"]:
        return {**rec, "state": "DEGRADED", "degraded_at": today}
    if st == "TRIGGERED" and ss(rec.get("triggered_at")) >= 1:
        return {**rec, "state": "DEGRADED", "degraded_at": today}
    if st == "ARMED" and ss(rec.get("armed_at")) >= LIFE["ARMED"]:
        nr = {k: rec[k] for k in ("stored_resistance", "resistance_atr", "invalid_price",
                                  "structure_type", "structure_started_at",
                                  "base_started_at", "route", "last_ready_at") if k in rec}
        return {**nr, "state": "FORMING", "forming_since": today,
                "last_refresh": rec.get("armed_at", today)}
    if st == "DEGRADED" and ss(rec.get("degraded_at")) >= LIFE["DEGRADED"]:
        nr = {k: rec[k] for k in ("stored_resistance", "resistance_atr", "invalid_price",
                                  "structure_type", "structure_started_at",
                                  "base_started_at", "route", "last_ready_at") if k in rec}
        return {**nr, "state": "FORMING", "forming_since": today,
                "last_refresh": rec.get("degraded_at", today)}
    if st == "FORMING" and ss(rec.get("last_refresh")) > LIFE["FORMING"]:
        return {"state": "OUT"}
    if st == "FAILED" and ss(rec.get("failed_at")) > LIFE["FAILED"]:
        return {"state": "OUT", "out_cooldown_from": today}
    return rec


def rebuild_ok(ctx, failed_at_idx: int, failure_low: float, old_resistance: float,
               atr_at_failure: float | None) -> bool:
    """FAILED 后最小重建 (Q1, 审核修订: 用失败日ATR/最低low/完整TR/贴近含上界)。"""
    n = ctx.n
    since = n - 1 - failed_at_idx
    if since < 6 or not atr_at_failure or not np.isfinite(atr_at_failure):
        return False
    low_since = float(np.nanmin(ctx.L[failed_at_idx + 1:]))
    tr5 = np.maximum(ctx.H[-5:] - ctx.L[-5:],
                     np.maximum(np.abs(ctx.H[-5:] - ctx.PC[-5:]), np.abs(ctx.L[-5:] - ctx.PC[-5:])))
    atr5 = float(np.nanmean(tr5))
    quiet = (np.nanmedian(ctx.turn[-3:]) <=
             0.75 * np.nanmedian(ctx.turn[max(0, failed_at_idx - 4):failed_at_idx]))
    near_cnt = int(np.sum(np.abs(ctx.C[-3:] - old_resistance) <= atr_at_failure))
    return (low_since >= failure_low and atr5 <= 0.8 * atr_at_failure
            and quiet and near_cnt >= 2)
