# -*- coding: utf-8 -*-
"""每日运行编排 (方案 §12; v1.1 2026-08-14: ARMED态/结构族/概念题材层/首萌冷却/复活指纹/焦点表).

流程: 域 → light pass(广度聚合+预筛) → L1 行业+概念双层五态(首萌20日冷却) → heavy pass
     (结构族+九特征+路径+ARMED/READY判定) → 状态机(含ARMED共振通道) → 焦点表≤15 → 一页输出
用法: python -m mainwave_engine.daily_run --date YYYY-MM-DD
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from . import features as F
from . import observation as O
from . import sectors as S
from . import state_machine as SM

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

OUT_ROOT = Path("results/mainwave_engine")
EMERGENCE_LOG = OUT_ROOT / "emergence_log.json"
LIMIT_SNAPSHOT_COLUMNS = ["date", "symbol", "pct_chg", "limit_pct"]
EMERGENCE_COOLDOWN = 20  # 首萌冷却(市场日), 防 EMERGING↔NEUTRAL 反复制造"首次" (P0-3)


def market_calendar(as_of: str) -> list[str]:
    p = Path("data/000001_SH_index.csv")
    d = pd.read_csv(p, usecols=["trade_date"])
    def norm(x) -> str:
        s = str(x)
        return s if "-" in s else f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    ds = sorted({norm(x) for x in d["trade_date"]})
    return [x for x in ds if x <= as_of]


def prefilter_channels(ctx: F.Ctx, fund_ok: bool) -> tuple[bool, str]:
    n = ctx.n
    C = ctx.C[-1]
    ma_ok = np.isfinite(ctx.ma120[-1])
    spread = ((max(ctx.ma20[-1], ctx.ma60[-1], ctx.ma120[-1])
               - min(ctx.ma20[-1], ctx.ma60[-1], ctx.ma120[-1])) / C) if ma_ok else 9.9
    ch1 = False
    if n >= 250 and ma_ok:
        pos = C / np.nanmax(ctx.H[-250:]) - 1
        ch1 = spread <= 0.08 and -0.45 <= pos <= -0.20
    rc = F.resistance_cluster(ctx)
    ch2 = bool(rc["ok"])
    slope60 = (ctx.ma60[-1] - ctx.ma60[-21]) if np.isfinite(ctx.ma60[-1]) and n >= 81 else -1
    atr_pct = ctx.atr20 / ctx.C
    hist = atr_pct[-120:]
    vol_q = float(np.mean(hist[np.isfinite(hist)] <= atr_pct[-1])) if np.isfinite(atr_pct[-1]) else 1.0
    ch3 = fund_ok and C >= ctx.ma60[-1] and slope60 > 0 and vol_q <= 0.40
    pit = F.pit_reclaim(ctx)
    ch4 = bool(pit["ok"]) and (pit.get("age") or 99) <= 20
    chans = [c for c, hit in zip("1234", (ch1, ch2, ch3, ch4)) if hit]
    return (len(chans) > 0, "+".join(chans))


def evaluate_symbol(sym: str, df: pd.DataFrame, run_date: str, ind_panel: pd.DataFrame,
                    group_state: dict, group_B: dict, first_emg: set, ignitions: dict,
                    mkt_med: dict, frozen: dict | None) -> dict:
    ctx = F.Ctx(sym, df)
    ind = D.industry_map().get(sym, "")
    groups = [ind] + [f"C:{c}" for c in D.concepts_map(run_date).get(sym, [])] if ind else \
             [f"C:{c}" for c in D.concepts_map(run_date).get(sym, [])]
    ind_df = ind_panel[ind_panel["industry"] == ind][["date", "ret"]] if ind else None
    sector_state = group_state.get(ind, "NEUTRAL")
    rel20 = F.align_rel(ctx, ind_df, 20)

    fund = F.fundamental_anchor(D.fundamental_view(D.code6(sym), run_date))
    imp = F.impulse_retention(ctx)
    fresh = F.fresh_impulse(ctx)   # v1.2裁决#2: 游资参与看近日涨停即可(仅路径签名)
    rc = F.resistance_cluster(ctx)
    pit = F.pit_reclaim(ctx)
    ip = F.inst_platform(ctx)
    va = F.volume_asymmetry(ctx, rel20)
    ss = F.silent_supply(ctx)
    ib = F.institutional_behavior(ctx, rel20)
    # 复审#8: 梯队点火取自个股所属全部分组(行业+概念), 且龙头不得响应自己的点火
    ign = None
    for g in groups:
        gi = ignitions.get(g)
        if gi and gi.get("leader") != sym and (ign is None or gi["date"] > ign["date"]):
            ign = gi
    relay = F.relay_response(ctx, ign, mkt_med)
    health = F.common_health(ctx, run_date)
    struct = F.structure_family(ctx, rc, pit, ip)

    # 结构口径: 已冻结→对冻结锚验证(含类型不变量, 复审#7); 未冻结→结构族当日判定
    if frozen and frozen.get("stored_resistance") is not None:
        structure_ready = F.structure_vs_frozen(ctx, frozen["stored_resistance"],
                                                frozen["resistance_atr"],
                                                frozen.get("structure_type"))
    else:
        structure_ready = bool(struct["ok"])

    supply_evidence = max([("pit", pit), ("asym", va), ("silent", ss)],
                          key=lambda kv: (kv[1]["ok"], kv[1]["score"]))
    supply_ready = bool(supply_evidence[1]["ok"])
    theme_emerging = any(group_state.get(g) == "EMERGING" for g in groups)
    # 复审#2: ACTIVE 板块=HOT锚延续(不构成首萌触发); 新鲜冲击既是路径签名也是方向锚(资金已来)
    theme_anchor = any(group_state.get(g) in ("EMERGING", "ACTIVE") for g in groups)
    hot_anchor = theme_anchor or imp["ok"] or relay["ok"] or fresh["ok"]
    inst_anchor = fund["ok"] and ib["ok"]
    hot_sig = (imp["ok"] or (pit["ok"] and (pit.get("age") or 99) <= 20)
               or relay["ok"] or fresh["ok"])
    route = ("HYBRID" if hot_sig and inst_anchor else "HOT" if hot_sig
             else "INST" if inst_anchor
             else "HOT" if theme_anchor else "UNCLASSIFIED")  # 题材锚型HOT(修不可达分支)
    armed_core = health["ok"] and structure_ready and supply_ready
    ready_core = armed_core and (hot_anchor or inst_anchor)   # 有锚⇒route必非UNCLASSIFIED

    # 证据新鲜度(复活指纹): 各事件日期的最大值
    ev_dates = [d for d in (imp.get("event_date"), pit.get("event_date"),
                            relay.get("event_date"), fresh.get("event_date"),
                            run_date if theme_emerging else None) if d]
    max_evidence_date = max(ev_dates) if ev_dates else None

    # 共振主题 = 个股所属分组中【板块热度(B)最高】者(v1.2用户裁决#1: 看板块热度, 不看归属主次);
    # 仅当该主题首萌才构成共振, 防多题材股重复触发
    themed = [(g, group_B.get(g)) for g in groups if group_B.get(g) is not None]
    resonance_theme = max(themed, key=lambda kv: kv[1])[0] if themed else None
    stock_first_emg = resonance_theme in first_emg if resonance_theme else False
    return {
        "symbol": sym, "industry": ind, "groups": groups, "sector_state": sector_state,
        "route": route, "close": float(ctx.C[-1]), "low": float(ctx.L[-1]),
        "atr_now": float(ctx.atr20[-1]) if np.isfinite(ctx.atr20[-1]) else None,
        "prefilter": True, "armed_core": bool(armed_core), "ready_core": bool(ready_core),
        "structure_ready": bool(structure_ready), "supply_ready": supply_ready,
        "route_anchor": bool(hot_anchor or inst_anchor), "health_ok": bool(health["ok"]),
        "supply_kind": supply_evidence[0] if supply_ready else "",
        "supply_score": float(supply_evidence[1]["score"]) if supply_ready else 0.0,
        "resistance": struct.get("anchor"), "resistance_atr": struct.get("atr"),
        "invalid_price_candidate": struct.get("invalid"),
        "structure_type": struct.get("type"),
        "structure_started_at": struct.get("started_at"),
        "imp": bool(imp["ok"]), "imp_age": imp.get("age"),
        "fresh": bool(fresh["ok"]), "fresh_age": fresh.get("age"),
        "pit": bool(pit["ok"]), "pit_age": pit.get("age"),
        "max_evidence_date": max_evidence_date,
        "stock_first_emg": bool(stock_first_emg),
        "resonance_theme": resonance_theme,
        "_ctx": ctx,
    }


def check_trigger(rec: dict, ctx: F.Ctx, first_emg: set, groups: list) -> tuple[bool, str, bool]:
    tp, res = rec.get("trigger_price"), rec.get("stored_resistance")
    if tp is None or res is None:
        return False, "", False
    C, H, L = ctx.C[-1], ctx.H[-1], ctx.L[-1]
    amt_med = float(np.nanmedian(ctx.amount[-21:-1]))
    breakout = C > tp and ctx.amount[-1] > amt_med
    limit_tr = ctx.pct[-1] >= ctx.lim * 0.985 and L < H
    # 板块共振 = 所属任一分组【首次】萌芽(含20日冷却) + 个股破5日高
    sector_sync = bool(any(g in first_emg for g in groups) and C > np.nanmax(ctx.H[-6:-1]))
    kind = "breakout" if breakout else "limit_up" if limit_tr else "sector_sync" if sector_sync else ""
    false_bo = bool(H >= tp and C < res
                    and (ctx.cloc[-1] < 0.45 or ctx.amount[-1] > 1.2 * amt_med))
    return bool(kind), kind, false_bo


def build_focus(ready_rows: list[dict], cap: int = 15) -> pd.DataFrame:
    """人工焦点表(P1): 字典序 距触发↑ → 证据新鲜度↓ → 供给质量↓ → 止损距离↑; HOT≤10, INST/HYBRID≤5, 互补填满。"""
    if not ready_rows:
        return pd.DataFrame()
    df = pd.DataFrame(ready_rows)
    df["dist_pct"] = (df["trigger_price"] / df["close"] - 1) * 100
    df["stop_pct"] = (1 - df["invalid_price"] / df["close"]) * 100
    df["ev_date"] = df["ev_date"].fillna("")
    df["预检"] = np.where(df["stop_pct"] > 8, "⚠止损远", "")  # §9.5四问预检: 止损距>8%执行层会拒买
    # 字典序(v1.2#4): 状态成熟度(ready_at越新越前) → 距触发 → 证据新鲜度 → 供给质量 → 止损距离
    df = df.sort_values(["ready_at", "dist_pct", "ev_date", "supply_score", "stop_pct"],
                        ascending=[False, True, False, False, True])
    hot = df[df["route"] == "HOT"].head(10)
    other = df[df["route"].isin(["INST", "HYBRID"])].head(5)
    focus = pd.concat([hot, other])
    if len(focus) < cap:  # 席位互补
        rest = df[~df.index.isin(focus.index)].head(cap - len(focus))
        focus = pd.concat([focus, rest])
    return focus.sort_values(["dist_pct"]).head(cap)


def write_limit_up_snapshot(per_stock: dict[str, pd.DataFrame], run_date: str,
                            path: Path) -> int:
    """Persist today's limit-up symbols from frames already loaded by the light pass."""
    rows = []
    for symbol, frame in per_stock.items():
        if frame is None or frame.empty or str(frame["date"].iloc[-1]) != run_date:
            continue
        pct_chg = pd.to_numeric(frame["pct_chg"].iloc[-1], errors="coerce")
        limit_pct = D.limit_pct(symbol)
        if np.isfinite(pct_chg) and pct_chg >= limit_pct * 0.985:
            rows.append({
                "date": run_date, "symbol": symbol,
                "pct_chg": float(pct_chg), "limit_pct": float(limit_pct),
            })
    snapshot = pd.DataFrame(rows, columns=LIMIT_SNAPSHOT_COLUMNS)
    if len(snapshot):
        snapshot = snapshot.sort_values("symbol")
    snapshot.to_csv(path, index=False, encoding="utf-8-sig")
    return len(snapshot)


def run_day(run_date: str, backfill: bool = False) -> int:
    out = OUT_ROOT / run_date.replace("-", "")
    out.mkdir(parents=True, exist_ok=True)

    store = SM.load_store()
    SM.guard_date(store, run_date)
    mdates = market_calendar(run_date)
    universe = D.list_universe()
    ind_map = D.industry_map()
    con_map = D.concepts_map(run_date)
    print(f"[1/6] 股票域 {len(universe)} 只 | 市场日历至 {mdates[-1] if mdates else '?'}")

    per_stock: dict[str, pd.DataFrame] = {}
    prefilter_hits: dict[str, str] = {}
    lim_of: dict[str, float] = {}
    chan_count = {"1": 0, "2": 0, "3": 0, "4": 0}
    for sym in universe:
        df = D.load_tail(sym, 300, as_of=run_date)
        if df is None:
            continue
        per_stock[sym] = df
        lim_of[sym] = D.limit_pct(sym)
        if str(df["date"].iloc[-1]) != run_date:
            continue
        try:
            ctx = F.Ctx(sym, df)
        except ValueError:
            continue
        fund_ok = bool(F.fundamental_anchor(D.fundamental_view(D.code6(sym), run_date))["ok"])
        hit, chans = prefilter_channels(ctx, fund_ok)
        if hit:
            prefilter_hits[sym] = chans
            for ch in chans.split("+"):
                chan_count[ch] += 1
    active_today = [s for s, d in per_stock.items() if str(d["date"].iloc[-1]) == run_date]
    print(f"[2/6] light pass: 载入 {len(per_stock)} (当日活跃 {len(active_today)}), 预筛 {len(prefilter_hits)} "
          f"(1低位={chan_count['1']} 2收敛={chan_count['2']} 3机构={chan_count['3']} 4挖坑={chan_count['4']})")
    if not active_today:
        print(f"[!] {run_date} 非交易日或数据未更新, 状态库不变, 退出")
        return 0

    # L1 双层(行业+概念)广度与五态
    limit_count = write_limit_up_snapshot(per_stock, run_date, out / "limit_up_snapshot.csv")
    print(f"[2.1/6] limit-up snapshot: {limit_count} stocks -> {out / 'limit_up_snapshot.csv'}")

    group_map = {s: ([ind_map[s]] if s in ind_map else []) + [f"C:{c}" for c in con_map.get(s, [])]
                 for s in per_stock}
    panel, mkt_med = S.build_breadth(per_stock, group_map, lim_of)
    sector_tbl = S.sector_states(panel)
    sector_tbl.to_csv(out / "sector_states.csv", index=False, encoding="utf-8-sig")
    group_state = dict(zip(sector_tbl["industry"], sector_tbl["state"]))
    group_B = dict(zip(sector_tbl["industry"], sector_tbl["B"]))  # 板块热度(共振主题判定用)
    prev_file = OUT_ROOT / "sector_states_prev.json"
    group_prev = json.loads(prev_file.read_text(encoding="utf-8")) if prev_file.exists() else {}
    emg_log = json.loads(EMERGENCE_LOG.read_text(encoding="utf-8")) if EMERGENCE_LOG.exists() else {}
    # 首萌集合: 今日EMERGING + 昨日非EMERGING + 距上轮萌芽≥冷却期
    first_emg = {g for g, st_ in group_state.items()
                 if st_ == "EMERGING" and group_prev.get(g) != "EMERGING"
                 and SM.mkt_sessions_since(emg_log.get(g), mdates) >= EMERGENCE_COOLDOWN}
    n_ind = int((~sector_tbl["industry"].astype(str).str.startswith("C:")).sum())
    print(f"[3/6] L1 双层: 行业{n_ind}+概念{len(sector_tbl) - n_ind} | "
          + ", ".join(f"{s}={int((sector_tbl['state'] == s).sum())}"
                      for s in ["EMERGING", "ACTIVE", "CROWDED", "RETREAT", "NEUTRAL"])
          + f" | 首萌(冷却后)={sorted(first_emg) if first_emg else '无'}")

    ignitions = S.leader_ignitions(per_stock, group_map, lim_of, run_date)  # 复审#8: 行业+概念双层点火

    tracked = set(prefilter_hits) | {s for s, r in store["symbols"].items()
                                     if r.get("state") not in (None, "OUT")}
    evals: dict[str, dict] = {}
    for sym in sorted(tracked):
        if sym in D.st_set():
            store["symbols"].pop(sym, None)
            continue
        df = per_stock.get(sym)
        if df is None:
            df = D.load_tail(sym, 300, as_of=run_date)
        if df is None or str(df["date"].iloc[-1]) != run_date:
            continue
        frozen = store["symbols"].get(sym)
        try:
            ev = evaluate_symbol(sym, df, run_date, panel, group_state, group_B, first_emg,
                                 ignitions, mkt_med, frozen)
        except ValueError:
            continue
        ev["prefilter"] = sym in prefilter_hits
        ev["prefilter_ch"] = prefilter_hits.get(sym, "")
        evals[sym] = ev
    print(f"[4/6] heavy pass: 评估 {len(evals)} 只")

    new_store, ready_rows, trig_rows, armed_rows, trajectory_rows = {}, [], [], [], []
    for sym, ev in evals.items():
        rec = store["symbols"].get(sym)
        ctx = ev.pop("_ctx")
        if rec and rec.get("state") == "READY":
            trig, kind, false_bo = check_trigger(rec, ctx, first_emg, ev["groups"])
            ev["trigger_event"], ev["trigger_kind"], ev["false_breakout"] = trig, kind, false_bo
        if rec and rec.get("state") == "ARMED":
            # 共振通道确认: 板块首萌 + 个股破5日高
            ev["armed_resonance"] = bool(ev["stock_first_emg"]
                                         and ctx.C[-1] > np.nanmax(ctx.H[-6:-1]))
        if rec and rec.get("state") == "FAILED":
            dts = ctx.dates
            f_idx = dts.index(rec["failed_at"]) if rec.get("failed_at") in dts else -1
            ev["rebuild_ok"] = f_idx > 0 and SM.rebuild_ok(
                ctx, f_idx, rec.get("failure_low", 0),
                rec.get("stored_resistance", 1e9), rec.get("atr_at_failure"))
        nrec = SM.step(sym, rec, ev, run_date, mdates)
        if nrec.get("state") != "OUT" or nrec.get("out_cooldown_from"):
            new_store[sym] = nrec
        st = nrec.get("state")
        trace_evidence = {
            "data_fresh": True,
            "health": ev.get("health_ok"),
            "structure_ready": ev.get("structure_ready"),
            "supply_ready": ev.get("supply_ready"),
            "imp": ev.get("imp"),
            "fresh": ev.get("fresh"),
            "pit": ev.get("pit"),
        }
        trajectory_rows.append({
            "date": run_date, "symbol": sym, "state": st,
            "prefilter_ch": ev.get("prefilter_ch", ""), "data_fresh": True,
            "health": bool(ev.get("health_ok")),
            "structure_ready": bool(ev.get("structure_ready")),
            "supply_ready": bool(ev.get("supply_ready")),
            "supply_kind": ev.get("supply_kind", ""),
            "imp": bool(ev.get("imp")), "imp_age": ev.get("imp_age"),
            "fresh": bool(ev.get("fresh")), "fresh_age": ev.get("fresh_age"),
            "pit": bool(ev.get("pit")), "pit_age": ev.get("pit_age"),
            "route": nrec.get("route", ev.get("route")),
            "armed_core": bool(ev.get("armed_core")),
            "ready_core": bool(ev.get("ready_core")),
            "max_evidence_date": ev.get("max_evidence_date"),
            "capital_stage": O.derive_capital_stage(trace_evidence, st),
            "capital_stage_version": O.CAPITAL_STAGE_VERSION,
        })
        base = {"symbol": sym, "name": D.name_map().get(sym, "") or sym, "industry": ev["industry"],
                "route": nrec.get("route", ev["route"]), "close": round(ev["close"], 2)}
        if st == "READY":
            ready_rows.append({**base, "ready_at": nrec.get("ready_at"),
                               "trigger_price": nrec.get("trigger_price"),
                               "invalid_price": nrec.get("invalid_price"),
                               "structure": nrec.get("structure_type", ""),
                               "supply": ev.get("supply_kind"),
                               "supply_score": ev.get("supply_score", 0.0),
                               "ev_date": ev.get("max_evidence_date"),
                               "sector": ev["sector_state"]})
        if st == "TRIGGERED":
            trig_rows.append({**base, "kind": nrec.get("trigger_kind"),
                              "trigger_price": nrec.get("trigger_price"),
                              "invalid_price": nrec.get("invalid_price")})
        if st == "ARMED":  # 复审#9: 最早发现层必须有名单, 不能只有数量
            res_, atr_ = nrec.get("stored_resistance"), nrec.get("resistance_atr")
            armed_rows.append({**base, "armed_at": nrec.get("armed_at"),
                               "structure": nrec.get("structure_type", ""),
                               "trigger_price": round(res_ + 0.2 * atr_, 2) if res_ and atr_ else None,
                               "invalid_price": nrec.get("invalid_price"),
                               "supply": ev.get("supply_kind"),
                               "ev_date": ev.get("max_evidence_date"),
                               "resonance_theme": ev.get("resonance_theme")})
    for sym, rec in store["symbols"].items():
        if sym not in evals and rec.get("state") not in (None, "OUT"):
            # 复审#5: 停牌/无当日数据 → 时间性转移照常推进(寿命不因停牌暂停), 价格判定留待复牌
            new_store[sym] = SM.time_step(rec, run_date, mdates)
    traced = {row["symbol"] for row in trajectory_rows}
    for sym in sorted(tracked - traced):
        state = new_store.get(sym, {}).get("state", "NOT_EVALUATED")
        trace_evidence = {"data_fresh": False}
        trajectory_rows.append({
            "date": run_date, "symbol": sym, "state": state,
            "prefilter_ch": prefilter_hits.get(sym, ""), "data_fresh": False,
            "health": "", "structure_ready": "", "supply_ready": "", "supply_kind": "",
            "imp": "", "imp_age": "", "fresh": "", "fresh_age": "", "pit": "", "pit_age": "",
            "route": new_store.get(sym, {}).get("route", ""), "armed_core": "", "ready_core": "",
            "max_evidence_date": "",
            "capital_stage": O.derive_capital_stage(trace_evidence, state),
            "capital_stage_version": O.CAPITAL_STAGE_VERSION,
        })
    O.write_trajectory(trajectory_rows, out / "trajectory.csv")
    SM.save_store({"symbols": new_store}, run_date)
    prev_file.write_text(json.dumps(group_state, ensure_ascii=False), encoding="utf-8")
    for g, st_ in group_state.items():
        if st_ == "EMERGING":
            emg_log[g] = run_date
    EMERGENCE_LOG.write_text(json.dumps(emg_log, ensure_ascii=False), encoding="utf-8")
    counts = pd.Series([r.get("state") for r in new_store.values()]).value_counts().to_dict()
    print(f"[5/6] 状态机: {counts}")

    # Keep the CSV schema even when a state has no candidates today.
    ready_df = pd.DataFrame(ready_rows, columns=[
        "symbol", "name", "industry", "route", "close", "ready_at",
        "trigger_price", "invalid_price", "structure", "supply", "supply_score",
        "ev_date", "sector",
    ])
    trig_df = pd.DataFrame(trig_rows, columns=[
        "symbol", "name", "industry", "route", "close", "kind",
        "trigger_price", "invalid_price",
    ])
    armed_df = pd.DataFrame(armed_rows, columns=[
        "symbol", "name", "industry", "route", "close", "armed_at", "structure",
        "trigger_price", "invalid_price", "supply", "ev_date", "resonance_theme",
    ])
    ready_df.to_csv(out / "ready_table.csv", index=False, encoding="utf-8-sig")  # 全量库存
    trig_df.to_csv(out / "triggered.csv", index=False, encoding="utf-8-sig")
    armed_df.to_csv(out / "armed_watch.csv", index=False, encoding="utf-8-sig")  # 埋伏全量(复审#9)
    armed_focus = pd.DataFrame()
    if len(armed_df):
        af = armed_df.dropna(subset=["trigger_price"]).copy()
        if len(af):
            af["dist_pct"] = (af["trigger_price"] / af["close"] - 1) * 100
            armed_focus = af.sort_values("dist_pct").head(10)
    focus = build_focus(ready_rows)
    focus_cols = ["symbol", "name", "industry", "route", "structure", "close",
                  "trigger_price", "invalid_price", "dist_pct", "stop_pct", "预检", "supply", "sector"]
    emg_tbl = (sector_tbl[sector_tbl["state"].isin(["EMERGING", "ACTIVE"])]
               .sort_values(["state", "B"], ascending=[False, False]).head(10))  # EMERGING(首萌)置顶不被截断
    bf_note = ([f"- ⚠ 断档补跑重放页(生成于 {pd.Timestamp.now().strftime('%Y-%m-%d')}): "
                "状态按历史数据逐日重建, 触发为追溯记录非实时观察"] if backfill else [])
    lines = [f"# 主升浪发现引擎 · 每日一页  {run_date}", ""] + bf_note + [
             f"- 域 {len(universe)} → 预筛 {len(prefilter_hits)} → 在池 {len(new_store)}"
             f" (ARMED {counts.get('ARMED', 0)} / READY {counts.get('READY', 0)}) → 触发 {len(trig_df)}",
             f"- 市场环境见 market_context.json，统一建议见 analysis_board.md；仅分析、不生成成交。READY全量见 ready_table.csv", "",
             "## 板块(萌芽/活跃, 行业+概念双层)", emg_tbl.to_markdown(index=False) if len(emg_tbl) else "(无)", "",
             "## 今日触发(次日执行窗口)", trig_df.to_markdown(index=False) if len(trig_df) else "(无)", "",
             f"## 人工焦点表(≤15, 字典序; 库存共 {len(ready_df)})",
             focus[[c for c in focus_cols if c in focus.columns]].round(2).to_markdown(index=False) if len(focus) else "(无)", "",
             f"## 埋伏观察(ARMED 前10, 只欠方向; 全量 {len(armed_df)} 见 armed_watch.csv)",
             armed_focus[[c for c in ["symbol", "name", "industry", "route", "structure", "close",
                                      "trigger_price", "invalid_price", "dist_pct", "supply",
                                      "resonance_theme"] if c in armed_focus.columns]].round(2).to_markdown(index=False)
             if len(armed_focus) else "(无)"]
    (out / "daily_page.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[6/6] 输出 -> {out}\\daily_page.md  (ARMED={counts.get('ARMED', 0)}, "
          f"READY={counts.get('READY', 0)}, 焦点={len(focus)}, TRIGGERED={len(trig_df)})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--trajectory-only", action="store_true",
                    help="rebuild trajectory for engine_as_of without mutating production state")
    ap.add_argument("--backfill-cap", type=int, default=10,
                    help="断档自动补跑上限(交易日数, 0=关闭)")
    a = ap.parse_args()
    if a.trajectory_only:
        from .trajectory_rebuild import rebuild_as_of
        normalized = (a.date if "-" in a.date else
                      f"{a.date[:4]}-{a.date[4:6]}-{a.date[6:]}")
        return rebuild_as_of(normalized, evaluate_symbol, prefilter_channels)
    # 断档检测(P0-1): 状态库上次推进日与本次目标日之间若有交易日空缺, 按序逐日重放,
    # 保证寿命时钟/首萌链/触发记录连续。日期守卫单调, 跳过即永久不可追溯。
    prev = SM.load_store().get("_meta", {}).get("engine_as_of", "")
    missed = [d for d in market_calendar(a.date) if prev < d < a.date] if prev else []
    if missed and 0 < a.backfill_cap and len(missed) <= a.backfill_cap:
        print(f"[补跑] 断档 {len(missed)} 个交易日 ({missed[0]} ~ {missed[-1]}), 按序重放...")
        for d in missed:
            print(f"\n----- 补跑 {d} -----")
            run_day(d, backfill=True)
    elif missed:
        print(f"[!] 断档 {len(missed)} 个交易日, 超过补跑上限 {a.backfill_cap} (或补跑已关闭), 直接跑 {a.date}")
        print(f"[!] 日期守卫单调: 本次跑完后断档期将永久不可追溯; 要补跑请中止, 改用 --backfill-cap {len(missed)} 重跑")
    return run_day(a.date)


if __name__ == "__main__":
    sys.exit(main())
