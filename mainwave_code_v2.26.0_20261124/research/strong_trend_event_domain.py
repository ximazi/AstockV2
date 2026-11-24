"""Historical event domain for the case study (screening quality task book v1.1 §9).

Every first-entry event (NEW / CHILD / RETRY) of CORE-SEMANTICS since START, each traded under ONE frozen
contract so cases are chosen from outcomes computed the same way for all:
  entry   next open only if inside [boundary, signal close + 0.5N]; otherwise the fill state is recorded
  size    example only: 1% of 100,000 at risk to the invalidation, whole lots, symbol cap 35%
  exit    CORE hard risk (low <= invalidation) or the 20-day low channel, both at the next open
  fees    NEW100K paper schedule: 3 bps (min 5 yuan) each side, 5 bps sell tax
Outcomes are descriptive, for choosing explanation cases and paired controls. They are not an optimisation
target and not a portfolio backtest (every event is an independent paper trade; seats and cash are ignored).
Unfinished trades are marked OPEN at the last close and kept apart.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mainwave_engine import strong_trend as ST, strong_trend_semantics as SEM  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
START = "2024-01-01"
CAPITAL, RISK, SYMBOL_CAP = 100000.0, 0.01, 0.35


def main(day, start=None, end=None, tag=""):
    global START
    START = start or START
    data_root = ROOT / "data"
    bench, label = ST.benchmarks(data_root), ST.names(ROOT)
    rows = []
    for n, path in enumerate(sorted(data_root.glob("s[hz]_*_daily.csv")), 1):
        symbol = path.name[:9]
        if symbol[:5] not in ST.BENCH:
            continue
        try:
            raw = SEM.load(path, day)
            a = SEM.analyse(raw, bench.get(symbol[:5]), symbol)
        except (OSError, ValueError, KeyError):
            continue
        if a is None:
            continue
        f, name = a["f"], label.get(symbol, "")
        amount = raw["amount"].astype(float).to_numpy() if "amount" in raw else np.full(len(raw), np.nan)
        for e in a["events"]:
            if e["kind"] == "PROGRESS" or e["date"] < START or (end and e["date"] > end) or e["index"] + 1 >= len(f):
                continue
            i = e["index"]
            if float(raw["isST"].iloc[i]) == 1:
                continue
            elig = SEM.eligibility_at(raw, f, i, bench.get(symbol[:5]), symbol, name)
            check = SEM.open_check(raw, f, e, symbol)
            q = SEM.quality(f, a["sig"], a["tr"], e)
            prototype = "A_BREAKOUT" if e["entry"] == "BREAKOUT" else "B_PULLBACK_REACCEL" if q.get("down_closes", 0) >= 1 else "C_SIDEWAYS_REACCEL"
            pre = float(raw["preclose"].iloc[i])
            row = {"symbol": symbol, "name": name, "date": e["date"], "kind": e["kind"], "entry_type": e["entry"], "prototype": prototype,
                   "setup_id": e["setup_id"], "domain": SEM.domain_of(symbol, name),
                   "research_eligible": elig["research_eligible"], "plan_eligible": elig["plan_eligible"],
                   "eligibility_reasons": "|".join(elig["reasons"]),
                   "wider_than_previous_stop": e.get("wider_than_previous_stop"),
                   "close_at_limit": float(raw["close"].iloc[i]) >= round(pre * (1 + SEM.limit_pct(symbol)), 2) - 0.001,
                   "rs5": SEM.rs_at(f, bench.get(symbol[:5]), i, 5), "rs20": SEM.rs_at(f, bench.get(symbol[:5]), i, 20),
                   "n0_pct": round(e["n0"] / float(f["close"].iloc[i]) * 100, 3),
                   "amount20_yi": round(float(np.nanmean(amount[max(0, i - 19):i + 1])) / 1e8, 3),
                   "open_state": check["state"],
                   "max_gain20_pct": round((float(f["high"].iloc[i + 1:i + 21].max()) / float(f["close"].iloc[i]) - 1) * 100, 2),
                   "matured20": i + 20 < len(f), **{f"q_{k}": v for k, v in q.items()}}
            if check["state"] == "IN_BAND":
                row.update(SEM.frozen_trade(f, e, symbol, CAPITAL, RISK, SYMBOL_CAP))
            else:
                row["trade"] = "NOT_FILLED_" + check["state"]
            rows.append(row)
        if n % 500 == 0:
            print(f"[event_domain] {n} files, {len(rows)} events", flush=True)
    domain = pd.DataFrame(rows)
    out = ROOT / "results/research/case_pool" / (day.replace("-", "") + tag)
    out.mkdir(parents=True, exist_ok=True)
    domain.to_csv(out / "event_domain.csv", index=False, encoding="utf-8-sig")
    summary = describe(domain)
    if tag:  # an extra validation window: no case shortlist is drawn from it
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return out
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    shortlist = pairs(domain)
    (out / "case_shortlist.json").write_text(json.dumps(shortlist, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "case_shortlist.md").write_text(render(shortlist, summary, day), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str)[:3000])
    return out


def describe(d):
    """Universes reconcile: all executable-domain events -> research eligible -> plan eligible (audit Q-01).
    Statistics use plan-eligible executable events only; unfinished trades are shown both apart and marked."""
    ex = d[d.domain == "EXECUTABLE"]
    research = ex[ex.research_eligible.astype(bool)]
    plan = research[research.plan_eligible.astype(bool)]
    closed = plan[plan.trade == "CLOSED"]
    marked = plan[plan.trade.isin(["CLOSED", "OPEN", "EXIT_PENDING"])]
    q = lambda s: [round(float(x), 3) for x in s.quantile([.1, .25, .5, .75, .9])] if len(s) else []
    stats = lambda g: {"n": len(g), "mean": round(float(g.net_r.mean()), 3), "median": round(float(g.net_r.median()), 3),
                       "win": round(float((g.net_r > 0).mean()), 3), "q": q(g.net_r)} if len(g) else {}
    fill = plan.groupby(["close_at_limit", "open_state"]).size().unstack(fill_value=0)
    top = closed.net_r.quantile(0.99) if len(closed) else None
    return {"universe": {"events_all": len(d), "executable_domain": len(ex), "research_eligible": len(research), "plan_eligible": len(plan),
                         "excluded_reasons": ex[~ex.plan_eligible.astype(bool)].eligibility_reasons.str.split("|").explode().value_counts().to_dict()},
            "trade_states_plan_eligible": plan.trade.value_counts().to_dict(),
            "exit_blocked_trades": int((plan.get("exit_blocked_opens", 0) > 0).sum()),
            "corporate_action_during_hold": int(plan.get("corporate_action_during_hold", False).fillna(False).astype(bool).sum()),
            "next_open_by_close_at_limit": {str(k): v for k, v in fill.to_dict("index").items()},
            "closed": stats(closed), "closed_plus_open_marked": stats(marked),
            "closed_without_top1pct": stats(closed[closed.net_r < top]) if top is not None else {},
            "closed_by_prototype": {p: stats(g) for p, g in closed.groupby("prototype")},
            "closed_by_kind": {k: stats(g) for k, g in closed.groupby("kind")},
            "note": "校准后研究台（资格统一、退出须可卖、MFE/MAE截至卖出、原价成交与复权分离）。逐事件独立纸面交易，"
                    "不含席位与共享资金，不是组合回测，不作优化目标。"}


FEATURES = ["rs20", "n0_pct", "amount20_yi"]


def pairs(d, per=3):
    """For each prototype: clearest winners, each with its nearest failed / stalled / unfillable control,
    matched only on facts known at the trigger (prototype, board, +/- 60 days, RS20, volatility, liquidity)."""
    ex = d[(d.domain == "EXECUTABLE") & d.matured20 & d.plan_eligible.astype(bool)].copy()
    ex["ts"] = pd.to_datetime(ex.date)
    for c in FEATURES:
        ex[c + "_z"] = (ex[c] - ex[c].mean()) / (ex[c].std() or 1)
    closed = ex[ex.trade == "CLOSED"]
    groups = {"failed": closed[(closed.net_r <= -0.7) & (closed.hold_bars <= 15)],
              "stalled": closed[(closed.hold_bars >= 25) & closed.net_r.between(-0.3, 0.5)],
              "unfillable": ex[ex.trade.str.startswith("NOT_FILLED") & (ex.max_gain20_pct >= 25)]}
    result, used_symbols = {}, set()
    # Positives are clear but not freak winners (3R-30R); a 100R eight-bagger is listed apart, not studied as typical.
    result["_extreme_winners_reference"] = [clean(r) for _, r in closed.sort_values("net_r", ascending=False).head(5).iterrows()]
    for proto, g in closed.groupby("prototype"):
        winners = g[g.net_r.between(3, 30)].sort_values("net_r", ascending=False)
        chosen = []
        for _, w in winners.iterrows():
            if w.symbol in used_symbols or any(abs((w.ts - pd.Timestamp(c["positive"]["date"])).days) < 60 for c in chosen):
                continue
            case = {"positive": clean(w)}
            for label, pool in groups.items():
                cand = pool[(pool.prototype == proto) & (pool.symbol.str[:5] == w.symbol[:5]) & (pool.symbol != w.symbol)
                            & ((pool.ts - w.ts).abs() <= pd.Timedelta(days=60))]
                if cand.empty:
                    case[label] = None
                    continue
                dist = sum((cand[c + "_z"] - w[c + "_z"]) ** 2 for c in FEATURES).pow(0.5)
                case[label] = clean(cand.loc[dist.idxmin()])
            chosen.append(case)
            used_symbols.add(w.symbol)
            if len(chosen) == per:
                break
        result[proto] = chosen
    return result


def clean(r):
    keep = ["symbol", "name", "date", "kind", "prototype", "setup_id", "close_at_limit", "rs5", "rs20", "n0_pct", "amount20_yi",
            "open_state", "trade", "entry", "exit_reason", "exit_date", "hold_bars", "net_r", "net_pnl", "mfe_r", "mae_r", "max_gain20_pct"]
    keep += [k for k in r.index if str(k).startswith("q_")]
    return {k: (None if isinstance(r.get(k), float) and math.isnan(r.get(k)) else r.get(k)) for k in keep if k in r.index}


def render(shortlist, summary, day):
    names = {"A_BREAKOUT": "A 平台后首次有效突破", "B_PULLBACK_REACCEL": "B 回调后再转强", "C_SIDEWAYS_REACCEL": "C 横向整理后再转强"}
    lines = [f"# 案例候选池 截至 {day}", "", summary["note"], "",
             "每个原型取净R在3到30之间、最高、时间相隔60天以上且股票不与其他原型重复的候选正例，并按触发当时已知特征（同原型、同板块、±60天、RS20、波动、成交额）"
             "配最近邻的三类对照：突破失败（15根内净R≤-0.7）、无进展（持有≥25根、净R在-0.3到0.5）、买不到（次日开盘不在区间且20日内最高涨≥25%）。"
             "这是给人工盲审挑选用的候选，不是已验证的优质案例。", ""]
    fmt = lambda c: "—" if not c else (f"{c['name']} {c['symbol']} {c['date']} {c.get('kind')} 净R {c.get('net_r')} 持有 {c.get('hold_bars')} "
                                       f"MFE {c.get('mfe_r')}R" if c.get("trade") == "CLOSED" else
                                       f"{c['name']} {c['symbol']} {c['date']} {c.get('open_state')} 20日最高 +{c.get('max_gain20_pct')}%")
    for proto, cases in shortlist.items():
        if proto.startswith("_"):
            continue
        lines += [f"## {names.get(proto, proto)}", "", "|正例|突破失败|无进展|买不到|", "|---|---|---|---|"]
        for c in cases:
            lines.append(f"|{fmt(c['positive'])}|{fmt(c['failed'])}|{fmt(c['stalled'])}|{fmt(c['unfillable'])}|")
        lines.append("")
    lines += ["## 极端赢家（仅参考，不作典型正例）", ""] + [f"- {fmt(c)}" for c in shortlist.get("_extreme_winners_reference", [])]
    return "\n".join(lines)


def from_csv(day):
    """Rebuild summary and shortlist from a saved domain without recomputing it."""
    out = ROOT / "results/research/case_pool" / day.replace("-", "")
    domain = pd.read_csv(out / "event_domain.csv", encoding="utf-8-sig")
    summary, shortlist = describe(domain), pairs(domain)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "case_shortlist.json").write_text(json.dumps(shortlist, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "case_shortlist.md").write_text(render(shortlist, summary, day), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str)[:3000])


if __name__ == "__main__":
    day = next((a for a in sys.argv[1:] if not a.startswith("--")), "2026-09-23")
    opt = {a.split("=")[0][2:]: a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--") and "=" in a}
    if "--from-csv" in sys.argv:
        from_csv(day)
    else:
        main(day, opt.get("start"), opt.get("end"), opt.get("tag", ""))
