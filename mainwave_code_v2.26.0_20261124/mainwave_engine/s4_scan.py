"""Daily S4 opportunity scan (pullback to EMA20 then relaunch) -> ranked list for manual selection into the virtual
portfolio with the S4_LAD exit. Report only: nothing is bought; `--pick` records a manual choice at that day's close.

Signal (research/four_dim_v2.py, S4研究总结_20260925.md), main board (60/00), non-ST, on continuous-adjusted prices:
  trend    MA20 > MA60, MA60 above its value 20 sessions ago, close > MA60
  pullback within the last 10 sessions a low <= EMA20 x 1.02 with that close >= EMA20 x 0.99, no close < EMA20 x 0.97,
           and a prior run (max high of the last 20 sessions >= EMA20 x 1.08)
  relaunch today up >= 2%, close above the last 5 closes and above MA5, close/EMA20 - 1 <= 12%; first day in 5
Exclusions (shown, not recommended): R1 MACD top divergence, R2 green bars growing, R3 close > 15% under the 60-session high.
Ranking: LEAD = the stock's 20-session return rank inside its industry (all boards). Weak evidence: it separated
outcomes in 2026 only (S4题材资金排序_20260925.md); there is no ranking that held in both years.
Event binding (external review 2026-09-26): a signal belongs to its pullback event, identified by the last qualifying
EMA20 touch before it. Only the first relaunch of an event is a new first entry; a later relaunch with no new touch since
the previous S4 day (within 10 sessions) is listed as progress of that event. A symbol whose S4 position of the same event already exists (open or closed) is not
offered again: after a stop, only a new pullback event can bring it back.
Exit contract carried to the virtual portfolio: pullback low (lowest low of the 10 sessions before the signal) and the
signal day's open, used by the graded denial exit (S4离场研究 / accept_failure.py B2h).
Market: the CSI 1000 trend-exit alert (position_monitor.market_risk) is printed on top; when it is red, no new buy is advised.
Index trend gate (index_trend, 2026-11-20): when the CSI 300 EMA100 slope over 10 sessions is positive, S4 candidates move to
"excluded" with the reason attached instead of being recommended. Reason: in that regime the S4 entry timing is measurably worse
than entering the same stock on a random day (AStockV2, docs/市场环境分层与入场时机_20261120.md). It is a separate, slower
filter from the CSI 1000 MA20 alert above.
python -m mainwave_engine.s4_scan --date YYYY-MM-DD [--pick SYMBOL]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import stock_analyzer as SA
from . import strong_trend as ST
from . import virtual_portfolio as VP
from . import theme_risk as TR
from . import theme_semantics as TS
from . import index_trend as IT
from .trading import position_monitor as PM

VERSION = "s4_scan_v1_3_20261120"
ROOT = Path(__file__).resolve().parents[1]
REF_RULE = ("参考排序（前向验证中，S4选股特征研究_20261011.md）：同日推荐里，单季营收同比增速越高、信号日收盘位置越低越好，"
            "两项的排名百分位等权平均（缺数据记中位）；不改正式排序，每日登记到 s4_reference_rank_log.csv。")
FIN_PATH = Path(__file__).resolve().parents[1] / "results/r4_fundamentals_panel.csv"
PRINCIPLE = ("执行原则：股票最多 2 只，ETF 另占 1 个名额；按账户风险预算而不是胜率分配仓位。"
             "S4 保留原分批止盈合约；G1 红灯或风险未知不开新仓；卖出建议在建议的开盘执行。"
             "核心指数 EMA100 中期趋势向上时也不开 S4 新仓（该环境下形态启动日入场的时点价值为负）。")
NOTICE = ("S4 机会扫描：只列出候选，不自动买入。人工选定后用 --pick 记入虚拟持仓（按当日收盘价记账），离场按分批止盈 S4_LAD 每日给出建议。"
          "排序依据（行业内 20 日强度）只在 2026 年有弱区分，两年都稳定的排序依据尚未找到。"
          "2026-11-20 起新增核心指数中期趋势环境门（沪深300 EMA100 斜率），被拦下的候选全部列在“已排除”里并注明原因，不静默丢弃。")


def signals(f):
    """S4 flags on an adjusted frame (open/high/low/close/volume). Returns a DataFrame aligned to f."""
    c, h, l = (f[k].to_numpy(float) for k in ("close", "high", "low"))
    s = pd.Series(c)
    ma5, ma20, ma60 = (s.rolling(n).mean().to_numpy() for n in (5, 20, 60))
    ema20 = s.ewm(span=20, adjust=False).mean().to_numpy()
    dif = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = (2 * (dif - dea)).to_numpy()
    prev = np.r_[np.nan, c[:-1]]
    trend = (ma20 > ma60) & (ma60 > np.r_[np.full(20, np.nan), ma60[:-20]]) & (c > ma60)
    bias = c / ema20 - 1
    touched = (l <= ema20 * 1.02) & (c >= ema20 * 0.99)
    touch = pd.Series(touched).shift(1).rolling(10).max().to_numpy() == 1
    idx = pd.Series(np.where(touched, np.arange(len(c)), np.nan))
    last_touch = idx.shift(1).ffill().to_numpy()  # the most recent touch before each day
    last_touch = np.where(np.arange(len(c)) - last_touch <= 10, last_touch, np.nan)
    pb_low = pd.Series(l).shift(1).rolling(10).min().to_numpy()
    no_break = pd.Series(c < ema20 * 0.97).shift(1).rolling(10).max().to_numpy() == 0
    run = pd.Series(h).shift(1).rolling(20).max().to_numpy() >= ema20 * 1.08
    close5 = s.shift(1).rolling(5).max().to_numpy()
    s4 = trend & touch & no_break & run & (c / prev - 1 >= 0.02) & (c > close5) & (c > ma5) & (bias <= 0.12)
    s4 = s4 & ~(pd.Series(s4).shift(1).rolling(5).max().to_numpy() == 1)
    # a relaunch with no new EMA20 touch after the previous S4 day (within 10 sessions) is progress of that event,
    # not a new first entry; a fresh touch after it starts a new event
    progress, last_s4 = np.zeros(len(c), bool), None
    for t in np.nonzero(s4)[0]:
        progress[t] = last_s4 is not None and t - last_s4 <= 10 and not touched[last_s4 + 1:t].any()
        last_s4 = t
    n = len(c)
    high20 = s.rolling(20).max().to_numpy()
    diverge = np.zeros(n, bool)
    for t in np.nonzero(c >= high20)[0]:
        if t >= 45:
            k = t - 40 + int(np.argmax(c[t - 40:t - 4]))
            diverge[t] = c[t] > c[k] and hist[k] > 0 and hist[t] < hist[k]
    green_down = (hist < 0) & ~(hist > np.r_[np.nan, np.nan, hist[:-2]])
    dd60 = c / pd.Series(h).rolling(60).max().to_numpy() - 1
    vr = f["volume"].astype(float) / f["volume"].astype(float).shift(1).rolling(20).mean()
    return pd.DataFrame({"s4": s4, "progress": progress, "last_touch": last_touch, "pb_low": pb_low, "R1": diverge, "R2": green_down, "R3": dd60 < -0.15, "dd60": dd60, "bias": bias,
                         "ema20": ema20, "pct": c / prev - 1, "vr": vr.to_numpy(),
                         "macd": np.where(hist >= 0, "红柱", "绿柱")})


def scan(day, data_root=None):
    data_root = Path(data_root or ROOT / "data")
    names = ST.names(ROOT)
    mp = json.loads((ROOT / "concept_mapping.json").read_text(encoding="utf-8")).get("mapping", {})
    industry = {s: v.get("industry") for s, v in mp.items()}
    r20, rows = {}, []
    for path in sorted(data_root.glob("s[hz]_*_daily.csv")):
        sym = path.name[:9]
        if sym[:5] not in ("sh_60", "sz_00", "sz_30", "sh_68"):
            continue
        try:
            raw = pd.read_csv(path, usecols=lambda c: c in {"date", "open", "high", "low", "close", "preclose", "volume",
                                                           "amount", "tradestatus", "isST"})
        except (OSError, ValueError):
            continue
        raw["date"] = raw["date"].astype(str).str[:10]
        raw = raw[(raw["date"] <= day)].drop_duplicates("date").tail(260).reset_index(drop=True)
        if len(raw) < 130 or raw["date"].iloc[-1] != day or float(raw["tradestatus"].iloc[-1]) != 1:
            continue
        f = ST.adjusted(raw)
        r20[sym] = float(f["close"].iloc[-1] / f["close"].iloc[-21] - 1)
        name = str(names.get(sym, ""))
        if sym[:5] not in ("sh_60", "sz_00") or float(raw["isST"].iloc[-1] or 0) == 1 or "ST" in name.upper() or "退" in name:
            continue
        x = signals(f)
        if not x["s4"].iloc[-1]:
            continue
        r = x.iloc[-1]
        why = [w for w, hit in (("R1 顶背离", r.R1), ("R2 绿柱放大", r.R2), ("R3 距60日高>15%", r.R3)) if hit]
        scale = float(f["adj_factor"].iloc[-1])
        touch_day = f["date"].iloc[int(r.last_touch)] if r.last_touch == r.last_touch else ""
        event = {"event_id": f"{sym}:{touch_day}", "touch_day": touch_day, "first_recovery_day": day,
                 "pullback_low": round(float(r.pb_low) / scale, 3), "signal_open": float(raw["open"].iloc[-1]),
                 "progress": bool(r.progress)}
        rows.append({"symbol": sym, "name": name, "industry": industry.get(sym) or "", "close": float(raw["close"].iloc[-1]),
                     "pct": round(float(r.pct) * 100, 2), "bias_ema20": round(float(r.bias) * 100, 1),
                     "dd60": round(float(r.dd60) * 100, 1), "macd": r.macd, "vol_ratio": round(float(r.vr), 2),
                     "amount_yi": round(float(raw["amount"].iloc[-1]) / 1e8, 2), "excluded": why,
                     "clv": clv(raw.iloc[-1]),
                     "ema20_raw": round(float(r.ema20) / scale, 3), "event": event})
    peers = pd.Series(r20)
    groups = pd.Series({s: industry.get(s) for s in peers.index})
    for row in rows:
        same = peers[groups == row["industry"]] if row["industry"] else pd.Series(dtype=float)
        row["lead"] = round(float(same.rank(pct=True)[row["symbol"]]), 2) if len(same) >= 5 else None
        c = row["close"]
        ev = row["event"]
        row["plan"] = {"entry_ref": c, "stop": round(c * 0.90, 2), "tp1": round(c * 1.10, 2), "tp2": round(c * 1.20, 2),
                       "trail": f"两日收盘 < {row['ema20_raw'] * 0.97:.2f}（EMA20×0.97，随日更新）",
                       "denial": f"3 日内收盘 < {ev['signal_open']:.2f}（信号日开盘）卖一半；收盘 < {ev['pullback_low']:.2f}（回踩低点）"
                                 f"或 5 日内两次收于 EMA20 下，卖出剩余"}
    taken = {p.get("s4_event_id") for p in VP.positions() if p.get("s4_event_id")}
    trend = IT.regime(day, data_root)
    gate_reason = None
    if trend.get("gate") != "ALLOW":
        gate_reason = f"环境门：{trend.get('text')}"
    for r in rows:
        if gate_reason:
            r["excluded"].append(gate_reason)
        if r["event"]["progress"]:
            r["excluded"].append(f"同一回踩事件（触及 {r['event']['touch_day']}）已出过首仓信号，本次是旧事件的进展")
        elif r["event"]["event_id"] in taken:
            r["excluded"].append("该回踩事件已记入过虚拟持仓（持有中或已结束）；止损后需新的回踩事件才能再入选")
    kept = sorted([r for r in rows if not r["excluded"]], key=lambda r: -(r["lead"] if r["lead"] is not None else -1))
    for i, r in enumerate(kept, 1):
        r["rank"] = i
    sales, fin_asof = fundamentals(day, [r["symbol"] for r in kept])
    reference(kept, sales)
    market = PM.market_risk(day, data_root)
    return {"as_of": day, "version": VERSION, "notice": NOTICE, "market_risk": market,
            "market_warning": PM.market_warning(day, data_root),
            "index_trend": trend,
            "buy_advised": market.get("level") == "NORMAL", "universe": len(r20), "reference_rule": REF_RULE,
            "fundamentals_latest_ann": fin_asof,
            "recommended": kept, "excluded": [r for r in rows if r["excluded"]]}


def clv(bar):
    """Close location of a daily bar: 0 at the low, 1 at the high."""
    rng = float(bar["high"]) - float(bar["low"])
    return round((float(bar["close"]) - float(bar["low"])) / rng, 3) if rng > 0 else None


def fundamentals(day, symbols, path=None):
    """Latest quarterly sales growth announced before day (the report of the latest announcement), per symbol."""
    path = Path(path or FIN_PATH)
    if not symbols or not path.exists():
        return {}, ""
    fin = pd.read_csv(path, usecols=["ts_code", "ann_date", "end_date", "q_sales_yoy"]).dropna(subset=["ann_date"])
    fin["ann"] = fin["ann_date"].astype(float).astype(int).astype(str)
    fin = fin[fin["ann"] < day.replace("-", "")]
    latest = fin["ann"].max() if len(fin) else ""
    fin["symbol"] = fin["ts_code"].str[-2:].str.lower() + "_" + fin["ts_code"].str[:6]
    last = fin[fin["symbol"].isin(symbols)].sort_values(["ann", "end_date"]).groupby("symbol").tail(1)
    return {s: (None if v != v else round(float(v), 2)) for s, v in zip(last["symbol"], last["q_sales_yoy"])}, latest


def reference(kept, sales):
    """Reference score among the day's recommendations: percentile ranks of high sales growth and low close location."""
    if not kept:
        return
    t = pd.DataFrame({"clv": [r.get("clv") for r in kept], "sales": [sales.get(r["symbol"]) for r in kept]}, dtype=float)
    score = ((-t["clv"]).rank(pct=True).fillna(0.5) + t["sales"].rank(pct=True).fillna(0.5)) / 2
    for r, s in zip(kept, score):
        r["q_sales_yoy"], r["ref_score"] = sales.get(r["symbol"]), round(float(s), 3)
    for i, r in enumerate(sorted(kept, key=lambda r: (-r["ref_score"], r["rank"])), 1):
        r["ref_rank"] = i


def log_reference(out, path):
    """Forward record of both orders, one block per day (a rerun replaces that day)."""
    rows = [{"date": out["as_of"], "symbol": r["symbol"], "name": r["name"], "rank": r["rank"], "ref_rank": r.get("ref_rank"),
             "ref_score": r.get("ref_score"), "clv": r.get("clv"), "q_sales_yoy": r.get("q_sales_yoy"),
             "fundamentals_latest_ann": out.get("fundamentals_latest_ann", ""), "buy_advised": out.get("buy_advised"),
             "index_trend_gate": (out.get("index_trend") or {}).get("gate"),
             "index_trend_slope": (out.get("index_trend") or {}).get("slope"),
             "version": out["version"]} for r in out["recommended"]]
    old = pd.read_csv(path, dtype={"date": str}) if Path(path).exists() else pd.DataFrame()
    if len(old):
        old = old[old["date"] != out["as_of"]]
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")


def render(out):
    m = out["market_risk"]
    lines = [f"# S4 机会 {out['as_of']}", ""]
    if m.get("level") == "TREND_EXIT":
        lines += ['<div style="background:#c00000;color:#fff;padding:12px 16px;font-weight:bold">🔴 大盘趋势清仓中：'
                  f"{m['text']}<br>不建议新开 S4 仓位。</div>", ""]
    elif m.get("level") == "CRITICAL":
        lines += [f'<div style="background:#b36b00;color:#fff;padding:12px 16px;font-weight:bold">🟠 {m["text"]}<br>不建议新开 S4 仓位。</div>', ""]
    elif m.get("level") != "NORMAL":
        lines += [f"🟠 大盘状态未确认：{m.get('text')}", ""]
    else:
        lines += [f"🟢 {m['text']}", ""]
    w = out.get("market_warning") or {}
    if w.get("level") == "WARNING":
        lines += ["🟠 短期回调预警（约一半误报，不追高）：" + "；".join(f"{r['index']}：{r['text']}" for r in w["indices"] if r["level"] == "WARNING"), ""]
    t = out.get("index_trend") or {}
    if t.get("gate") == "BLOCK":
        lines += ['<div style="background:#8a5a00;color:#fff;padding:12px 16px;font-weight:bold">🟠 中期趋势门：'
                  f"{t.get('text')}</div>", ""]
    elif t.get("gate") == "ALLOW":
        lines += [f"🟡 中期趋势门：{t.get('text')}", ""]
    else:
        lines += [f"🟠 中期趋势门：{t.get('text')}", ""]
    lines += [out["notice"], "", PRINCIPLE, "", f"扫描 {out['universe']} 只；形态候选 {len(out['recommended'])} 只；排除 {len(out['excluded'])} 只。", ""]
    if out["recommended"]:
        ref = sorted(out["recommended"], key=lambda r: r.get("ref_rank") or 999)
        lines += ["## 形态候选（非入场许可；行业内20日强度为弱排序依据，右侧另列参考排名）", "",
                  f"{out.get('reference_rule', '')}财务数据最新公告日：{out.get('fundamentals_latest_ann') or '缺失'}。", "",
                  "参考排序前列：" + "、".join(f"{r['name']}{r['symbol'][3:]}" for r in ref[:5]), "",
                  "|排名|参考排名|股票|行业|收盘|涨幅|偏离EMA20|距60日高|MACD|量比|行业内强度|单季营收增速|收盘位置|止损 −10%|止盈 +10% / +20%|其余离场|",
                  "|---:|---:|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|"]
        for r in out["recommended"]:
            p = r["plan"]
            lead = "—" if r["lead"] is None else f"{r['lead']:.0%}"
            sales = "—" if r.get("q_sales_yoy") is None else f"{r['q_sales_yoy']:+.1f}%"
            loc = "—" if r.get("clv") is None else f"{r['clv']:.2f}"
            lines.append(f"|{r['rank']}|{r.get('ref_rank', '—')}|{r['name']} {r['symbol'][3:]}|{r['industry']}|{r['close']:.2f}|{r['pct']:+.1f}%|"
                         f"{r['bias_ema20']:+.1f}%|{r['dd60']:.1f}%|{r['macd']}|{r['vol_ratio']}|{lead}|{sales}|{loc}|{p['stop']}|"
                         f"{p['tp1']} / {p['tp2']}|{p['denial']}；{p['trail']}；盈利≥10%后放量大阴/长上影|")
        lines += ["", "人工选定后记入虚拟持仓（按当日收盘价，离场按 S4_LAD 每日给建议）：", "",
                  f"`python -m mainwave_engine.s4_scan --date {out['as_of']} --pick <代码>`", ""]
    if out["excluded"]:
        lines += ["## 已排除（研究显示止损率更高）", ""] + [
            f"- {r['name']} {r['symbol'][3:]}：{'、'.join(r['excluded'])}（距60日高 {r['dd60']:.1f}%，MACD {r['macd']}）"
            for r in out["excluded"]] + [""]
    return "\n".join(lines)


def pick(day, symbol, out_root=None, db=None):
    path = Path(out_root or SA.OUT_ROOT) / day.replace("-", "") / "s4_opportunities.json"
    if not path.exists():
        raise ValueError(f"{day} 还没有 S4 扫描结果，先运行扫描")
    out = json.loads(path.read_text(encoding="utf-8"))
    symbol = SA.normalize_symbol(symbol)
    if not any(r["symbol"] == symbol for r in out["recommended"]):
        reason = next(("、".join(r["excluded"]) for r in out["excluded"] if r["symbol"] == symbol), "不在当日 S4 推荐中")
        raise ValueError(f"{symbol} 不能按 S4 记入：{reason}")
    row = next(r for r in out["recommended"] if r["symbol"] == symbol)
    market = out.get("market_risk") or {}
    trend = out.get("index_trend") or {}
    if (out.get("as_of") != day or out.get("buy_advised") is not True
            or market.get("as_of") != day or market.get("level") != "NORMAL"):
        raise ValueError("S4 入场复核未通过：大盘红灯、未知或数据过期，不接受新加入")
    if trend.get("gate") != "ALLOW":
        raise ValueError("S4 入场复核未通过：中期趋势门未放行（" + str(trend.get("text"))[:120] + "）")
    mapping_path = ROOT / "concept_mapping.json"
    info = json.loads(mapping_path.read_text(encoding="utf-8")).get("mapping", {}).get(symbol, {}) if mapping_path.exists() else {}
    corrected, _ = TS.apply_membership_evidence({symbol: info}, day)
    info = corrected[symbol]
    risk = TR.evaluate_many([row.get("industry", "")] + info.get("concepts", []), day)
    if risk["theme_risk_action"] != "ALLOW":
        raise ValueError("S4 入场复核未通过：" + risk["theme_risk_reason"])
    ev = row.get("event") or {}
    return VP.add(symbol, day, db or VP.DB_PATH, strategy="S4_LAD",
                  meta={"s4_event_id": ev.get("event_id"), "s4_touch_day": ev.get("touch_day"),
                        "s4_pullback_low": ev.get("pullback_low"), "s4_signal_open": ev.get("signal_open")} if ev else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--pick")
    args = parser.parse_args()
    if args.pick:
        result = pick(args.date, args.pick)
        p = result["position"]
        print(f"[s4_scan] {'已记入' if result['created'] else '已存在'}虚拟持仓 {p['name']} {p['symbol']} "
              f"{p['entry_date']} 收盘 {p['entry_price']:.2f}，离场 {p.get('strategy', '默认')}", flush=True)
        return 0
    out = scan(args.date)
    folder = SA.OUT_ROOT / args.date.replace("-", "")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "s4_opportunities.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (folder / "s4_opportunities.md").write_text(render(out), encoding="utf-8")
    log_reference(out, SA.OUT_ROOT / "s4_reference_rank_log.csv")
    top = "、".join(f"{r['name']}{r['symbol'][3:]}" for r in out["recommended"][:5]) or "无"
    print(f"[s4_scan] {args.date} 推荐 {len(out['recommended'])} 排除 {len(out['excluded'])} 大盘 {out['market_risk'].get('level')}"
          f"；前列：{top} -> {folder / 's4_opportunities.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
