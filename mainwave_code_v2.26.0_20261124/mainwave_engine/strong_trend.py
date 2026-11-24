"""STRONG_TREND core, shadow only (strong_trend_core_v1 §04: CT20 + RA3, EOD research profile).

One strategy, two entries, one trend exit:
  BREAKOUT        C_t > K_t = max(H over the 20 completed bars before t); invalidation K_t - 2*N0
  REACCELERATION  a breakout b within the last 30 completed bars (b <= t-4), consolidation t-3..t-1
                  (a down close, or close range <= 1*N0), consolidation low above b's invalidation,
                  C_t > consolidation high; invalidation = consolidation low - 0.01
  both            REACCELERATION is primary, BREAKOUT kept as a label
  entry band      [boundary, C_t + 0.5*N0], only the next session's open; unfilled = expired
  trend exit      20-day low channel, one close below the protection in force; a new protection is
                  computed after the bar completes and applies from the next session; never lowered
N0 = Wilder ATR20 of the previous completed session. Prices are ex-rights adjusted to the latest basis
from the file's own preclose, so a dividend or split is not read as a breakout or a breach.
Research values, not an account authorisation: this module places no order and changes no rank or gate.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "strong_trend_ct20_ra3_shadow_v1_20260923"
ROOT = Path(__file__).resolve().parents[1]
PARAMS = {"breakout_lookback": 20, "atr_lookback": 20, "atr_smoothing": "WILDER_SMA_SEED_LAST_400_BARS",
          "strength_windows": [5, 20], "reacceleration_anchor_lookback": 30, "consolidation_bars": 3,
          "sideways_close_range_max_atr": 1.0, "entry_upper_buffer_atr": 0.5,
          "breakout_invalidation_below_level_atr": 2.0, "trend_exit_lookback": 20, "trend_exit_confirm_closes": 1,
          "min_history": 61, "tick": 0.01, "data_basis": "FINAL_EOD", "execution_basis": "NEXT_SPECIFIED_OPEN",
          "fixed_profit_targets_enabled": False, "automatic_holding_age_exit_enabled": False,
          "add_experiment_enabled": False, "ai_decision_authority": "EXPLANATION_ONLY"}
BENCH = {"sh_60": "000001_SH", "sz_00": "399001_SZ", "sz_30": "399006_SZ", "sh_68": "000688_SH"}
NOTICE = ("强势趋势核心影子（研究参数 CT20/RA3，日线收盘确认、下一交易日开盘才有模拟资格）："
          "不下单、不改排名与闸门；数值为研究起点，未经收益验证，也不是账户授权。")


def adjusted(frame):
    """Scale history to the latest price basis using each day's preclose against the prior close."""
    f = frame.reset_index(drop=True).copy()
    ratio = (f["preclose"] / f["close"].shift()).fillna(1.0)
    ratio = ratio.where((ratio - 1).abs() > 0.001, 1.0)
    factor = ratio[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    for c in ("open", "high", "low", "close"):
        f[c] = f[c] * factor
    f["adj_factor"] = factor  # raw price on bar t = adjusted / adj_factor[t]; lets research recover traded prices
    return f


def wilder_atr(f, n=20):
    prev = f["close"].shift()
    tr = pd.concat([f["high"] - f["low"], (f["high"] - prev).abs(), (f["low"] - prev).abs()], axis=1).max(axis=1)
    atr = pd.Series(np.nan, index=f.index)
    if len(f) <= n:
        return atr
    seeded = tr.iloc[n:].copy()
    seeded.iloc[0] = tr.iloc[1:n + 1].mean()
    atr.iloc[n:] = seeded.ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    return atr


def breakout_at(f, i, atr, n=20):
    """Close above the prior n completed highs; K excludes bar i itself."""
    if i < n or not math.isfinite(atr[i - 1]) or atr[i - 1] <= 0:
        return None
    level = float(f["high"].iloc[i - n:i].max())
    if not f["close"].iloc[i] > level:
        return None
    n0 = float(atr[i - 1])
    return {"index": i, "level": level, "n0": n0, "invalidation": level - PARAMS["breakout_invalidation_below_level_atr"] * n0}


def reacceleration_at(f, i, atr):
    """RA3: most recent valid anchor breakout, fixed three-bar consolidation before t, then a close above it."""
    k = PARAMS["consolidation_bars"]
    if i < PARAMS["breakout_lookback"] + k + 1 or not math.isfinite(atr[i - 1]) or atr[i - 1] <= 0:
        return None
    window = f.iloc[i - k:i]
    top, bottom = float(window["high"].max()), float(window["low"].min())
    closes = f["close"].iloc[i - k - 1:i].to_numpy()
    paused = bool((np.diff(closes) < 0).any()) or float(window["close"].max() - window["close"].min()) <= PARAMS["sideways_close_range_max_atr"] * atr[i - 1]
    if not paused or not f["close"].iloc[i] > top:
        return None
    for b in range(i - k - 1, max(i - PARAMS["reacceleration_anchor_lookback"], PARAMS["breakout_lookback"]) - 1, -1):
        anchor = breakout_at(f, b, atr)
        if anchor and bottom > anchor["invalidation"]:
            return {"index": i, "anchor_index": b, "anchor_level": anchor["level"], "anchor_invalidation": anchor["invalidation"],
                    "level": top, "n0": float(atr[i - 1]), "invalidation": round(bottom - PARAMS["tick"], 4),
                    "consolidation": [str(d) for d in window["date"]]}
    return None


def strength(f, bench, n):
    """RS_n: stock log return minus the matched index log return over the same n sessions (dates aligned)."""
    if bench is None or len(f) <= n:
        return None
    end, start = f["date"].iloc[-1], f["date"].iloc[-1 - n]
    if end not in bench.index or start not in bench.index:
        return None
    return float(math.log(f["close"].iloc[-1] / f["close"].iloc[-1 - n]) - math.log(bench[end] / bench[start]))


def protection_step(protection, close, lows_window):
    """One completed bar. Test against the protection already in force, then ratchet for the next session."""
    if close < protection:
        return True, protection
    if len(lows_window) < PARAMS["trend_exit_lookback"]:
        return False, protection  # incomplete window: keep the protection in force, candidate not updated
    return False, max(protection, float(min(lows_window)))


def replay_exit(f, entry_index, hard):
    """EOD replay from an entry at bar `entry_index`'s open. A breach known only at the close exits at the next
    open; no bar is ever filled at the protection price itself."""
    protection = hard
    n = PARAMS["trend_exit_lookback"]
    for j in range(entry_index, len(f)):
        if f["low"].iloc[j] <= hard:
            reason = "HARD_RISK"
        else:
            broken, protection = protection_step(protection, float(f["close"].iloc[j]),
                                                 f["low"].iloc[max(0, j - n + 1):j + 1].tolist() if j >= n - 1 else [])
            reason = "TREND_EXIT" if broken else None
        if reason:
            nxt = j + 1 if j + 1 < len(f) else None
            return {"signal_index": j, "reason": reason, "protection": protection,
                    "exit_index": nxt, "exit_price": None if nxt is None else float(f["open"].iloc[nxt])}
    return {"signal_index": None, "reason": "OPEN", "protection": protection, "exit_index": None, "exit_price": None}


def evaluate(raw, bench, day, symbol, name=""):
    """One security on one completed session. Returns None if nothing to report."""
    f = raw[raw["date"] <= day].sort_values("date").drop_duplicates("date", keep="last").tail(400).reset_index(drop=True)
    if len(f) < PARAMS["min_history"] or f["date"].iloc[-1] != day:
        return None
    last = f.iloc[-1]
    if not all(float(last[c]) > 0 for c in ("open", "high", "low", "close", "preclose")) or float(last.get("volume", 0)) <= 0:
        return None
    if float(last.get("tradestatus", 1)) != 1 or float(last.get("isST", 0)) == 1:
        return None
    raw_close, raw_pre = float(last["close"]), float(last["preclose"])
    f = adjusted(f)
    atr = wilder_atr(f, PARAMS["atr_lookback"]).to_numpy()
    i = len(f) - 1
    a, b = breakout_at(f, i, atr), reacceleration_at(f, i, atr)
    if not (a or b):
        return None
    primary = "REACCELERATION" if b else "BREAKOUT"
    sig = b or a
    close = float(f["close"].iloc[i])
    upper = close + PARAMS["entry_upper_buffer_atr"] * sig["n0"]
    lower = sig["level"]
    rs5, rs20 = strength(f, bench, 5), strength(f, bench, 20)
    eligible = any(x is not None and x > 0 for x in (rs5, rs20))
    board_limit = 0.2 if symbol[3:5] in ("30", "68") else 0.1
    problems = []
    if not 0 < sig["invalidation"] < lower:
        problems.append("失效位不低于买入区间下界，计划非法")
    if rs5 is None or rs20 is None:
        problems.append("相对强度缺基准数据")
    return {"symbol": symbol, "name": name, "day": day, "entry": primary, "labels": [x for x, y in (("BREAKOUT", a), ("REACCELERATION", b)) if y],
            "close": round(raw_close, 3), "pct_chg": round((raw_close / raw_pre - 1) * 100, 2),
            "boundary": round(lower, 3), "band": [round(lower, 3), round(upper, 3)], "invalidation": round(sig["invalidation"], 3),
            "n0": round(sig["n0"], 4), "risk_per_share": round(upper - sig["invalidation"], 3),
            "risk_pct_at_upper": round((upper - sig["invalidation"]) / upper * 100, 2),
            "rs5": None if rs5 is None else round(rs5, 4), "rs20": None if rs20 is None else round(rs20, 4),
            "strength_eligible": eligible, "plan_legal": not problems, "problems": problems,
            "close_at_limit": raw_close >= round(raw_pre * (1 + board_limit), 2) - 0.001,
            "reacceleration": b and {k: b[k] for k in ("anchor_level", "anchor_invalidation", "consolidation")},
            "paper_board": bool(symbol[:5] in ("sh_60", "sz_00")) and not any(x in str(name).upper() for x in ("ST", "退")),
            "valid_for": "仅下一交易日开盘；开盘不在区间内即过期"}


def benchmarks(data_root):
    out = {}
    for prefix, code in BENCH.items():
        path = Path(data_root) / f"{code}_index.csv"
        if path.exists():
            b = pd.read_csv(path)
            out[prefix] = pd.Series(b["close"].to_numpy(), index=pd.to_datetime(b["trade_date"].astype(str)).dt.strftime("%Y-%m-%d"))
    return out


def names(root):
    path = Path(root) / "data/stock_list.csv"
    if not path.exists():
        return {}
    f = pd.read_csv(path, dtype=str)
    code = next((c for c in ("code", "ts_code", "symbol") if c in f), None)
    label = next((c for c in ("code_name", "name") if c in f), None)
    if not code or not label:
        return {}
    def norm(x):
        x = str(x).lower().replace(".", "_")
        if x.endswith(("_sh", "_sz")):
            x = x[-2:] + "_" + x[:6]
        return x
    return {norm(c): n for c, n in zip(f[code], f[label])}


def held_symbols(root):
    """Symbols with an actual position or open buy in any book: they belong to the add contract, not a new first entry."""
    import sqlite3
    from .trading import books as B, store
    held = {}
    for book, db in (("MANUAL", store.DB_PATH), ("AUTO", Path(root) / B.AUTO_DB), ("NEW100K", Path(root) / B.NEW100K_DB)):
        if not Path(db).exists():
            continue
        try:
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            for (s,) in conn.execute("SELECT DISTINCT symbol FROM paper_lots WHERE remaining>0"):
                held.setdefault(s, []).append(book)
            conn.close()
        except sqlite3.Error:
            continue
    return held


def frozen_status(root, day):
    """What the frozen FIRST_PULLBACK path says about the same symbols, for the design's regression check."""
    path = Path(root) / "results/mainwave_engine" / day.replace("-", "") / "trading_closure.json"
    if not path.exists():
        return None
    snap = json.loads(path.read_text(encoding="utf-8"))
    return {r["symbol"]: ("PAPER_ELIGIBLE" if r.get("paper_plan_eligible") else r.get("state") or "CANDIDATE") for r in snap.get("records", [])}


def scan(day, root=ROOT, data_root=None):
    data_root = Path(data_root or Path(root) / "data")
    bench, label, held = benchmarks(data_root), names(root), held_symbols(root)
    rows, scanned, skipped = [], 0, 0
    for path in sorted(data_root.glob("s[hz]_*_daily.csv")):
        symbol = path.name[:9]
        if symbol[:5] not in BENCH:
            continue
        try:
            raw = pd.read_csv(path, usecols=lambda c: c in {"date", "open", "high", "low", "close", "preclose", "volume", "tradestatus", "isST"})
            raw["date"] = raw["date"].astype(str).str[:10]
            scanned += 1
            row = evaluate(raw, bench.get(symbol[:5]), day, symbol, label.get(symbol, ""))
        except (OSError, ValueError, KeyError) as exc:
            skipped += 1
            continue
        if row:
            row["held_in"] = held.get(symbol, [])
            rows.append(row)
    frozen = frozen_status(root, day)
    for r in rows:
        r["frozen_first_pullback"] = None if frozen is None else frozen.get(r["symbol"], "NOT_IN_CANDIDATES")
    key = lambda r: (-(r["rs20"] if r["rs20"] is not None else -9), -(r["rs5"] if r["rs5"] is not None else -9), r["symbol"])
    attention = sorted([r for r in rows if r["strength_eligible"]], key=key)
    review = [r for r in attention if r["plan_legal"]]
    allocation = [r for r in review if r["paper_board"] and not r["held_in"]]
    summary = {"scanned": scanned, "unreadable": skipped, "signals": len(rows),
               "breakout": sum(r["entry"] == "BREAKOUT" for r in rows), "reacceleration": sum(r["entry"] == "REACCELERATION" for r in rows),
               "attention": len(attention), "review": len(review), "allocation": len(allocation),
               "allocation_close_at_limit": sum(r["close_at_limit"] for r in allocation),
               "frozen_pullback_would_block": None if frozen is None else sum(r["frozen_first_pullback"] != "PAPER_ELIGIBLE" for r in allocation)}
    return {"version": VERSION, "as_of": day, "notice": NOTICE, "params": PARAMS, "orders_enabled": False,
            "account_authorisation": "CONFIG_PENDING：新核心尚未获准进入任何模拟账户；NEW100K 仍按原冻结规则运行",
            "summary": summary,
            "boards": {"attention": attention, "review": review, "allocation": allocation},
            "ranking": "计划分配榜：先合法完整，再 RS20 降序、RS5 降序、证券代码；透明基准，未验证最优"}


def render(result, top=20):
    s = result["summary"]
    lines = [f"# 强势趋势核心影子 {result['as_of']}", "", result["notice"], "", result["account_authorisation"], "",
             f"扫描 {s['scanned']} 只；信号 {s['signals']}（突破 {s['breakout']}，再转强 {s['reacceleration']}）；"
             f"注意力榜 {s['attention']}，机会复核榜 {s['review']}，计划分配榜 {s['allocation']}"
             f"（其中收盘涨停 {s['allocation_close_at_limit']}，次日开盘可能无法成交）。", ""]
    if s.get("frozen_pullback_would_block") is not None:
        lines += [f"对照：计划分配榜中 {s['frozen_pullback_would_block']} 只在旧首次回踩分支下不具备模拟资格。", ""]
    lines += ["|#|股票|入口|收盘|涨幅|买入区间|失效位|每股风险%|RS20|RS5|旧分支|", "|---|---|---|---:|---:|---|---:|---:|---:|---:|---|"]
    for n, r in enumerate(result["boards"]["allocation"][:top], 1):
        lines.append(f"|{n}|{r['name']} {r['symbol']}|{'再转强' if r['entry'] == 'REACCELERATION' else '突破'}{'（涨停）' if r['close_at_limit'] else ''}|"
                     f"{r['close']}|{r['pct_chg']}%|{r['band'][0]}–{r['band'][1]}|{r['invalidation']}|{r['risk_pct_at_upper']}|"
                     f"{r['rs20']}|{r['rs5']}|{r['frozen_first_pullback'] or '—'}|")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    day = pd.Timestamp(args.date).date().isoformat()
    result = scan(day)
    out = ROOT / "results/mainwave_engine" / day.replace("-", "")
    out.mkdir(parents=True, exist_ok=True)
    (out / "strong_trend_shadow.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "strong_trend_shadow.md").write_text(render(result), encoding="utf-8")
    print(f"[strong_trend] {day} {result['summary']} -> {out / 'strong_trend_shadow.json'}", flush=True)
    try:  # CORE-SEMANTICS sits beside CORE-V1; its failure never touches the baseline output
        from . import strong_trend_semantics as SEM
        sem = SEM.run(day, v1=result)
        (out / "strong_trend_semantics.json").write_text(json.dumps(sem, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        (out / "strong_trend_semantics.md").write_text(SEM.render(sem), encoding="utf-8")
        print(f"[strong_trend] semantics {sem['counts']['events_by_kind']} conditional={sem['counts']['conditional_plans']}", flush=True)
    except Exception as exc:
        print(f"[strong_trend] semantics FAILED: {str(exc)[:300]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
