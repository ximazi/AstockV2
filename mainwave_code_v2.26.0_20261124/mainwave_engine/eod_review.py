"""After-close review, fact layer (P0): limit boards, yesterday's frozen pool, ladder, theme structure,
activity and loss effects. Every fact carries its domain, denominators and gaps. Nothing here ranks,
values a position or places an order; the module never imports the paper ledger."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import time

import pandas as pd

from .pipeline import _write
from . import first_board_watch as FW, theme_semantics as TS

VERSION = "eod_review_v1_3_20261120"
OUT_ROOT = Path("results/mainwave_engine")
# Reference lists share the intraday cache so the U list fetched at 14:30 is the one the review reconciles.
REFERENCE_ROOT = Path("results/intraday")
SESSIONS = 25
TOL = 0.005
COLS = ["date", "open", "high", "low", "close", "preclose", "volume", "amount", "turnover", "tradestatus", "isST", "adjustflag"]
DOMAIN = {"id": "cn_a_nonst_v1", "label": "沪深非ST A股可核验域",
          "boards": {"MAIN": ("sh_60", "sz_00"), "CHINEXT": ("sz_30",), "STAR": ("sh_68",)},
          "limit_pct": {"MAIN": .10, "CHINEXT": .20, "STAR": .20},
          "excluded": "ST（当日 isST=1）、北交所与指数代码、当日停牌或无成交、无涨跌幅限制日（只能靠参考价识别）",
          "price_basis": "未复权原始价；涨跌停价 = 前收 ×(1±幅度) 四舍五入到分，优先用交易所参考价核对"}
# Index-membership and financing tags are memberships, not themes; they stay counted but never rank.
GENERIC_CONCEPTS = ("融资融券", "深股通", "沪股通", "沪深300样本股", "中证500成分股", "上证180", "上证50", "创业板综",
                    "科创50", "证金持股", "转融券标的", "MSCI", "富时罗素", "标普道琼斯", "高股息精选", "预盈预增",
                    "预亏预减", "股权转让", "参股银行", "参股券商", "参股保险", "基金重仓", "QFII重仓", "社保重仓",
                    "创业板重组松绑", "深成500", "中证100", "上证380", "AB股", "AH股", "B股", "ST板块", "次新股")
ROLES = {"HEIGHT_LEADER": "最高板", "THEME_LEADER": "题材领涨", "TREND_CORE": "趋势核心",
         "RELAY_FAILURE": "接力失败代表", "LIMIT_DOWN": "跌停代表"}
NO_SAMPLE = "无可比样本"


# ---------------------------------------------------------------- helpers
def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def file_sha(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def board_of(symbol):
    for board, prefixes in DOMAIN["boards"].items():
        if symbol.startswith(prefixes):
            return board
    return ""


def limit_price(preclose, pct):
    return float((Decimal(str(preclose)) * (Decimal(1) + Decimal(str(pct)))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def number(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def clean(value):
    """JSON-safe: numpy scalars become Python scalars, NaN becomes null, never a crash at write time."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def ratio(numerator, denominator):
    """A ratio with its denominator on the record; zero denominator reads as no sample, never 0%."""
    if not denominator:
        return {"value": None, "numerator": numerator, "denominator": denominator, "status": NO_SAMPLE}
    return {"value": round(numerator / denominator, 4), "numerator": numerator, "denominator": denominator, "status": "OK"}


def distribution(values):
    valid = [v for v in values if v is not None]
    if not valid:
        return {"n": 0, "status": NO_SAMPLE}
    return {"n": len(valid), "mean": round(statistics.fmean(valid), 3), "median": round(statistics.median(valid), 3),
            "up": sum(v > 0 for v in valid), "worst": round(min(valid), 3), "best": round(max(valid), 3), "status": "OK"}


def calendar(data_root):
    path = Path(data_root) / "000300_SH_index.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path, usecols=["trade_date"])
    return sorted({str(x) if "-" in str(x) else f"{str(x)[:4]}-{str(x)[4:6]}-{str(x)[6:8]}" for x in frame.trade_date})


def index_return(data_root, day):
    path = Path(data_root) / "000300_SH_index.csv"
    if not path.exists():
        return None
    f = pd.read_csv(path)
    f["day"] = pd.to_datetime(f.trade_date.astype(str)).dt.strftime("%Y-%m-%d")
    f = f[f.day.le(day)].sort_values("day").drop_duplicates("day")
    if f.empty or f.day.iloc[-1] != day or len(f) < 2:
        return None
    c = pd.to_numeric(f.close, errors="coerce")
    if any(number(value) is None or value <= 0 for value in c.iloc[-2:]):
        return None
    return {"close": float(c.iloc[-1]), "return_pct": round(float(c.iloc[-1] / c.iloc[-2] - 1) * 100, 3),
            "return5_pct": round(float(c.iloc[-1] / c.iloc[-6] - 1) * 100, 3) if len(f) >= 6 else None,
            "return20_pct": round(float(c.iloc[-1] / c.iloc[-21] - 1) * 100, 3) if len(f) >= 21 else None}


# ---------------------------------------------------------------- panel
def load_panel(data_root, day, sessions=SESSIONS):
    """Last `sessions` raw bars per domain symbol up to `day`; unreadable files are listed, never dropped silently."""
    frames, unreadable = {}, []
    for path in sorted(Path(data_root).glob("*_daily.csv")):
        symbol = path.name[:-len("_daily.csv")]
        if not board_of(symbol):
            continue
        try:
            f = pd.read_csv(path, usecols=lambda c: c in COLS)
            f["date"] = f["date"].astype(str).str[:10]
            f = f[f.date.le(day)].sort_values("date").drop_duplicates("date", keep="last").tail(sessions).reset_index(drop=True)
        except (OSError, ValueError, KeyError):
            unreadable.append(symbol)
            continue
        if not f.empty:
            frames[symbol] = f
    return frames, unreadable


def annotate(frame, board):
    """Per-bar limit prices and board flags from the stock's own history, rounded the exchange way."""
    pct = DOMAIN["limit_pct"][board]
    f = frame.copy()
    for col in ("open", "high", "low", "close", "preclose", "volume", "amount", "turnover", "tradestatus", "isST"):
        f[col] = pd.to_numeric(f.get(col, pd.Series(float("nan"), index=f.index)), errors="coerce").replace([float("inf"), -float("inf")], float("nan"))
    f["metadata_valid"] = f.isST.isin([0, 1]) & f.tradestatus.isin([0, 1])
    f["st"] = f.isST.eq(1)
    f["traded"] = f.volume.gt(0) & f.tradestatus.eq(1)
    prices = f[["open", "high", "low", "close", "preclose"]]
    f["valid"] = (f.metadata_valid & prices.notna().all(axis=1) & prices.gt(0).all(axis=1)
                  & f.high.ge(f[["open", "close", "low"]].max(axis=1))
                  & f.low.le(f[["open", "close"]].min(axis=1)) & f.volume.ge(0) & f.amount.ge(0))
    if "adjustflag" in f:
        f["valid"] &= pd.to_numeric(f.adjustflag, errors="coerce").eq(3)
    f["up_limit"] = [limit_price(p, pct) if number(p) is not None and p > 0 else None for p in f.preclose]
    f["down_limit"] = [limit_price(p, -pct) if number(p) is not None and p > 0 else None for p in f.preclose]
    eligible = f.valid & f.traded & ~f.st
    f["is_U"] = (f.close - f.up_limit).abs().le(TOL) & eligible & f.up_limit.notna()
    f["is_Z"] = ~f.is_U & f.high.ge(f.up_limit - TOL) & eligible & f.up_limit.notna()
    f["is_D"] = (f.close - f.down_limit).abs().le(TOL) & eligible & f.down_limit.notna()
    f["is_DZ"] = ~f.is_D & f.low.le(f.down_limit + TOL) & eligible & f.down_limit.notna()
    f["limit_basis"] = "COMPUTED_FROM_PRECLOSE"
    return f


# ---------------------------------------------------------------- reference data (Tushare) via tail_observation cache
def fetch_reference(day, root, use_tushare=True):
    """U/D/Z lists and exchange limit prices, cached per type; absence is a named gap, not zero."""
    result = {"status": "NOT_REQUESTED", "limit_prices": {}, "lists": {}, "issues": [], "files": {}}
    if not use_tushare:
        result["issues"].append("未请求参考数据（--no-tushare）")
        return result
    from . import tail_observation as TO, intraday_quotes as Q
    try:
        client = Q.Client()
    except Exception as exc:  # no token, no network library: the review still runs on local bars
        result["status"] = "UNAVAILABLE"
        result["issues"].append(f"参考数据客户端不可用：{exc}")
        return result
    reference = Path(root) / "reference"
    limits = TO.fetch_source(client, "stk_limit", day, root)
    result["limit_prices_status"] = limits.get("status")
    result["files"]["stk_limit"] = str(reference / f"stk_limit_{day.replace('-', '')}.json")
    if limits.get("status") in ("AVAILABLE", "PARTIAL"):
        result["limit_prices"] = {Q.symbol_of(r["ts_code"]): {"up": number(r.get("up_limit")), "down": number(r.get("down_limit"))}
                                  for r in limits.get("records", [])}
    if limits.get("status") != "AVAILABLE":
        result["issues"].extend(f"stk_limit: {x}" for x in limits.get("issues", []) or [limits.get("status", "")])
    for kind in ("U", "D", "Z"):
        source = TO.fetch_source(client, "limit_list_d", day, root, limit_type=kind)
        result["files"][f"limit_list_d_{kind}"] = str(reference / (f"limit_list_d_{day.replace('-', '')}.json" if kind == "U"
                                                                     else f"limit_list_d_{kind}_{day.replace('-', '')}.json"))
        result["lists"][kind] = {"status": source.get("status"), "records": source.get("records", []),
                                 "issues": source.get("issues", [])}
        if source.get("status") not in ("AVAILABLE", "EMPTY"):
            result["issues"].extend(f"limit_list_d_{kind}: {x}" for x in source.get("issues", []) or [source.get("status", "UNAVAILABLE")])
    complete = limits.get("status") == "AVAILABLE" and all(v["status"] in ("AVAILABLE", "EMPTY") for v in result["lists"].values())
    result["status"] = "AVAILABLE" if complete else "PARTIAL" if result["limit_prices"] or any(v["status"] in ("AVAILABLE", "EMPTY", "PARTIAL") for v in result["lists"].values()) else "UNAVAILABLE"
    return result


# ---------------------------------------------------------------- facts
def day_rows(panel, day, previous_day=None):
    """One annotated row per domain symbol for `day`, with the reason when it does not qualify.
    A symbol whose last bar is older than the previous session has missed two or more sessions: it is long
    suspended or delisted, not a gap in today's data. Only a symbol missing just today's bar is a data gap."""
    rows, excluded = {}, Counter()
    for symbol, frame in panel.items():
        f = annotate(frame, board_of(symbol))
        last = f.iloc[-1]
        if last.date != day and previous_day and last.date < previous_day:
            excluded["LONG_SUSPENDED"] += 1
            rows[symbol] = {"symbol": symbol, "board": board_of(symbol), "frame": f, "in_domain": False, "reason": "LONG_SUSPENDED"}
            continue
        if last.date != day:
            excluded["NOT_TRADED_OR_MISSING"] += 1
            rows[symbol] = {"symbol": symbol, "board": board_of(symbol), "frame": f, "in_domain": False, "reason": "NOT_TRADED_OR_MISSING"}
            continue
        if not bool(last.metadata_valid) or (last.tradestatus == 1 and not bool(last.valid)):
            excluded["INVALID_DATA"] += 1
            rows[symbol] = {"symbol": symbol, "board": board_of(symbol), "frame": f, "in_domain": False, "reason": "INVALID_DATA"}
            continue
        if bool(last.st):
            excluded["ST"] += 1
            rows[symbol] = {"symbol": symbol, "board": board_of(symbol), "frame": f, "in_domain": False, "reason": "ST"}
            continue
        if not bool(last.traded):
            excluded["SUSPENDED"] += 1
            rows[symbol] = {"symbol": symbol, "board": board_of(symbol), "frame": f, "in_domain": False, "reason": "SUSPENDED"}
            continue
        rows[symbol] = {"symbol": symbol, "board": board_of(symbol), "frame": f, "in_domain": True, "reason": ""}
    return rows, dict(excluded)


def apply_reference_prices(rows, limit_prices, reference_complete=False):
    """Exchange limit prices override the computed ones for the day; disagreements are listed for audit."""
    disagreements = []
    for symbol, row in rows.items():
        if not row["in_domain"]:
            continue
        f = row["frame"]
        ref = limit_prices.get(symbol)
        computed = {"up": f.up_limit.iloc[-1], "down": f.down_limit.iloc[-1]}
        if ref and number(ref.get("up")) is not None and number(ref.get("down")) is not None and ref["up"] > ref["down"] > 0:
            if abs(ref["up"] - computed["up"]) > TOL or abs(ref["down"] - computed["down"]) > TOL:
                disagreements.append({"symbol": symbol, "computed": computed, "reference": ref,
                                      "note": "参考价与前收推算不符：可能是无涨跌幅限制日或特殊幅度，按参考价计"})
            last = f.index[-1]
            f.loc[last, "up_limit"], f.loc[last, "down_limit"] = ref["up"], ref["down"]
            close, high, low = f.close.iloc[-1], f.high.iloc[-1], f.low.iloc[-1]
            f.loc[last, "is_U"] = abs(close - ref["up"]) <= TOL
            f.loc[last, "is_Z"] = (not f.is_U.iloc[-1]) and high >= ref["up"] - TOL
            f.loc[last, "is_D"] = abs(close - ref["down"]) <= TOL
            f.loc[last, "is_DZ"] = (not f.is_D.iloc[-1]) and low <= ref["down"] + TOL
            row["price_basis"] = "EXCHANGE_REFERENCE"
        else:
            row["price_basis"] = "LIMIT_RULE_UNKNOWN" if ref is not None or reference_complete else "COMPUTED_FROM_PRECLOSE"
            if row["price_basis"] == "LIMIT_RULE_UNKNOWN":
                f.loc[f.index[-1], ["up_limit", "down_limit"]] = float("nan")
                f.loc[f.index[-1], ["is_U", "is_Z", "is_D", "is_DZ"]] = False
        f.loc[f.index[-1], "limit_basis"] = row["price_basis"]
    return disagreements


def boards_fact(rows, day, reference):
    """U/D/Z from prices, by board, reconciled stock by stock against the reference lists."""
    sets = {k: [] for k in ("U", "Z", "D", "DZ")}
    by_board = defaultdict(lambda: Counter())
    for symbol, row in rows.items():
        if not row["in_domain"]:
            continue
        last = row["frame"].iloc[-1]
        for kind, flag in (("U", "is_U"), ("Z", "is_Z"), ("D", "is_D"), ("DZ", "is_DZ")):
            if bool(last[flag]):
                sets[kind].append(symbol)
                by_board[row["board"]][kind] += 1
        by_board[row["board"]]["domain"] += 1
    recon = {}
    for kind in ("U", "Z", "D"):
        source = reference.get("lists", {}).get(kind) or {}
        if source.get("status") not in ("AVAILABLE", "EMPTY", "PARTIAL"):
            recon[kind] = {"status": "REFERENCE_UNAVAILABLE", "issues": source.get("issues", [])}
            continue
        from . import intraday_quotes as Q
        theirs = {Q.symbol_of(r["ts_code"]) for r in source.get("records", []) if board_of(Q.symbol_of(r["ts_code"]))}
        ours = set(sets[kind])
        recon[kind] = {"status": "RECONCILED" if ours == theirs and source.get("status") != "PARTIAL" else "RECONCILIATION_PENDING", "local": len(ours), "reference": len(theirs), "both": len(ours & theirs),
                       "local_only": sorted(ours - theirs)[:50], "reference_only": sorted(theirs - ours)[:50],
                       "note": "参考口径不含ST；本地口径同样剔除ST。差异按股票列出，供逐股核对。"}
    open_rate = {"status": NO_SAMPLE, "note": "开板过程只在参考数据的 open_times 字段中；缺字段不计算"}
    source_u = reference.get("lists", {}).get("U") or {}
    if source_u.get("status") in ("AVAILABLE", "PARTIAL") and source_u.get("records"):
        from . import intraday_quotes as Q
        scoped = {Q.symbol_of(r["ts_code"]): r for r in source_u["records"]
                  if rows.get(Q.symbol_of(r["ts_code"]), {}).get("in_domain")}
        opened = [r for r in scoped.values() if number(r.get("open_times")) is not None and number(r["open_times"]) >= 0]
        if opened:
            open_rate = {**ratio(sum(1 for r in opened if number(r["open_times"]) > 0), len(opened)),
                         "definition": "收盘涨停股中当日有过开板者 / 有开板字段的收盘涨停股（含开板后回封）",
                         "open_times_total": int(sum(number(r["open_times"]) for r in opened)),
                         "reference_domain_members": len(scoped), "missing_fields": len(scoped) - len(opened),
                         "local_U_without_reference_U": sorted(set(sets["U"]) - set(scoped)),
                         "note": "open_times 是次数，不是股票数"}
    return {"U": sorted(sets["U"]), "Z": sorted(sets["Z"]), "D": sorted(sets["D"]), "DZ": sorted(sets["DZ"]),
            "counts": {k: len(v) for k, v in sets.items()},
            "by_board": {b: dict(c) for b, c in sorted(by_board.items())},
            "seal_failure_rate": {**ratio(len(sets["Z"]), len(sets["U"]) + len(sets["Z"])),
                                  "definition": "触及涨停但收盘未封住的股票数 / 触及涨停的股票数，按股票去重"},
            "open_rate": open_rate, "reconciliation": recon,
            "definitions": {"U": "收盘价等于涨停价", "Z": "最高价触及涨停价但收盘未封住", "D": "收盘价等于跌停价",
                            "DZ": "最低价触及跌停价但收盘未封死"}}


def history_series(rows, sessions, day):
    """Daily U/D/Z counts over the panel window, computed the same way every day."""
    per_day = defaultdict(Counter)
    for row in rows.values():
        f = row["frame"]
        for _, bar in f.iterrows():
            if bar.date > day or bool(bar.st) or not bool(bar.traded) or not bool(bar.valid):
                continue
            per_day[bar.date]["domain"] += 1
            for kind, flag in (("U", "is_U"), ("Z", "is_Z"), ("D", "is_D")):
                if bool(bar[flag]):
                    per_day[bar.date][kind] += 1
    days = [d for d in sessions if d <= day][-20:]
    return [{"day": d, **{k: per_day[d].get(k, 0) for k in ("U", "Z", "D", "domain")},
             "basis": "MIXED_REFERENCE_AND_COMPUTED" if d == day else "COMPUTED_FROM_PRECLOSE"} for d in days if d in per_day]


SENTIMENT_METRICS = {
    "limit_up": ("收盘涨停家数", True), "seal_failure_rate": ("封板失败率（炸板÷触及涨停）", False),
    "limit_down": ("收盘跌停家数", False), "pool_mean_pct": ("昨日涨停股今日平均涨幅", True),
    "pool_up_share": ("昨日涨停股今日收红比例", True), "open_premium_pct": ("昨日涨停股今日开盘溢价", True),
    "promotion_1_to_2": ("一进二晋级率", True), "max_height": ("最高连板高度", True),
}
READINGS = ("冷", "偏冷", "中性", "偏热", "热")


def sentiment_fact(rows, sessions, day, window=20):
    """Relay sentiment as readings: each metric against its own last `window` sessions, none averaged into a score.

    'Hot' means high for metrics where high is hot (limit-ups, pool gains, promotion), and low for the ones
    where high is cold (seal failures, limit-downs). The reading is relative to this window only.
    """
    position = {d: i for i, d in enumerate(sessions)}
    days = [d for d in sessions if d <= day][-window:]
    per = {d: {"U": 0, "Z": 0, "D": 0, "pool": [], "pool_up": 0, "open": [], "h1_pool": 0, "h1_cont": 0, "max_h": 0} for d in days}
    for row in rows.values():
        f = row["frame"]
        dates, is_u, is_z, is_d = f.date.tolist(), f.is_U.tolist(), f.is_Z.tolist(), f.is_D.tolist()
        st, traded = f.st.tolist(), f.traded.tolist()
        close, pre, opn = f.close.tolist(), f.preclose.tolist(), f.open.tolist()
        heights, height = [], 0
        for i, d in enumerate(dates):
            consecutive = i > 0 and position.get(d, -9) - position.get(dates[i - 1], -9) == 1
            height = (height + 1 if (consecutive and height > 0) else 1) if (is_u[i] and traded[i] and not st[i]) else 0
            heights.append(height)
        for i, d in enumerate(dates):
            if d not in per or st[i] or not traded[i]:
                continue
            p = per[d]
            p["U"] += bool(is_u[i]); p["Z"] += bool(is_z[i]); p["D"] += bool(is_d[i]); p["max_h"] = max(p["max_h"], heights[i])
            consecutive = i > 0 and position.get(d, -9) - position.get(dates[i - 1], -9) == 1
            if (consecutive and is_u[i - 1] and traded[i - 1] and not st[i - 1] and pre[i] and close[i - 1]
                    and abs(pre[i] / close[i - 1] - 1) <= .001):
                ret = (close[i] / pre[i] - 1) * 100
                p["pool"].append(ret); p["pool_up"] += ret > 0; p["open"].append((opn[i] / pre[i] - 1) * 100)
                if heights[i - 1] == 1:
                    p["h1_pool"] += 1; p["h1_cont"] += bool(is_u[i])
    series = {name: [] for name in SENTIMENT_METRICS}
    for d in days:
        p = per[d]
        touched = p["U"] + p["Z"]
        series["limit_up"].append(p["U"]); series["limit_down"].append(p["D"])
        series["seal_failure_rate"].append(round(p["Z"] / touched, 4) if touched else None)
        series["pool_mean_pct"].append(round(statistics.fmean(p["pool"]), 3) if p["pool"] else None)
        series["pool_up_share"].append(round(p["pool_up"] / len(p["pool"]), 4) if p["pool"] else None)
        series["open_premium_pct"].append(round(statistics.fmean(p["open"]), 3) if p["open"] else None)
        series["promotion_1_to_2"].append(round(p["h1_cont"] / p["h1_pool"], 4) if p["h1_pool"] else None)
        series["max_height"].append(p["max_h"])
    metrics = {}
    for name, (label, hotter_when_high) in SENTIMENT_METRICS.items():
        values = series[name]
        today = values[-1] if values else None
        valid = [v for v in values if v is not None]
        if today is None or len(valid) < 5:
            metrics[name] = {"label": label, "today": today, "status": NO_SAMPLE, "series": values}
            continue
        below = sum(1 for v in valid if v < today); ties = sum(1 for v in valid if v == today)
        percentile = round((below + ties / 2) / len(valid), 3)
        hot = percentile if hotter_when_high else 1 - percentile
        reading = READINGS[min(4, int(hot * 5))]
        metrics[name] = {"label": label, "today": today, "percentile": percentile, "reading": reading,
                         "hotter_when_high": hotter_when_high, "n": len(valid), "median": round(statistics.median(valid), 3),
                         "min": min(valid), "max": max(valid), "status": "OK", "series": values}
    return {"window_days": days, "metrics": metrics,
            "readings": "冷 / 偏冷 / 中性 / 偏热 / 热 = 今日在自身近 %d 日中的分位五等分；炸板率与跌停数方向取反" % window,
            "note": "接力情绪与指数趋势分列，不合成一个总分；这是相对近 20 日的读法，不是绝对阈值，也不是预测。"}


def breadth_fact(rows):
    up = down = flat = 0
    for row in rows.values():
        if not row["in_domain"]:
            continue
        last = row["frame"].iloc[-1]
        if last.preclose and last.close > last.preclose:
            up += 1
        elif last.preclose and last.close < last.preclose:
            down += 1
        else:
            flat += 1
    return {"up": up, "down": down, "flat": flat, "domain": up + down + flat}


def pool_performance(pool, rows, day):
    """Yesterday's members, today: open, close and intraday high each against the previous close, and
    close against open. Suspended, missing and corporate-action members are listed, never counted as zero."""
    members, excluded = [], Counter()
    for symbol in pool:
        row = rows.get(symbol)
        f = row["frame"] if row else None
        if f is None or f.iloc[-1].date != day or not bool(f.iloc[-1].traded):
            excluded["SUSPENDED_OR_MISSING"] += 1
            members.append({"symbol": symbol, "status": "SUSPENDED_OR_MISSING"})
            continue
        last = f.iloc[-1]
        if not bool(last.valid) or bool(last.st) or last.limit_basis == "LIMIT_RULE_UNKNOWN":
            excluded["INVALID_DATA"] += 1
            members.append({"symbol": symbol, "status": "INVALID_DATA"})
            continue
        prior_close = f.close.iloc[-2] if len(f) >= 2 else None
        if prior_close and last.preclose and abs(last.preclose / prior_close - 1) > .001:
            excluded["CORPORATE_ACTION_PENDING"] += 1
            members.append({"symbol": symbol, "status": "CORPORATE_ACTION_PENDING"})
            continue
        pre = last.preclose
        members.append({"symbol": symbol, "status": "OK", "board": row["board"],
                        "open_vs_pre": round((last.open / pre - 1) * 100, 3), "close_vs_pre": round((last.close / pre - 1) * 100, 3),
                        "close_vs_open": round((last.close / last.open - 1) * 100, 3) if last.open else None,
                        "high_vs_pre": round((last.high / pre - 1) * 100, 3),
                        "limit_today": "U" if bool(last.is_U) else "Z" if bool(last.is_Z) else "D" if bool(last.is_D) else ""})
    ok = [m for m in members if m["status"] == "OK"]
    return {"performance": members, "n_pool": len(pool), "n_valid": len(ok), "excluded": dict(excluded),
            "open_vs_pre": distribution([m["open_vs_pre"] for m in ok]),
            "close_vs_pre": distribution([m["close_vs_pre"] for m in ok]),
            "close_vs_open": distribution([m["close_vs_open"] for m in ok]),
            "high_vs_pre": {**distribution([m["high_vs_pre"] for m in ok]), "note": "日内最高是事后最大幅度，不是可实现收益"},
            "continued_U": sum(1 for m in ok if m["limit_today"] == "U")}


def ladder_fact(rows, sessions, day, previous_ladder):
    """Consecutive limit-up heights on consecutive market sessions; yesterday's ladder judged member by member."""
    position = {d: i for i, d in enumerate(sessions)}
    heights = {}
    frozen_heights = {s: int(h) for h, members in (previous_ladder or {}).items() for s in members}
    for symbol, row in rows.items():
        if not row["in_domain"] or not bool(row["frame"].iloc[-1].is_U):
            continue
        f = row["frame"]
        if previous_ladder is not None:
            heights[symbol] = frozen_heights.get(symbol, 0) + 1
            continue
        height, i = 0, len(f) - 1
        while i >= 0 and bool(f.iloc[i].is_U):
            if i < len(f) - 1 and position.get(f.iloc[i + 1].date, -9) - position.get(f.iloc[i].date, -9) != 1:
                break  # a gap in the calendar breaks the chain; N days M boards is not a height
            height += 1
            i -= 1
        heights[symbol] = height
    by_height = Counter(heights.values())
    promotion = {}
    for h, members in sorted((previous_ladder or {}).items()):
        continued = [s for s in members if heights.get(s, 0) >= int(h) + 1]
        checkable = [s for s in members if s in rows and rows[s]["in_domain"] and rows[s]["frame"].iloc[-1].date == day
                     and rows[s].get("price_basis") != "LIMIT_RULE_UNKNOWN"]
        broke = [s for s in checkable if s not in continued]
        promotion[str(h)] = {**ratio(len(continued), len(checkable)), "pool": len(members), "uncheckable": len(members) - len(checkable),
                             "continued": continued, "broke": broke[:50]}
    return {"heights": {s: h for s, h in sorted(heights.items(), key=lambda kv: (-kv[1], kv[0]))},
            "by_height": {str(k): v for k, v in sorted(by_height.items())},
            "max_height": max(by_height) if by_height else 0,
            "promotion_from_previous": promotion,
            "definition": "连续市场交易日收盘涨停的天数；停牌或缺日打断计数。晋级率 = 昨日n板母池中今日继续封板者 / 今日可核验者"}


def theme_fact(rows, day, mapping, registry, limit_set):
    """Theme structure from the full membership table: breadth, limit-ups, MA20 and amount share, overlap disclosed."""
    domain_amount = sum(r["frame"].iloc[-1].amount or 0 for r in rows.values() if r["in_domain"])
    members = defaultdict(list)
    for symbol, info in mapping.items():
        if symbol in rows and rows[symbol]["in_domain"]:
            for concept in info.get("concepts", []):
                members[concept].append(symbol)
    for theme in registry:
        if theme.get("active_from", "9999") <= day:
            members[f"紧链:{theme.get('label', theme.get('id'))}"] = [s for s in theme.get("members", {}) if s in rows and rows[s]["in_domain"]]

    def stats(name, symbols):
        up = above = limit = 0
        amount = 0.
        for s in symbols:
            f = rows[s]["frame"]
            last = f.iloc[-1]
            up += bool(last.close > last.preclose)
            limit += s in limit_set
            amount += last.amount or 0
            if len(f) >= 20:
                above += bool(last.close > f.close.tail(20).mean())
        return {"theme": name, "members": len(symbols), "up": up, "limit_up": limit, "above_ma20": above,
                "breadth": ratio(up, len(symbols)), "limit_share": ratio(limit, len(symbols)),
                "amount_share": ratio(amount, domain_amount) if domain_amount else ratio(0, 0),
                "generic": name.startswith(GENERIC_CONCEPTS) or any(g in name for g in GENERIC_CONCEPTS),
                "chain": name.startswith("紧链:")}
    # Registered chains are small by design and always shown; open concepts need five members to mean anything.
    table = [stats(name, syms) for name, syms in members.items() if len(syms) >= 5 or (name.startswith("紧链:") and syms)]
    ranked = [t for t in table if not t["generic"]]
    overlap = Counter()
    for symbol in limit_set:
        overlap[len([c for c in mapping.get(symbol, {}).get("concepts", []) if not any(g in c for g in GENERIC_CONCEPTS)])] += 1
    return {"domain_amount": domain_amount, "themes_counted": len(table),
            "top_by_limit_up": sorted(ranked, key=lambda t: (-t["limit_up"], -t["breadth"]["value"] if t["breadth"]["value"] is not None else 0, t["theme"]))[:15],
            "top_by_amount_share": sorted(ranked, key=lambda t: -(t["amount_share"]["value"] or 0))[:15],
            # Size-aware view: a raw count favours thousand-member concepts; the share needs at least three limit-ups
            # and a bounded membership to say anything. Display filters, not production thresholds.
            "top_by_limit_share": sorted([t for t in ranked if t["limit_up"] >= 3 and t["members"] <= 300],
                                         key=lambda t: (-(t["limit_share"]["value"] or 0), -t["limit_up"], t["theme"]))[:15],
            "chains": [t for t in table if t["chain"]],
            "overlap_of_limit_ups": {str(k): v for k, v in sorted(overlap.items())},
            "notes": ["题材成交额占比按去重成员计算；跨概念重叠使各题材占比不可直接相加",
                      "指数样本、融资融券等成员标签只计数不排序", "题材成员为当前映射版本，无历史时点版本",
                      "涨停占比榜只列 ≥3 只涨停且成员 ≤300 的题材：这是展示过滤，不是生产门槛",
                      "已核验题材关联按生效和获知时间叠加；市场叙事关联不代表主营或业绩兑现"]}


def activity_fact(rows, day):
    """Amount concentration, relative turnover, close location and 20-day position for the active names."""
    table = []
    for symbol, row in rows.items():
        if not row["in_domain"]:
            continue
        f = row["frame"]
        last = f.iloc[-1]
        prior = f.turnover.iloc[:-1].tail(20)
        prior = prior[prior > 0]
        rel = round(float(last.turnover / prior.median()), 3) if len(prior) >= 10 and last.turnover and last.turnover > 0 else None
        span = (f.high.tail(20).max() - f.low.tail(20).min()) if len(f) >= 20 else None
        table.append({"symbol": symbol, "board": row["board"], "amount": float(last.amount or 0),
                      "pct_chg": round((last.close / last.preclose - 1) * 100, 3) if last.preclose else None,
                      "close_location": round(float((last.close - last.low) / (last.high - last.low)), 3) if last.high > last.low else None,
                      "relative_turnover": rel,
                      "position20": round(float((last.close - f.low.tail(20).min()) / span), 3) if span else None,
                      "limit": "U" if bool(last.is_U) else "Z" if bool(last.is_Z) else "D" if bool(last.is_D) else ""})
    total = sum(t["amount"] for t in table)
    top = sorted(table, key=lambda t: -t["amount"])[:20]
    buckets = Counter("strong" if (t["close_location"] or 0) >= .7 else "weak" if t["close_location"] is not None and t["close_location"] <= .3 else "mid"
                      for t in table if t["close_location"] is not None)
    return {"domain_amount": total, "top20": top,
            "top20_share": ratio(sum(t["amount"] for t in top), total) if total else ratio(0, 0),
            "close_location_buckets": dict(buckets),
            "scatter": [{k: t[k] for k in ("symbol", "board", "position20", "relative_turnover", "close_location", "pct_chg", "limit")}
                        for t in table if t["relative_turnover"] is not None and t["position20"] is not None
                        and (t["relative_turnover"] >= 2 or t["limit"] or t in top)][:400],
            "notes": ["成交额是交易活跃，不是净买入", "相对换手 = 当日换手 / 前20日换手中位数；不同板块与体量不用同一绝对阈值"]}


def loss_fact(rows, day, pool_perf, boards, activity):
    """Where money was lost today: yesterday's strong pool, limit-downs, failed seals, gap-up reversals, breakdowns —
    layered by board and 20-day position, never pressed into one score."""
    pos = {t["symbol"]: t["position20"] for t in activity["scatter"]}
    for row in rows.values():
        if row["in_domain"] and row["symbol"] not in pos:
            f = row["frame"]
            span = (f.high.tail(20).max() - f.low.tail(20).min()) if len(f) >= 20 else None
            pos[row["symbol"]] = round(float((f.close.iloc[-1] - f.low.tail(20).min()) / span), 3) if span else None
    layer = lambda s: "HIGH" if (pos.get(s) or 0) >= .8 else "LOW" if pos.get(s) is not None and pos[s] <= .2 else "MID"
    events = {"yesterday_pool_negative": [m["symbol"] for m in pool_perf["performance"] if m["status"] == "OK" and m["close_vs_pre"] < 0],
              "limit_down": boards["D"], "seal_failed": boards["Z"], "gap_up_reversal": [], "breakdown_20d": []}
    for symbol, row in rows.items():
        if not row["in_domain"]:
            continue
        f = row["frame"]
        last = f.iloc[-1]
        prior_close = f.close.iloc[-2] if len(f) >= 2 else None
        if prior_close and last.preclose and abs(last.preclose / prior_close - 1) > .001:
            continue  # a corporate action moves the price scale; that is reconciliation, not a loss event
        if last.preclose and last.open >= last.preclose * 1.02 and last.close < last.open and last.close < last.preclose:
            events["gap_up_reversal"].append(symbol)
        if len(f) >= 21 and last.close < f.low.iloc[-21:-1].min():
            events["breakdown_20d"].append(symbol)
    layered = {}
    for name, symbols in events.items():
        layered[name] = {"count": len(symbols),
                         "by_board": dict(Counter(rows[s]["board"] for s in symbols if s in rows)),
                         "by_position": dict(Counter(layer(s) for s in symbols)),
                         "symbols": sorted(symbols)[:60]}
    return {"events": layered, "position_layers": "HIGH ≥ 0.8 / LOW ≤ 0.2 of the 20-day range",
            "notes": ["价格大跌不能单独证明主力出货或业绩原因", "分层展示，不压成一个亏钱效应总分"]}


# ---------------------------------------------------------------- lists
def nominate_bellwethers(rows, ladder, themes, pool_perf, boards, mapping):
    """Rule-based representatives, each tied to a question; risk names belong here, buy lists do not."""
    picks, seen = [], set()

    def add(symbol, role, question, facts):
        if symbol and symbol not in seen and len(picks) < 8:
            seen.add(symbol)
            picks.append({"symbol": symbol, "name": mapping.get(symbol, {}).get("name", ""), "role": role,
                          "role_label": ROLES[role], "question": question, "facts": facts,
                          "note": "机械提名，非买入推荐；高位风险股可以入选，只作观察"})
    for symbol, height in list(ladder["heights"].items())[:2]:
        add(symbol, "HEIGHT_LEADER", "最高板明日能否继续晋级；断板后是否修复", {"height": height})
    for theme in themes["top_by_limit_up"][:2]:
        if theme["limit_up"]:
            candidates = [s for s in boards["U"] if theme["theme"] in mapping.get(s, {}).get("concepts", [])]
            if candidates:
                lead = max(candidates, key=lambda s: ladder["heights"].get(s, 0))
                add(lead, "THEME_LEADER", f"{theme['theme']} 领涨能否带动冻结成员广度改善", {"theme": theme["theme"], "limit_up": theme["limit_up"]})
    core = []
    for symbol, row in rows.items():
        f = row["frame"]
        if row["in_domain"] and len(f) >= 21 and not bool(f.iloc[-1].is_U):
            r20 = f.close.iloc[-1] / f.close.iloc[-21] - 1
            span = f.high.tail(20).max() - f.low.tail(20).min()
            if r20 > .15 and span and (f.close.iloc[-1] - f.low.tail(20).min()) / span >= .7 and f.close.iloc[-1] > f.close.tail(20).mean():
                core.append((r20, symbol))
    for r20, symbol in sorted(core, reverse=True)[:2]:
        add(symbol, "TREND_CORE", "趋势核心结构是否保持；回踩是否有承接", {"return20_pct": round(r20 * 100, 2)})
    worst = min((m for m in pool_perf["performance"] if m["status"] == "OK"), key=lambda m: m["close_vs_pre"], default=None)
    if worst and worst["close_vs_pre"] < 0:
        add(worst["symbol"], "RELAY_FAILURE", "昨日涨停接力失败是否扩散到低位与其他板块", {"close_vs_pre": worst["close_vs_pre"]})
    if boards["D"]:
        add(boards["D"][0], "LIMIT_DOWN", "跌停延续还是单日事件；同板块是否受影响", {"limit": "D"})
    return picks


def candidate_lists(out_dir):
    """Candidates and review-eligible names are references to the frozen daily deliverable, not a new ranking."""
    brief = read_json(out_dir / "daily_brief.json")
    ledgers = read_json(out_dir / "books_daily.json")
    pool, seen = [], set()
    for row in (brief.get("sections", {}).get("top", {}).get("strength", []) + brief.get("sections", {}).get("top", {}).get("participation", [])):
        if row["symbol"] not in seen:
            seen.add(row["symbol"])
            pool.append({k: row.get(k) for k in ("symbol", "name", "theme", "route_ids", "state", "blockers", "verdict_reading",
                                                  "strength_rank", "participation_rank", "continuity_event")})
    chosen = (ledgers.get("signal") or {}).get("trade_set") or []
    return {"continuing_candidates": pool, "source": "daily_brief.json 强势榜/参与榜（冻结产物引用）" if brief else "缺 daily_brief.json",
            "review_eligible": chosen, "review_note": "暂无条件齐备标的" if not chosen else "现有规则下条件齐备，须同时看资格、闸门、位置与阻断",
            "auto_book": {k: (ledgers.get("auto") or {}).get(k) for k in ("live", "plans", "open_positions", "fills")} if ledgers else {}}


# ---------------------------------------------------------------- assembly
def read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}


def previous_review(out_root, previous_day, known_at):
    if not previous_day:
        return {}, {}
    folder = Path(out_root) / previous_day.replace("-", "") / "eod_review"
    cutoff = pd.Timestamp(known_at)
    for version in sorted(folder.glob("v[0-9][0-9][0-9]"), reverse=True):
        if not all((version / name).exists() for name in ("facts.json", "manifest.json", "report.json", "report.md")):
            continue
        facts = read_json(version / "facts.json")
        available = pd.Timestamp(facts.get("known_at"))
        if facts.get("as_of") != previous_day or pd.isna(available) or available.tzinfo is None or available > cutoff:
            continue
        return facts, {"path": str(version / "facts.json"), "sha256": file_sha(version / "facts.json"),
                       "version": version.name, "known_at": facts["known_at"]}
    return {}, {}


def build(day, data_root="data", out_root=OUT_ROOT, root=REFERENCE_ROOT, use_tushare=True, mapping_path="concept_mapping.json",
          registry_path="theme_registry.json", now=None):
    started = time.perf_counter()
    timing = {}
    day = pd.Timestamp(day).date().isoformat()
    known_at = (now or datetime.now().astimezone()).isoformat(timespec="seconds")
    out_root, data_root = Path(out_root), Path(data_root)
    sessions = calendar(data_root)
    if day not in sessions:
        raise ValueError(f"{day} 不在市场日历中，不能做盘后复盘")
    previous_day = next((d for d in reversed(sessions) if d < day), "")
    t0 = time.perf_counter()
    panel, unreadable = load_panel(data_root, day)
    timing["load_panel_s"] = round(time.perf_counter() - t0, 1)
    rows, excluded = day_rows(panel, day, previous_day)
    in_domain = sum(r["in_domain"] for r in rows.values())
    if not in_domain:
        raise ValueError(f"DATA_WAIT: {day} 没有可核验当日行情，不能输出零涨跌停；读取 {len(panel)}，排除 {excluded}，不可读 {len(unreadable)}")
    coverage_status = "PARTIAL" if unreadable or excluded.get("NOT_TRADED_OR_MISSING") or excluded.get("INVALID_DATA") else "READY"
    t0 = time.perf_counter()
    reference = fetch_reference(day, root, use_tushare)
    timing["reference_s"] = round(time.perf_counter() - t0, 1)
    disagreements = apply_reference_prices(rows, reference.get("limit_prices", {}), reference.get("limit_prices_status") == "AVAILABLE")
    t0 = time.perf_counter()
    boards = boards_fact(rows, day, reference)
    series = history_series(rows, sessions, day)
    breadth = breadth_fact(rows)
    prior, prior_source = previous_review(out_root, previous_day, known_at)
    if prior.get("as_of") == previous_day:
        before_open = pd.Timestamp(day + "T09:30:00+08:00")
        pool = prior["boards"]["U"]
        pool_basis = "FROZEN_PREVIOUS_REVIEW" if pd.Timestamp(prior["known_at"]) < before_open else "HISTORICAL_PREVIOUS_REVIEW"
        previous_ladder = defaultdict(list)
        for s, h in prior["ladder"]["heights"].items():
            previous_ladder[str(h)].append(s)
    else:
        previous_rows = {}
        for symbol, row in rows.items():
            frame = row["frame"][row["frame"].date.le(previous_day)]
            if not frame.empty and frame.iloc[-1].date == previous_day:
                last = frame.iloc[-1]
                previous_rows[symbol] = {**row, "frame": frame, "in_domain": bool(last.valid and last.traded and not last.st)}
        pool = sorted(s for s, r in previous_rows.items() if r["in_domain"] and bool(r["frame"].iloc[-1].is_U))
        pool_basis = "RECONSTRUCTED_FROM_BARS_NOT_FROZEN"
        previous_ladder = defaultdict(list)
        for s, h in ladder_fact(previous_rows, sessions, previous_day, None)["heights"].items():
            previous_ladder[str(h)].append(s)
    pool_perf = pool_performance(pool, rows, day)
    ladder = ladder_fact(rows, sessions, day, previous_ladder)
    mapping = read_json(mapping_path).get("mapping", {})
    mapping, membership = TS.apply_membership_evidence(mapping, day, known_at)
    registry = read_json(registry_path).get("themes", [])
    themes = theme_fact(rows, day, mapping, registry, set(boards["U"]))
    activity = activity_fact(rows, day)
    losses = loss_fact(rows, day, pool_perf, boards, activity)
    sentiment = sentiment_fact(rows, sessions, day)
    timing["facts_s"] = round(time.perf_counter() - t0, 1)
    market_context = read_json(out_root / day.replace("-", "") / "market_context.json")
    lists = {"bellwethers": nominate_bellwethers(rows, ladder, themes, pool_perf, boards, mapping),
             **candidate_lists(out_root / day.replace("-", ""))}
    sources = {"bars": {"path": str(data_root), "symbols": len(panel), "unreadable": unreadable,
                        "slice_sha256": digest({s: digest(f.to_json(orient="split", double_precision=15)) for s, f in panel.items()})},
               "index_calendar": {"path": str(data_root / "000300_SH_index.csv"), "sha256": file_sha(data_root / "000300_SH_index.csv")},
               "previous_review": prior_source,
               "market_context": {"path": str(out_root / day.replace("-", "") / "market_context.json"),
                                  "sha256": file_sha(out_root / day.replace("-", "") / "market_context.json")},
               "concept_mapping": {"path": mapping_path, "sha256": file_sha(mapping_path), "generated": read_json(mapping_path).get("generated", "")},
               "membership_evidence": {**membership, "sha256": file_sha(TS.MEMBERSHIP_EVIDENCE)},
               "theme_registry": {"path": registry_path, "sha256": file_sha(registry_path)},
               "reference": {"status": reference["status"], "issues": reference["issues"],
                             "files": {k: {"path": v, "sha256": file_sha(v)} for k, v in reference.get("files", {}).items()}},
               "daily_brief": {"sha256": file_sha(out_root / day.replace("-", "") / "daily_brief.json")},
               "books_daily": {"sha256": file_sha(out_root / day.replace("-", "") / "books_daily.json")}}
    market_index = index_return(data_root, day)
    gaps = list(reference["issues"]) + membership["errors"]
    if coverage_status == "PARTIAL":
        gaps.append(f"当日行情覆盖不完整：缺行 {excluded.get('NOT_TRADED_OR_MISSING', 0)}，无效 {excluded.get('INVALID_DATA', 0)}，不可读 {len(unreadable)}")
    unknown_limits = [s for s, r in rows.items() if r.get("price_basis") == "LIMIT_RULE_UNKNOWN"]
    if unknown_limits:
        gaps.append(f"{len(unknown_limits)} 只涨跌停规则无法核验，不套用普通幅度；见 coverage.limit_rule_unknown")
    if not market_context:
        gaps.append("缺当日 market_context.json")
    if not market_index:
        gaps.append("缺可核验的当日指数与前收价格")
    if boards["open_rate"]["status"] == NO_SAMPLE:
        gaps.append("开板率：缺开板过程字段，未计算")
    if pool_basis == "RECONSTRUCTED_FROM_BARS_NOT_FROZEN":
        gaps.append("昨日涨停母池由日线重建，不是前一日冻结记录")
    elif pool_basis == "HISTORICAL_PREVIOUS_REVIEW":
        gaps.append("昨日母池来自事后归档，不是今日开盘前已有的前向冻结记录")
    gaps.append("历史日线重建与20日序列使用普通板块幅度，未逐日核验历史特殊幅度；冻结梯队存在时优先逐日推进")
    gaps.append("盘中先后顺序、封单稳定性只有日线时保持未知")
    gaps.append("题材成员为当前映射版本，无历史时点版本")
    known_at = (now or datetime.now().astimezone()).isoformat(timespec="seconds")
    prior_days = [d for d in sessions if d < day][-FW.LOOKBACK:]
    p5_pools = FW.load_pools(out_root, prior_days + [day], known_at)
    p5 = FW.limit_up_matches(day, prior_days, p5_pools, boards, ladder["heights"], rows)
    p5_prior, p5_source = previous_review(out_root, previous_day, day + "T09:14:59.999999+08:00")
    p5_feedback = FW.limit_up_followup(p5_prior.get("p5_limit_up", {}), day, rows)
    sources["p5_preparation"] = {"logic_sha256": file_sha(FW.__file__),
                                 "pool_hashes": {d: digest(pool) for d, pool in p5_pools.items()},
                                 "prior_review": p5_source}
    facts = {"version": VERSION, "as_of": day, "previous_day": previous_day, "known_at": known_at, "domain": DOMAIN,
             "coverage": {"status": coverage_status, "symbols_read": len(panel), "in_domain": in_domain,
                          "excluded": excluded, "unreadable": len(unreadable), "limit_rule_unknown": unknown_limits,
                          "expected_local_files": len(panel) + len(unreadable)},
             "market": {"index": market_index, "breadth": breadth,
                        "context": {k: market_context.get(k) for k in ("structure", "temperature", "state", "trend_review_allowed", "data_status")},
                        "series_20d": series},
             "boards": boards, "limit_price_disagreements": disagreements[:50],
             "yesterday_pool": {"basis": pool_basis, "source": prior_source, "members": pool, **pool_perf},
             "ladder": ladder, "themes": themes, "activity": activity, "loss_effects": losses, "lists": lists,
             "sentiment": sentiment, "p5_limit_up": p5, "p5_limit_up_feedback": p5_feedback}
    manifest = {"version": VERSION, "as_of": day, "known_at": known_at, "domain_id": DOMAIN["id"], "sources": sources,
                "gaps": gaps, "timing": {**timing, "total_s": round(time.perf_counter() - started, 1)},
                "code_sha256": file_sha(__file__),
                "input_hash": digest([VERSION, file_sha(__file__), DOMAIN, day, sources]),
                "stage": "PARTIAL_DATA" if coverage_status == "PARTIAL" or unknown_limits or not market_index or not market_context else "REVIEW_READY" if reference["status"] == "AVAILABLE" else "PRICE_FINAL_REFERENCE_PENDING",
                "sections": {"market": coverage_status if market_context and market_index else "MISSING_INPUT",
                             "boards": "PARTIAL" if unknown_limits else coverage_status,
                             "ladder": "PARTIAL" if unknown_limits else coverage_status,
                             "themes": coverage_status if mapping else "NO_MAPPING",
                             "activity": coverage_status, "loss_effects": coverage_status,
                             "lists": "READY" if read_json(out_root / day.replace('-', '') / "books_daily.json") else "MISSING_INPUT",
                             "conclusions": "NOT_IMPLEMENTED_P1", "verification": "NOT_IMPLEMENTED_P1", "next_session_plan": "NOT_IMPLEMENTED_P1"}}
    return facts, manifest


def persist(day, facts, manifest, out_root=OUT_ROOT):
    """Immutable versions: same inputs reuse the existing version, changed inputs get the next one."""
    folder = Path(out_root) / day.replace("-", "") / "eod_review"
    latest = read_json(folder / "latest.json")
    if (latest.get("input_hash") == manifest["input_hash"] and latest.get("path")
            and all((Path(latest["path"]) / name).exists() for name in ("facts.json", "manifest.json", "report.json", "report.md"))):
        return {**latest, "reused": True}
    existing = sorted(p.name for p in folder.glob("v???") if p.is_dir())
    next_version = max((int(v[1:]) for v in existing if re.fullmatch(r"v\d+", v)), default=0) + 1
    version = f"v{next_version:03d}"
    target = folder / version
    target.mkdir(parents=True, exist_ok=False)
    manifest = clean({**manifest, "review_version": version,
                      "previous_version": latest.get("version", ""), "changed_from_previous": bool(latest)})
    facts = clean(facts)
    _write(target / "facts.json", facts)
    _write(target / "manifest.json", manifest)
    (target / "report.md").write_text(render(facts, manifest), encoding="utf-8")
    _write(target / "report.json", compact(facts, manifest))
    pointer = {"version": version, "path": str(target), "input_hash": manifest["input_hash"], "as_of": day,
               "known_at": manifest["known_at"], "stage": manifest["stage"]}
    _write(folder / "latest.json", pointer)
    return {**pointer, "reused": False}


def compact(facts, manifest):
    """What the page needs; the full facts stay in facts.json for drill-down."""
    b, l, t, a, y, m = facts["boards"], facts["ladder"], facts["themes"], facts["activity"], facts["yesterday_pool"], facts["market"]
    return {"version": VERSION, "as_of": facts["as_of"], "previous_day": facts["previous_day"], "known_at": facts["known_at"],
            "review_version": manifest.get("review_version"), "stage": manifest["stage"], "gaps": manifest["gaps"],
            "coverage": facts["coverage"], "domain": {"id": DOMAIN["id"], "label": DOMAIN["label"]},
            "market": {"index": m["index"], "breadth": m["breadth"], "context": m["context"], "series_20d": m["series_20d"]},
            "boards": {"counts": b["counts"], "by_board": b["by_board"], "seal_failure_rate": b["seal_failure_rate"],
                       "open_rate": b["open_rate"], "reconciliation": {k: {kk: vv for kk, vv in v.items() if kk not in ("local_only", "reference_only")} | {"local_only_n": len(v.get("local_only", [])), "reference_only_n": len(v.get("reference_only", []))} for k, v in b["reconciliation"].items()},
                       "U": b["U"][:100], "Z": b["Z"][:100], "D": b["D"][:100]},
            "yesterday_pool": {k: y[k] for k in ("basis", "n_pool", "n_valid", "excluded", "open_vs_pre", "close_vs_pre", "close_vs_open", "high_vs_pre", "continued_U")},
            "ladder": {"by_height": l["by_height"], "max_height": l["max_height"], "heights": dict(list(l["heights"].items())[:40]),
                       "promotion_from_previous": {h: {k: v for k, v in p.items() if k != "broke"} | {"broke_n": len(p["broke"])} for h, p in l["promotion_from_previous"].items()}},
            "themes": {"top_by_limit_up": t["top_by_limit_up"], "top_by_amount_share": t["top_by_amount_share"][:10],
                       "top_by_limit_share": t["top_by_limit_share"],
                       "chains": t["chains"], "overlap_of_limit_ups": t["overlap_of_limit_ups"], "notes": t["notes"]},
            "activity": {"top20": a["top20"], "top20_share": a["top20_share"], "close_location_buckets": a["close_location_buckets"],
                         "scatter": a["scatter"][:250], "notes": a["notes"]},
            "loss_effects": {"events": {k: {kk: vv for kk, vv in v.items() if kk != "symbols"} | {"symbols": v["symbols"][:12]} for k, v in facts["loss_effects"]["events"].items()},
                             "notes": facts["loss_effects"]["notes"]},
            "lists": facts["lists"], "sections": manifest["sections"],
            "p5_limit_up": facts.get("p5_limit_up", {}), "p5_limit_up_feedback": facts.get("p5_limit_up_feedback", []),
            "sentiment": {"window_days": facts.get("sentiment", {}).get("window_days", []),
                          "metrics": {k: {kk: vv for kk, vv in v.items() if kk != "series"} | {"series": v.get("series", [])[-20:]}
                                      for k, v in facts.get("sentiment", {}).get("metrics", {}).items()},
                          "readings": facts.get("sentiment", {}).get("readings", ""), "note": facts.get("sentiment", {}).get("note", "")}}


def _pct(x):
    return "--" if x is None else f"{x:.2f}%"


def _ratio_text(r):
    """Percent with its fraction; money fractions read in 亿, counts stay counts."""
    if r.get("value") is None:
        return NO_SAMPLE
    n, d = r["numerator"], r["denominator"]
    if isinstance(d, float) and d > 1e6:
        return f"{r['value']:.1%}（{n / 1e8:.1f} 亿 / {d / 1e8:.1f} 亿）"
    return f"{r['value']:.1%}（{n}/{d}）"


LABELS = {
    "kind": {"U": "收盘涨停", "Z": "炸板", "D": "收盘跌停", "DZ": "触跌停未封死"},
    "board": {"MAIN": "主板", "CHINEXT": "创业板", "STAR": "科创板"},
    "pos": {"HIGH": "高位", "MID": "中位", "LOW": "低位"},
    "loss": {"yesterday_pool_negative": "昨日涨停股今日收跌", "limit_down": "收盘跌停", "seal_failed": "炸板",
             "gap_up_reversal": "高开低走", "breakdown_20d": "跌破20日低点"},
    "structure": {"RANGE": "震荡", "BULL": "上行", "BEAR": "下行", "TRANSITION": "过渡"},
    "temperature": {"NORMAL": "正常", "HOT": "过热", "COLD": "偏冷", "OVERHEATED": "过热"},
    "state": {"GREEN": "绿灯（常态）", "ORANGE": "橙灯（警戒）", "RED": "红灯（避险）"},
    "basis": {"HISTORICAL_PREVIOUS_REVIEW": "取自前一日归档", "FROZEN_PREVIOUS_REVIEW": "取自前一日冻结记录",
              "RECONSTRUCTED_FROM_BARS_NOT_FROZEN": "由日线重建（非冻结）"},
    "recon": {"RECONCILED": "已对齐", "RECONCILIATION_PENDING": "有差异，待逐股核对", "REFERENCE_UNAVAILABLE": "无参考数据"},
    "excluded": {"ST": "ST", "NOT_TRADED_OR_MISSING": "缺当日行情", "SUSPENDED": "停牌", "LONG_SUSPENDED": "连续两日以上无行情（长期停牌/退市）", "INVALID_ROW": "无效行"},
    "pool_excluded": {"SUSPENDED_OR_MISSING": "停牌或缺数据", "CORPORATE_ACTION_PENDING": "公司行动待核", "MISSING": "缺数据"},
    "bucket": {"strong": "强收盘", "mid": "中性", "weak": "弱收盘"},
}
GLOSSARY = ("收盘涨停＝收盘价等于涨停价；炸板＝盘中触及涨停价但收盘未封住；收盘跌停＝收盘价等于跌停价；"
            "触跌停未封死＝盘中触及跌停价但收盘高于跌停价。涨跌停价按主板 10%、创业板/科创板 20% 从前收推算，有交易所参考价时以参考价为准。")


def _lab(group, key):
    return LABELS.get(group, {}).get(key, key)


def _signed(x):
    return "--" if x is None else f"{x:+.2f}%"


def _pairs(mapping, group):
    return " / ".join(f"{_lab(group, k)} {v}" for k, v in (mapping or {}).items()) or "无"


def _dist_row(name, d):
    if not d or not d.get("n"):
        return f"| {name} | 无可比样本 | | | | | |"
    return f"| {name} | {d['n']} | {_signed(d['mean'])} | {_signed(d['median'])} | {d['up']} | {_signed(d['worst'])} | {_signed(d['best'])} |"


def render(facts, manifest):
    b, l, t, a, y, m, lo, li = (facts[k] for k in ("boards", "ladder", "themes", "activity", "yesterday_pool", "market", "loss_effects", "lists"))
    ctx = m.get("context") or {}
    idx = m.get("index") or {}
    lines = [f"# 盘后复盘 {facts['as_of']}（{manifest.get('review_version', '')}，事实层）", "",
             f"> 统计域：{DOMAIN['label']}；生成时间 {facts['known_at']}；数据完整度 {manifest['stage']}。",
             "> 这是事实层，不是结论：解释、假设、次日核对属 P1，未实施。", f"> 术语：{GLOSSARY}", "",
             "## 顶栏", "",
             f"- 覆盖：读到 {facts['coverage']['symbols_read']} 只，可核验 {facts['coverage']['in_domain']} 只，排除 {_pairs(facts['coverage']['excluded'], 'excluded')}，不可读 {facts['coverage']['unreadable']}",
             f"- 缺口：{'；'.join(manifest['gaps']) or '无'}",
             f"- 耗时（秒）：{manifest['timing']}", "",
             "## 市场与接力", "",
             (f"- 沪深300：今日 {_signed(idx.get('return_pct'))}，5 日 {_signed(idx.get('return5_pct'))}，20 日 {_signed(idx.get('return20_pct'))}" if idx else "- 沪深300：缺当日数据"),
             f"- 市场状态：{_lab('state', ctx.get('state'))}，结构 {_lab('structure', ctx.get('structure'))}，温度 {_lab('temperature', ctx.get('temperature'))}，"
             f"趋势类复核 {'允许' if ctx.get('trend_review_allowed') else '不允许'}，数据 {'新鲜' if ctx.get('data_status') == 'FRESH' else ctx.get('data_status')}",
             f"- 宽度：上涨 {m['breadth']['up']} / 下跌 {m['breadth']['down']} / 平 {m['breadth']['flat']}（共 {m['breadth']['domain']} 只）",
             f"- 涨跌停：收盘涨停 {b['counts']['U']}，炸板 {b['counts']['Z']}，收盘跌停 {b['counts']['D']}，触跌停未封死 {b['counts']['DZ']}",
             f"- 封板失败率（炸板 ÷ 触及涨停）：{_ratio_text(b['seal_failure_rate'])}；涨停股曾开板比例：{_ratio_text(b['open_rate']) if b['open_rate'].get('value') is not None else b['open_rate']['status']}", "",
             "| 板块 | 可核验只数 | 收盘涨停 | 炸板 | 收盘跌停 | 触跌停未封死 |", "|---|---|---|---|---|---|"]
    lines += [f"| {_lab('board', bd)} | {v.get('domain', 0)} | {v.get('U', 0)} | {v.get('Z', 0)} | {v.get('D', 0)} | {v.get('DZ', 0)} |" for bd, v in b["by_board"].items()]
    lines += ["", "与 Tushare 参考名单逐股对账（两边都不含 ST；差异只列出供核对）：", "",
              "| 类型 | 本地计算 | 参考名单 | 两边都有 | 仅本地 | 仅参考 | 状态 |", "|---|---|---|---|---|---|---|"]
    for k, v in b["reconciliation"].items():
        if v.get("local") is None:
            lines.append(f"| {_lab('kind', k)} | {_lab('recon', v.get('status'))} | | | | | |")
        else:
            lines.append(f"| {_lab('kind', k)} | {v['local']} | {v['reference']} | {v['both']} | {len(v.get('local_only', []))} | {len(v.get('reference_only', []))} | {_lab('recon', v.get('status'))} |")
    p5 = facts.get("p5_limit_up", {})
    if p5:
        lines += ["", "## P5 入池后涨停", "",
                  f"- 收盘封板 {p5['closing_limit_up_count']} 只，P5 交集 {len(p5['rows'])} 只，先入池后首板重点 {p5['focus_count']} 只。只观察，不改变风险许可。"]
        if p5["unverified_dates"]:
            lines.append("- P5 档案未通过时点核验：" + "、".join(p5["unverified_dates"]) + "；不能把未核日期当成没有入池。")
        lines += ["", "| 分类 | 股票 | 证据 | 状态 | 边界 |", "|---|---|---|---|---|"]
        lines += ["| " + " | ".join(r[i] for i in (0, 1, 3, 6, 7)) + " |"
                  for r in FW.limit_up_table_rows(p5, facts.get("p5_limit_up_feedback", []))] or ["| 暂无可确认交集 | | | | |"]
    lines += ["", "### 昨日涨停股今日表现", "",
              f"- 昨日收盘涨停 {y['n_pool']} 只（母池{_lab('basis', y['basis'])}），今日可核验 {y['n_valid']} 只"
              + (f"，未核验：{_pairs(y['excluded'], 'pool_excluded')}" if y.get("excluded") else "") + f"；今日继续封板 {y['continued_U']} 只。日内最高对前收是事后幅度，不是可实现收益。", "",
              "| 口径 | 样本 | 均值 | 中位 | 上涨只数 | 最差 | 最好 |", "|---|---|---|---|---|---|---|",
              _dist_row("开盘价对昨收", y["open_vs_pre"]), _dist_row("收盘价对昨收", y["close_vs_pre"]),
              _dist_row("收盘价对开盘价", y["close_vs_open"]), _dist_row("日内最高对昨收（事后）", y["high_vs_pre"]), "",
              "### 近 20 个交易日", "", "| 日期 | 收盘涨停 | 炸板 | 收盘跌停 | 可核验只数 |", "|---|---|---|---|---|"]
    lines += [f"| {s['day']} | {s['U']} | {s['Z']} | {s['D']} | {s['domain']} |" for s in m["series_20d"]]
    se = facts.get("sentiment") or {}
    lines += ["", "## 接力情绪（每项与自身近 20 日比较，不合成总分）", "", f"- {se.get('note', '')}", f"- 读法：{se.get('readings', '')}", "",
              "| 指标 | 今日 | 近20日中位 | 近20日最低–最高 | 今日分位 | 读法 |", "|---|---|---|---|---|---|"]
    for name, mt in (se.get("metrics") or {}).items():
        if mt.get("status") != "OK":
            lines.append(f"| {mt['label']} | {_sent_val(name, mt.get('today'))} | {NO_SAMPLE} | | | |")
        else:
            lines.append(f"| {mt['label']} | {_sent_val(name, mt['today'])} | {_sent_val(name, mt['median'])} | {_sent_val(name, mt['min'])}–{_sent_val(name, mt['max'])} | {mt['percentile']:.0%} | {mt['reading']} |")
    lines += ["", "## 连板梯队", "",
              f"- 今日高度分布：{'，'.join(f'{h} 板 {n} 只' for h, n in l['by_height'].items()) or '无涨停'}；最高 {l['max_height']} 板",
              f"- 最高板：{'、'.join(f'{s}（{h} 板）' for s, h in list(l['heights'].items())[:8]) or '无'}",
              "- 连板按市场日历连续计数，停牌或缺日打断。昨日梯队今日晋级（按高度）：", "",
              "| 昨日高度 | 昨日母池 | 今日可核验 | 继续封板（晋级） | 断板 | 晋级率 |", "|---|---|---|---|---|---|"]
    lines += [f"| {h} 板 | {p['pool']} | {p['denominator']} | {p['numerator']} | {len(p['broke'])} | {_ratio_text(p)} |" for h, p in l["promotion_from_previous"].items()]
    lines += ["", "## 题材地图（按完整成员表统计）", "", "按涨停只数（大概念占优）：", "",
              "| 题材 | 成员 | 上涨 | 涨停 | 站上MA20 | 广度（上涨/成员） | 成交额占比 |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {x['theme']} | {x['members']} | {x['up']} | {x['limit_up']} | {x['above_ma20']} | {_ratio_text(x['breadth'])} | {_ratio_text(x['amount_share'])} |" for x in t["top_by_limit_up"][:12]]
    lines += ["", "按涨停占成员比例（≥3 只涨停且成员 ≤300，只是展示过滤）：", "", "| 题材 | 成员 | 涨停 | 涨停占比 | 广度 | 成交额占比 |", "|---|---|---|---|---|---|"]
    lines += [f"| {x['theme']} | {x['members']} | {x['limit_up']} | {_ratio_text(x['limit_share'])} | {_ratio_text(x['breadth'])} | {_ratio_text(x['amount_share'])} |" for x in t["top_by_limit_share"][:10]]
    chains_text = "；".join(f"{c['theme'].replace('紧链:', '', 1)} 成员 {c['members']}，上涨 {c['up']}，涨停 {c['limit_up']}" for c in t["chains"]) or "无"
    lines += ["", f"- 登记的紧链题材：{chains_text}",
              f"- 涨停股所属题材数：{'，'.join(f'{k} 个题材 {v} 只' for k, v in t['overlap_of_limit_ups'].items()) or '无'}", f"- 注：{'；'.join(t['notes'])}", "",
              "## 活跃与风险", "", f"- 成交额前 20 只占全市场 {_ratio_text(a['top20_share'])}；收盘位置分布：{_pairs(a['close_location_buckets'], 'bucket')}。成交额是交易活跃度，不是净买入。",
              "", "| 股票 | 板块 | 成交额(亿) | 涨跌 | 收盘位置 | 相对换手 | 20日位置 | 涨跌停 |", "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {x['symbol']} | {_lab('board', x['board'])} | {x['amount'] / 1e8:.1f} | {_signed(x['pct_chg'])} | {x['close_location']} | {x['relative_turnover']} | {x['position20']} | {_lab('kind', x['limit']) if x['limit'] else ''} |" for x in a["top20"]]
    lines += ["", "- 亏钱事件分层（位置按 20 日区间：高位 ≥ 0.8，低位 ≤ 0.2）：", "", "| 事件 | 数量 | 按板块 | 按位置 | 样例 |", "|---|---|---|---|---|"]
    lines += [f"| {_lab('loss', k)} | {v['count']} | {_pairs(v['by_board'], 'board')} | {_pairs(v['by_position'], 'pos')} | {'、'.join(v['symbols'][:6])} |" for k, v in lo["events"].items()]
    lines += ["", "## 三类清单（不混用）", "", "### 风向标（机械提名，非买入推荐）", "", "| 股票 | 角色 | 观察问题 | 事实 |", "|---|---|---|---|"]
    lines += [f"| {p['symbol']} {p['name']} | {p['role_label']} | {p['question']} | {_facts_text(p['facts'])} |" for p in li["bellwethers"]] or ["| 无 | | | |"]
    lines += ["", f"### 持续候选池（{li['source']}）", "", "| 股票 | 题材 | 路径 | 状态 | 阻断 | 结论 | 强势# | 参与# | 今日事件 |", "|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {c['symbol']} {c.get('name') or ''} | {c.get('theme')} | {c.get('route_ids')} | {c.get('state')} | {'；'.join(c.get('blockers') or [])} | {c.get('verdict_reading')} | {c.get('strength_rank')} | {c.get('participation_rank')} | {c.get('continuity_event')} |" for c in li["continuing_candidates"]] or ["| 无 | | | | | | | | |"]
    lines += ["", f"### 参与复核清单：{li['review_note']}", ""]
    lines += [f"- {c['symbol']} 参与#{c.get('participation_rank')} {c.get('route_ids')}" for c in li["review_eligible"]]
    lines += ["", "## 持仓与执行", "", f"- 自动账（引用 books_daily.json）：{li.get('auto_book')}；人工账与三本账详情见每日交付第五节。", "",
              "## 昨日兑现 / 今日结论 / 明日计划", "", "- 未实施（P1）：本版只交付事实层；解释、假设与次日核对不在此版生成。", ""]
    return "\n".join(lines)


def _sent_val(name, v):
    if v is None:
        return "--"
    if name in ("seal_failure_rate", "pool_up_share", "promotion_1_to_2"):
        return f"{v:.1%}"
    if name in ("pool_mean_pct", "open_premium_pct"):
        return f"{v:+.2f}%"
    return f"{v:g}"


def _facts_text(facts):
    parts = []
    for k, v in (facts or {}).items():
        parts.append(f"{v} 板" if k == "height" else f"20日 {_signed(v)}" if k == "return20_pct" else f"今日 {_signed(v)}" if k == "close_vs_pre"
                     else f"题材 {v}" if k == "theme" else f"涨停 {v} 只" if k == "limit_up" else _lab("kind", v) if k == "limit" else f"{k} {v}")
    return "，".join(parts)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out-root", default=str(OUT_ROOT))
    parser.add_argument("--no-tushare", action="store_true")
    args = parser.parse_args()
    folder = Path(args.out_root) / args.date.replace("-", "") / "eod_review"
    attempted_at = datetime.now().astimezone().isoformat(timespec="seconds")
    _write(folder / "status.json", {"as_of": args.date, "status": "RUNNING", "attempted_at": attempted_at})
    try:
        facts, manifest = build(args.date, args.data_root, args.out_root, use_tushare=not args.no_tushare)
        pointer = persist(args.date, facts, manifest, args.out_root)
        facts = read_json(Path(pointer["path"]) / "facts.json")
        manifest = read_json(Path(pointer["path"]) / "manifest.json")
        _write(folder / "status.json", {"as_of": args.date, "status": "SUCCESS" if pointer["stage"] == "REVIEW_READY" else "PARTIAL",
                                       **pointer, "attempted_at": attempted_at, "gaps": manifest["gaps"], "timing": manifest["timing"]})
        print(f"[eod_review] {args.date} {pointer['version']} reused={pointer['reused']} stage={pointer['stage']} "
              f"U={facts['boards']['counts']['U']} Z={facts['boards']['counts']['Z']} D={facts['boards']['counts']['D']} "
              f"gaps={len(manifest['gaps'])} timing={manifest['timing']}", flush=True)
        p5 = facts.get("p5_limit_up", {})
        if p5:
            print(f"[eod_review] P5+U matches={len(p5['rows'])} first_board_focus={p5['focus_count']} "
                  f"status={p5['status']} (observation only)", flush=True)
    except Exception as exc:  # the review must never block the main pipeline or hide its own failure
        _write(folder / "status.json", {"as_of": args.date, "status": "DATA_WAIT" if str(exc).startswith("DATA_WAIT:") else "FAILED",
                                       "attempted_at": attempted_at, "error": str(exc)[:500]})
        print(f"[eod_review] FAILED {args.date}: {exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
