"""Dated security profile for paper matching. The adaptation scope is stated, not assumed."""
from functools import lru_cache
from pathlib import Path
import pandas as pd
from .. import data as D
from . import store

VERSION = "security_master_v1_20260920"
LOT_SIZE = 100
# Paper matching is enabled for ordinary main-board names only; everything else is observed.
PAPER_SCOPE = "ORDINARY_MAIN_BOARD_ONLY"
NO_LIMIT_SESSIONS = {"STAR": 5, "CHINEXT": 5, "MAIN": 1}
MASTER_BASIS = "CURRENT_SNAPSHOT_NOT_POINT_IN_TIME"


def board(symbol):
    return "STAR" if symbol.startswith("sh_68") else "CHINEXT" if symbol.startswith("sz_30") else "MAIN"


@lru_cache(maxsize=1)
def _listing():
    path = Path(store.ROOT) / "data/stock_list.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path, dtype=str)
    out = {}
    for row in frame.to_dict("records"):
        code = str(row.get("ts_code", ""))
        if "." not in code:
            continue
        exchange, number = code.split(".", 1)
        symbol = f"{exchange}_{number}"
        raw = str(row.get("list_date", "") or "")
        listed = f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}" if len(raw) == 8 else ""
        out[symbol] = {"name": row.get("name", ""), "list_date": listed,
                       "trade_status": str(row.get("tradeStatus", "") or "")}
    return out


def _sessions_between(calendar, start, end):
    return len([d for d in calendar if start < d <= end])


def profile(symbol, day, calendar=None, bars=None):
    """Lot size, limit band, ST, listing age, halt suspicion and whether paper matching may touch it."""
    listing = _listing().get(symbol, {})
    name = listing.get("name") or D.name_map().get(symbol, "")
    is_st = ("ST" in name.upper()) or ("退" in name)
    kind = board(symbol)
    base_limit = D.limit_pct(symbol)
    limit = 5.0 if (is_st and kind == "MAIN") else base_limit
    listed = listing.get("list_date", "")
    early = False
    if listed and calendar:
        early = _sessions_between(calendar, listed, day) < NO_LIMIT_SESSIONS[kind]
    halt = "UNKNOWN"
    if bars is not None and calendar and day in calendar:
        halt = "SUSPECTED_HALT_OR_MISSING" if day not in set(bars) else "TRADED"
    in_scope = kind == "MAIN" and not is_st and not early and halt != "SUSPECTED_HALT_OR_MISSING"
    return {"version": VERSION, "symbol": symbol, "name": name, "day": day, "board": kind,
            "lot_size": LOT_SIZE, "limit_pct": limit, "base_limit_pct": base_limit, "is_st": is_st,
            "list_date": listed, "early_listing_no_limit": early, "halt": halt,
            "paper_scope": PAPER_SCOPE, "in_paper_scope": in_scope,
            "master_basis": MASTER_BASIS,
            "notes": ["主数据为当前快照，名称与上市日不按历史时点回溯；缺日不单独作为停牌证据。"]}


def scope_report(symbols, day, calendar=None):
    rows = [profile(s, day, calendar) for s in symbols]
    counts = {}
    for row in rows:
        key = row["board"] + ("_ST" if row["is_st"] else "")
        counts[key] = counts.get(key, 0) + 1
    return {"version": VERSION, "day": day, "scope": PAPER_SCOPE,
            "in_scope": sum(r["in_paper_scope"] for r in rows), "total": len(rows), "by_board": counts,
            "rows": rows}
