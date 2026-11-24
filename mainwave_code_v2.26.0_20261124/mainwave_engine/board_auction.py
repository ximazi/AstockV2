"""Opening-auction alerts for an immutable prior-session watch. No orders."""

import argparse
from datetime import date, datetime
from pathlib import Path
from statistics import median
import time

import pandas as pd

from . import board_relay as W, eod_review as E, intraday_quotes as Q
from .review_clock import CST, instant

VERSION = "board_auction_v1_20261124"
WINDOW = ("09:26", "09:30")
FIELDS = "ts_code,trade_date,vol,price,amount,pre_close"


def _compact(day):
    return day.replace("-", "")


def frozen(root, prev, target):
    before = W.clock(target, "09:15")
    for path in sorted((W.folder(root, prev) / "forward").glob("*.json"), reverse=True):
        out = E.read_json(path)
        stamp = instant(out.get("published_at"))
        forecast = out.get("target") or ""
        # A holiday forecast can roll only when the caller checks the official previous-session identity.
        planned = forecast == target or (out.get("target_basis") == "PROVISIONAL_WEEKDAY" and prev < forecast < target)
        if (out.get("version") == W.VERSION and out.get("mode") == "FORWARD" and out.get("as_of") == prev
                and planned and stamp and W.clock(prev, "15:00") <= stamp < min(before, W.clock(forecast, "09:15"))
                and out.get("status") in ("READY", "PARTIAL") and out.get("orders_enabled") is False):
            return out, path
    return {}, None


def focus_symbols(watch):
    return set(watch["focus"].get("FIRST", [])) | set(watch["focus"].get("RELAY", []))


def snapshot_rows(frame, day):
    if not set(FIELDS.split(",")).issubset(frame.columns) or frame.empty or len(frame) >= 8000:
        return {}
    result, duplicate, seen = {}, set(), set()
    for row in frame.to_dict("records"):
        code = str(row["ts_code"])
        if str(row["trade_date"]) != _compact(day) or not code.endswith((".SH", ".SZ")):
            continue
        parts = code.split(".")
        if len(parts) != 2 or len(parts[0]) != 6 or not parts[0].isascii() or not parts[0].isdigit():
            continue
        symbol = Q.symbol_of(code)
        if symbol in seen:
            duplicate.add(symbol)
        seen.add(symbol)
        price, previous, amount, vol = (E.number(row[k]) for k in ("price", "pre_close", "amount", "vol"))
        if any(v is None or v <= 0 for v in (price, previous, amount, vol)):
            continue
        if abs(amount / vol - price) > max(.02, price * .005):
            continue
        result[symbol] = {"price": price, "preclose": previous, "amount": amount,
                          "gap_pct": (price / previous - 1) * 100}
    return {s: r for s, r in result.items() if s not in duplicate}


def compare(watch, quotes, notices=None):
    notices = notices or {}
    rows = {r["symbol"]: r for r in watch["rows"]}
    outcomes = []
    for s in sorted(focus_symbols(watch)):
        r, q = rows[s], quotes.get(s, {})
        p, raw = r["price"], q.get("price")
        status, metrics = "DATA_WAIT", {}
        issuer = notices.get(s) or {}
        old = r.get("notices", [])
        risk = [n["kind"] for n in old + issuer.get("notices", [])]
        if "REGULATORY_REVIEW" in risk:
            status = "REGULATORY_REVIEW"
        elif q and abs(q["preclose"] - p["close"]) < .011:
            gap = q["gap_pct"]
            share = q["amount"] / p["amount"] * 100
            group = r.get("peer_symbols", [])
            values = [quotes[x]["gap_pct"] for x in group if x in quotes and x in rows and
                      rows[x].get("lane") == r["lane"] and rows[x].get("board") == r.get("board") and
                      abs(quotes[x]["preclose"] - rows[x].get("price", {}).get("close", -1)) < .011]
            peer = median(values) if values else None
            pct = E.DOMAIN["limit_pct"].get(E.board_of(s))
            up = E.limit_price(q["preclose"], pct) if pct is not None else None
            # Prior weakness is observed (failed prior seal attempts or a below-preclose excursion), not inferred from today's gap.
            weak_yesterday = (r.get("open_times") or 0) > 0 or p.get("gap_intraday_low_pct", 0) < 0
            condition = 2 <= round(gap, 4) <= 5 and 5 <= round(share, 4) <= 10
            aligned = condition and peer is not None and gap > peer and peer > 0
            metrics = {"auction_price": raw, "auction_amount": q["amount"], "gap_pct": round(gap, 4),
                       "amount_share_pct": round(share, 4), "peer_median_gap_pct": peer, "peer_count": len(values),
                       "test_band_hit": condition, "weak_yesterday_observed": weak_yesterday}
            if up is not None and raw >= up - .005:
                status = "AT_LIMIT_FILL_UNPROVEN"
            elif raw < p["low"]:
                status = "STRUCTURE_LOST"
            elif issuer.get("status") != "AVAILABLE":
                status = "NOTICES_DATA_WAIT"
            elif "RISK_ALERT" in risk or "CLARIFICATION" in risk:
                status = "NEWS_RISK_WATCH"
            elif aligned:
                status = "WEAK_TO_STRONG_WATCH" if weak_yesterday else "CONTINUATION_WATCH"
            elif condition and peer is None:
                status = "PEERS_DATA_WAIT"
            else:
                status = "WAIT_OPEN"
        elif q:
            status = "REFERENCE_MISMATCH"
        outcomes.append({"symbol": s, "name": r["name"], "lane": r["lane"], "state": status,
                         "theme": r.get("theme"), "structure_floor": p.get("low"), "risk": risk,
                         "notice_status": issuer.get("status", "DATA_WAIT"), **metrics})
    return outcomes


def calendar_previous(client, day, deadline):
    f = client.query("trade_cal", fields="cal_date,is_open,pretrade_date", deadline=deadline,
                     exchange="SSE", start_date=_compact(day), end_date=_compact(day))
    if len(f) != 1 or str(f.iloc[0].get("cal_date")) != _compact(day):
        return None, "CALENDAR_DATA_WAIT"
    r = f.iloc[0]
    if str(r.get("is_open")) != "1":
        return None, "NOT_TRADING_DAY"
    try:
        return datetime.strptime(str(r["pretrade_date"]), "%Y%m%d").date().isoformat(), None
    except (ValueError, KeyError):
        return None, "CALENDAR_DATA_WAIT"


def prepare_notices(day, root, data_root, now=None, collector=None):
    from .limit_theme.sources import Collector

    now = now or datetime.now(CST)
    loc = W.folder(root, day)
    prev = next((s for s in reversed(E.calendar(data_root)) if s < day), None)
    watch, path = frozen(root, prev, day) if prev else ({}, None)
    payload = {"as_of": day, "orders_enabled": False, "rows": {},
               "watch_sha256": E.file_sha(path) if path else None, "status": "NO_FROZEN_WATCH"}
    if watch and now.date().isoformat() == day and W.clock(day, "09:00") <= now < W.clock(day, "09:25"):
        source = collector or Collector(loc, day, now=now)
        for s in sorted(focus_symbols(watch)):
            try:
                record = source.announcements(s[3:])
                data = record.get("data") or {}
                seen = instant(record.get("collected_at"))
                available = (record.get("status") == "AVAILABLE" and not data.get("truncated") and seen
                             and W.clock(day, "09:00") <= seen <= now and isinstance(data.get("records"), list))
                materials = []
                for r in data["records"] if available else []:
                    disclosed = pd.to_datetime(r.get("announcementTime"), unit="ms", utc=True, errors="coerce")
                    if pd.notna(disclosed) and disclosed.tz_convert(CST).date().isoformat() >= prev:
                        materials.append({"kind": "ISSUER_TITLE", "title": r.get("announcementTitle"),
                                          "url": "https://www.cninfo.com.cn/" + str(r.get("adjunctUrl") or "").lstrip("/"),
                                          "collected_at": seen.isoformat()})
                payload["rows"][s] = {"status": "AVAILABLE" if available else "DATA_WAIT",
                                      "notices": W.watch_notices(materials, now)}
            except (ValueError, TypeError, OSError, AttributeError):
                payload["rows"][s] = {"status": "DATA_WAIT", "notices": []}
        payload["status"] = "READY" if all(r["status"] == "AVAILABLE" for r in payload["rows"].values()) else "PARTIAL"
    elif watch:
        payload["status"] = "OUTSIDE_NOTICE_WINDOW"
    payload["published_at"] = (now if collector else datetime.now(CST)).isoformat()
    # A late slow request is not eligible for the opening alert.
    if instant(payload["published_at"]) >= W.clock(day, "09:25"):
        payload["status"] = "OUTSIDE_NOTICE_WINDOW"
        payload["rows"] = {}
    E._write(loc / "overnight.json", payload)
    return payload


def run(day, root=E.OUT_ROOT, data_root="data", client=None, now=None, wait=False):
    get_now = now or (lambda: datetime.now(CST))
    start, end = (W.clock(day, t) for t in WINDOW)
    at = get_now()
    out = {"as_of": day, "version": VERSION, "status": "OUTSIDE_WINDOW", "mode": "OBSERVATION",
           "orders_enabled": False, "rows": [], "attempted_at": at.isoformat()}
    if wait and W.clock(day, "09:15") <= at < start:
        if at < W.clock(day, "09:23"):
            prepare_notices(day, root, data_root)
            at = get_now()
        if at < start:
            time.sleep((start - at).total_seconds())
        at = get_now()
    if not start <= at < end:
        return out
    loc = W.folder(root, day)
    valid = E.read_json(loc / "auction.json")
    if valid.get("as_of") == day and valid.get("status") == "READY":
        return valid
    deadline = time.monotonic() + (end - at).total_seconds()
    try:
        client = client or Q.Client()
        prev, issue = calendar_previous(client, day, deadline)
    except (Q.ApiError, OSError, ValueError) as exc:
        return {**out, "status": "CALENDAR_DATA_WAIT", "source_error": getattr(exc, "code", type(exc).__name__)}
    if issue:
        return {**out, "status": issue}
    if not prev or prev >= day or prev not in E.calendar(data_root):
        return {**out, "status": "REFERENCE_CALENDAR_STALE"}
    watch, source = frozen(root, prev, day)
    if not watch:
        return {**out, "status": "NO_FROZEN_WATCH", "previous_day": prev}
    out.update(previous_day=prev, watch_path=str(source), watch_sha256=E.file_sha(source))
    targets = focus_symbols(watch)
    if not targets:
        return {**out, "status": "EMPTY_WATCH"}
    overnight = E.read_json(loc / "overnight.json")
    seen = instant(overnight.get("published_at"))
    notices = overnight.get("rows", {}) if (overnight.get("as_of") == day and
        overnight.get("watch_sha256") == out["watch_sha256"] and seen and W.clock(day, "09:00") <= seen < W.clock(day, "09:25")) else {}
    quotes = {}
    attempts = 3 if wait else 1
    for n in range(attempts):
        try:
            frame = client.query("stk_auction", fields=FIELDS, deadline=deadline, trade_date=_compact(day))
        except (Q.ApiError, OSError, ValueError) as exc:
            code = getattr(exc, "code", type(exc).__name__)
            return {**out, "status": "PERMISSION_DENIED" if str(code) == "40203" else "SOURCE_UNAVAILABLE",
                    "source_error": code, "rows": compare(watch, {}, notices)}
        at = get_now()
        if at < start or at >= end:
            return {**out, "status": "LATE_DATA", "received_at": at.isoformat(), "rows": compare(watch, {}, notices)}
        quotes = snapshot_rows(frame, day)
        if targets <= quotes.keys() or n == attempts - 1 or (end - at).total_seconds() <= 10:
            break
        time.sleep(5)
    results = compare(watch, quotes, notices)
    ready = targets <= quotes.keys() and all(r["state"] not in {"DATA_WAIT", "NOTICES_DATA_WAIT",
                                                              "PEERS_DATA_WAIT", "REFERENCE_MISMATCH"} for r in results)
    alerts = [r["symbol"] for r in results if r["state"] == "WEAK_TO_STRONG_WATCH"]
    return {**out, "status": "READY" if ready else "PARTIAL" if quotes else "DATA_WAIT", "alerts": alerts,
            "source": "stk_auction", "received_at": at.isoformat(),
            "quote_coverage": {"expected": len(targets), "received": len(targets & quotes.keys())}, "rows": results}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date", required=True, type=lambda s: date.fromisoformat(s).isoformat())
    p.add_argument("--out-root", default=str(E.OUT_ROOT))
    p.add_argument("--data-root", default="data")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--watch", action="store_true")
    mode.add_argument("--prepare", action="store_true")
    a = p.parse_args()
    if a.prepare:
        out = prepare_notices(a.date, a.out_root, a.data_root)
    else:
        out = run(a.date, a.out_root, a.data_root, wait=a.watch)
        loc = W.folder(a.out_root, a.date)
        # An out-of-window manual check cannot replace the opening record.
        if out["status"] != "OUTSIDE_WINDOW":
            attempt = loc / "auction_attempts" / (datetime.now(CST).strftime("%Y%m%d_%H%M%S_%f") + ".json")
            E._write(attempt, out)
            E._write(loc / "auction.json", out)
            if out.get("watch_sha256") and out.get("previous_day"):
                E._write(W.folder(a.out_root, out["previous_day"]) / "auction_target.json", {"as_of": a.date})
    print(f"[board_auction] {a.date} {out['status']} -> {W.folder(a.out_root, a.date)}", flush=True)
    for r in out["rows"] if not a.prepare else []:
        print(f"  {r['symbol']} {r['lane']} {r['state']} gap={r.get('gap_pct', '--')} amount_pct={r.get('amount_share_pct', '--')}", flush=True)
    return 0 if out["status"] in {"READY", "PARTIAL", "EMPTY_WATCH", "NOT_TRADING_DAY"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
