"""Timestamped Tushare observations; minute bars are never daily snapshots."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from collections import deque
import json
import math
import os
import time

import pandas as pd
import requests

CHINA = timezone(timedelta(hours=8))
COLLECTION_START = "09:30"
COLLECTION_END = "16:00"
LUNCH = ("11:31", "13:00")  # no fresh quotes can exist in the midday break; runs there are refused, watch loops wait


def in_collection_window(at):
    """Same-day collection is allowed 09:30-11:30 and 13:00-16:00 China time (after 15:00 is the closing catch-up)."""
    hm = stamp(at).strftime("%H:%M:%S")
    return COLLECTION_START <= hm < COLLECTION_END and not (LUNCH[0] <= hm < LUNCH[1])
INDEX_CODES = ("000001.SH", "399001.SZ", "399006.SZ", "000300.SH")
SNAPSHOT_FIELDS = "ts_code,name,pre_close,open,high,low,close,vol,amount,trade_time"


def now_china():
    return datetime.now(CHINA)


def stamp(value):
    t = pd.Timestamp(value)
    if pd.isna(t):
        raise ValueError("Missing timestamp")
    return t.tz_localize(CHINA) if t.tzinfo is None else t.tz_convert(CHINA)


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def ts_code(symbol):
    exchange, code = symbol.split("_")
    return f"{code}.{exchange.upper()}"


def symbol_of(code):
    code, exchange = code.split(".")
    return f"{exchange.lower()}_{code}"


def session_minutes(t):
    minute = stamp(t).hour * 60 + stamp(t).minute
    return min(120, max(0, minute - 570)) + min(120, max(0, minute - 780))


class ApiError(RuntimeError):
    def __init__(self, api, code, message):
        super().__init__(f"{api}: {code} {message}")
        self.code = code
        self.daily_limit = any(s in message.lower() for s in ("次/天", "次/日", "daily limit", "daily quota"))
        self.rate_limited = not self.daily_limit and (code == 429 or any(
            s in message.lower() for s in ("频率超限", "频次超限", "rate limit", "too many requests")))


class Client:
    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv()
        self.token = os.environ.get("TUSHARE_TOKEN", "")
        if not self.token:
            raise ValueError("TUSHARE_TOKEN is not configured")
        self.session = requests.Session()
        self.last_call = 0.
        self.audit = []
        self.denied = {}
        self.minute_calls = deque(maxlen=5)
        self.cooldown_until = {}
        self.daily_exhausted = {}

    def query(self, api, fields="", deadline=None, **params):
        if api in self.denied:
            raise self.denied[api]
        if api in self.daily_exhausted:
            day, error = self.daily_exhausted[api]
            if day == now_china().date():
                raise error
            del self.daily_exhausted[api]
        now = time.monotonic()
        ready = max(self.last_call + 1.3, self.cooldown_until.get(api, 0))
        # This account's observed rt_min limit is 5 requests/minute, not 50.
        if api == "rt_min" and len(self.minute_calls) == 5:
            ready = max(ready, self.minute_calls[0] + 61)
        if deadline is not None and max(now, ready) >= deadline:
            raise ApiError(api, "BUDGET", "collection budget or window end reached before next allowed request")
        if ready - now >= 2:
            print(f"[intraday_quotes] {api} wait {ready - now:.1f}s; then resume pending request", flush=True)
        time.sleep(max(0, ready - now))
        self.last_call = time.monotonic()
        if deadline is not None and self.last_call >= deadline:
            raise ApiError(api, "BUDGET", "collection deadline reached")
        if api == "rt_min":
            self.minute_calls.append(self.last_call)
        try:
            response = self.session.post("https://api.tushare.pro", json={
                "api_name": api, "token": self.token, "params": params, "fields": fields}, timeout=(3, 12))
            response.raise_for_status()
            payload = response.json()
            if payload.get("code") != 0:
                raise ApiError(api, payload.get("code"), str(payload.get("msg", "")).replace(self.token, "[redacted]"))
            data = payload.get("data") or {}
            frame = pd.DataFrame(data.get("items") or [], columns=data.get("fields") or [])
            self.audit.append({"api": api, "status": "OK", "rows": len(frame)})
            return frame
        except (requests.RequestException, ValueError, ApiError) as exc:
            self.audit.append({"api": api, "status": "FAILED", "error": str(exc).replace(self.token, "[redacted]")})
            if isinstance(exc, ApiError):
                if exc.daily_limit:
                    self.daily_exhausted[api] = (now_china().date(), exc)
                elif exc.rate_limited:
                    self.cooldown_until[api] = time.monotonic() + 61
                elif exc.code == 40203:
                    self.denied[api] = exc
                raise
            raise ApiError(api, getattr(exc, "code", "TRANSPORT"), str(exc).replace(self.token, "[redacted]")) from exc


def normalize(frame, source, at, max_age_seconds=120):
    """Reject wrong-day, stale, future, malformed and nontrading observations."""
    at = stamp(at)
    good, rejected = {}, []
    minute = source == "rt_min"
    for row in frame.to_dict("records"):
        code = str(row.get("ts_code", row.get("code", "")))
        reason = ""
        try:
            observed = stamp(row.get("time") if minute else row.get("trade_time"))
            age = (at - observed).total_seconds()
            if observed.date() != at.date():
                reason = "WRONG_DAY"
            elif age < 0:
                reason = "FUTURE_QUOTE"
            elif age > max_age_seconds:
                reason = "STALE_QUOTE"
            elif not ("09:30" <= observed.strftime("%H:%M") <= "11:30" or "13:00" <= observed.strftime("%H:%M") <= "15:00"):
                reason = "OUTSIDE_SESSION"
            prices = {k: number(row.get(k)) for k in ("open", "high", "low", "close")}
            if any(v is None or v <= 0 for v in prices.values()):
                reason = reason or "INVALID_PRICE"
            elif not (prices["low"] <= min(prices["open"], prices["close"]) <= max(prices["open"], prices["close"]) <= prices["high"]):
                reason = reason or "INVALID_OHLC"
            if not code or "." not in code:
                reason = reason or "INVALID_CODE"
            volume, amount = number(row.get("vol")), number(row.get("amount"))
            if volume is None or amount is None or volume < 0 or amount < 0:
                reason = reason or "INVALID_VOLUME"
            elif source == "rt_k" and (volume == 0 or amount == 0):
                reason = reason or "NO_TRADES"
            elif source == "rt_k" and not prices.get("low") is None and not prices.get("high") is None:
                tolerance = max(.02, prices["high"] * .005)
                if not prices["low"] - tolerance <= amount / volume <= prices["high"] + tolerance:
                    reason = reason or "VOLUME_AMOUNT_UNIT_MISMATCH"
            if reason:
                rejected.append({"ts_code": code, "reason": reason})
                continue
            result = {"ts_code": code, "source": source, "observed_at": observed.isoformat(),
                      "age_seconds": age, "close": prices["close"], "quality": "PRICE_ONLY" if minute else "DAILY_SNAPSHOT",
                      "pre_close": None if minute else number(row.get("pre_close")),
                      "open": None if minute else prices["open"], "high": None if minute else prices["high"],
                      "low": None if minute else prices["low"], "volume": None if minute else volume,
                      "amount": None if minute else amount, "vwap": None if minute or not volume else amount / volume}
            if not minute and (result["pre_close"] is None or result["pre_close"] <= 0):
                rejected.append({"ts_code": code, "reason": "MISSING_PRE_CLOSE"})
                continue
            if code not in good or result["observed_at"] > good[code]["observed_at"]:
                good[code] = result
        except (TypeError, ValueError):
            rejected.append({"ts_code": code, "reason": "MISSING_OR_INVALID_TIMESTAMP"})
    return good, rejected


def minute_context(frame, code, at):
    """Require a full morning-to-now sequence before claiming full-day VWAP."""
    empty = {"status": "INCOMPLETE", "full_day_vwap": None, "above_vwap_ratio": None,
             "late_return_pct": None, "late_drawdown_pct": None, "late_pattern": "WAIT_MINUTES", "reclaim_confirmed": False}
    if frame.empty or not {"time", "open", "high", "low", "close", "vol", "amount"}.issubset(frame.columns):
        return empty
    at = stamp(at)
    f = frame.rename(columns={"code": "ts_code"}).copy()
    if "ts_code" not in f or set(f.ts_code) != {code}:
        return empty
    times = pd.to_datetime(f.time, errors="coerce")
    if times.isna().any():
        return empty
    f["time"] = times.map(stamp)
    f = f[f.time.le(at) & f.time.dt.date.eq(at.date())].sort_values("time").drop_duplicates("time", keep="last")
    if f.empty:
        return empty
    # The optional 09:30/13:00 opening prints are retained in sums, not required bars.
    start = at.normalize()
    expected = pd.date_range(start + pd.Timedelta(hours=9, minutes=31), start + pd.Timedelta(hours=11, minutes=30), freq="min").union(
        pd.date_range(start + pd.Timedelta(hours=13, minutes=1), at.floor("min"), freq="min"))
    expected = expected[expected <= at]
    missing = len(expected.difference(pd.DatetimeIndex(f.time)))
    latest = f.time.iloc[-1]
    result = {**empty, "minute_count": len(f), "missing_minutes": missing, "observed_at": latest.isoformat()}
    numeric = f[["open", "high", "low", "close", "vol", "amount"]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not numeric.map(math.isfinite).all().all() or (numeric < 0).any().any():
        return result
    if missing or (at - latest).total_seconds() > 120 or f.time.iloc[0].strftime("%H:%M") > "09:31":
        return result
    if (numeric[["open", "high", "low", "close"]] <= 0).any().any() or numeric.vol.sum() <= 0:
        return result
    if (numeric.low > numeric[["open", "close"]].min(axis=1)).any() or (numeric.high < numeric[["open", "close"]].max(axis=1)).any():
        return result
    if ((numeric.vol.eq(0) & numeric.amount.ne(0)) | (numeric.vol.gt(0) & numeric.amount.le(0))).any():
        return result
    vwap = numeric.amount.cumsum() / numeric.vol.cumsum().replace(0, float("nan"))
    late = f[f.time.dt.strftime("%H:%M").ge("14:30")]
    latest_price = float(numeric.close.iloc[-1])
    result.update(status="COMPLETE", full_day_vwap=float(vwap.iloc[-1]),
                  above_vwap_ratio=float(numeric.close.ge(vwap).mean()))
    if len(late) >= 3:
        rise = (latest_price / float(late.close.iloc[0]) - 1) * 100
        drawdown = (latest_price / float(late.high.max()) - 1) * 100
        result.update(late_return_pct=rise, late_drawdown_pct=drawdown,
                      late_pattern="WEAK_BELOW_VWAP" if latest_price < vwap.iloc[-1] else
                      "PULLBACK_ABOVE_VWAP" if drawdown < -.5 else "ABOVE_VWAP")
    # Completed minutes only: dip, hold the low, then break the intervening high.
    finished = f[f.time.lt(at.floor("min"))].tail(15).reset_index(drop=True)
    result["reclaim_confirmed"] = False
    if len(finished) >= 6:
        trough = int(finished.low.idxmin())
        if 1 <= trough <= len(finished) - 3:
            result["reclaim_confirmed"] = bool(
                finished.low.iloc[trough] < finished.low.iloc[:trough].min()
                and finished.low.iloc[trough + 1:].min() > finished.low.iloc[trough]
                and finished.close.iloc[-1] > finished.high.iloc[trough:-1].max()
                and finished.close.iloc[-1] >= vwap.iloc[-2]
                and latest_price >= finished.close.iloc[-1])
    return result


def collect(client, symbols, minute_symbols, at, budget_seconds=None, minute_selector=None, priority_symbols=()):
    """Finish the quote sweep through throttling, bounded by today's collection window."""
    start = time.monotonic()
    until_end = (stamp(f"{stamp(at).date()} {COLLECTION_END}") - stamp(at)).total_seconds()
    seconds = max(0, until_end)
    if budget_seconds is not None:
        seconds = min(budget_seconds, seconds)
    deadline = start + seconds
    deadline_reason = "WINDOW_END" if seconds == max(0, until_end) else "BUDGET"
    raw, issues = [], []

    def fetch(api, **params):
        transport_failures = 0
        while True:
            if time.monotonic() >= deadline:
                raise ApiError(api, "BUDGET", "collection budget or window end reached")
            try:
                frame = client.query(api, deadline=deadline, **params)
                break
            except ApiError as exc:
                if exc.code == "TRANSPORT":
                    transport_failures += 1
                    if transport_failures < 3:
                        print(f"[intraday_quotes] {api} transport failure; wait up to 5s and retry current batch "
                              f"({transport_failures}/2)", flush=True)
                        time.sleep(min(5, max(0, deadline - time.monotonic())))
                        continue
                if not exc.rate_limited:
                    raise
                # Client enforces the cooldown on retry, including across watch cycles.
                print(f"[intraday_quotes] {api} rate limited; keep current batch and retry after cooldown", flush=True)
        raw.append({"api": api, "received_at": now_china().isoformat(), "records": json.loads(frame.to_json(orient="records"))})
        return frame

    universe = set(symbols)
    ordered = dict.fromkeys(list(priority_symbols) + list(minute_symbols) + list(symbols))
    codes = [ts_code(s) for s in ordered if s in universe]
    api = "rt_k"
    pending, received, missing_attempts = list(codes), set(), {}
    stop_reason = ""
    print(f"[intraday_quotes] start requested={len(codes)}; wait through rate limits; "
          f"deadline={deadline_reason} remaining={max(0, seconds):.1f}s (no requests after {COLLECTION_END} CST)", flush=True)
    while pending:
        # rt_min officially accepts up to 1,000 rows; verify returned coverage below.
        requested = pending[:1000 if api == "rt_min" else 300]
        batch = ",".join(requested)
        try:
            frame = fetch(api, ts_code=batch, **({"freq": "1MIN"} if api == "rt_min" else {"fields": SNAPSHOT_FIELDS}))
        except ApiError as exc:
            issues.append(str(exc))
            if api == "rt_k" and exc.code != "BUDGET":
                api = "rt_min"
                continue
            stop_reason = "DAILY_QUOTA" if exc.daily_limit else deadline_reason if exc.code == "BUDGET" else str(exc.code)
            break
        returned = set(frame.get("ts_code", frame.get("code", pd.Series(dtype=str))).astype(str))
        received.update(set(requested) & returned)
        missing = [code for code in requested if code not in returned]
        pending = pending[len(requested):]
        if missing:
            # Suspended/absent symbols must not consume the rest of the daily quota forever.
            for code in missing:
                missing_attempts[code] = missing_attempts.get(code, 0) + 1
                if missing_attempts[code] < 3:
                    pending.append(code)
        print(f"[intraday_quotes] {api} received={len(received)}/{len(codes)} pending={len(pending)} "
              f"missing_in_batch={len(missing)} elapsed={time.monotonic() - start:.1f}s", flush=True)
    missing = [code for code in codes if code not in received]
    coverage = {"status": "PARTIAL" if missing else "COMPLETE", "requested": len(codes), "received": len(received),
                "missing_codes": missing, "stop_reason": stop_reason or ("MISSING_QUOTES" if missing else "ALL_RECEIVED")}
    print(f"[intraday_quotes] stock sweep {coverage['status']} received={len(received)}/{len(codes)} "
          f"missing={len(missing)} reason={coverage['stop_reason']} (timestamps still require freshness checks)", flush=True)
    if missing:
        issues.append(f"Stock quote sweep incomplete: missing {len(missing)}/{len(codes)}; "
                      f"reason={coverage['stop_reason']}; examples={','.join(missing[:5])}")
    try:
        fetch("rt_idx_k", ts_code=",".join(INDEX_CODES))
    except ApiError as exc:
        issues.append(str(exc))
    if minute_selector is not None:
        minute_symbols = list(dict.fromkeys(minute_selector(raw, now_china())))
    for symbol in minute_symbols:
        try:
            fetch("rt_min_daily", ts_code=ts_code(symbol), freq="1MIN")
        except ApiError as exc:
            issues.append(str(exc))
            if exc.code in {40203, "BUDGET"} or exc.daily_limit:
                break
    return {"started_at": stamp(at).isoformat(), "received_at": now_china().isoformat(),
            "raw": raw, "issues": issues, "api_audit": client.audit, "minute_symbols": minute_symbols,
            "quote_collection": coverage}
