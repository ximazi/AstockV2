"""Versioned, observation-only first-board and relay watch."""

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from . import eod_review as E
from .hotspot_relay import _frozen
from .intraday_quotes import symbol_of
from .review_clock import CST, instant

VERSION = "board_relay_v1_20261124"
LANES = {1: "FIRST", 2: "RELAY", 3: "HEIGHT"}
MAX_FOCUS = 3


def folder(root, day):
    return Path(root) / day.replace("-", "") / "board_relay"


def clock(day, time):
    return datetime.fromisoformat(f"{day}T{time}+08:00")


def files_before(base, day, cutoff, filename, receipt=False):
    for path in sorted(base.glob(f"v[0-9]*/{filename}"), reverse=True):
        data = E.read_json(path)
        if receipt:
            stamp = E.read_json(path.with_name("published.json"))
            if stamp.get("as_of") != day or stamp.get("sha256") != E.file_sha(path):
                continue
            known = instant(stamp.get("available_at"))
        else:
            known = instant(data.get("known_at"))
        if data.get("as_of") == day and known and clock(day, "15:00") <= known <= cutoff:
            return data, {"path": str(path), "known_at": known.isoformat(), "sha256": E.file_sha(path)}
    return {}, None


def next_session(sessions, day):
    later = next((d for d in sessions if d > day), None)
    if later:
        return later
    d = date.fromisoformat(day) + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d.isoformat()


def seal_time(value):
    text = str(value if value is not None else "").strip().replace(":", "")
    if not text.isascii() or not text.isdigit() or len(text) not in (5, 6):
        return None
    try:
        at = datetime.strptime(text.zfill(6), "%H%M%S").time()
    except ValueError:
        return None
    value = at.hour * 3600 + at.minute * 60 + at.second
    return value if 9 * 3600 + 25 * 60 <= value <= 15 * 3600 else None


def bars(path, day):
    if not path.is_file():
        return {}
    cols = ["high", "low", "close", "preclose", "amount", "turnover", "pct_chg"]
    try:
        f = pd.read_csv(path, usecols=["date", *cols], dtype={"date": str})
    except (OSError, ValueError):
        return {}
    f = f.loc[f.date.le(day)].sort_values("date").drop_duplicates("date", keep="last").tail(81)
    if f.empty or f.date.iloc[-1] != day:
        return {}
    f[cols] = f[cols].apply(pd.to_numeric, errors="coerce")
    last = f.iloc[-1]
    low, high, close, preclose, amount = map(E.number, (last.low, last.high, last.close, last.preclose, last.amount))
    if any(v is None or v <= 0 for v in (low, high, close, preclose, amount)) or not low <= close <= high:
        return {}
    prev = f.iloc[:-1]
    result = {"close": close, "preclose": preclose, "amount": amount, "low": low, "one_price": high == low,
              "gap_intraday_low_pct": round((low / preclose - 1) * 100, 2),
              "turnover": E.number(last.turnover)}
    if len(prev) >= 20:
        mid = E.number(prev.turnover.tail(20).median())
        result["turnover_relative"] = round(last.turnover / mid, 3) if mid and mid > 0 and result["turnover"] is not None else None
    # Chain raw daily returns to keep the 60-session prices comparable through corporate actions.
    pct = pd.to_numeric(f.pct_chg, errors="coerce")
    if len(prev) >= 60 and pct.between(-99.9999, 1000).all() and f[["high", "low", "close"]].gt(0).all().all():
        index = (1 + pct / 100).cumprod()
        adj = index / f.close
        hi = (f.high * adj).iloc[:-1]
        lo = (f.low * adj).iloc[:-1]
        before, current = index.iloc[-2], index.iloc[-1]
        highest, lowest = hi.tail(60).max(), lo.tail(60).min()
        result.update(position60=round((before - lowest) / (highest - lowest), 4) if highest > lowest else None,
                      pre_ret20_pct=round((before / index.iloc[-22] - 1) * 100, 2),
                      headroom60_pct=round((highest / current - 1) * 100, 2),
                      ema20_bias_pct=round((before / index.iloc[:-1].ewm(span=20, adjust=False).mean().iloc[-1] - 1) * 100, 2),
                      ema60_bias_pct=round((before / index.iloc[:-1].ewm(span=60, adjust=False).mean().iloc[-1] - 1) * 100, 2),
                      local_breakout=bool(current > hi.tail(10).max()))
    return E.clean(result)


def watch_notices(materials, cutoff, subject=()):
    out = []
    for m in materials:
        seen, published = instant(m.get("collected_at")), instant(m.get("published_at"))
        if not seen or seen > cutoff or (published and published > cutoff):
            continue
        title = m.get("title") or ""
        kind = next((kind for terms, kind in (
            (("监管工作函", "监管函", "重点监控", "停牌核查"), "REGULATORY_REVIEW"),
            (("澄清", "辟谣"), "CLARIFICATION"),
            (("风险提示",), "RISK_ALERT"),
            (("异常波动",), "ABNORMAL_MOVE"),
        ) if any(t in title for t in terms)), None)
        official = m.get("kind") in ("ISSUER_BODY", "ISSUER_TITLE")
        named_media = (m.get("kind") == "MEDIA_BODY" and kind == "REGULATORY_REVIEW" and
                       published and any(name and name in title for name in subject))
        if kind and (official or named_media):
            out.append({"kind": kind, "title": title, "source": m.get("url"), "known_at": seen.isoformat(),
                        "basis": "ISSUER" if official else "MEDIA_PENDING_OFFICIAL_CHECK"})
    return out


def seat_inputs(base, day, cutoff):
    key = ["top_inst", day, {"trade_date": day.replace("-", "")}]
    p = base / "limit_theme/source_cache" / (E.digest(key)[:24] + ".json")
    src = E.read_json(p)
    at = instant(src.get("collected_at"))
    if (src.get("key") != key or not at or not clock(day, "15:00") <= at <= cutoff
            or src.get("status") != "AVAILABLE" or not isinstance(src.get("data"), list)):
        return None, None
    by_symbol = defaultdict(list)
    for row in src["data"]:
        by_symbol[row.get("ts_code")].append(row)
    return by_symbol, {"path": str(p), "known_at": at.isoformat(), "sha256": E.file_sha(p)}


def seat_summary(rows, amount=None):
    if rows is None:
        return {"status": "DATA_WAIT", "disclosures": []}
    groups, nets = defaultdict(dict), defaultdict(dict)
    for r in rows:
        if str(r.get("side")) not in ("0", "BUY"):
            continue
        reason, name = r.get("reason"), r.get("exalter")
        buy = E.number(r.get("buy"))
        if not reason or not name or buy is None or buy <= 0:
            continue
        groups[reason][name] = max(groups[reason].get(name, 0), buy)
        net = E.number(r.get("net_buy"))
        if net is not None:
            nets[reason][name] = net
    disclosures = []
    for reason, by_name in sorted(groups.items()):
        top = sorted(by_name.items(), key=lambda x: (-x[1], x[0]))[:5]
        matching = [(n, v) for n, v in top if "东方财富" in n and "拉萨" in n]
        disclosures.append({"reason": reason, "seat_count": len(matching),
                            "buy_share": round(sum(v for _, v in matching) / sum(v for _, v in top), 4),
                            "net_to_day_amount": round(sum(nets[reason][n] for n, _ in matching) / amount, 4)
                            if amount and all(n in nets[reason] for n, _ in matching) else None})
    return {"status": "AVAILABLE" if disclosures else ("NO_BUY_DETAIL" if rows else "NOT_ON_LIST"),
            "disclosures": disclosures}


def compile_watch(day, facts, attrs, reference, quotes, seats, cutoff, names=None):
    board = facts["boards"]
    heights = facts["ladder"]["heights"]
    disputed = set(board.get("reconciliation", {}).get("U", {}).get("local_only", []))
    theme_rows = {r["symbol"]: r for r in attrs.get("rows", [])}
    chains = defaultdict(list)
    for s in board["U"]:
        row = theme_rows.get(s, {})
        if s not in disputed and row.get("status") == "CORROBORATED" and row.get("primary_theme"):
            chains[row["primary_theme"]].append(s)
    market = facts.get("market", {}).get("context", {})
    p5 = {r["symbol"]: r["priority"] for r in facts.get("p5_limit_up", {}).get("rows", [])}
    rows = []
    for s in sorted(set(board["U"])):
        height = heights.get(s)
        q, a, ref = quotes.get(s, {}), theme_rows.get(s, {}), reference.get(s, {})
        theme = a.get("primary_theme")
        peers = [x for x in chains.get(theme, []) if x != s] if s in chains.get(theme, []) else []
        ladder = dict(sorted(Counter(str(heights[x]) for x in peers if x in heights).items()))
        first, last = seal_time(ref.get("first_time")), seal_time(ref.get("last_time"))
        if first is not None and last is not None and last < first:
            first = last = None
        opened, fd = E.number(ref.get("open_times")), E.number(ref.get("fd_amount"))
        name = a.get("name") or ref.get("name") or (names or {}).get(s) or s
        notices = watch_notices(attrs.get("materials", {}).get(s, []), cutoff, (name, s[3:]))
        regulatory = any(n["kind"] == "REGULATORY_REVIEW" for n in notices)
        lane = LANES.get(height, "HEIGHT") if isinstance(height, int) and height >= 1 else "UNKNOWN"
        confirmed_chain = s in chains.get(theme, [])
        has_structure = q.get("position60") is not None
        # This is a phase/position observation, not a fixed price, market cap or turnover gate.
        low_restart = has_structure and q["position60"] <= .5
        support = confirmed_chain or q.get("local_breakout", False)
        risk = ("BOARD_DISPUTED" if s in disputed else "DATA_WAIT" if not q or lane == "UNKNOWN" else
                "REGULATORY_REVIEW" if regulatory else "HEIGHT_WATCH" if lane == "HEIGHT" else
                "ONE_PRICE_UNPROVEN" if q["one_price"] else "OBSERVATION")
        eligible = risk == "OBSERVATION" and (
            (low_restart and support) or (q.get("local_breakout") and bool(peers)) if lane == "FIRST"
            else bool(peers) or (first is not None and q.get("local_breakout", False))
        )
        rows.append({"symbol": s, "name": name, "lane": lane, "height": height,
                     "board": E.board_of(s),
                     "theme": theme if confirmed_chain else None, "peer_symbols": peers, "peer_ladder": ladder,
                     "p5": p5.get(s), "price": q, "first_seal_s": first, "last_seal_s": last,
                     "open_times": int(opened) if opened is not None and opened >= 0 and opened.is_integer() else None,
                     "seal_to_amount": round(fd / q["amount"], 4) if fd is not None and fd >= 0 and q.get("amount") else None,
                     "state": risk, "focus_eligible": bool(eligible), "notices": notices,
                     "seat_evidence": seat_summary(None if seats is None else seats.get(s[3:] + "." + s[:2].upper(), []), q.get("amount"))})
    focus, baseline = {}, {}
    for lane in LANES.values():
        lane_rows = [r for r in rows if r["lane"] == lane and r["price"] and r["state"] != "BOARD_DISPUTED"
                     and (lane == "HEIGHT" or r["state"] != "REGULATORY_REVIEW")]
        baseline[lane] = [r["symbol"] for r in sorted(lane_rows, key=lambda r: (
            -(r["height"] or 0), r["first_seal_s"] or 99999, r["symbol"]))[:MAX_FOCUS]]
        ranked = sorted((r for r in lane_rows if r["focus_eligible"] or lane == "HEIGHT"), key=lambda r: (
            -r["height"] if lane == "HEIGHT" else 0, -len(r["peer_symbols"]), not bool(r["theme"]),
            not r["price"].get("local_breakout", False),
            r["first_seal_s"] or 99999, r["last_seal_s"] or 99999,
            r["price"].get("position60") if r["price"].get("position60") is not None else 2, r["symbol"]))
        seen, chosen = set(), []
        for row in ranked:
            tag = row["theme"] or row["symbol"]
            if tag not in seen:
                chosen.append(row["symbol"])
                seen.add(tag)
            if len(chosen) == MAX_FOCUS:
                break
        focus[lane] = chosen
    return {"as_of": day, "version": VERSION, "orders_enabled": False, "market": market,
            "rows": rows, "focus": focus, "baseline": baseline, "universe": len(rows),
            "failed_seals": len(board.get("Z", [])), "unmatched_attribution": sum(not r["theme"] for r in rows)}


def build(day, root, data_root, reference_root, cutoff, names=None):
    base = Path(root) / day.replace("-", "")
    facts, fact_source = files_before(base / "eod_review", day, cutoff, "facts.json")
    if not facts:
        return {"as_of": day, "version": VERSION, "status": "DATA_WAIT", "reason": "NO_DATED_EOD_FACTS",
                "orders_enabled": False, "rows": [], "focus": {}, "baseline": {}}
    attr, attr_source = files_before(base / "limit_theme", day, cutoff, "report.json", receipt=True)
    archived_attr = not attr
    if archived_attr:
        core, path = _frozen(base, day, cutoff.isoformat())
        if path:
            # Older reports only prove the attribution that actually appeared in the frozen review.
            attr = {"rows": [{"symbol": r["symbol"], "name": r.get("name"),
                              "status": a.get("status"), "primary_theme": a.get("theme")}
                             for r in core.get("rows", []) for a in [r.get("attribution") or {}]
                             if instant(a.get("known_at")) and instant(a["known_at"]) <= cutoff]}
            attr_source = {"path": str(path), "basis": "FROZEN_CORE_ONLY", "sha256": E.file_sha(path)}
    ref_path = Path(reference_root) / "reference" / f"limit_list_d_{day.replace('-', '')}.json"
    ref = E.read_json(ref_path)
    stamp = instant(ref.get("retrieved_at"))
    disciplined = (ref.get("as_of") == day and stamp and clock(day, "15:00") <= stamp <= cutoff
                   and ref.get("status") in ("AVAILABLE", "PARTIAL", "EMPTY"))
    ref_rows = {symbol_of(r["ts_code"]): r for r in ref["records"] if "." in r.get("ts_code", "")} if disciplined else {}
    seats, seat_source = seat_inputs(base, day, cutoff)
    quotes = {s: bars(Path(data_root) / f"{s}_daily.csv", day) for s in set(facts["boards"]["U"]) if E.board_of(s)}
    output = compile_watch(day, facts, attr, ref_rows, quotes, seats, cutoff, names)
    gaps = [k for k, status in (("LIMIT_TIMES", disciplined), ("THEME_ATTRIBUTION", attr_source),
                                ("LHB_SEATS", seat_source)) if not status]
    if any(not row["price"] for row in output["rows"]):
        gaps.append("PRICE_ROWS")
    if archived_attr and attr_source:
        gaps.append("ATTRIBUTION_PARTIAL_ARCHIVE")
    if disciplined and ref["status"] == "PARTIAL":
        gaps.append("LIMIT_TIMES_PARTIAL")
    if attr and not archived_attr and not attr.get("scope", {}).get("full_limit_up_coverage"):
        gaps.append("THEME_SCOPE_PARTIAL")
    output.update(status="PARTIAL" if gaps else "READY", gaps=gaps, sources={"eod": fact_source, "attribution": attr_source,
                  "seats": seat_source, "limit_times": {"path": str(ref_path), "known_at": stamp.isoformat(),
                  "sha256": E.file_sha(ref_path)} if disciplined else None})
    return output


def followup(previous, target, facts, data_root):
    result = []
    if not previous or not facts:
        return result
    u, z = set(facts["boards"]["U"]), set(facts["boards"]["Z"])
    disputed = set(facts["boards"].get("reconciliation", {}).get("U", {}).get("local_only", []))
    chosen = {s for ss in list(previous["focus"].values()) + list(previous["baseline"].values()) for s in ss}
    for r in previous["rows"]:
        s = r["symbol"]
        if s not in chosen:
            continue
        q = bars(Path(data_root) / f"{s}_daily.csv", target)
        h = facts["ladder"]["heights"].get(s)
        state = ("DATA_WAIT" if not q or s in disputed or abs(q["preclose"] - r["price"]["close"]) >= .011 else
                 "ONE_PRICE_PROMOTION" if s in u and h == r["height"] + 1 and q["one_price"] else
                 "PROMOTION" if s in u and h == r["height"] + 1 else
                 "FAILED_SEAL" if s in z else "NO_PROMOTION")
        result.append({"symbol": s, "lane": r["lane"], "state": state,
                       "close_change_pct": round((q["close"] / q["preclose"] - 1) * 100, 2) if q else None})
    return result


def run(day, root=E.OUT_ROOT, data_root="data", reference_root=E.REFERENCE_ROOT, now=None):
    now = now or datetime.now(CST)
    days = E.calendar(data_root)
    if day not in days or now < clock(day, "15:00"):
        raise ValueError("Completed session is not available")
    target = next_session(days, day)
    end = clock(target, "09:15") - timedelta(microseconds=1)
    names = {s: r.get("name") for s, r in E.read_json(Path(data_root).parent / "concept_mapping.json").get("mapping", {}).items()}
    out = build(day, root, data_root, reference_root, min(now, end), names)
    live = now <= end
    out.update(mode="FORWARD" if live else "REPLAY", published_at=now.isoformat(), target=target,
               target_basis="DATED_CALENDAR" if target in days else "PROVISIONAL_WEEKDAY")
    prev = days[days.index(day) - 1] if days.index(day) else None
    prior = E.read_json(folder(root, prev) / "latest.json") if prev else {}
    if prior.get("mode") == "FORWARD" and prior.get("target") == day:
        stamp = instant(prior.get("published_at"))
        if stamp and stamp < clock(day, "09:15"):
            facts, _ = files_before(Path(root) / day.replace("-", "") / "eod_review", day, min(now, end), "facts.json")
            out["prior_followup"] = followup(prior, day, facts, data_root)
    directory = folder(root, day)
    archive = directory / ("forward" if live else "replay") / f"{now.strftime('%Y%m%d_%H%M%S_%f')}.json"
    E._write(archive, out)
    E._write(directory / ("latest.json" if live else "replay.json"), out)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date", required=True, type=lambda s: date.fromisoformat(s).isoformat())
    p.add_argument("--out-root", default=str(E.OUT_ROOT))
    p.add_argument("--data-root", default="data")
    p.add_argument("--reference-root", default=str(E.REFERENCE_ROOT))
    a = p.parse_args()
    directory = folder(a.out_root, a.date)
    try:
        out = run(a.date, a.out_root, a.data_root, a.reference_root)
    except (OSError, ValueError, KeyError) as exc:
        E._write(directory / "status.json", {"as_of": a.date, "status": "FAILED", "error": str(exc), "orders_enabled": False})
        print(f"[board_relay] FAILED: {exc}", flush=True)
        return 1
    E._write(directory / "status.json", {k: out.get(k) for k in ("as_of", "status", "mode", "published_at", "gaps")})
    print(f"[board_relay] {out['status']} {out['mode']} U={len(out['rows'])} focus={out['focus']} -> {directory}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
