"""Broad, dated preparation pool. Membership is evidence to review, not a buy signal."""
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D, features as F, phase_events as PE
from . import intraday_quotes as Q
from .pipeline import _write

VERSION = "potential_pool_v1_20260919"
RULES = {"liquidity_amount": 30_000_000, "touch_atr": .25, "penetration_atr": .75,
         "lower_wick": .25, "close_location": .65, "pullback_distance_atr": 1.,
         "ignition_distance_atr": 2.5, "ignition_activity": 1.3, "pool_bonus": 2}
NOTICE = "潜在关注池，不证明主力身份，不等于买点。部分证据即可入池；分数为待校准观察分，不是收益率或胜率。"


def entry_checks(seed, bar):
    """One rule for final daily bars and provisional cumulative snapshots."""
    empty = {"ignition": False, "pullback": False, "combined": False, "risk_flags": [],
             "lower_wick_ratio": None, "ema10": None, "checks": {}}
    if seed.get("status") != "READY":
        return empty
    p = {k: Q.number(bar.get(k)) for k in ("open", "high", "low", "close", "pre_close", "volume")}
    if any(v is None or v <= 0 for v in p.values()) or not p["low"] <= min(p["open"], p["close"]) <= max(p["open"], p["close"]) <= p["high"]:
        return empty
    scale = p["pre_close"] / seed["close"]
    atr, prior_ema = seed["atr20"] * scale, seed["ema10"] * scale
    if atr <= 0 or seed["volume20"] <= 0:
        return empty
    ema10 = prior_ema * 9 / 11 + p["close"] * 2 / 11
    ema20 = seed["ema20"] * scale * 19 / 21 + p["close"] * 2 / 21
    width = p["high"] - p["low"]
    wick = (min(p["open"], p["close"]) - p["low"]) / width if width else 0
    location = (p["close"] - p["low"]) / width if width else 0
    activity = p["volume"] / seed["volume20"]
    distance = (p["close"] - ema10) / atr
    risks = []
    if p["close"] < seed["low20"] * scale:
        risks.append("跌破前20日结构低点")
    if seed.get("pressure"):
        risks.append("前收盘供给压力尚待解除")
    if distance > RULES["ignition_distance_atr"]:
        risks.append("偏离EMA10过大，等待位置修复")
    if activity >= 1.3 and (location < .4 or (width and (p["high"] - max(p["open"], p["close"])) / width >= .45)):
        risks.append("放量弱收或长上影，警惕供给压力")
    if not width:
        risks.append("一字价格，未核验可成交性")
    trend = bool(seed["trend_up"] and p["pre_close"] > prior_ema and p["close"] >= ema20)
    checks = {"trend": trend,
              "touch_ema10": prior_ema - RULES["penetration_atr"] * atr <= p["low"] <= prior_ema + RULES["touch_atr"] * atr,
              "lower_wick": wick >= RULES["lower_wick"],
              "recovered": p["close"] >= max(ema10, prior_ema) and location >= RULES["close_location"],
              "near_entry": 0 <= distance <= RULES["pullback_distance_atr"]}
    pullback = all(checks.values()) and not risks
    ignition = (p["close"] > seed["high20"] * scale + .1 * atr and p["close"] > p["pre_close"]
                and p["close"] >= ema20 and seed["ema20_rising"] and location >= .7
                and activity >= RULES["ignition_activity"] and not risks)
    prior_ignition = seed.get("ignition_date", "")
    held_ignition = seed.get("ignition_floor") is not None and p["low"] >= seed["ignition_floor"] * scale
    return {"ignition": bool(ignition), "pullback": bool(pullback),
            "combined": bool(pullback and prior_ignition and held_ignition),
            "risk_flags": risks, "lower_wick_ratio": wick, "ema10": ema10,
            "close_location": location, "distance_atr": distance, "activity": activity,
            "checks": checks, "ignition_date": prior_ignition, "orders_enabled": False}


def feature_seed(history, symbol, as_of):
    """Reuse the return-index phase engine, with only bars known by as_of."""
    required = {"date", "open", "high", "low", "close", "amount", "volume", "pct_chg"}
    if not required.issubset(history.columns):
        return {"status": "HISTORY_WAIT"}
    raw = history.copy()
    raw["date"] = pd.to_datetime(raw.date)
    raw = raw[raw.date.le(as_of)].sort_values("date").drop_duplicates("date", keep="last").tail(320).reset_index(drop=True)
    if len(raw) < 70 or raw.date.iloc[-1].strftime("%Y-%m-%d") != as_of:
        return {"status": "HISTORY_WAIT"}
    cols = list(required - {"date"})
    raw[cols] = raw[cols].apply(pd.to_numeric, errors="coerce")
    if (not np.isfinite(raw[cols]).all().all() or raw[["open", "high", "low", "close"]].le(0).any().any()
            or raw[["amount", "volume"]].lt(0).any().any() or raw.pct_chg.le(-100).any()
            or raw.high.lt(raw[["open", "close"]].max(axis=1)).any()
            or raw.low.gt(raw[["open", "close"]].min(axis=1)).any()):
        return {"status": "HISTORY_WAIT"}
    f = PE.classify_frame(raw)
    adjusted = f.copy()
    for key in ("open", "high", "low", "close"):
        adjusted[key] = f[key] * f.return_index / f.close
    adjusted["preclose"] = adjusted.close.shift().fillna(adjusted.close)
    if "turnover" not in adjusted:
        adjusted["turnover"] = np.nan
    adjusted["date"] = f.date.dt.strftime("%Y-%m-%d")
    c, h, low, op = (adjusted[k] for k in ("close", "high", "low", "open"))
    ctx = F.Ctx(symbol, adjusted)
    retained, supply = F.impulse_retention(ctx), F.silent_supply(ctx)
    ema10, ema20 = (c.ewm(span=n, adjust=False).mean() for n in (10, 20))
    turn = pd.to_numeric(adjusted.turnover, errors="coerce").where(lambda s: s > 0)
    vol = raw.volume.where(raw.volume > 0)
    if vol.tail(20).notna().sum() < 15:
        return {"status": "HISTORY_WAIT"}
    base = lambda s: s.shift().rolling(20, min_periods=15).median()
    relative_turn, relative_vol = turn / base(turn), vol / base(vol)
    # A changing float denominator invalidates the turnover signal, not all price evidence.
    denominator_jump = (vol / turn).pct_change(fill_method=None).abs() > .1
    relative_turn = relative_turn.mask(denominator_jump)
    bullish = c > op
    consecutive = bullish.astype(int).groupby((~bullish).cumsum()).cumsum()
    proxy_limit = raw.pct_chg >= D.limit_pct(symbol) * .98
    persistent_volume = (relative_vol >= 2) & (f.close_location >= .65) & bullish
    low_active = (f.range_position60 <= .35) & (relative_turn >= 1.5) & (f.amount_ratio >= 1.5) & (f.close_location >= .65)
    gap_days = [j for j in range(len(f) - 10, len(f)) if low.iloc[j] > h.iloc[j - 1] and low.iloc[j:].min() > h.iloc[j - 1]]
    evidence = []
    def add(mask, count, label):
        recent = mask.tail(count)
        if recent.any():
            j = recent[recent].index[-1]
            evidence.append({"key": label, "date": adjusted.date.iloc[j]})
    add(proxy_limit, 20, "近20日涨停强度代理")
    add(consecutive >= 4, 10, "近期至少四连阳")
    if gap_days:
        evidence.append({"key": "近期向上缺口仍未回补", "date": adjusted.date.iloc[gap_days[-1]]})
    if persistent_volume.tail(5).sum() >= 2:
        add(persistent_volume, 5, "近五日至少两次倍量强收")
    add(low_active, 5, "低位相对高换手高成交额且强收")
    if retained["ok"]:
        evidence.append({"key": "历史冲击效果保留", "date": str(retained["event_date"])[:10]})
    if supply["ok"]:
        evidence.append({"key": "充分换手后供给收缩", "date": as_of})
    score = (2 if proxy_limit.tail(20).any() or retained["ok"] else 0)
    score += min(2, int((consecutive >= 4).tail(10).any()) + int(bool(gap_days)))
    score += 2 if persistent_volume.tail(5).sum() >= 2 or low_active.tail(5).any() else 0
    score += 2 if supply["ok"] else 0
    raw_scale = float(raw.close.iloc[-1] / c.iloc[-1])
    def seed_at(i):
        prior_sos = [j for j in range(max(0, i - 19), i + 1) if f.SOS.iloc[j]
                     and low.iloc[j:i + 1].min() >= low.iloc[j]]
        j = prior_sos[-1] if prior_sos else None
        return {"status": "READY", "as_of": adjusted.date.iloc[i], "close": float(c.iloc[i] * raw_scale),
                "ema10": float(ema10.iloc[i] * raw_scale), "ema20": float(ema20.iloc[i] * raw_scale),
                "atr20": float(f.atr20.iloc[i] * raw_scale), "volume20": float(vol.iloc[max(0, i-19):i+1].median()),
                "high20": float(h.iloc[i-19:i+1].max() * raw_scale), "low20": float(low.iloc[i-19:i+1].min() * raw_scale),
                "trend_up": bool(ema10.iloc[i] > ema10.iloc[i-3] and ema20.iloc[i] > ema20.iloc[i-3]),
                "ema20_rising": bool(ema20.iloc[i] > ema20.iloc[i-3]),
                "pressure": bool(f.UTAD.iloc[i] or f.SOW.iloc[i] or f.effort_result.iloc[i] == "DISTRIBUTION_PRESSURE"),
                "ignition_date": adjusted.date.iloc[j] if j is not None else "",
                "ignition_floor": float(low.iloc[j] * raw_scale) if j is not None else None}
    seed = seed_at(len(f)-1)
    today = {k: float(adjusted[k].iloc[-1] * raw_scale) for k in ("open", "high", "low", "close")}
    today.update(pre_close=float(c.iloc[-2] * raw_scale), volume=float(raw.volume.iloc[-1]))
    entry = entry_checks(seed_at(len(f)-2), today)
    risks = list(entry["risk_flags"])
    if seed["pressure"] and "前收盘供给压力尚待解除" not in risks:
        risks.append("当日放量受阻或结构破坏")
    liquid = bool(raw.amount.tail(20).median() >= RULES["liquidity_amount"] and raw.amount.iloc[-1] > 0 and raw.volume.iloc[-1] > 0)
    return {**seed, "previous_day": adjusted.date.iloc[-2], "evidence": evidence, "score": score, "liquid": liquid, "risk_flags": risks, "entry": entry,
            "metrics": {"close": float(raw.close.iloc[-1]), "ema10": seed["ema10"],
                        "turnover": Q.number(turn.iloc[-1]), "relative_turnover": Q.number(relative_turn.iloc[-1]),
                        "relative_volume": Q.number(relative_vol.iloc[-1]), "position60": Q.number(f.range_position60.iloc[-1])},
            "missing": ["换手率缺失或流通股本分母变化"] if pd.isna(relative_turn.iloc[-1]) else []}


def load(root, day):
    path = Path(root) / "reference" / f"potential_pool_{day.replace('-', '')}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def latest(root, day, at):
    for path in sorted((Path(root) / "reference").glob("potential_pool_????????.json"), reverse=True):
        if path.stem[-8:] > day.replace('-', ''):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") == VERSION and Q.stamp(data["known_at"]) <= Q.stamp(at):
            return data
    return {}


def build(seeds, as_of, previous=None, mapping=None, at=None):
    previous, mapping = previous or {}, mapping or {}
    old = {r["symbol"]: r for r in previous.get("records", [])} if previous.get("version") == VERSION and previous.get("as_of", as_of) < as_of else {}
    records = []
    for symbol in sorted(set(seeds) | set(old)):
        seed, before = seeds.get(symbol, {}), old.get(symbol, {})
        ready = seed.get("status") == "READY" and seed.get("as_of") == as_of
        evidence = seed.get("evidence", []) if ready else []
        admitted = bool(ready and evidence and seed.get("liquid"))
        if not admitted and not before:
            continue
        state = "DATA_WAIT" if not ready else "EXITED" if not admitted else "RISK_WATCH" if seed.get("risk_flags") else "POTENTIAL"
        prior_member = before.get("state") == "POTENTIAL" and previous.get("as_of") == seed.get("previous_day")
        info = mapping.get(symbol, {})
        entry = seed.get("entry", {}) if ready else {}
        bonus = RULES["pool_bonus"] if prior_member and entry.get("ignition") and state == "POTENTIAL" else 0
        records.append({"symbol": symbol, "name": info.get("name", before.get("name", "")),
                        "theme": " / ".join(info.get("concepts", [])[:3]), "state": state,
                        "first_seen": before.get("first_seen", as_of), "last_qualified": as_of if admitted else before.get("last_qualified"),
                        "evidence": [f"{r['date']} {r['key']}" for r in evidence], "risk_flags": seed.get("risk_flags", []),
                        "score": seed.get("score", 0) if ready else 0, "metrics": seed.get("metrics", {}) if ready else {},
                        "entry": entry, "prior_member": prior_member, "ignition_bonus": bonus,
                        "missing": seed.get("missing", []) if ready else ["当日行情缺失或历史数据不完整"],
                        "exit_reason": "证据已过观察窗或流动性不足" if state == "EXITED" else "",
                        "orders_enabled": False})
    order = {"POTENTIAL": 0, "RISK_WATCH": 1, "DATA_WAIT": 2, "EXITED": 3}
    records.sort(key=lambda r: (order[r["state"]], -(r["score"] + r["ignition_bonus"]), r["symbol"]))
    counts = Counter(r["state"] for r in records)
    ready = sum(s.get("status") == "READY" and s.get("as_of") == as_of for s in seeds.values())
    return {"version": VERSION, "as_of": as_of, "known_at": Q.stamp(at or Q.now_china()).isoformat(),
            "status": "AVAILABLE" if seeds and ready == len(seeds) else "PARTIAL", "notice": NOTICE,
            "rules": RULES, "orders_enabled": False, "records": records,
            "summary": {"universe": len(seeds), "ready": ready, "active": counts["POTENTIAL"],
                        "risk": counts["RISK_WATCH"], "exited": counts["EXITED"], "data_wait": counts["DATA_WAIT"]}}


def publish(seeds, as_of, root, mapping=None, at=None):
    at = at or Q.now_china()
    yesterday = (pd.Timestamp(as_of) - pd.Timedelta(days=1)).date().isoformat()
    previous = latest(root, yesterday, at)
    payload = build(seeds, as_of, previous, mapping, at)
    path = Path(root) / "reference" / f"potential_pool_{as_of.replace('-', '')}.json"
    _write(path, payload)
    lines = [f"# 潜在标的池 {as_of}", NOTICE, f"\n汇总：{payload['summary']}",
             "\n|股票|状态|观察分|首次入池|点火/EMA10回踩/组合|证据/风险|", "|---|---|---|---|---|---|"]
    for r in payload["records"]:
        e = r["entry"]
        lines.append(f"|{r['symbol']} {r['name']}|{r['state']}|{r['score']}+{r['ignition_bonus']}|{r['first_seen']}|{e.get('ignition', False)}/{e.get('pullback', False)}/{e.get('combined', False)}|{'；'.join(r['evidence'] + r['risk_flags'] + r['missing'])}|")
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[potential_pool] {as_of} {payload['summary']} -> {path}", flush=True)
    return payload
