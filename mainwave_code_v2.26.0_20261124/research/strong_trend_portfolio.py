"""Two-stock-seat shared-account replay of the calibrated event domain (audit v1.2 Q-11, with the Q-10 baseline).

Each session: exits at the open first (calibrated exits: only on a tradable open), then last night's picks are
bought at the open if the open is inside the frozen band (no substitute if not), sized on current equity at 1%
risk to the invalidation, 35% symbol cap, cash limited; then tonight's picks fill only the free seats, never a
symbol already held. Positions are marked to the adjusted close every session; unfinished ones at the end.
Policies: R0 = current night order (non-limit first, RS20 desc, RS5 desc, code) and RANDOM = same-day random
picks among the same plan-eligible executable events (baseline for whether R0 adds anything).
The ETF seat is not modelled. Research only: no parameter here is chosen from these results.
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
CAPITAL, RISK, CAP, SEATS = 100000.0, 0.01, 0.35, 2
FILLED = ("CLOSED", "OPEN", "EXIT_PENDING")


def fee(notional, sell):
    return max(5.0, notional * 3e-4) + (notional * 5e-4 if sell else 0.0)


class Prices:
    """Adjusted closes per symbol, loaded once; ratios are basis-free so any later data does not matter."""
    def __init__(self, day):
        self.day, self.cache = day, {}

    def frame(self, symbol):
        if symbol not in self.cache:
            f = ST.adjusted(SEM.load(ROOT / "data" / f"{symbol}_daily.csv", self.day))
            self.cache[symbol] = (dict(zip(f["date"], f["close"])), dict(zip(f["date"], f["open"])))
        return self.cache[symbol]


def r0_key(r):
    return (bool(r.close_at_limit), -(r.rs20 if pd.notna(r.rs20) else -9), -(r.rs5 if pd.notna(r.rs5) else -9), r.symbol)


def replay(events, calendar, prices, policy="R0", seed=0):
    rng = np.random.default_rng(seed)
    by_day = {d: g for d, g in events.groupby("date")}
    cash, positions, pending, curve, trades, unfilled, seat_days = CAPITAL, {}, [], [], [], 0, 0
    for t, day in enumerate(calendar):
        # 1. exits at the open
        for sym in [s for s, p in positions.items() if p["exit_date"] == day and p["trade"] == "CLOSED"]:
            p = positions.pop(sym)
            proceeds = p["qty"] * p["entry"] * p["gross_ratio"]
            cash += proceeds - fee(proceeds, True)
            pnl = proceeds - fee(proceeds, True) - p["cost"]
            trades.append({**{k: p[k] for k in ("symbol", "signal_date", "entry_date", "prototype", "kind")}, "exit_date": day,
                           "qty": p["qty"], "pnl": round(pnl, 1), "net_r": round(pnl / (p["qty"] * p["per_share_risk"]), 3)})
        # 2. last night's picks at the open
        equity = cash + sum(mark(p, prices, calendar[t - 1] if t else day) for p in positions.values())
        for ev in pending:
            if ev.open_state != "IN_BAND" or ev.trade not in FILLED or ev.symbol in positions or ev.entry_date != day:
                unfilled += 1
                continue
            qty = math.floor(min(RISK * equity / ev.per_share_risk, CAP * equity / ev.entry, (cash - 10) / ev.entry) / 100) * 100
            if qty < 100:
                unfilled += 1
                continue
            cost = qty * ev.entry + fee(qty * ev.entry, False)
            cash -= cost
            positions[ev.symbol] = {"symbol": ev.symbol, "signal_date": ev.date, "entry_date": ev.entry_date, "prototype": ev.prototype,
                                    "kind": ev.kind, "qty": qty, "entry": ev.entry, "cost": cost, "per_share_risk": ev.per_share_risk,
                                    "gross_ratio": ev.gross_ratio, "exit_date": ev.exit_date, "trade": ev.trade,
                                    "entry_adj": prices.frame(ev.symbol)[1].get(ev.entry_date)}
        pending = []
        # 3. tonight's picks for free seats
        free = SEATS - len(positions)
        cands = by_day.get(day)
        if free > 0 and cands is not None:
            cands = cands[~cands.symbol.isin(list(positions))]
            if len(cands):
                ordered = (sorted(cands.itertuples(), key=policy) if callable(policy)
                           else sorted(cands.itertuples(), key=r0_key) if policy == "R0"
                           else [cands.iloc[i] for i in rng.permutation(len(cands))])
                pending = list(ordered[:free])
        # 4. mark at the close
        seat_days += len(positions)
        curve.append({"date": day, "equity": cash + sum(mark(p, prices, day) for p in positions.values()), "positions": len(positions)})
    final = calendar[-1]
    for p in positions.values():
        value = mark(p, prices, final)
        trades.append({**{k: p[k] for k in ("symbol", "signal_date", "entry_date", "prototype", "kind")}, "exit_date": None,
                       "qty": p["qty"], "pnl": round(value - p["cost"], 1), "net_r": round((value - p["cost"]) / (p["qty"] * p["per_share_risk"]), 3),
                       "open_at_end": True})
    return pd.DataFrame(curve), pd.DataFrame(trades), {"unfilled_picks": unfilled, "seat_utilisation": round(seat_days / (SEATS * len(calendar)), 3)}


def mark(p, prices, day):
    closes = prices.frame(p["symbol"])[0]
    close = closes.get(day)
    if close is None:  # suspended: last known close
        earlier = [d for d in closes if d <= day]
        close = closes[max(earlier)] if earlier else p["entry_adj"]
    return p["qty"] * p["entry"] * close / p["entry_adj"] if p["entry_adj"] else p["qty"] * p["entry"]


def metrics(curve, trades, extra):
    eq = curve.equity.to_numpy()
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1
    years = len(curve) / 244
    closed = trades[trades.get("open_at_end", pd.Series(False, index=trades.index)).fillna(False) == False] if len(trades) else trades  # noqa: E712
    top5 = trades.pnl.nlargest(5).sum() if len(trades) else 0
    total = eq[-1] - CAPITAL
    return {"final_equity": round(float(eq[-1]), 0), "total_return": round(float(eq[-1] / CAPITAL - 1), 4),
            "cagr": round(float((eq[-1] / CAPITAL) ** (1 / years) - 1), 4) if years > 0 and eq[-1] > 0 else None,
            "max_drawdown": round(float(dd.min()), 4), "trades": int(len(trades)), "closed": int(len(closed)),
            "win_rate": round(float((trades.pnl > 0).mean()), 3) if len(trades) else None,
            "mean_net_r": round(float(trades.net_r.mean()), 3) if len(trades) else None,
            "top5_share_of_pnl": round(float(top5 / total), 3) if total > 0 else None, **extra}


def main(day="2026-09-23", start="2024-01-01", seeds=20):
    pool = pd.read_csv(ROOT / "results/research/case_pool" / day.replace("-", "") / "event_domain.csv", encoding="utf-8-sig")
    ev = pool[(pool.domain == "EXECUTABLE") & pool.plan_eligible.astype(bool) & (pool.date >= start)].copy()
    index = pd.read_csv(ROOT / "data/000001_SH_index.csv")
    calendar = sorted(d for d in pd.to_datetime(index.trade_date.astype(str)).dt.strftime("%Y-%m-%d") if start <= d <= day)
    prices = Prices(day)
    out = ROOT / "results/research/portfolio" / day.replace("-", "")
    out.mkdir(parents=True, exist_ok=True)
    curve, trades, extra = replay(ev, calendar, prices, "R0")
    curve.to_csv(out / "r0_curve.csv", index=False)
    trades.to_csv(out / "r0_trades.csv", index=False, encoding="utf-8-sig")
    r0 = metrics(curve, trades, extra)
    random = []
    for s in range(seeds):
        c, t, x = replay(ev, calendar, prices, "RANDOM", seed=s)
        random.append(metrics(c, t, x))
    rnd = pd.DataFrame(random)
    summary = {"window": [calendar[0], calendar[-1]], "sessions": len(calendar), "eligible_events": len(ev),
               "R0": r0,
               "RANDOM": {k: {"mean": round(float(rnd[k].mean()), 4), "median": round(float(rnd[k].median()), 4),
                              "min": round(float(rnd[k].min()), 4), "max": round(float(rnd[k].max()), 4)}
                          for k in ("total_return", "max_drawdown", "win_rate", "mean_net_r", "trades", "seat_utilisation")},
               "R0_rank_among_random_total_return": int((rnd.total_return < r0["total_return"]).sum()), "random_seeds": seeds,
               "r0_by_prototype": trades.groupby("prototype").agg(n=("pnl", "size"), pnl=("pnl", "sum"), mean_r=("net_r", "mean")).round(3).to_dict("index") if len(trades) else {},
               "r0_by_year": trades.assign(y=trades.entry_date.str[:4]).groupby("y").agg(n=("pnl", "size"), pnl=("pnl", "sum")).round(1).to_dict("index") if len(trades) else {},
               "note": "两股票席位共享账户逐日回放（研究台校准后）；ETF席位未建模；不作参数选择依据。RANDOM为同日期同资格随机选择的基准。"}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main(*(a for a in sys.argv[1:]))
