"""逐笔交易路径分解: 把两席账本里的每一笔拆成可归因的失败模式。

目的不是重新选股, 而是回答一个具体问题:
亏损到底来自信号错了, 还是退出错了。
判据只用入场与离场之后的实际价格路径, 不引入未来信息做选股。
"""
from __future__ import annotations

import pandas as pd

from .bars import forward_path
from .config import FWD_BARS, MFE_MATERIAL, MFE_WEAK

LEDGER_COLS = [
    "day", "symbol", "status", "entry", "price", "qty",
    "notional", "risk", "exit", "pnl", "reason",
]


def load_ledger(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    for c in LEDGER_COLS:
        if c not in df.columns:
            df[c] = None
    for c in ("price", "qty", "notional", "risk", "pnl"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[LEDGER_COLS].copy()


def _max_high(df: pd.DataFrame) -> float:
    return float(df["high"].max()) if len(df) else float("nan")


def _min_low(df: pd.DataFrame) -> float:
    return float(df["low"].min()) if len(df) else float("nan")


def trade_path(symbol, entry_date, entry_price, exit_date, bars: int = FWD_BARS):
    """一笔交易的入场后价格路径指标。

    离场后的价格仅用于事后归因, 不参与任何当时的决策。
    返回 (指标字典, 持仓段日线, 全窗口日线)。
    """
    path = forward_path(symbol, entry_date, bars)
    if path.empty or not entry_price:
        return {}, path, path
    ex = pd.Timestamp(exit_date) if exit_date else None
    hold, after = path, path.iloc[0:0]
    if ex is not None:
        mask = path["date"] <= ex
        hold, after = path.loc[mask], path.loc[~mask]
    out = {
        "bars_held": int(len(hold)),
        "mfe": _max_high(path) / entry_price - 1.0,
        "mae": _min_low(path) / entry_price - 1.0,
        "mfe_hold": _max_high(hold) / entry_price - 1.0,
        "post_max": (_max_high(after) / entry_price - 1.0) if len(after) else float("nan"),
        "post_bars": int(len(after)),
    }
    return out, hold, path


def r_metrics(entry_price: float, risk_per_share: float, risk_total: float, pnl: float,
              hold: pd.DataFrame, full: pd.DataFrame) -> dict:
    """把一笔交易换算成风险倍数 R。

    R 的分母是初始止损距离。账本里的 risk 列除以 qty 就是每股风险,
    因此 stop = entry - risk/qty, 不需要重新推导 ATR。
    realized_r 用 pnl/risk, 与 v1 的 net_R 同口径。

    mfe_r_hold 是持仓期内最大浮盈, mfe_r_full 是入场后 20 根内最大浮盈;
    capture = realized_r / mfe_r_hold 衡量涨到的钱留下了多少。
    """
    if not risk_per_share or not entry_price:
        return {}
    den = float(risk_per_share)
    out = {"realized_r": float(pnl) / float(risk_total) if risk_total else float("nan")}
    if len(full):
        out["mfe_r_full"] = (_max_high(full) - entry_price) / den
        out["mae_r_full"] = (_min_low(full) - entry_price) / den
    if len(hold):
        out["mfe_r_hold"] = (_max_high(hold) - entry_price) / den
        out["mae_r_hold"] = (_min_low(hold) - entry_price) / den
        m = out.get("mfe_r_hold")
        out["capture"] = out["realized_r"] / m if m and m > 0.05 else float("nan")
    return out


def classify(pnl: float, mfe_hold: float, post_max: float) -> str:
    """把一笔交易归入一个失败模式。

    判定顺序有讲究: 离场后仍涨过入场价, 说明信号方向没错, 是退出问题;
    这类必须排在持仓期 MFE 之前, 否则会被持仓期 MFE 掩盖。
    """
    nan = post_max != post_max
    if pnl > 0:
        if not nan and post_max >= MFE_MATERIAL:
            return "WINNER_TRUNCATED"
        return "WINNER_OK"
    if not nan and post_max >= MFE_MATERIAL:
        return "EXIT_TOO_EARLY"
    if mfe_hold == mfe_hold and mfe_hold >= MFE_MATERIAL:
        return "HOLD_MFE_GAVE_BACK"
    if mfe_hold == mfe_hold and mfe_hold >= MFE_WEAK:
        return "WEAK_FOLLOWTHROUGH"
    return "ENTRY_NO_FOLLOWTHROUGH"


def enrich(ledger: pd.DataFrame) -> pd.DataFrame:
    """给账本每一行补上路径指标与失败模式。"""
    rows = []
    for r in ledger.itertuples(index=False):
        rec = {
            "day": r.day, "symbol": r.symbol, "status": r.status,
            "entry": r.entry, "price": r.price, "qty": r.qty,
            "notional": r.notional, "risk": r.risk,
            "exit": r.exit, "pnl": r.pnl, "reason": r.reason,
        }
        for extra in ("year", "arm"):
            if hasattr(r, extra):
                rec[extra] = getattr(r, extra)
        if r.status == "CLOSED" and r.entry and r.price:
            metrics, hold, full = trade_path(r.symbol, r.entry, r.price, r.exit)
            rec.update(metrics)
            risk_ps = (r.risk / r.qty) if (r.qty and r.risk == r.risk) else 0.0
            rec.update(r_metrics(r.price, risk_ps, r.risk, r.pnl, hold, full))
            rec["stop_price"] = r.price - risk_ps if risk_ps else float("nan")
            rec["mode"] = classify(r.pnl, rec.get("mfe_hold", float("nan")),
                                   rec.get("post_max", float("nan")))
        else:
            rec["mode"] = ""
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize(enriched: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    """按给定维度汇总: 笔数, 盈亏, 失败模式分布。"""
    d = enriched[enriched["status"] == "CLOSED"].copy()
    if not len(d):
        return pd.DataFrame(), pd.DataFrame()

    def _row(d):
        p = d["pnl"]
        w = p[p > 0]
        l = p[p <= 0]
        return {
            "trades": int(len(p)),
            "wins": int(len(w)),
            "pnl": float(p.sum()),
            "avg_pnl": float(p.mean()) if len(p) else 0.0,
            "avg_win": float(w.mean()) if len(w) else 0.0,
            "avg_loss": float(l.mean()) if len(l) else 0.0,
        }

    if group:
        rows = []
        for key, g in d.groupby(group):
            key = key if isinstance(key, tuple) else (key,)
            rec = dict(zip(group, key))
            rec.update(_row(g))
            rows.append(rec)
        out = pd.DataFrame(rows)
        keys = group + ["mode"]
    else:
        out = pd.DataFrame([_row(d)])
        keys = ["mode"]
    mode = (d.assign(mode=d["mode"].replace("", "NA"))
              .groupby(keys)
              .agg(n=("pnl", "size"), pnl=("pnl", "sum"))
              .reset_index())
    return out, mode
