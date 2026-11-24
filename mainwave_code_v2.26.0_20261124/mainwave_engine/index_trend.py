"""核心指数中期趋势环境门。

只做多的市场里, 同一个形态在指数中期趋势向上和向下时含义不同。AStockV2 的
分层检验 (D:/AICode/AStockV2/docs/市场环境分层与入场时机_20261120.md) 显示:
S4 追启动日次日开盘的入场时点, 在核心指数中期上行时比"同一只股票随机某天
入场"系统性差约 1%/笔; 3 个指数 x 3 种 EMA 长度 x 3 种回看 = 27 组参数
全部显著为负、零组反向, 去掉最大赢家不变。指数下行时方向相反, 但样本只有
3-4 段独立行情, 不足以作为正面结论。

所以这里只做一件事: 把"核心指数 EMA100 斜率为正"标成不开新仓的环境。
判据只用当日收盘已知的信息 (EMA 是因果的, 斜率只用当日与 lookback 根之前
的值)。数据不足时按未知处理并拒绝开新仓, 与中证1000 趋势门一致: 风险门
不因数据缺失而放行。

与 mainwave_engine.trading.position_monitor.market_risk 的分工:
  market_risk  中证1000 收盘 vs MA20, 快门, 管短期趋势破位;
  index_trend  沪深300 EMA100 斜率, 慢门, 管中期趋势方向下的形态入场时机。
两者不是同一个判据, 也不是同一个周期。

python -c "from mainwave_engine import index_trend as T; print(T.regime('2026-11-19'))"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

VERSION = "index_trend_v1_20261120"
INDEX_FILE = "000300_SH_index.csv"
INDEX_NAME = "沪深300"
SPAN = 100
LOOKBACK = 10
MIN_BARS = 250
RULE = ("核心指数中期趋势门: 沪深300 EMA100 与 10 个交易日前的比值。"
        "斜率为正 -> 不开 S4 新仓 (该环境下形态启动日入场的时点价值为负); "
        "斜率为负 -> 允许 (但尚无充分样本证明更好); 数据不足 -> 未知, 不开新仓。")
EVIDENCE = ("依据 AStockV2 分层检验 (2026-11-20): 27 组参数下 S4 入场时点"
            "在该环境显著劣于同股随机入场; 指数下行方向的证据样本不足, "
            "仅作观察, 不作为正面结论。")


def _frame(as_of: str, data_root) -> pd.DataFrame:
    path = Path(data_root) / INDEX_FILE
    f = pd.read_csv(path)
    f["date"] = pd.to_datetime(f["trade_date"].astype(str)).dt.strftime("%Y-%m-%d")
    f["close"] = pd.to_numeric(f["close"], errors="coerce")
    f = f.dropna(subset=["date", "close"]).sort_values("date")
    return f[f["date"] <= as_of].drop_duplicates("date").reset_index(drop=True)


def regime(as_of: str, data_root=None, span: int = SPAN,
           lookback: int = LOOKBACK) -> dict:
    """核心指数中期趋势环境。返回的 gate 取 BLOCK / ALLOW / UNKNOWN。"""
    base = {"as_of": as_of, "index": INDEX_NAME, "version": VERSION, "rule": RULE,
            "evidence": EVIDENCE, "report_only": False,
            "gate": "UNKNOWN", "regime": "UNKNOWN", "status": "UNKNOWN",
            "data_date": "", "close": None, "ema": None, "slope": None, "text": ""}
    try:
        f = _frame(as_of, data_root or Path("data"))
    except (OSError, ValueError, KeyError, pd.errors.EmptyDataError) as exc:
        return {**base, "status": "MISSING",
                "text": f"{INDEX_NAME} 日线不可用 ({str(exc)[:80]}), 中期趋势未知, 不开 S4 新仓"}
    if len(f) < MIN_BARS:
        return {**base, "status": "INSUFFICIENT_HISTORY", "data_date": str(f["date"].iloc[-1]) if len(f) else "",
                "text": f"{INDEX_NAME} 历史不足 {MIN_BARS} 个交易日, 中期趋势未知, 不开 S4 新仓"}
    c = f["close"].astype(float).reset_index(drop=True)
    ema = c.ewm(span=span, adjust=False).mean()
    slope = float(ema.iloc[-1] / ema.iloc[-1 - lookback] - 1)
    close = float(c.iloc[-1])
    data_date = str(f["date"].iloc[-1])
    fresh = data_date == as_of
    if not fresh:
        return {**base, "status": "STALE", "data_date": data_date, "close": round(close, 2),
                "ema": round(float(ema.iloc[-1]), 2), "slope": round(slope, 4),
                "text": f"{INDEX_NAME} 最新日线为 {data_date}, 不是 {as_of}, 中期趋势未确认, 不开 S4 新仓"}
    up = slope > 0
    return {**base, "status": "FRESH", "data_date": data_date, "close": round(close, 2),
            "ema": round(float(ema.iloc[-1]), 2), "slope": round(slope, 4),
            "regime": "UP" if up else "DOWN",
            "gate": "BLOCK" if up else "ALLOW",
            "text": (f"{INDEX_NAME} EMA{span} 中期趋势向上 (10 日斜率 {slope*100:+.2f}%), "
                     f"该环境下形态启动日入场的时点价值为负, 不开 S4 新仓"
                     if up else
                     f"{INDEX_NAME} EMA{span} 中期趋势向下 (10 日斜率 {slope*100:+.2f}%), 允许 S4 新仓"
                     "(该环境的正面证据样本不足, 只作观察)")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--data-root", default="data")
    args = ap.parse_args()
    out = regime(args.date, args.data_root)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
