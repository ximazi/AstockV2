"""本地日线归档读取。

v1 工作区 data/ 下每个标的一份 {symbol}_daily.csv。
只做只读读取与最小缓存, 不做任何复权或重算 —— 必须与 v1 记账时使用的
价格序列完全一致, 否则 MFE/MAE 与账本不可比。
"""
from __future__ import annotations

import functools

import pandas as pd

from .config import V1_DATA


def _path(symbol: str):
    return V1_DATA / f"{symbol}_daily.csv"


def has_bars(symbol: str) -> bool:
    return _path(symbol).exists()


COLS = [
    "date", "open", "high", "low", "close",
    "preclose", "volume", "amount", "tradestatus", "turnover", "pct_chg", "isST",
]


@functools.lru_cache(maxsize=512)
def load(symbol: str) -> pd.DataFrame:
    """返回该标的的日线, 按日期升序, 索引重置。"""
    p = _path(symbol)
    if not p.exists():
        raise FileNotFoundError(f"缺少日线档案: {p}")
    df = pd.read_csv(p, dtype={"date": str}, usecols=COLS)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def window(symbol: str, start: str, end: str) -> pd.DataFrame:
    """取 [start, end] 闭区间日线。"""
    df = load(symbol)
    m = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    return df.loc[m].reset_index(drop=True)


def forward_path(symbol: str, entry_date: str, bars: int) -> pd.DataFrame:
    """入场日(含)起的 bars 根日线。

    入场按次日开盘成交, 因此入场日当天的高低点都在路径内。
    数据不足时返回现有全部行。
    """
    df = load(symbol)
    idx = df.index[df["date"] >= pd.Timestamp(entry_date)]
    if len(idx) == 0:
        return df.iloc[0:0]
    start = int(idx[0])
    return df.iloc[start : start + bars].reset_index(drop=True)


def bars(symbol: str) -> pd.DataFrame:
    '''带指标与涨跌停幅度的日线, 供成交模型使用。'''
    from .execution import limit_pct, with_indicators
    df = with_indicators(load(symbol))
    st = pd.to_numeric(df.get('isST', pd.Series(0.0, index=df.index)), errors='coerce')
    df['isST'] = st.fillna(0)
    df['limit_pct'] = [limit_pct(symbol, bool(v)) for v in df['isST']]
    df.attrs['limit_pct'] = float(df['limit_pct'].iloc[-1]) if len(df) else 0.10
    return df


def index_of(bars: pd.DataFrame, date: str) -> int:
    '''日期在日线中的下标; 找不到返回 -1。'''
    hit = bars.index[bars['date'] == pd.Timestamp(date)]
    return int(hit[0]) if len(hit) else -1
