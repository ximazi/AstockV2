from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd


STOCK_RE = re.compile(
    r"^(?:sh_(?:60|68)\d{4}|sz_(?:00|30)\d{4}|bj_[489]\d{5})$",
    re.IGNORECASE,
)


def normalize_file_symbol(value: Any) -> str:
    return str(value or "").strip().lower().replace(".", "_")


def is_stock_symbol(symbol: Any) -> bool:
    return bool(STOCK_RE.fullmatch(normalize_file_symbol(symbol)))


def _mapping_names(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8")).get("mapping", {})
    except Exception:
        return {}
    return {
        normalize_file_symbol(key): str(value.get("name", "") or "")
        for key, value in raw.items()
        if isinstance(value, dict)
    }


def _stock_list_names(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="gbk")
    except Exception:
        return {}
    if "ts_code" not in df.columns or "name" not in df.columns:
        return {}
    return {
        normalize_file_symbol(symbol): str(name or "")
        for symbol, name in zip(df["ts_code"], df["name"])
    }


@lru_cache(maxsize=8)
def load_st_delist_set(
    stock_list: str = "data/stock_list.csv",
    concept_mapping: str = "concept_mapping.json",
) -> frozenset[str]:
    names = _stock_list_names(Path(stock_list))
    for symbol, name in _mapping_names(Path(concept_mapping)).items():
        if name:
            names[symbol] = name
    blocked = {
        symbol
        for symbol, name in names.items()
        if "ST" in name.upper() or "退" in name
    }
    return frozenset(blocked)


def universe_rejection_reason(symbol: Any) -> str:
    normalized = normalize_file_symbol(symbol)
    if not is_stock_symbol(normalized):
        return "non_stock_symbol"
    if normalized in load_st_delist_set():
        return "st_or_delisting"
    return ""


def latest_tradeability(
    symbol: Any,
    data_dir: Path = Path("data"),
    as_of: str | None = None,
    min_amount: float = 5e7,
) -> tuple[bool, str, float]:
    normalized = normalize_file_symbol(symbol)
    path = data_dir / f"{normalized}_daily.csv"
    if not path.exists():
        return False, "price_data_missing", 0.0
    try:
        df = pd.read_csv(path, usecols=["date", "open", "high", "low", "close", "amount"])
    except Exception:
        return False, "tradeability_data_missing", 0.0
    if df.empty:
        return False, "tradeability_data_empty", 0.0
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if as_of:
        df = df[df["date"] <= pd.Timestamp(as_of)]
    if df.empty:
        return False, "tradeability_date_missing", 0.0
    row = df.sort_values("date").iloc[-1]
    amount_raw = pd.to_numeric(row.get("amount"), errors="coerce")
    amount = 0.0 if pd.isna(amount_raw) else float(amount_raw)
    if amount < min_amount:
        return False, "amount_below_50m", amount
    prices = [float(row.get(col, 0.0) or 0.0) for col in ["open", "high", "low", "close"]]
    if min(prices) > 0 and max(prices) / min(prices) - 1.0 < 0.001:
        return False, "one_price_board", amount
    return True, "", amount
