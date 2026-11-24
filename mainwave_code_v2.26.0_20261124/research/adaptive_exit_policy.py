"""Research variants; the approved evidence rules are shared with production."""
from __future__ import annotations

from mainwave_engine.exit_policy import market_frame, enrich, warning

VERSION = "adaptive_exit_v3_20260911"
ARMS = {
    "original20": ("stage36_20", None),
    "duration60": ("stage36_60", None),
    "trend20": ("stage36_20", "TREND"),
    "trend60": ("stage36_60", "TREND"),
    "risk20": ("stage36_20", "RISK"),
    "risk60": ("stage36_60", "RISK"),
    "wide_risk60": ("stage36_60", "WIDE_RISK"),
    "close_risk20": ("stage36_20", "CLOSE_RISK"),
    "close_risk60": ("stage36_60", "CLOSE_RISK"),
    "strong_core60": ("stage36_60", "CLOSE_CORE"),
    "review21": ("stage36_60", "REVIEW_ONLY"),
    "extend_risk60": ("stage36_60", "EXTEND_RISK"),
    "extend_close60": ("stage36_60", "EXTEND_CLOSE"),
}
