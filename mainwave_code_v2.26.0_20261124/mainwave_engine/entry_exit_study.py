"""Read-only research attachment. Never changes frozen cases or human reviews."""
import json
from pathlib import Path
import re

ROOT = Path("results/mainwave_engine/review_workbench")
DIRECTORY = "entry_exit_study_20260911"


def load(summary, root=ROOT):
    symbol, day = summary["symbol"], summary["signal_date"]
    if not re.fullmatch(r"(?:sh|sz|bj)_\d{6}", symbol) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
        raise ValueError("Invalid research attachment identity")
    directory = root / DIRECTORY
    if not (directory / "manifest.json").is_file():
        return None
    path = directory / f"{symbol}_{day}.json"
    if not path.is_file():
        return None
    study = json.loads(path.read_text(encoding="utf-8"))
    if study.get("symbol") != symbol or study.get("original_signal") != day:
        raise ValueError("Research attachment identity mismatch")
    return study
