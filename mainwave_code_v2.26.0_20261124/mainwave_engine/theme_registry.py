# -*- coding: utf-8 -*-
"""Point-in-time tight-chain registry used by the observation layer."""
from __future__ import annotations

import json
import re
from pathlib import Path


REGISTRY = Path("theme_registry.json")
SYMBOL_RE = re.compile(r"^(sh_60|sh_68|sz_00|sz_30)\d{4}$")


class RegistryError(RuntimeError):
    pass


def load_registry(path: Path = REGISTRY) -> dict:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RegistryError(f"题材注册库不可读: {path} ({exc})") from exc
    if raw.get("schema_version") != 1 or not isinstance(raw.get("themes"), list):
        raise RegistryError(f"题材注册库格式错误: {path}")
    for row in raw["themes"]:
        required = {"id", "label", "active_from", "source", "members"}
        if not required.issubset(row) or not isinstance(row["members"], dict):
            raise RegistryError(f"题材注册项缺字段: {row}")
        bad = [s for s in row["members"] if not SYMBOL_RE.match(s)]
        if bad:
            raise RegistryError(f"题材 {row['id']} 含非法股票代码: {bad[:3]}")
    return raw


def themes_as_of(as_of: str, path: Path = REGISTRY) -> list[dict]:
    """Return the latest registered version of each theme known on ``as_of``."""
    rows = [r for r in load_registry(path)["themes"] if r["active_from"] <= as_of]
    latest: dict[str, dict] = {}
    for row in sorted(rows, key=lambda x: (x["id"], x["active_from"])):
        latest[row["id"]] = row
    return [latest[k] for k in sorted(latest)]


def latest_themes(path: Path = REGISTRY) -> list[dict]:
    """Return latest versions regardless of active date, for labelled backfill only."""
    latest: dict[str, dict] = {}
    for row in sorted(load_registry(path)["themes"], key=lambda x: (x["id"], x["active_from"])):
        latest[row["id"]] = row
    return [latest[k] for k in sorted(latest)]
