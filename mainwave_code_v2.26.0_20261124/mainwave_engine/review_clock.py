"""Publication receipts for dated review inputs; filesystem times are not evidence."""
import hashlib
import json
from datetime import datetime, timedelta, timezone

from .pipeline import _write

CST = timezone(timedelta(hours=8))


def instant(value):
    if not value or "T" not in str(value):
        return None
    try:
        t = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return t.astimezone(CST) if t.tzinfo is not None else None


def publish(path, day):
    receipt = {"as_of": day, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "available_at": datetime.now(CST).isoformat()}
    _write(path.with_suffix(".published.json"), receipt)


def visible(path, day, cutoff):
    stamp = path.with_suffix(".published.json")
    if not path.is_file() or not stamp.is_file():
        return False
    try:
        receipt = json.loads(stamp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(receipt, dict):
        return False
    published, until = instant(receipt.get("available_at")), instant(cutoff)
    return bool(receipt.get("as_of") == day and published and until and published <= until
                and hashlib.sha256(path.read_bytes()).hexdigest() == receipt.get("sha256"))
