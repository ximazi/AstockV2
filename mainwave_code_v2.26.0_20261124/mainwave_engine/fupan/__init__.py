"""Daily review V2 (双层复盘): local facts → review.report.v2 → validated dual-view HTML, plus a bounded AI
evidence packet. The page is read-only: it never gets order authority, and every number must be
recomputable from the artefacts the closure already froze."""
from pathlib import Path

VERSION = "fupan_v2_5_20261029"
SCHEMA_VERSION = "review.report.v2"
AI_SCHEMA_VERSION = "review.ai.input.v2"
ASSETS = Path(__file__).resolve().parent / "assets"
