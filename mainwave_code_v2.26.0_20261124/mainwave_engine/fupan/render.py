"""Render a validated report into one self-contained dual-view HTML (CSS, JS, schema and data embedded)."""
import json
from pathlib import Path

from . import ASSETS
from .contract import validate


def render_html(data, template=None):
    errors = validate(data)
    if errors:
        raise ValueError("报告校验失败:\n" + "\n".join(errors))
    html = (template or ASSETS / "report_template.html").read_text(encoding="utf-8")
    # Escaping '<' prevents a value containing </script> from escaping the JSON data block.
    packed = (json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
              .replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
              .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))
    replacements = {"__REPORT_SCHEMA__": (ASSETS / "report.schema.json").read_text(encoding="utf-8").replace("<", "\\u003c"),
                    "/*__REPORT_CSS__*/": (ASSETS / "report.css").read_text(encoding="utf-8"),
                    "/*__REPORT_JS__*/": (ASSETS / "report.js").read_text(encoding="utf-8"),
                    "__REPORT_DATA__": packed}
    for marker, value in replacements.items():
        if html.count(marker) != 1:
            raise ValueError("模板标记必须且仅能出现一次: " + marker)
        html = html.replace(marker, value, 1)
    return html


def render(data, out, template=None):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(data, template), encoding="utf-8")
    return out
