"""CLI: python -m mainwave_engine.fupan --date YYYY-MM-DD  (pipeline step after eod_review; never blocks the pipeline)."""
import argparse
import json
from datetime import datetime
from pathlib import Path

from .. import eod_review as R
from .build import apply_analysis, build, persist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out-root", default=str(R.OUT_ROOT))
    parser.add_argument("--analysis", default="", help="analyzer JSON to merge into the latest report (no rebuild)")
    parser.add_argument("--analyze", action="store_true", help="after building, send the packet to the configured model and merge its answer")
    args = parser.parse_args()
    folder = Path(args.out_root) / args.date.replace("-", "") / "fupan"
    attempted_at = datetime.now().astimezone().isoformat(timespec="seconds")
    R._write(folder / "status.json", {"as_of": args.date, "status": "RUNNING", "attempted_at": attempted_at})
    try:
        if args.analysis:
            pointer = apply_analysis(args.date, json.loads(Path(args.analysis).read_text(encoding="utf-8-sig")), args.out_root)
        else:
            report, meta = build(args.date, args.out_root, args.data_root)
            pointer = persist(args.date, report, meta, args.out_root)
        print(f"[fupan] {args.date} {pointer['version']} reused={pointer['reused']} mode={pointer['mode']} "
              f"stocks={pointer['stocks']} issues={pointer['issues']} required={pointer['required_symbols']}", flush=True)
        ai = {"status": "NOT_REQUESTED"}
        if args.analyze:
            from .analyzer import run as analyze
            ai = analyze(args.date, args.out_root)
            if ai.get("pointer"):
                pointer = ai["pointer"]
            print(f"[fupan] analyzer {ai.get('status')} provider={ai.get('provider', '')} model={ai.get('model', '')} "
                  f"latency={ai.get('latency_s', '')}s usage={ai.get('usage', '')} {ai.get('error', '')}", flush=True)
        degraded = args.analyze and ai.get("status") not in ("MERGED", "ALREADY_MERGED")
        R._write(folder / "status.json", {"as_of": args.date, **pointer, "status": "PARTIAL" if degraded else "SUCCESS",
                                          "delivery_status": "DEGRADED" if degraded else "SUCCESS", "rule_report_available": True,
                                          "attempted_at": attempted_at,
                                          "analyzer": {k: v for k, v in ai.items() if k != "pointer"}})
    except Exception as exc:  # the review page must never block the main pipeline or hide its own failure
        R._write(folder / "status.json", {"as_of": args.date, "status": "DATA_WAIT" if str(exc).startswith("DATA_WAIT:") else "FAILED",
                                          "attempted_at": attempted_at, "error": str(exc)[:500]})
        print(f"[fupan] FAILED {args.date}: {exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
