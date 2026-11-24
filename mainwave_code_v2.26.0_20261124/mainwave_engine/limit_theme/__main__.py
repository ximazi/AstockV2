import argparse
from pathlib import Path

from .. import eod_review as R, intraday_quotes as Q, model_cli
from .review import run


def main():
    parser = argparse.ArgumentParser(description="涨停股多源证据与本地Codex归因，独立只读模块")
    parser.add_argument("--date", required=True)
    parser.add_argument("--out-root", default=str(R.OUT_ROOT))
    parser.add_argument("--symbols", default="", help="可选逗号分隔内部代码；子集不发布全市场汇总")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--refresh-sources", action="store_true")
    args = parser.parse_args()
    day = Q.stamp(args.date).date().isoformat() if "-" in args.date else f"{args.date[:4]}-{args.date[4:6]}-{args.date[6:]}"
    cfg = model_cli.config()
    if args.collect_only:
        cfg["provider"] = "off"
    try:
        run(day, args.out_root, args.symbols.split(",") if args.symbols else None, cfg=cfg, refresh=args.refresh_sources)
    except Exception as exc:
        R._write(Path(args.out_root) / day.replace("-", "") / "limit_theme/status.json",
                 {"as_of": day, "status": "FAILED", "error": str(exc)[:400]})
        print(f"[limit_theme] FAILED {exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
