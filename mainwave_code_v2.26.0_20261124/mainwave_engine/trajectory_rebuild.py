# -*- coding: utf-8 -*-
"""Read-only reconstruction of the feature trajectory at production as-of.

This command never advances or rewrites the production state machine. The
production state attached to each row is read from the already-closed store,
so an exact reconstruction is only allowed for ``engine_as_of``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import data as D
from . import features as F
from . import observation as O
from . import sectors as S
from . import state_machine as SM


def rebuild_as_of(run_date: str, evaluate=None, prefilter=None) -> int:
    if evaluate is None or prefilter is None:
        from .daily_run import evaluate_symbol, prefilter_channels
        evaluate = evaluate_symbol
        prefilter = prefilter_channels
    store = SM.load_store()
    engine_as_of = store.get("_meta", {}).get("engine_as_of", "")
    if run_date != engine_as_of:
        raise SM.StoreError(
            "read-only trajectory reconstruction requires date == engine_as_of "
            f"({engine_as_of or 'EMPTY'}); use research.theme_ignition.audit_replay "
            "for historical feature/state replay"
        )

    universe = D.list_universe()
    ind_map = D.industry_map()
    con_map = D.concepts_map(run_date)
    per_stock = {}
    prefilter_hits = {}
    lim_of = {}
    for symbol in universe:
        frame = D.load_tail(symbol, 300, as_of=run_date)
        if frame is None:
            continue
        per_stock[symbol] = frame
        lim_of[symbol] = D.limit_pct(symbol)
        if str(frame["date"].iloc[-1]) != run_date:
            continue
        try:
            context = F.Ctx(symbol, frame)
        except ValueError:
            continue
        fund_ok = bool(F.fundamental_anchor(
            D.fundamental_view(D.code6(symbol), run_date))["ok"])
        hit, channels = prefilter(context, fund_ok)
        if hit:
            prefilter_hits[symbol] = channels

    group_map = {
        symbol: ([ind_map[symbol]] if symbol in ind_map else [])
        + [f"C:{concept}" for concept in con_map.get(symbol, [])]
        for symbol in per_stock
    }
    panel, market_median = S.build_breadth(per_stock, group_map, lim_of)
    sector_table = S.sector_states(panel)
    group_state = dict(zip(sector_table["industry"], sector_table["state"]))
    group_breadth = dict(zip(sector_table["industry"], sector_table["B"]))
    ignitions = S.leader_ignitions(per_stock, group_map, lim_of, run_date)

    tracked = set(prefilter_hits) | {
        symbol for symbol, record in store["symbols"].items()
        if record.get("state") not in (None, "OUT")
    }
    rows = []
    for symbol in sorted(tracked):
        frame = per_stock.get(symbol)
        state = store["symbols"].get(symbol, {}).get("state", "OUT")
        if frame is None or str(frame["date"].iloc[-1]) != run_date:
            rows.append(O.trajectory_row(
                run_date, symbol, state, {}, data_fresh=False,
                prefilter_ch=prefilter_hits.get(symbol, "")))
            continue
        try:
            evidence = evaluate(
                symbol, frame, run_date, panel, group_state, group_breadth, set(),
                ignitions, market_median, store["symbols"].get(symbol))
        except ValueError:
            continue
        evidence.pop("_ctx", None)
        evidence["prefilter_ch"] = prefilter_hits.get(symbol, "")
        rows.append(O.trajectory_row(
            run_date, symbol, state, evidence, data_fresh=True,
            prefilter_ch=evidence["prefilter_ch"]))

    output = Path("results/mainwave_engine") / run_date.replace("-", "") / "trajectory.csv"
    O.write_trajectory(rows, output)
    print(f"[trajectory-only] {run_date}: {len(rows)} rows -> {output}")
    print("[trajectory-only] production state store was not modified")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    date = args.date if "-" in args.date else f"{args.date[:4]}-{args.date[4:6]}-{args.date[6:]}"
    return rebuild_as_of(date)


if __name__ == "__main__":
    raise SystemExit(main())
