"""Check the actual daily writer without advancing production state."""
from pathlib import Path
import subprocess
import sys


def test_zero_candidate_states_keep_csv_headers(tmp_path):
    # daily_run wraps stdout at import, so isolate it from pytest capture.
    script = r'''
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
from mainwave_engine import daily_run as run

date = "2026-09-16"
symbol = "sh_600001"
bars = pd.DataFrame([{"date": date, "close": 10.0, "pct_chg": 0.0}])
sectors = pd.DataFrame([{"industry": "Test", "state": "NEUTRAL", "B": 0.5}])
evidence = {"_ctx": SimpleNamespace(), "industry": "Test", "route": "test",
            "close": 10.0, "sector_state": "NEUTRAL"}
os.chdir(sys.argv[1])
with ExitStack() as stack:
    mocks = [
        (run, "market_calendar", [date]), (run.D, "list_universe", [symbol]),
        (run.D, "industry_map", {symbol: "Test"}), (run.D, "concepts_map", {}),
        (run.D, "load_tail", bars), (run.D, "limit_pct", 10.0),
        (run.D, "fundamental_view", {}), (run.D, "st_set", set()), (run.D, "name_map", {}),
        (run.F, "Ctx", SimpleNamespace()), (run.F, "fundamental_anchor", {"ok": False}),
        (run, "prefilter_channels", (True, "1")), (run, "evaluate_symbol", evidence),
        (run.S, "build_breadth", (pd.DataFrame(), None)), (run.S, "sector_states", sectors),
        (run.S, "leader_ignitions", {}), (run.SM, "load_store", {"symbols": {}}),
        (run.SM, "save_store", None), (run.SM, "step", {"state": "FORMING"}),
    ]
    for owner, name, result in mocks:
        stack.enter_context(patch.object(owner, name, return_value=result))
    assert run.run_day(date) == 0
    required = {"symbol", "name", "industry", "route", "close", "trigger_price", "invalid_price"}
    extra = {
        "ready_table.csv": {"ready_at", "structure", "supply", "supply_score", "ev_date", "sector"},
        "armed_watch.csv": {"armed_at", "structure", "supply", "ev_date", "resonance_theme"},
        "triggered.csv": {"kind"},
    }
    for name, columns in extra.items():
        frame = pd.read_csv(run.OUT_ROOT / "20260916" / name)
        assert frame.empty, name
        assert set(frame.columns) == required | columns, (name, frame.columns.tolist())
        print(f"PASS {name}: 0 rows, {len(frame.columns)} columns")
'''
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, encoding="utf-8", timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("PASS ") == 3
