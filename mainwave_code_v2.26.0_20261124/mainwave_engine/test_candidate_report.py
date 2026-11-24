"""Empty observation pools must not interrupt the daily report pipeline."""
import json
import sys

import pandas as pd
import pytest

from mainwave_engine import candidate_report


@pytest.fixture
def reports(tmp_path, monkeypatch):
    out = tmp_path / "results/mainwave_engine/20260916"
    out.mkdir(parents=True)
    row = {
        "symbol": "sh_600001", "name": "Test", "industry": "Test industry",
        "route": "R1", "structure": "BASE", "supply": "CONTRACTING",
        "close": 10.0, "trigger_price": 10.2, "invalid_price": 9.5,
    }
    for name in ("ready_table.csv", "armed_watch.csv"):
        pd.DataFrame([row]).to_csv(out / name, index=False, encoding="utf-8-sig")
    pd.DataFrame(columns=["symbol"]).to_csv(out / "intersection_candidates.csv", index=False)
    (out / "p1_closure_report.md").write_text("# Canonical focus\nNo final candidates.\n", encoding="utf-8")
    pd.DataFrame([{"industry": "Test industry", "state": "NEUTRAL", "B": 0.5, "dB2": 0.0}]).to_csv(
        out / "sector_states.csv", index=False)
    (tmp_path / "external_theme_prior.json").write_text(json.dumps({"themes": []}), encoding="utf-8")
    pd.DataFrame([{
        "ts_code": "600001.SH", "ann_date": 20260801,
        "q_profit_yoy": 10.0, "or_yoy": 5.0, "netprofit_yoy": 10.0,
    }]).to_csv(tmp_path / "results/r4_fundamentals_panel.csv", index=False)
    monkeypatch.setattr(candidate_report, "ROOT", tmp_path)
    monkeypatch.setattr(candidate_report, "_stock_stats", lambda *_: {})
    monkeypatch.setattr(sys, "argv", ["candidate_report", "--date", "2026-09-16"])
    return out


@pytest.mark.parametrize("empty_names", [
    ("armed_watch.csv",), ("ready_table.csv",), ("ready_table.csv", "armed_watch.csv"),
])
@pytest.mark.parametrize("empty_format", ["zero_bytes", "bom_newline", "header_only"])
def test_empty_pools_still_publish_reports(reports, capsys, empty_names, empty_format):
    for name in empty_names:
        path = reports / name
        if empty_format == "header_only":
            pd.read_csv(path).iloc[:0].to_csv(path, index=False, encoding="utf-8-sig")
        else:
            path.write_bytes(b"" if empty_format == "zero_bytes" else b"\xef\xbb\xbf\r\n")
    candidate_report.main()
    assert f"raw_pool={2 - len(empty_names)}" in capsys.readouterr().out
    assert (reports / "candidate_report.md").read_bytes() == (reports / "p1_closure_report.md").read_bytes()
    assert "Carrier hard-gate exclusions" in (reports / "candidate_pool_diagnostics.md").read_text(encoding="utf-8")


def test_populated_pools_keep_all_candidates(reports, capsys):
    candidate_report.main()
    assert "raw_pool=2" in capsys.readouterr().out
    diagnostics = (reports / "candidate_pool_diagnostics.md").read_text(encoding="utf-8")
    assert "READY" in diagnostics and "ARMED" in diagnostics


@pytest.mark.parametrize("name", ["ready_table.csv", "armed_watch.csv"])
def test_missing_pool_is_not_treated_as_zero_candidates(reports, name):
    (reports / name).unlink()
    with pytest.raises(FileNotFoundError):
        candidate_report.main()
    assert not (reports / "candidate_report.md").exists()


@pytest.mark.parametrize("name", ["ready_table.csv", "armed_watch.csv"])
def test_invalid_schema_is_not_treated_as_zero_candidates(reports, name):
    pd.DataFrame([{"symbol": "sh_600001"}]).to_csv(reports / name, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        candidate_report.main()
    assert not (reports / "candidate_report.md").exists()
