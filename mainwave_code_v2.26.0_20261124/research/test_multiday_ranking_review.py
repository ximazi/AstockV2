import copy

import pandas as pd
import pytest

from research import multiday_ranking_review as study
from research.test_recent_ranking_review import raw_prices


def row(day, **changes):
    result = dict(symbol="sh_600000", name="fixture", date=day, tracking_only=False, value_rank=1,
                  value_tier="MARKET_WAIT", structural_advice="WAIT_CONFIRMATION", route_ids="R1",
                  price_date=day, current_core=True, price_features=True, stage="PULLBACK", close=12.,
                  cycle_id="cycle-a", setup_id="setup-a", setup_kind="FIRST_PULLBACK", setup_status="FORMING",
                  retained=True, supply=True, demand=False, strong_close=False, position60=.3,
                  turnover_rel=1., amount_rel=1., close_location=.5, route_bias_atr=1.,
                  new_low20=False, in_band=False, route_ready=True, market_allowed=False,
                  initial_review=False, supply_pressure=False, stock_lifecycle_label="fixture",
                  core_version=study.CONFIG["snapshot_version"])
    result.update(changes)
    return result


def step(day, session, rows, pool, history):
    return study.advance_pool(day, session, rows, pool, history)


def test_refreshed_tracking_stock_competes_without_rediscovery():
    pool, history = {}, {}
    step("2026-09-14", 0, [row("2026-09-14")], pool, history)
    result = step("2026-09-15", 1, [row("2026-09-15", tracking_only=True, demand=True,
                                       setup_status="CONFIRMED", structural_advice="PRIORITY_REVIEW", in_band=True)], pool, history)[0]
    assert not result["today_candidate"]
    assert result["pool_rank"] == 1
    assert result["components"]["retest_confirmed"] == 2
    assert result["score_delta"] > 0
    assert result["structure_review"]
    assert not result["formal_initial_review"]
    assert result["first_seen"] == "2026-09-14"
    assert result["last_discovery"] == "2026-09-14"


def test_no_same_day_or_unbounded_repetition_bonus():
    pool, history = {}, {}
    scores = []
    for session in range(8):
        day = f"2026-09-{session + 10:02}"
        r = step(day, session, [row(day)], pool, history)[0]
        scores.append(r["score"])
    assert scores == [4, 5, 6, 6, 6, 6, 6, 6]
    history_before = copy.deepcopy(history)
    # Rebuilding a report starts with empty state, not yesterday's accumulated score.
    pool2, history2 = {}, {}
    for session in range(8):
        day = f"2026-09-{session + 10:02}"
        step(day, session, [row(day)], pool2, history2)
    assert history2 == history_before


def test_missing_update_is_not_confirmation_and_eventually_expires():
    pool, history = {}, {}
    step("2026-09-09", 0, [row("2026-09-09")], pool, history)
    missing = step("2026-09-10", 1, [], pool, history)[0]
    assert missing["pool_status"] == "STALE" and missing["score"] is None and missing["pool_rank"] is None
    fresh = step("2026-09-11", 2, [row("2026-09-11", tracking_only=True)], pool, history)[0]
    assert fresh["history_reset"] and "persistent" not in fresh["components"]
    expired = step("2026-09-18", 7, [], pool, history)[0]
    assert expired["pool_status"] == "EXPIRED"


@pytest.mark.parametrize("changes,status", [({"stage": "FAILED"}, "INVALID"),
                                          ({"value_tier": "RISK_WATCH"}, "BLOCKED"),
                                          ({"supply_pressure": True}, "BLOCKED")])
def test_current_risk_overrides_historical_score(changes, status):
    pool, history = {}, {}
    step("2026-09-14", 0, [row("2026-09-14", demand=True, strong_close=True)], pool, history)
    r = step("2026-09-15", 1, [row("2026-09-15", **changes)], pool, history)[0]
    assert r["pool_status"] == status and r["score"] is None
    assert r["pool_rank"] is None and not r["structure_review"]


def test_new_cycle_cannot_inherit_points_and_weak_activity_loses_points():
    pool, history = {}, {}
    step("2026-09-14", 0, [row("2026-09-14")], pool, history)
    r = step("2026-09-15", 1, [row("2026-09-15", cycle_id="cycle-b", setup_status="CONFIRMED",
                                  demand=True, close_location=.1, turnover_rel=2., amount_rel=2.)], pool, history)[0]
    assert r["history_reset"] and "persistent" not in r["components"]
    assert "retest_confirmed" not in r["components"]
    assert r["components"]["weak_active"] == -3
    assert r["score_delta"] is None


def test_loss_of_retention_and_extension_are_penalized():
    pool, history = {}, {}
    step("2026-09-14", 0, [row("2026-09-14")], pool, history)
    r = step("2026-09-15", 1, [row("2026-09-15", retained=False, route_bias_atr=4.)], pool, history)[0]
    assert r["components"]["retention_lost"] == -2
    assert r["components"]["extended"] == -3
    assert r["score_delta"] < 0 and not r["structure_review"]


def test_rank_does_not_use_future_returns_or_tracking_flag():
    first = row("2026-09-15", score=6, tracking_only=True, ret1=-10)
    second = row("2026-09-15", symbol="sh_600001", score=2, tracking_only=False, ret1=10)
    assert study.order([second, first], True)[0]["symbol"] == first["symbol"]
    first["ret1"], second["ret1"] = 100, -100
    assert study.order([second, first], True)[0]["symbol"] == first["symbol"]


def test_research_freezes_inputs_and_separates_new_sample(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "ENGINE", tmp_path / "engine")
    prices = raw_prices()
    prices["date"] = pd.bdate_range(end="2026-09-18", periods=90).strftime("%Y-%m-%d")
    data = tmp_path / "data"
    data.mkdir()
    prices.to_csv(data / "sh_600000_daily.csv", index=False)
    prices[["date"]].rename(columns={"date": "trade_date"}).to_csv(data / "000001_SH_index.csv", index=False)
    for day in ["2026-09-15", "2026-09-16"]:
        folder = study.ENGINE / day.replace("-", "")
        r = row(day, tracking_only=day == "2026-09-16")
        core = dict(as_of=day, version=study.CONFIG["snapshot_version"],
                    state_vector={"stage": "PULLBACK", "setup_id": "setup-a", "setup_kind": "FIRST_PULLBACK"},
                    route_evidence={"retained": True, "supply": True, "demand": True})
        study.base.write_json(folder / "analysis_universe.json", dict(
            as_of=day, version=study.CONFIG["snapshot_version"], market={"structure": "BEAR"}, records=[r]))
        study.base.write_json(folder / "analysis_evidence.json", dict(
            as_of=day, version=study.CONFIG["snapshot_version"], records={r["symbol"]: core}))
    output = tmp_path / "study"
    first = study.run("2026-09-18", output)
    assert {r["sample_period"] for r in first["ranking_results"]} == {"INITIAL_EXPLORATORY", "FORWARD_FROZEN_RULES"}
    assert study.run("2026-09-18", output) == first
    source = study.ENGINE / "20260915/analysis_universe.json"
    changed = study.base.read_json(source)
    changed["records"][0]["value_tier"] = "RISK_WATCH"
    study.base.write_json(source, changed)
    assert study.run("2026-09-18", output) == first
    observations = study.base.read_json(output / "reviews/20260918/observations.json")
    forward = [r for r in observations if r["date"] == "2026-09-16"][0]
    assert not forward["today_candidate"] and forward["pool_status"] == "ACTIVE"
    assert forward["score"] > 0
