import numpy as np
import pandas as pd
import pytest

from research import recent_ranking_review as study
from research.recent_ranking_review import features, outcomes, rank


def raw_prices():
    dates = pd.bdate_range("2026-01-01", periods=90).strftime("%Y-%m-%d")
    close = np.arange(90, dtype=float) * .03 + 10
    return pd.DataFrame(dict(date=dates, close=close, open=close - .03, high=close + .1,
                             low=close - .1, pct_chg=np.full(90, .3), turnover=np.full(90, 2.),
                             volume=np.full(90, 2000000.), amount=np.full(90, 22000000.)))


def test_features_ignore_future_bars():
    raw = raw_prices()
    day = raw.iloc[70].date
    before = features(raw, day)
    changed = raw.copy()
    changed.loc[71:, ["close", "volume", "turnover", "amount", "high", "low", "pct_chg"]] *= 50
    assert before == features(changed, day)


def test_baseline_excludes_signal_day_and_strong_volume_is_not_strong_close():
    raw = raw_prices()
    raw.loc[89, "turnover"] = 4
    raw.loc[89, "volume"] = 4000000
    raw.loc[89, "open"] = raw.loc[89, "high"]
    value = features(raw, raw.iloc[-1].date)
    assert value["turnover_rel"] == 2
    assert not value["strong_close"]


def test_float_change_does_not_become_turnover_signal():
    raw = raw_prices()
    raw.loc[89, "turnover"] = 4
    value = features(raw, raw.iloc[-1].date)
    assert value["denominator_changed"]
    assert np.isnan(value["turnover_rel"])


def test_fixed_horizon_excludes_pending_and_missing_dates():
    raw = raw_prices()
    dates = raw.date.tolist()
    assert outcomes(raw, dates[-1], dates[-1], 1, dates) == {}
    assert outcomes(raw, dates[-3], dates[-1], 3, dates) == {}
    result = outcomes(raw, dates[-3], dates[-1], 2, dates)
    assert abs(result["ret2"] - ((1.003 ** 2 - 1) * 100)) < 1e-10
    assert outcomes(raw.drop(index=88), dates[-3], dates[-1], 2, dates) == {}


def test_ranking_does_not_consume_returns_and_stage_policy_is_explicit():
    frame = pd.DataFrame([
        dict(symbol="a", value_rank=1, tier_order=5, stage_order=0, quality_grade=0, r20=0, ret1=100),
        dict(symbol="b", value_rank=2, tier_order=5, stage_order=4, quality_grade=2, r20=20, ret1=-50),
    ])
    assert rank(frame, "quality_within_stage").symbol.tolist() == ["a", "b"]
    assert rank(frame, "quality_cross_stage").symbol.tolist() == ["b", "a"]
    frame.ret1 *= -1
    assert rank(frame, "quality_cross_stage").symbol.tolist() == ["b", "a"]


def test_review_is_frozen_repeatable_and_separates_forward_dates(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "ENGINE", tmp_path / "engine")
    data = tmp_path / "data"
    data.mkdir()
    prices = raw_prices()
    prices["date"] = pd.bdate_range(end="2026-09-18", periods=90).strftime("%Y-%m-%d")
    prices.to_csv(data / "sh_600000_daily.csv", index=False)
    prices[["date"]].rename(columns={"date": "trade_date"}).to_csv(data / "000001_SH_index.csv", index=False)
    for day in ["2026-09-15", "2026-09-16"]:
        folder = study.ENGINE / day.replace("-", "")
        row = dict(symbol="sh_600000", name="fixture", route_ids="R1", price_date=day,
                   value_rank=1, value_tier="MARKET_WAIT", structural_advice="PRIORITY_REVIEW", close=12.)
        study.write_json(folder / "analysis_universe.json", dict(
            as_of=day, version=study.CONFIG["snapshot_version"], market={"structure": "BEAR"}, records=[row]))
        study.write_json(folder / "stock_analysis_sh_600000.json", dict(
            meta={"price_date": day, "report_date": day, "version": "fixture", "engine_version": "fixture"},
            route_evidence={"retained": True, "supply": True, "demand": True}))
    output = tmp_path / "study"
    first = study.run("2026-09-09", "2026-09-18", output)
    assert {r["sample_period"] for r in first["ranking_results"]} == {"INITIAL_EXPLORATORY", "FORWARD_FROZEN_RULES"}
    assert all(r["days"] == 1 for r in first["ranking_results"])
    saved = output / "reviews/20260918/observations.json"
    before = saved.read_bytes()
    assert study.run("2026-09-09", "2026-09-18", output) == first
    assert saved.read_bytes() == before
    source = study.ENGINE / "20260915/analysis_universe.json"
    changed = study.read_json(source)
    changed["records"][0]["value_rank"] = 100
    study.write_json(source, changed)
    assert study.run("2026-09-09", "2026-09-18", output) == first
    assert saved.read_bytes() == before
    changed_config = dict(study.CONFIG, activity_high=2.)
    study.write_json(output / "protocol.json", changed_config)
    with pytest.raises(ValueError, match="Frozen protocol differs"):
        study.run("2026-09-09", "2026-09-18", output)
