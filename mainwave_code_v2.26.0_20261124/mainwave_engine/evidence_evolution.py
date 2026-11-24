"""Versioned, prior-date evidence comparisons, separate from trading permission."""
from __future__ import annotations

import json
from pathlib import Path

VERSION = "evidence_evolution_v1_20260908"
CHANGE_LABELS = {"NEW": "新增", "PERSISTENT": "延续", "IMPROVED": "改善",
                 "WEAKENED": "减弱或时效消退", "INVALID": "条件不再成立", "UNKNOWN": "未知"}


def compare(core: dict, previous: dict | None = None) -> dict:
    previous = previous or {}
    compatible = (previous.get("version") == core["version"] and previous.get("symbol") == core["symbol"]
                  and str(previous.get("as_of", "")) < core["as_of"]
                  and previous.get("observation_mode") != "HISTORICAL_RECONSTRUCTION"
                  and core.get("observation_mode") != "HISTORICAL_RECONSTRUCTION")
    prior = previous.get("evolution", {}).get("latest", {}) if compatible else {}
    old_cycle = prior.get("cycle_id", "")
    current_cycle = core["state_vector"].get("cycle_id", "")
    new_cycle = bool(old_cycle and current_cycle and old_cycle != current_cycle)
    if new_cycle:
        prior = {}
    latest = {"as_of": core["as_of"], "cycle_id": current_cycle,
              "stage": core["state_vector"]["stage"], "advice": core["advice"],
              "setup_id": core["state_vector"]["setup_id"],
              "stage_label": core["state_vector"]["stage_label"], "advice_label": core["advice_label"],
              "theme_phase": core["state_vector"].get("theme_phase") or "未确认",
              "breakout_label": core.get("breakout", {}).get("breakout_status_label", "未建立"),
              "checks": {c["id"]: c["status"] for c in core["checks"]}}
    origin = previous["evolution"].get("origin", prior) if prior else latest
    changes = []
    state_changes = [{"label": label, "before": prior.get(key), "now": latest[key]}
                     for key, label in (("stage_label", "连续阶段"), ("advice_label", "建议"),
                                        ("theme_phase", "题材阶段"), ("breakout_label", "突破状态"))
                     if prior and prior.get(key) != latest[key]]
    for check in core["checks"]:
        before = prior.get("checks", {}).get(check["id"])
        now = check["status"]
        change = ("UNKNOWN" if now == "UNKNOWN" else "NEW" if before is None else
                  "PERSISTENT" if before == now else "INVALID" if now == "FAIL" else
                  "WEAKENED" if before == "PASS" else "IMPROVED" if now == "PASS" else "NEW")
        changes.append({"id": check["id"], "label": check["label"], "before": before, "now": now,
                        "change": change, "change_label": CHANGE_LABELS[change]})
    state, window = core["state_vector"], core["review_window"]
    event_key = (f"{state['setup_id']}:{window['confirmed_at']}"
                 if state["setup_status"] == "CONFIRMED" and window["confirmed_at"] == core["as_of"] else "")
    summary = ("新周期，重新建立证据基线" if new_cycle else
               "首份同版证据快照，尚无可比前值" if not prior else
               "；".join([f"{r['label']}：{r['before']} → {r['now']}" for r in state_changes]
                         + [f"{r['label']}：{r['change_label']}" for r in changes
                            if r["change"] not in {"PERSISTENT", "UNKNOWN"}]) or "相较上次有效快照，已观测条件延续")
    return {"version": VERSION, "origin": origin, "previous": prior or None, "latest": latest,
            "previous_as_of": prior.get("as_of", ""), "baseline_reset": new_cycle,
            "changes": changes, "state_changes": state_changes, "summary": summary, "confirmation_event_key": event_key,
            "scope": "候选发现起持续留档；同版本、严格早于当前日期的快照比较，不伪造昨日状态",
            "boundary": "确认时效消退不等于持有逻辑失效；正常缩量不自动判为资金离场；不生成加减仓指令"}


def load_previous(root: Path, symbol: str, as_of: str, core_version: str) -> dict | None:
    cutoff = as_of.replace("-", "")
    for folder in sorted(root.glob("????????"), reverse=True):
        if not folder.is_dir() or not folder.name.isdigit() or folder.name >= cutoff:
            continue
        for name in ("analysis_evidence.json", f"stock_analysis_{symbol}.json"):
            path = folder / name
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            core = payload.get("records", {}).get(symbol, {}) if name == "analysis_evidence.json" else payload.get("analysis_core", {})
            if (core.get("version") == core_version and core.get("symbol") == symbol
                    and core.get("as_of", "").replace("-", "") == folder.name and core.get("evolution")
                    and core.get("observation_mode") != "HISTORICAL_RECONSTRUCTION"):
                return core
    return None
