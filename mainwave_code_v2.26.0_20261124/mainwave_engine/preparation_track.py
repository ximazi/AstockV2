"""Causal preparation identities, independent of daily threshold crossings.

This is an observation ledger, not a replacement for discovery thresholds.
TTL and the frozen floor reuse research hypotheses, not calibrated buy rules.
"""
from __future__ import annotations

import pandas as pd

VERSION = "preparation_track_v1_20260910"
TTL = 30  # Active stock sessions from first observation, never refreshed daily.
FIELDS = ("preparation_id", "preparation_first_seen", "preparation_origin",
          "preparation_state", "preparation_parent_id", "preparation_valid",
          "preparation_age", "preparation_reason", "activity_contraction",
          "volatility_contraction", "contraction_disagreement")
LABELS = {"PRE_ATTEMPT": "启动前准备", "ATTEMPT_UNDERWAY": "突破尝试中形成准备",
          "POST_ACCEPTANCE": "已启动后的整理", "UNKNOWN": "准备前因未知"}


def annotate(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    records, seen, track = [], set(), None
    for i, row in enumerate(frame.itertuples(index=False)):
        day = str(row.date.date())
        status, parent = row.breakout_status, row.breakout_id
        # A failed child ends this local preparation, not every larger stock cycle.
        if track and track["preparation_valid"]:
            reason = ("FROZEN_FLOOR_BROKEN" if row.return_index < track["floor"] else
                      "BOUND_PARENT_FAILED" if parent == track["preparation_parent_id"] and
                      status in {"FAILED", "REJECTED", "EXPIRED"} else
                      "BOUND_PARENT_REPLACED" if track["preparation_parent_id"] and parent and
                      parent != track["preparation_parent_id"] else
                      "FIXED_TTL_EXPIRED" if i - track["index"] > TTL else "")
            if reason:
                track.update(preparation_valid=False, preparation_reason=reason,
                             preparation_state="EXPIRED" if reason == "FIXED_TTL_EXPIRED" else "INVALID")
        anchor = row.probe_at
        if (track is None or not track["preparation_valid"]) and row.prepared and row.eligible and anchor and anchor not in seen:
            origin = ("POST_ACCEPTANCE" if status in {"ACCEPTED", "ADVANCING"} else
                      "ATTEMPT_UNDERWAY" if status == "ATTEMPT" else
                      "PRE_ATTEMPT" if status in {"NONE", "FAILED", "REJECTED", "EXPIRED"} else "UNKNOWN")
            track = {"preparation_id": f"{symbol}:preparation:{anchor}:{day}",
                     "preparation_first_seen": day, "preparation_origin": origin,
                     "preparation_state": "OBSERVING", "preparation_valid": True,
                     "preparation_parent_id": parent if status in {"ATTEMPT", "ACCEPTED", "ADVANCING"} else "",
                     "preparation_reason": "FIRST_OBSERVATION", "index": i,
                     "floor": row.structure_floor - .5 * row.atr20}
            seen.add(anchor)
        if track and track["preparation_valid"]:
            if not track["preparation_parent_id"] and row.breakout_event == "ATTEMPT":
                track["preparation_parent_id"] = parent
            linked = parent == track["preparation_parent_id"] and bool(parent)
            track["preparation_state"] = (status if linked and status in {"ATTEMPT", "ACCEPTED", "ADVANCING"}
                                          else "EVIDENCE_STRONG" if row.prepared else "OBSERVING")
            track["preparation_reason"] = "SAME_TRACK_UPDATED"
            if anchor:
                seen.add(anchor)  # New pulses strengthen this track; they do not reset its age.
        activity = row.amount_contract <= .85 and row.turnover_contract <= .85
        volatility = row.range_contract <= .85
        record = {k: track[k] if track else "" for k in FIELDS[:6]}
        record.update(preparation_valid=bool(track and track["preparation_valid"]),
                      preparation_age=i - track["index"] if track else None,
                      preparation_reason=track["preparation_reason"] if track else "NO_VALID_PREPARATION",
                      activity_contraction=activity, volatility_contraction=volatility,
                      contraction_disagreement=(row.amount_contract <= .85) != (row.turnover_contract <= .85))
        records.append(record)
    return pd.DataFrame(records, index=frame.index, columns=FIELDS)
