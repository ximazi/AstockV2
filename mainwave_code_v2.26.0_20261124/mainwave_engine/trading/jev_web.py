"""Read-only endpoints for the Jev shadow layer; served from the day's frozen artefacts, never recomputed."""
from flask import Blueprint, jsonify, request
from . import store, daily, jev

bp = Blueprint("jev", __name__)


def _latest_day():
    with store.connect(store.DB_PATH) as conn:
        snap = store.latest(conn)
    return (snap or {}).get("as_of", "")


@bp.get("/api/trading/jev")
def jev_view():
    day = request.args.get("as_of") or _latest_day()
    folder = store.ROOT / "results/mainwave_engine" / day.replace("-", "") if day else None
    shadow = daily.read(folder / "jev_shadow.json") if folder and (folder / "jev_shadow.json").exists() else None
    review = daily.read(folder / "jev_validation.json") if folder and (folder / "jev_validation.json").exists() else None
    if not shadow:
        return jsonify({"status": "NO_RECORD", "as_of": day, "notice": jev.NOTICE})
    return jsonify({"status": "READY", "shadow": shadow, "validation": review})


@bp.get("/api/trading/jev/questions")
def jev_questions():
    return jsonify({"question_set": jev.QUESTION_SET_VERSION, "model": jev.MODEL, "global": jev.GLOBAL_QUESTIONS,
                    "R1": jev.R1_QUESTIONS, "R2": jev.R2_QUESTIONS, "state_notes": jev.STATE_NOTES, "notice": jev.NOTICE})
