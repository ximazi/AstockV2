"""Frozen rule dispatch. An old plan runs its registered rules, or waits for review."""
import hashlib
import json
from pathlib import Path
from . import store

VERSION = "rule_dispatch_v1_20260920"
# Entry, adds and exit are frozen for forward validation. Historical exit evidence flipped between common and
# full samples and on removing the largest case; tuning further would only fit the researcher to the sample.
FREEZE = {"frozen_at": "2026-09-21", "parts": ("entry", "exit", "pullback", "mode"),
          "reason": "历史证据不稳定（常见样本略好、全样本反转、去掉最大案例反转），继续调参只会形成研究者过拟合",
          "lift_condition": "AUTO 账前向有效样本（题材×信号日簇）达到登记实验的 n_effective_min 后，按经济门禁决定",
          "allowed": "影子版本（shadow_only）可以登记与观察，不改动生效规则"}
# Parameters can be restored from the registry; executable code cannot be rebuilt from a hash.
RESTORABLE = "PARAMETERS_ONLY_CODE_NOT_REBUILDABLE"
SOURCES = {
    "entry": "mainwave_engine/entry_policy.py",
    "exit": "mainwave_engine/exit_policy.py",
    "pullback": "mainwave_engine/pullback_plan.py",
    "mode": "mainwave_engine/trading/modes.py",
    "execution": "mainwave_engine/trading/paper.py",
    "dependency": "mainwave_engine/route_evidence.py",
}


def _table(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS rule_versions ("
                 "id TEXT PRIMARY KEY, kind TEXT NOT NULL, version TEXT NOT NULL, "
                 "created_at TEXT NOT NULL, payload TEXT NOT NULL)")


def _hash(path, root=None):
    target = Path(root or store.ROOT) / path
    return hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else ""


def build(root=None):
    """Current rule set: versions, parameters and direct file hashes in one summary."""
    from .. import entry_policy as EP, exit_policy as XP, pullback_plan as PP
    from . import modes, paper
    from . import VERSION as SOFTWARE
    parts = {
        "entry": {"version": EP.VERSION, "rules": EP.RULES,
                  "parameters": {"initial": EP.INITIAL, "add_weights": list(EP.ADD_WEIGHTS),
                                 "pullbacks": list(EP.PULLBACKS), "window": EP.WINDOW}},
        "exit": {"version": XP.VERSION, "rules": XP.RULES, "parameters": {}},
        "pullback": {"version": getattr(PP, "VERSION", ""), "rules": {}, "parameters": {}},
        "mode": {"version": modes.VERSION, "rules": {}, "parameters": modes.CONTRACT_SUMMARY},
        "execution": {"version": paper.VERSION, "rules": {}, "parameters": {"model": paper.MODEL}},
        "dependency": {"version": _dependency_version(), "rules": {}, "parameters": {}},
    }
    for kind, item in parts.items():
        item["source"] = SOURCES[kind]
        item["sha256"] = _hash(SOURCES[kind], root)
    payload = {"registry_version": VERSION, "software_version": SOFTWARE, "parts": parts,
               "restorable": RESTORABLE, "freeze": FREEZE,
               "note": "软件版本号变化不等于策略冻结；策略、数据与研究协议版本分别记录。"}
    payload["fingerprint"] = store.digest(payload)
    return payload


def _dependency_version():
    from .. import route_evidence as RE
    return getattr(RE, "VERSION", "")


def register(db=store.DB_PATH, root=None):
    """Record the current rule set so a later plan can look its own version back up."""
    payload = build(root)
    with store.connect(db) as conn:
        _table(conn)
        for kind, item in payload["parts"].items():
            identity = f"{kind}:{item['version']}:{item['sha256'][:12]}"
            conn.execute("INSERT OR IGNORE INTO rule_versions VALUES (?,?,?,?,?)",
                         (identity, kind, item["version"], store.now(), store.encoded(item)))
    return payload


def lookup(kind, version, sha256, db=store.DB_PATH, conn=None):
    """Inside a ledger transaction the caller's connection is reused; a second writer would self-lock."""
    if conn is not None:
        return _lookup(conn, kind, version, sha256)
    with store.connect(db) as own:
        return _lookup(own, kind, version, sha256)


def _lookup(conn, kind, version, sha256):
    _table(conn)
    row = conn.execute("SELECT payload FROM rule_versions WHERE id=?",
                       (f"{kind}:{version}:{sha256[:12]}",)).fetchone()
    if row:
        return json.loads(row[0])
    row = conn.execute("SELECT payload FROM rule_versions WHERE kind=? AND version=? ORDER BY rowid DESC LIMIT 1",
                       (kind, version)).fetchone()
    return json.loads(row[0]) if row else None


def dispatch(plan, db=store.DB_PATH, root=None, conn=None):
    """Decide which rules an existing plan runs under, and say so out loud."""
    current = build(root)
    frozen = plan.get("rule_fingerprint") or {}
    legacy = plan.get("exit_code_hashes") or {}
    parts, drift, unrestorable = {}, [], []
    for kind, item in current["parts"].items():
        want_version = frozen.get(kind, {}).get("version") or _legacy_version(plan, kind)
        want_hash = frozen.get(kind, {}).get("sha256") or _legacy_hash(legacy, kind)
        if not want_version and not want_hash:
            parts[kind] = {**item, "dispatch": "CURRENT_NO_FROZEN_RECORD"}
            continue
        same = (want_version in ("", item["version"])) and (want_hash in ("", item["sha256"]))
        if same:
            parts[kind] = {**item, "dispatch": "CURRENT_MATCHES_FROZEN"}
            continue
        drift.append(kind)
        registered = lookup(kind, want_version, want_hash or "", db, conn)
        if registered and registered.get("sha256") == want_hash:
            parts[kind] = {**registered, "dispatch": "FROZEN_PARAMETERS_FROM_REGISTRY",
                           "limit": RESTORABLE}
        else:
            unrestorable.append(kind)
            parts[kind] = {"version": want_version, "sha256": want_hash,
                           "dispatch": "UNRESTORABLE_PENDING_REVIEW", "limit": RESTORABLE}
    # Parameters can be read back from the registry; the code that ran them cannot. So any drift
    # means the frozen version is not executable: no new adds, deterministic risk floors keep running.
    status = ("PENDING_REVIEW" if unrestorable else "PARAMETERS_ON_RECORD_CODE_CHANGED" if drift else "CURRENT")
    return {"version": VERSION, "plan_id": plan.get("id"), "status": status, "parts": parts,
            "drift": drift, "unrestorable": unrestorable,
            "executable_as_frozen": not drift, "new_adds_allowed": not drift,
            "risk_tasks_retained": True,
            "note": ("规则已变更且登记表没有原版本，该计划待人工核对；不补仓，已有持仓只跑确定性硬底线与结构失效。"
                     if unrestorable else
                     "规则已变更：原参数可从登记表取回，但原代码不可重建，不算按冻结版本执行；"
                     "不补仓，已有持仓只跑确定性硬底线与结构失效，其余等待复核。"
                     if drift else "规则未变更，按当前版本运行。")}


def _legacy_version(plan, kind):
    return {"exit": plan.get("exit_version"), "entry": plan.get("entry_version"),
            "execution": plan.get("version")}.get(kind, "")


def _legacy_hash(legacy, kind):
    return {"exit": legacy.get("exit_policy.py"), "pullback": legacy.get("pullback_plan.py")}.get(kind, "")


def fingerprint(root=None):
    """Compact per-part version+hash record stored on every new plan."""
    current = build(root)
    return {kind: {"version": item["version"], "sha256": item["sha256"], "source": item["source"]}
            for kind, item in current["parts"].items()}
