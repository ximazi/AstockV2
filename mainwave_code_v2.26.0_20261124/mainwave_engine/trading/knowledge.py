"""Build a verifiable registry from the user's reviewed register and actual sources."""
import hashlib
import re
from . import store

VERSION = "knowledge_registry_v1_20260919"


def build(root=store.ROOT):
    doc = root / "02_知识点覆盖与待验证.md"
    text = doc.read_text(encoding="utf-8-sig")
    actual = {}
    for path in sorted((root / "talk/transcripts").rglob("*.transcript.reviewed.md")):
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        actual[sha] = path
    sources, items = [], []
    for match in re.finditer(r"^## (F\d+)｜([^\n]+)\n(.*?)(?=^## |\Z)", text, re.M | re.S):
        sid, name, section = match.groups()
        sha = re.search(r"SHA-256：`([a-f0-9]{64})`", section).group(1)
        if sha not in actual:
            raise ValueError(f"{sid}: 来源哈希未匹配，禁止静默修改知识来源")
        path = actual[sha]
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        sources.append({"source_id": sid, "name": name.strip(), "path": path.relative_to(root).as_posix(),
                        "sha256": sha, "status": "REGISTERED"})
        for row in section.splitlines():
            if not re.match(r"\|F\d+-K\d+；", row):
                continue
            cells = row.strip("|").split("|")
            location = re.fullmatch(r"(F\d+-K\d+)；L(\d+)[–—-]L(\d+)", cells[0])
            if len(cells) != 5 or not location:
                raise ValueError(f"知识登记格式无效: {row}")
            kid, start, end = location.groups()
            start, end = int(start), int(end)
            if not 1 <= start <= end <= len(lines):
                raise ValueError(f"{kid}: 源行越界")
            items.append({"knowledge_id": kid, "source_id": sid, "start_line": start, "end_line": end,
                          "source_sha256": sha, "excerpt": "\n".join(lines[start-1:end]), "claim": cells[1],
                          "disposition": cells[2], "module": cells[3], "validation": cells[4],
                          "empirical_status": "NOT_VERIFIED_BY_REGISTRATION", "live_authorized": False})
    if not items or len({i["knowledge_id"] for i in items}) != len(items):
        raise ValueError("知识登记为空或编号重复")
    registered = {s["sha256"] for s in sources}
    for sha, path in actual.items():
        if sha not in registered:
            sources.append({"source_id": "PENDING_" + sha[:12], "name": path.parent.name,
                            "path": path.relative_to(root).as_posix(), "sha256": sha, "status": "PENDING_REVIEW"})
    return {"version": VERSION, "register_sha256": hashlib.sha256(doc.read_bytes()).hexdigest(),
            "sources": sources, "items": items,
            "summary": {"registered_sources": len(registered), "pending_sources": len(actual) - len(registered),
                        "knowledge_items": len(items)}, "live_orders_enabled": False}


def persist(registry, db=store.DB_PATH):
    identity = store.digest(registry)
    with store.connect(db) as conn:
        conn.execute("INSERT OR IGNORE INTO registry_versions VALUES (?,?,?)", (identity, store.now(), store.encoded(registry)))
    return identity
