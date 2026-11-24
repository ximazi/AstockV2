"""Content-addressed input snapshots for reproducible P1 runs."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_inputs(paths: list[Path], output_dir: Path, metadata: dict) -> tuple[Path, dict]:
    hashes = {str(path): sha256(path) for path in paths}
    identity = hashlib.sha256(
        json.dumps({"metadata": metadata, "hashes": hashes}, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    target = output_dir / "input_snapshots" / identity
    target.mkdir(parents=True, exist_ok=True)
    copied = {}
    for index, path in enumerate(paths):
        if not path.exists():
            continue
        name = f"{index:02d}_{path.name}"
        destination = target / name
        if not destination.exists():
            shutil.copy2(path, destination)
        copied[str(path)] = {"snapshot": str(destination), "sha256": hashes[str(path)]}
    manifest = {"snapshot_id": identity, "metadata": metadata, "inputs": copied, "missing": [
        str(path) for path in paths if not path.exists()
    ]}
    manifest_path = target / "snapshot_manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return target, manifest
