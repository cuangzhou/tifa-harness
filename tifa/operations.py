from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any


class RunLock:
    def __init__(self, path: Path) -> None: self.path, self.acquired = path, False
    def acquire(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(descriptor, str(os.getpid()).encode()); os.close(descriptor); self.acquired = True; return self
        except FileExistsError as exc: raise RuntimeError(f"run is already locked: {self.path.stem}") from exc
    def release(self) -> None:
        if self.acquired: self.path.unlink(missing_ok=True); self.acquired = False


def continuation_lineage(workspace: Path, run_id: str) -> list[dict[str, Any]]:
    runs = workspace / ".tifa" / "runs"; lineage = []; current = run_id; seen = set()
    while current and current not in seen:
        seen.add(current); report_path = runs / current / "report.json"
        if not report_path.exists(): break
        report = json.loads(report_path.read_text(encoding="utf-8")); lineage.append({"run_id": current, "parent_run_id": report.get("parent_run_id"), "source_checkpoint_id": report.get("source_checkpoint_id"), "stop_reason": report.get("stop_reason")}); current = report.get("parent_run_id")
    return lineage


def collect_garbage(workspace: Path, *, apply: bool = False, retain_days: int = 30, retain_latest: int = 20) -> dict[str, Any]:
    runs = workspace / ".tifa" / "runs"; candidates = sorted((path for path in runs.glob("*") if path.is_dir()), key=lambda path: path.stat().st_mtime, reverse=True) if runs.exists() else []
    cutoff = datetime.now(timezone.utc) - timedelta(days=retain_days); removable = [path for index, path in enumerate(candidates) if index >= retain_latest and datetime.fromtimestamp(path.stat().st_mtime, timezone.utc) < cutoff]
    reclaimed = sum(sum(file.stat().st_size for file in path.rglob("*") if file.is_file()) for path in removable)
    if apply:
        for path in removable: shutil.rmtree(path)
    return {"mode": "apply" if apply else "dry-run", "run_count": len(removable), "reclaimed_bytes": reclaimed, "runs": [path.name for path in removable]}


def migrate_artifact(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8")); version = payload.get("schema_version")
    mapping = {"tifa-task-state.v2": "tifa-task-state.v3", "tifa-report.v2": "tifa-report.v3"}
    if version not in mapping: return {"path": str(path), "status": "unchanged", "schema_version": version}
    payload["schema_version"] = mapping[version]
    if "phase" not in payload and version == "tifa-task-state.v2": payload["phase"] = "FAILED" if payload.get("status") == "failed" else "COMPLETE"
    temporary = path.with_suffix(path.suffix + ".tmp"); temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8"); temporary.replace(path)
    return {"path": str(path), "status": "migrated", "schema_version": payload["schema_version"]}
