from __future__ import annotations

from html import escape
import json
from pathlib import Path
from typing import Any


def load_run(workspace: Path, run_id: str) -> dict[str, Any]:
    run = workspace.resolve() / ".tifa" / "runs" / run_id
    if not run.is_dir(): raise FileNotFoundError(f"run not found: {run_id}")
    payload: dict[str, Any] = {"run_id": run_id, "path": str(run)}
    for name in ("task_state.json", "report.json", "metrics.json", "evidence_bundle.json"):
        path = run / name
        if path.exists(): payload[name.removesuffix(".json")] = json.loads(path.read_text(encoding="utf-8"))
    payload["checkpoints"] = [path.stem for path in sorted((run / "checkpoints").glob("*.json"))] if (run / "checkpoints").exists() else []
    return payload


def render_report(workspace: Path, run_id: str, output: Path, format: str = "json") -> Path:
    payload = load_run(workspace, run_id); output.parent.mkdir(parents=True, exist_ok=True)
    if format == "json": output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    elif format == "html":
        report = payload.get("report", {}); metrics = payload.get("metrics", {})
        body = escape(json.dumps({"report": report, "metrics": metrics, "checkpoints": payload["checkpoints"]}, ensure_ascii=False, indent=2))
        output.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>Tifa run {escape(run_id)}</title><style>body{{font:15px system-ui;max-width:1000px;margin:40px auto;padding:0 20px}}pre{{white-space:pre-wrap;background:#f4f4f5;padding:20px;border-radius:8px}}</style></head><body><h1>Tifa run {escape(run_id)}</h1><pre>{body}</pre></body></html>", encoding="utf-8")
    else: raise ValueError("format must be json or html")
    return output
