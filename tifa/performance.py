from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, TypeVar

from .workspace import WorkspaceContext


T = TypeVar("T")


def _measure(action: Callable[[], T], repeats: int = 3) -> tuple[list[float], T]:
    values = []; result: T | None = None
    for _ in range(repeats):
        started = time.perf_counter(); result = action(); values.append((time.perf_counter() - started) * 1000)
    assert result is not None
    return values, result


def _rss_mb() -> float:
    try:
        import psutil  # type: ignore[import-not-found]
        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def benchmark_workspace(output: Path | None = None, sizes: tuple[int, ...] = (1000, 5000, 10000), repeats: int = 3) -> dict:
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="tifa-workspace-benchmark-") as folder:
        root = Path(folder)
        for count in sizes:
            target = root / str(count); target.mkdir()
            for index in range(count): (target / f"file_{index:05d}.py").write_text(f"VALUE_{index} = {index}\n", encoding="utf-8")
            before_memory = _rss_mb(); cold, context = _measure(lambda: WorkspaceContext.build(target), 1); peak = max(before_memory, _rss_mb())
            incremental, context = _measure(lambda: WorkspaceContext.build(target), repeats)
            search, _ = _measure(lambda: [path for path in context.index if path.endswith("999.py")], repeats)
            context_build, _ = _measure(context.text, repeats)
            rows.append({"file_count": count, "cold_start_ms": cold[0], "incremental_ms": incremental, "incremental_p95_ms": max(incremental), "search_p95_ms": max(search), "context_build_p95_ms": max(context_build), "process_rss_mb": peak})
    ten_k = next(row for row in rows if row["file_count"] == max(sizes)); thresholds = {"incremental_fingerprint_p95_ms": 2000, "search_p95_ms": 3000, "context_build_p95_ms": 500}
    passed = ten_k["incremental_p95_ms"] <= 2000 and ten_k["search_p95_ms"] <= 3000 and ten_k["context_build_p95_ms"] <= 500
    report = {"schema_version": "tifa-workspace-benchmark.v1", "status": "passed" if passed else "failed", "sizes": rows, "thresholds": thresholds, "repeats": repeats}
    if output: output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
