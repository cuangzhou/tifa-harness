from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import statistics
from typing import Any

from .replay import ReplayDiffReport, ReplayRunner
from .result_contract import make_result, write_result


def evaluation_root() -> Path:
    return Path(__file__).resolve().parents[1] / "evaluation"


def run_replay_benchmark(mode: str, output: Path | None = None) -> dict[str, Any]:
    if mode not in {"smoke", "full"}: raise ValueError("mode must be smoke or full")
    root = evaluation_root(); fixtures = sorted((root / "fixtures").glob("*.json"))
    matrix = json.loads((root / "replay_benchmark_matrix.json").read_text(encoding="utf-8"))["tasks"]
    repetitions = 1 if mode == "smoke" else 3
    reports = [ReplayRunner().replay(path) for _ in range(repetitions) for path in fixtures]
    typed = [r for r in reports if isinstance(r, ReplayDiffReport)]
    if not typed: raise ValueError("no replay fixtures found")
    n = len(typed)
    metrics = {"offline_replay_consistency_rate": sum(r.replay_consistent for r in typed) / n, "verifier_reproduction_rate": sum(r.verifier_match for r in typed) / n, "artifact_digest_match_rate": sum(r.artifact_digest_match for r in typed) / n, "report_reconstruction_rate": sum(r.report_digest_match for r in typed) / n, "mean_replay_duration_ms": statistics.mean(r.duration_ms for r in typed), "p95_replay_duration_ms": sorted(r.duration_ms for r in typed)[max(0, int(n * .95) - 1)], "failure_type_distribution": dict(Counter(r.failure_category or "none" for r in typed)), "planned_fixture_count": len(matrix), "executed_fixture_count": len(fixtures), "unimplemented_fixture_count": len(matrix) - len(fixtures), "experimental_groups": {"offline_replay": "MEASURED", "replay_only": "NOT_IMPLEMENTED", "verified_case_assisted": "NOT_IMPLEMENTED", "irrelevant_case_injected": "NOT_IMPLEMENTED"}, "case_assistance_delta": "NOT_IMPLEMENTED", "checkpoint_reexecution_rate": "NOT_IMPLEMENTED", "forked_replay": "NOT_IMPLEMENTED", "counterfactual_replay": "NOT_IMPLEMENTED"}
    result = make_result(project="Tifa", benchmark=f"offline-replay-{mode}", dataset_version="pico-offline-fixtures-v1", result_kind="measured", implementation_status="integrated", git_commit="unversioned", command=f"tifa benchmark replay --mode {mode}", case_count=len(fixtures) if mode == "smoke" else len(matrix), repetitions=repetitions, seed=20260714, metrics=metrics, limitations=["仅验证 12 个历史 Pico Offline Replay fixture", "full 的 case_count=24 是规划矩阵规模，executed_fixture_count 才是实际执行数", "确定性 fixture 不代表真实模型任务成功率", "Forked、Counterfactual 与案例辅助尚未实现"])
    target = output or root / "artifacts" / f"tifa_offline_replay_{mode}_v1.json"
    write_result(result, target); target.with_name(target.stem + "_cases.json").write_text(json.dumps([asdict(r) for r in typed], ensure_ascii=False, indent=2), encoding="utf-8")
    return result
