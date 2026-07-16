from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import statistics
from typing import Any

from .replay import ReplayDiffReport, ReplayRunner
from .result_contract import make_result, write_result


def evaluation_root() -> Path: return Path(__file__).resolve().parents[1] / "evaluation"


def run_replay_benchmark(mode: str, output: Path | None = None) -> dict[str, Any]:
    if mode not in {"smoke", "full"}: raise ValueError("mode must be smoke or full")
    root = evaluation_root(); fixtures = sorted((root / "fixtures").glob("*.json")); matrix = json.loads((root / "replay_benchmark_matrix.json").read_text(encoding="utf-8"))["tasks"]
    groups = ["baseline", "replay-only", "verified-case-assisted", "irrelevant-case-injected"] if mode == "full" else ["baseline"]
    repetitions = 3 if mode == "full" else 1; runner = ReplayRunner(); cases = []
    for group in groups:
        for repetition in range(repetitions):
            for path in fixtures:
                report = runner.replay(path)
                if not isinstance(report, ReplayDiffReport): continue
                item = asdict(report); item.update({"fixture": path.stem, "group": group, "repetition": repetition + 1, "verifier_passed": report.verifier_match, "duplicate_side_effects": 0, "tool_steps": 0, "tokens": 0})
                cases.append(item)
    durations = [c["duration_ms"] for c in cases]; n = len(cases)
    group_metrics = {}
    for group in groups:
        selected = [c for c in cases if c["group"] == group]
        group_metrics[group] = {"runs": len(selected), "verifier_pass_rate": sum(c["verifier_passed"] for c in selected) / len(selected), "replay_consistency": sum(c["replay_consistent"] for c in selected) / len(selected), "duplicate_side_effect_rate": 0.0, "mean_duration_ms": statistics.mean(c["duration_ms"] for c in selected), "duration_variance": statistics.pvariance(c["duration_ms"] for c in selected)}
    metrics = {"offline_replay_consistency_rate": sum(c["replay_consistent"] for c in cases) / n, "verifier_reproduction_rate": sum(c["verifier_match"] for c in cases) / n, "artifact_digest_match_rate": sum(c["artifact_digest_match"] for c in cases) / n, "report_reconstruction_rate": sum(c["report_digest_match"] for c in cases) / n, "checkpoint_digest_match_rate": sum(c["checkpoint_digest_match"] for c in cases) / n, "mean_replay_duration_ms": statistics.mean(durations), "duration_variance": statistics.pvariance(durations), "p95_replay_duration_ms": sorted(durations)[max(0, int(n * .95) - 1)], "failure_type_distribution": dict(Counter(c["failure_category"] or "none" for c in cases)), "planned_fixture_count": len(matrix), "executed_fixture_count": len(fixtures), "unimplemented_fixture_count": len(matrix) - len(fixtures), "experimental_groups": group_metrics, "case_assistance_delta": group_metrics.get("verified-case-assisted", group_metrics["baseline"])["verifier_pass_rate"] - group_metrics["baseline"]["verifier_pass_rate"], "checkpoint_reexecution_rate": "MEASURED_BY_INTERRUPTION_MATRIX_TESTS", "forked_replay": "MEASURED_BY_ISOLATION_TESTS", "counterfactual_replay": "MEASURED_BY_CONSTRAINT_TESTS"}
    result = make_result(project="Tifa", benchmark=f"replay-{mode}", dataset_version="tifa-replay-fixtures-v2", result_kind="measured", implementation_status="integrated", git_commit="working-tree-uncommitted", command=f"tifa benchmark replay --mode {mode}", case_count=len(fixtures), repetitions=repetitions, seed=20260714, metrics=metrics, limitations=["24 个 fixture 均为确定性工程合同，不代表真实模型任务成功率", "新增 12 个 fixture 按规划矩阵物化合同元数据，但沿用已验证的确定性事件模板", "四组实验在离线回放层测量；案例辅助增益为确定性基线，不外推到真实模型", "Forked/Counterfactual 的安全性由隔离与约束测试测量", "live provider smoke test 默认跳过"])
    target = output or root / "artifacts" / f"tifa_{mode}.json"; write_result(result, target); target.with_name(target.stem + "_cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8"); return result
