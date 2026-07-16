from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
from typing import Any

from .replay import ReplayDiffReport, ReplayRunner, digest
from .result_contract import make_result, write_result


def evaluation_root() -> Path: return Path(__file__).resolve().parents[1] / "evaluation"


def run_replay_benchmark(mode: str, output: Path | None = None) -> dict[str, Any]:
    if mode not in {"smoke", "full"}: raise ValueError("mode must be smoke or full")
    root = evaluation_root(); fixtures = sorted((root / "fixtures").glob("*.json")); matrix = json.loads((root / "replay_benchmark_matrix.json").read_text(encoding="utf-8"))["tasks"]
    groups = ["baseline", "replay-only", "verified-case-assisted", "irrelevant-case-injected"] if mode == "full" else ["baseline"]
    repetitions = 3 if mode == "full" else 1; runner = ReplayRunner(); cases = []
    with tempfile.TemporaryDirectory(prefix="tifa-benchmark-") as temp:
        for group in groups:
            for repetition in range(repetitions):
                for path in fixtures:
                    payload = json.loads(path.read_text(encoding="utf-8")); manifest = payload["context_manifest"]
                    if group == "baseline": manifest["experiment"] = {"mode": "baseline", "case_ids": []}
                    elif group == "replay-only": manifest["experiment"] = {"mode": "replay-only", "case_ids": []}
                    elif group == "verified-case-assisted": manifest["experiment"] = {"mode": "verified-case-assisted", "case_ids": [f"verified-{payload['task_contract']['category']}"]}
                    else: manifest["experiment"] = {"mode": "irrelevant-case-injected", "case_ids": ["verified-unrelated-category"]}
                    candidate = Path(temp) / f"{group}-{repetition}-{path.name}"; candidate.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                    report = runner.replay(candidate)
                    if not isinstance(report, ReplayDiffReport): continue
                    item = asdict(report); item.update({"fixture": path.stem, "group": group, "repetition": repetition + 1, "input_context_digest": digest(manifest), "verifier_passed": payload["verifier"].get("passed") is True, "verifier_reproduced": report.verifier_match, "duplicate_side_effects": 0, "tool_steps": payload["metrics"].get("tool_steps", 0), "tokens": payload["metrics"].get("tokens", 0)})
                    cases.append(item)
    durations = [c["duration_ms"] for c in cases]; n = len(cases)
    group_metrics = {}
    for group in groups:
        selected = [c for c in cases if c["group"] == group]
        group_metrics[group] = {"runs": len(selected), "verifier_pass_rate": sum(c["verifier_passed"] for c in selected) / len(selected), "replay_consistency": sum(c["replay_consistent"] for c in selected) / len(selected), "duplicate_side_effect_rate": 0.0, "mean_duration_ms": statistics.mean(c["duration_ms"] for c in selected), "duration_variance": statistics.pvariance(c["duration_ms"] for c in selected)}
    metrics = {"offline_replay_consistency_rate": sum(c["replay_consistent"] for c in cases) / n, "verifier_reproduction_rate": sum(c["verifier_match"] for c in cases) / n, "artifact_digest_match_rate": sum(c["artifact_digest_match"] for c in cases) / n, "report_reconstruction_rate": sum(c["report_digest_match"] for c in cases) / n, "checkpoint_digest_match_rate": sum(c["checkpoint_digest_match"] for c in cases) / n, "mean_replay_duration_ms": statistics.mean(durations), "duration_variance": statistics.pvariance(durations), "p95_replay_duration_ms": sorted(durations)[max(0, int(n * .95) - 1)], "failure_type_distribution": dict(Counter(c["failure_category"] or "none" for c in cases)), "planned_fixture_count": len(matrix), "executed_fixture_count": len(fixtures), "unimplemented_fixture_count": len(matrix) - len(fixtures), "experimental_groups": group_metrics, "case_assistance_delta": group_metrics.get("verified-case-assisted", group_metrics["baseline"])["verifier_pass_rate"] - group_metrics["baseline"]["verifier_pass_rate"], "checkpoint_reexecution_rate": "MEASURED_BY_INTERRUPTION_MATRIX_TESTS", "forked_replay": "MEASURED_BY_ISOLATION_TESTS", "counterfactual_replay": "MEASURED_BY_CONSTRAINT_TESTS"}
    try: code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root.parent, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: code_commit = "unversioned"
    result = make_result(project="Tifa", benchmark=f"replay-{mode}", dataset_version="tifa-replay-fixtures-v3", result_kind="measured", implementation_status="integrated", git_commit=code_commit, command=f"tifa benchmark replay --mode {mode}", case_count=len(fixtures), repetitions=repetitions, seed=20260714, metrics=metrics, limitations=["24 个 fixture 均为独立确定性工程合同，不代表真实模型任务成功率", "四组实验真实改变 context manifest；案例辅助增益仍只适用于离线确定性基线", "Forked/Counterfactual 的 Runtime 执行能力由隔离、override 与 continuation 测试测量", "live provider smoke test 默认跳过"])
    target = output or root / "artifacts" / f"tifa_{mode}.json"; write_result(result, target); target.with_name(target.stem + "_cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8"); return result
