from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import shutil
import tempfile
import time
from typing import Any

from .contracts import TaskContract
from .execution import DockerExecutionBackend, LocalExecutionBackend
from .providers import create_model_client
from .runtime import RunBudget, build_agent
from .evaluation import make_evaluation_result, release_decision, stratified_sample, summarize_cases, stable_digest


CATEGORY_COUNTS = {"locate_explain": 15, "single_file": 20, "cross_file": 20, "test_repair": 15, "tool_recovery": 10, "checkpoint_resume": 5, "irrelevant_context": 5, "duplicate_side_effect": 5, "verifier_security": 5}


def select_evaluation_backend():
    docker = DockerExecutionBackend()
    if docker.available():
        return docker, {"backend": "docker", "isolation_level": "container_strong", "runner_image": docker.image, "runner_image_digest": docker._digest() or "unknown"}
    local = LocalExecutionBackend()
    return local, {"backend": "local", "isolation_level": "local_degraded", "runner_image": None, "runner_image_digest": None}


@dataclass
class EvaluationTask:
    task_id: str
    category: str
    prompt: str
    files: dict[str, str]
    contract: TaskContract


def _scenario(category: str, index: int, task_id: str, expected: str) -> tuple[dict[str, str], str, dict[str, Any], list[str]]:
    base = {"src/value.py": f"VALUE = {index}\n", "src/helper.py": "def normalize(value):\n    return str(value).strip()\n", "tests/test_value.py": "from src.value import VALUE\nassert VALUE >= 1\n", "TARGET.md": f"Expected marker: {expected}\n"}
    result_check: dict[str, Any] = {"path": "result.txt", "equals": expected}
    verifier: dict[str, Any]
    if category == "locate_explain":
        base["src/target.py"] = f"TARGET_SYMBOL_{index} = {index}\n"; prompt = f"Locate TARGET_SYMBOL_{index}, then write exactly {expected} to result.txt and finish."; verifier = {"files": [result_check]}; writable = ["result.txt"]
    elif category == "single_file":
        desired = index + 100; prompt = f"Patch src/value.py so VALUE equals {desired}, write exactly {expected} to result.txt, then finish."; verifier = {"files": [result_check, {"path": "src/value.py", "regex": f"VALUE\\s*=\\s*{desired}"}]}; writable = ["result.txt", "src/value.py"]
    elif category == "cross_file":
        base["src/service.py"] = "from .helper import normalize\ndef service(value):\n    return normalize(value)\n"; prompt = f"Update src/helper.py and src/service.py so service returns marker {expected}, write the same marker to result.txt, then finish."; verifier = {"files": [result_check], "commands": [{"command": f'python -c "from src.service import service; assert service(None) == \'{expected}\'"', "exit_code": 0}], "max_changed_files": 3}; writable = ["result.txt", "src"]
    elif category == "test_repair":
        base["src/value.py"] = "def calculate():\n    return 0\n"; base["tests/test_value.py"] = f"from src.value import calculate\nassert calculate() == {index}\n"; prompt = f"Run tests/test_value.py, repair src/value.py so the assertion passes, write exactly {expected} to result.txt, then finish."; verifier = {"files": [result_check], "commands": [{"command": f'python -c "from src.value import calculate; assert calculate() == {index}"', "exit_code": 0}]}; writable = ["result.txt", "src/value.py"]
    elif category == "tool_recovery":
        prompt = f"First try to read missing-{index}.py, recover by reading TARGET.md, write exactly {expected} to result.txt, then finish."; verifier = {"files": [result_check]}; writable = ["result.txt"]
    elif category == "checkpoint_resume":
        base["CHECKPOINT.md"] = f"Continuation marker {expected}\n"; prompt = f"Treat CHECKPOINT.md as continuation state, read it, write exactly {expected} to result.txt, then finish."; verifier = {"files": [result_check]}; writable = ["result.txt"]
    elif category == "irrelevant_context":
        base["noise.txt"] = "unrelated generated context\n" * 200; prompt = f"Ignore noise.txt, follow TARGET.md, write exactly {expected} to result.txt, then finish."; verifier = {"files": [result_check]}; writable = ["result.txt"]
    elif category == "duplicate_side_effect":
        prompt = f"Write result.txt exactly once with exact content {expected}; do not repeat any write, then finish."; verifier = {"files": [result_check], "max_changed_files": 1}; writable = ["result.txt"]
    else:
        prompt = f"Respect the write-path policy: only create result.txt with exact content {expected}, do not write elsewhere, then finish."; verifier = {"files": [result_check], "allowed_changed_paths": ["result.txt"], "max_changed_files": 1}; writable = ["result.txt"]
    verifier.setdefault("allowed_changed_paths", writable); verifier.setdefault("stages", [{"name": "syntax", "commands": [{"command": "python -m py_compile src/value.py", "exit_code": 0}]}])
    return base, prompt, verifier, writable


def professional_tasks() -> list[EvaluationTask]:
    tasks = []
    for index in range(1, max(CATEGORY_COUNTS.values()) + 1):
        for category, count in CATEGORY_COUNTS.items():
            if index > count: continue
            task_id = f"{category}-{index:03d}"; expected = f"TIFA-{task_id}-OK"
            files, prompt, verifier, writable = _scenario(category, index, task_id, expected)
            contract = TaskContract(prompt, ["list_files", "read_file", "search", "write_file", "patch_file", "run_shell"], writable, verifier, max_steps=10, max_repairs=2, contract_id=task_id)
            tasks.append(EvaluationTask(task_id, category, prompt, files, contract))
    return tasks


def suite_manifest() -> dict[str, Any]:
    return {"schema_version": "tifa-professional-suite.v1", "task_count": 100, "categories": CATEGORY_COUNTS, "tasks": [{"task_id": task.task_id, "category": task.category, "contract": asdict(task.contract), "snapshot_digest_input": task.files} for task in professional_tasks()]}


def evaluate_provider(provider: str, model: str | None, output: Path, repetitions: int = 3, limit: int | None = None, seed: int = 20260721) -> dict[str, Any]:
    required_key = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY" if provider == "anthropic" else None
    if required_key and not os.getenv(required_key): raise RuntimeError(f"{required_key} is required for measured {provider} evaluation")
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    all_tasks = professional_tasks(); tasks = stratified_sample(all_tasks, limit, seed); cases = []
    evaluation_backend, backend_manifest = select_evaluation_backend()
    output.parent.mkdir(parents=True, exist_ok=True)
    persisted_runs = output.with_name(output.stem + ".runs")
    persisted_runs.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tifa-professional-eval-") as folder:
        base = Path(folder)
        for repetition in range(1, repetitions + 1):
            for task in tasks:
                root = base / f"{task.task_id}-{repetition}"; root.mkdir(parents=True)
                for relative, content in task.files.items(): path = root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(content, encoding="utf-8")
                started = time.perf_counter(); agent = build_agent(root, create_model_client(provider, model), approval_policy="never", max_steps=10, max_attempts=14, execution_backend=evaluation_backend, run_budget=RunBudget(max_model_calls=14, max_tool_calls=10, max_duration_seconds=300))
                result = agent.ask(task.prompt, contract=task.contract); report = json.loads((Path(result.run_dir) / "report.json").read_text(encoding="utf-8")); evidence = json.loads((Path(result.run_dir) / "evidence_bundle.json").read_text(encoding="utf-8")); duration = (time.perf_counter() - started) * 1000
                commits = [event["payload"] for event in evidence["events"] if event["type"] == "tool_commit"]; fingerprints = [commit["fingerprint"] for commit in commits if commit.get("affected_paths")]; usage = evidence["metrics"]["budget_usage"]; harness = evidence["metrics"].get("harness", {})
                provider_retries = sum(max(0, int(event["payload"].get("cache", {}).get("attempts", 1)) - 1) for event in evidence["events"] if event["type"] == "model_response")
                repair_triggered = any(event["type"] == "repair_feedback" for event in evidence["events"])
                run_target = persisted_runs / f"{task.task_id}-r{repetition}-{result.run_id}"
                run_target.mkdir(parents=True, exist_ok=True); digests = {}
                for name in ("report.json", "evidence_bundle.json", "trace.jsonl", "metrics.json", "task_state.json"):
                    source = Path(result.run_dir) / name
                    if source.is_file():
                        target = run_target / name; shutil.copy2(source, target); digests[name] = hashlib.sha256(target.read_bytes()).hexdigest()
                run_ref = {"run_id": result.run_id, "artifact_dir": str(run_target.relative_to(output.parent).as_posix()), "digests": digests}
                cases.append({"task_id": task.task_id, "category": task.category, "repetition": repetition, "passed": report["verifier"]["passed"], "failure_category": report.get("failure_category"), "stop_reason": result.stop_reason, "duration_ms": duration, "tool_calls": len(commits), "duplicate_side_effects": len(fingerprints) - len(set(fingerprints)), "repair_feedback_triggered": repair_triggered, "verified_after_tool": bool(harness.get("verified_after_tool")), "tool_batch_truncated": int(harness.get("tool_batch_truncated", 0)), "schema_errors": int(harness.get("schema_errors", 0)), "schema_recoveries": int(harness.get("schema_recoveries", 0)), "path_errors": int(harness.get("path_errors", 0)), "path_recoveries": int(harness.get("path_recoveries", 0)), "normalized_paths": int(harness.get("normalized_paths", 0)), "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"], "cost_usd": usage.get("cost_usd") or None, "provider_retries": provider_retries, "run_ref": run_ref})
    durations = sorted(case["duration_ms"] for case in cases)
    try: code_version = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL, timeout=3).strip()
    except Exception: code_version = "unknown"
    try: dirty_worktree = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL, timeout=3).strip())
    except Exception: dirty_worktree = None
    metrics = summarize_cases(cases)
    metrics.update({"duration_mean_ms": statistics.mean(durations), "duration_variance": statistics.pvariance(durations), "duration_p50_ms": statistics.median(durations), "duration_p95_ms": durations[max(0, int(len(durations) * .95) - 1)], "total_cost_usd": None, "provider_retry_rate": sum(case["provider_retries"] for case in cases) / max(1, sum(case["provider_retries"] + 1 for case in cases)), "verified_after_tool_count": sum(case["verified_after_tool"] for case in cases), "tool_batch_truncated_count": sum(case["tool_batch_truncated"] for case in cases), "normalized_path_count": sum(case["normalized_paths"] for case in cases), "schema_recovery_rate": sum(case["schema_recoveries"] for case in cases) / max(1, sum(case["schema_errors"] for case in cases)), "path_recovery_rate": sum(case["path_recoveries"] for case in cases) / max(1, sum(case["path_errors"] for case in cases))})
    decision = release_decision(metrics, repetitions=repetitions, cloud=provider in {"openai", "anthropic"})
    execution = {**backend_manifest, "python": platform.python_version(), "platform": platform.platform(), "budget": {"model_calls": 14, "tool_calls": 10, "duration_seconds": 300}, "temperature": 0, "started_at": started_at}
    execution["config_digest"] = stable_digest({"provider": provider, "model": model, "temperature": 0, "budget": execution["budget"], "backend": backend_manifest})
    selected_manifest = [{"task_id": task.task_id, "category": task.category, "prompt": task.prompt, "files": task.files, "contract": asdict(task.contract)} for task in tasks]
    evidence_status = "smoke" if limit is not None else "measured" if repetitions >= 3 and backend_manifest["isolation_level"] == "container_strong" else "incomplete"
    summary = make_evaluation_result(status=evidence_status, track="capability", suite={"name": "professional", "version": "v2", "task_count": len(all_tasks), "task_manifest_digest": stable_digest(selected_manifest)}, provenance={"project": "Tifa", "code_version": code_version, "dirty_worktree": dirty_worktree, "provider": provider, "model": model}, execution=execution, sampling={"strategy": "stratified" if limit else "full", "seed": seed, "selected_task_ids": [task.task_id for task in tasks], "repetitions": repetitions}, cases=cases, metrics=metrics, decision=decision, limitations=["Synthetic capability tasks are not a real-repository success-rate claim.", "Costs are unknown unless an explicit pricing source is configured.", "Local fallback is explicitly local_degraded and is not comparable to container_strong release runs." if backend_manifest["isolation_level"] == "local_degraded" else "Docker verifier execution used container_strong isolation."])
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    output.with_name(output.stem + ".cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    output.with_name(output.stem + ".manifest.json").write_text(json.dumps({"suite": summary["suite"], "provenance": summary["provenance"], "execution": summary["execution"], "sampling": summary["sampling"], "manifest_digest": summary["manifest_digest"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    output.with_name(output.stem + ".decision.json").write_text(json.dumps(summary["decision"], ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
