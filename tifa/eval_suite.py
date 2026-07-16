from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile
import time
from typing import Any

from .contracts import TaskContract
from .execution import DockerExecutionBackend
from .providers import create_model_client
from .runtime import RunBudget, build_agent


CATEGORY_COUNTS = {"locate_explain": 15, "single_file": 20, "cross_file": 20, "test_repair": 15, "tool_recovery": 10, "checkpoint_resume": 5, "irrelevant_context": 5, "duplicate_side_effect": 5, "verifier_security": 5}


@dataclass
class EvaluationTask:
    task_id: str
    category: str
    prompt: str
    files: dict[str, str]
    contract: TaskContract


def _scenario(category: str, index: int, task_id: str, expected: str) -> tuple[dict[str, str], str, dict[str, Any], list[str]]:
    base = {"src/value.py": f"VALUE = {index}\n", "src/helper.py": "def normalize(value):\n    return str(value).strip()\n", "tests/test_value.py": "from src.value import VALUE\nassert VALUE >= 1\n", "TARGET.md": f"Expected marker: {expected}\n"}
    result_check = {"path": "result.txt", "equals": expected}
    if category == "locate_explain":
        base["src/target.py"] = f"TARGET_SYMBOL_{index} = {index}\n"; prompt = f"Locate TARGET_SYMBOL_{index}, then write exactly {expected} to result.txt and finish."; verifier = {"files": [result_check]}; writable = ["result.txt"]
    elif category == "single_file":
        desired = index + 100; prompt = f"Patch src/value.py so VALUE equals {desired}, write exactly {expected} to result.txt, then finish."; verifier = {"files": [result_check, {"path": "src/value.py", "regex": f"VALUE\\s*=\\s*{desired}"}]}; writable = ["result.txt", "src/value.py"]
    elif category == "cross_file":
        base["src/service.py"] = "from .helper import normalize\ndef service(value):\n    return normalize(value)\n"; prompt = f"Update src/helper.py and src/service.py so service returns marker {expected}, write the same marker to result.txt, then finish."; verifier = {"files": [result_check, {"path": "src/helper.py", "contains": expected}, {"path": "src/service.py", "contains": expected}], "max_changed_files": 3}; writable = ["result.txt", "src"]
    elif category == "test_repair":
        base["src/value.py"] = "def calculate():\n    return 0\n"; base["tests/test_value.py"] = f"from src.value import calculate\nassert calculate() == {index}\n"; prompt = f"Run tests/test_value.py, repair src/value.py so the assertion passes, write exactly {expected} to result.txt, then finish."; verifier = {"files": [result_check], "commands": [{"command": "python tests/test_value.py", "exit_code": 0}]}; writable = ["result.txt", "src/value.py"]
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


def evaluate_provider(provider: str, model: str | None, output: Path, repetitions: int = 3, limit: int | None = None) -> dict[str, Any]:
    required_key = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY" if provider == "anthropic" else None
    if required_key and not os.getenv(required_key): raise RuntimeError(f"{required_key} is required for measured {provider} evaluation")
    tasks = professional_tasks()[:limit] if limit else professional_tasks(); cases = []
    with tempfile.TemporaryDirectory(prefix="tifa-professional-eval-") as folder:
        base = Path(folder)
        for repetition in range(1, repetitions + 1):
            for task in tasks:
                root = base / f"{task.task_id}-{repetition}"; root.mkdir(parents=True)
                for relative, content in task.files.items(): path = root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(content, encoding="utf-8")
                started = time.perf_counter(); agent = build_agent(root, create_model_client(provider, model), approval_policy="never", max_steps=10, max_attempts=14, execution_backend=DockerExecutionBackend(), run_budget=RunBudget(max_model_calls=14, max_tool_calls=10, max_duration_seconds=300))
                result = agent.ask(task.prompt, contract=task.contract); report = json.loads((Path(result.run_dir) / "report.json").read_text(encoding="utf-8")); evidence = json.loads((Path(result.run_dir) / "evidence_bundle.json").read_text(encoding="utf-8")); duration = (time.perf_counter() - started) * 1000
                commits = [event["payload"] for event in evidence["events"] if event["type"] == "tool_commit"]; fingerprints = [commit["fingerprint"] for commit in commits if commit.get("affected_paths")]; usage = evidence["metrics"]["budget_usage"]
                provider_retries = sum(max(0, int(event["payload"].get("cache", {}).get("attempts", 1)) - 1) for event in evidence["events"] if event["type"] == "model_response")
                cases.append({"task_id": task.task_id, "category": task.category, "repetition": repetition, "passed": report["verifier"]["passed"], "failure_category": report.get("failure_category"), "stop_reason": result.stop_reason, "duration_ms": duration, "tool_calls": len(commits), "duplicate_side_effects": len(fingerprints) - len(set(fingerprints)), "recovered": any(event["type"] == "repair_feedback" for event in evidence["events"]) and report["verifier"]["passed"] is True, "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"], "cost_usd": usage["cost_usd"], "provider_retries": provider_retries})
    durations = sorted(case["duration_ms"] for case in cases); passed = sum(case["passed"] is True for case in cases); failures = [case for case in cases if not case["passed"]]
    recovery_cases = [case for case in cases if case["category"] == "tool_recovery"]
    try: code_version = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL, timeout=3).strip()
    except Exception: code_version = "unknown"
    summary = {"schema_version": "tifa-provider-eval.v1", "result_kind": "measured", "code_version": code_version, "provider": provider, "model": model, "task_count": len(tasks), "executed_case_count": len(cases), "repetitions": repetitions, "verifier_pass_rate": passed / len(cases), "duplicate_side_effect_rate": sum(case["duplicate_side_effects"] for case in cases) / max(1, sum(case["tool_calls"] for case in cases)), "recovery_success_rate": sum(case["recovered"] for case in recovery_cases) / max(1, len(recovery_cases)), "provider_schema_failure_rate": sum(case["failure_category"] in {"invalid_response", "transport", "auth"} for case in cases) / len(cases), "provider_retry_rate": sum(case["provider_retries"] for case in cases) / max(1, sum(case["provider_retries"] + 1 for case in cases)), "total_input_tokens": sum(case["input_tokens"] for case in cases), "total_output_tokens": sum(case["output_tokens"] for case in cases), "total_cost_usd": sum(case["cost_usd"] for case in cases), "duration_mean_ms": statistics.mean(durations), "duration_variance": statistics.pvariance(durations), "duration_p50_ms": statistics.median(durations), "duration_p95_ms": durations[max(0, int(len(durations) * .95) - 1)], "failure_distribution": {str(category): sum(case["failure_category"] == category for case in failures) for category in sorted({case["failure_category"] for case in failures}, key=str)}, "environment": {"python": platform.python_version(), "platform": platform.platform(), "runner_image": "tifa-runner:0.5.0"}, "release_gate_passed": passed / len(cases) >= .8 and not any(case["duplicate_side_effects"] for case in cases) and (not recovery_cases or sum(case["recovered"] for case in recovery_cases) / len(recovery_cases) >= .9), "cases": cases}
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"); return summary
