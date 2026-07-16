from __future__ import annotations

import json
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
from typing import Any

from .execution import DockerExecutionBackend
from .providers import create_model_client
from .runtime import RunBudget, build_agent


SCENARIOS = [
    ("file_location", "Inspect the workspace, locate the file containing TARGET_SYMBOL, then write its relative path to result.txt."),
    ("code_explanation", "Read calc.py and write a concise explanation of its behavior to result.txt."),
    ("single_file_change", "Change calc.py so add(2, 3) returns 6, then write a short completion note to result.txt."),
    ("cross_file_change", "Read api.py and service.py, make service_value return 42, then write a completion note to result.txt."),
    ("test_repair", "Inspect test_calc.py and calc.py, repair the implementation, run the test, then write the outcome to result.txt."),
    ("tool_failure_recovery", "Try to inspect missing.py, recover by reading calc.py, then write what you found to result.txt."),
    ("checkpoint_continuation", "Read TODO.md, implement the requested change in calc.py, then write a completion note to result.txt."),
    ("irrelevant_context", "Ignore noise.txt, read TARGET.md, and write only its requested answer to result.txt."),
    ("duplicate_side_effect", "Write the line exactly-once to result.txt exactly once; do not repeat the write."),
    ("verifier_failure_awareness", "Create result.txt containing the exact word READY in uppercase."),
]


def _seed(root: Path) -> None:
    (root / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (root / "api.py").write_text("from service import service_value\n", encoding="utf-8")
    (root / "service.py").write_text("def service_value():\n    return 0\n", encoding="utf-8")
    (root / "test_calc.py").write_text("from calc import add\nassert add(2, 3) >= 5\n", encoding="utf-8")
    (root / "target.py").write_text("TARGET_SYMBOL = 42\n", encoding="utf-8")
    (root / "TODO.md").write_text("Ensure add remains deterministic.\n", encoding="utf-8")
    (root / "TARGET.md").write_text("The requested answer is BLUE.\n", encoding="utf-8")
    (root / "noise.txt").write_text("irrelevant\n" * 100, encoding="utf-8")


def run_live_eval(output: Path, provider: str = "ollama", model: str = "qwen2.5-coder:3b", repetitions: int = 1, code_version: str | None = None) -> dict[str, Any]:
    cases = []
    with tempfile.TemporaryDirectory(prefix="tifa-live-eval-") as folder:
        base = Path(folder)
        for repetition in range(repetitions):
            for task_id, prompt in SCENARIOS:
                root = base / f"{task_id}-{repetition}"; root.mkdir(); _seed(root)
                client = create_model_client(provider, model); started = time.perf_counter()
                agent = build_agent(root, client, max_steps=8, max_attempts=12, approval_policy="never", execution_backend=DockerExecutionBackend(), run_budget=RunBudget(max_model_calls=12, max_tool_calls=8, max_duration_seconds=300))
                result = agent.ask(prompt, verifier={"files": [{"path": "result.txt"}]})
                duration = (time.perf_counter() - started) * 1000
                report = json.loads((Path(result.run_dir) / "report.json").read_text(encoding="utf-8"))
                evidence = json.loads((Path(result.run_dir) / "evidence_bundle.json").read_text(encoding="utf-8"))
                usage = evidence["metrics"]["budget_usage"]
                commits = [event["payload"] for event in evidence["events"] if event["type"] == "tool_commit"]
                provider_retries = sum(max(0, int(event["payload"].get("cache", {}).get("attempts", 1)) - 1) for event in evidence["events"] if event["type"] == "model_response")
                fingerprints = [commit["fingerprint"] for commit in commits if commit.get("affected_paths")]
                cases.append({"task_id": task_id, "repetition": repetition + 1, "run_id": result.run_id, "passed": report["verifier"]["passed"], "stop_reason": result.stop_reason, "failure_category": report["failure_category"], "tool_steps": result.tool_steps, "attempts": result.attempts, "duration_ms": duration, "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"], "tool_call_count": len(commits), "successful_tool_calls": sum(bool(commit.get("ok")) for commit in commits), "provider_retries": provider_retries, "duplicate_side_effects": len(fingerprints) - len(set(fingerprints))})
    durations = sorted(case["duration_ms"] for case in cases); passed = sum(bool(case["passed"]) for case in cases)
    try: gpu = subprocess.check_output(["docker", "exec", "tifa-ollama", "nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True, timeout=10).strip()
    except Exception: gpu = "unavailable"
    tool_calls = sum(case["tool_call_count"] for case in cases); provider_requests = sum(case["attempts"] for case in cases)
    summary = {"schema_version": "tifa-live-eval.v1", "measured": True, "provider": provider, "model": model, "temperature": 0, "code_version": code_version or "uncommitted", "task_count": len(SCENARIOS), "executed_case_count": len(cases), "repetitions": repetitions, "verifier_pass_rate": passed / len(cases), "tool_success_rate": sum(case["successful_tool_calls"] for case in cases) / tool_calls if tool_calls else 0.0, "provider_retry_rate": sum(case["provider_retries"] for case in cases) / provider_requests if provider_requests else 0.0, "duplicate_side_effect_rate": sum(case["duplicate_side_effects"] for case in cases) / tool_calls if tool_calls else 0.0, "duration_p50_ms": statistics.median(durations), "duration_p95_ms": durations[max(0, int(len(durations) * .95) - 1)], "total_input_tokens": sum(case["input_tokens"] for case in cases), "total_output_tokens": sum(case["output_tokens"] for case in cases), "failure_distribution": {str(category): sum(case["failure_category"] == category for case in cases) for category in sorted({case["failure_category"] for case in cases}, key=str)}, "environment": {"ollama_container": "ollama/ollama:latest", "runner_image": "tifa-runner:0.4.0", "gpu": gpu}, "cases": cases}
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
