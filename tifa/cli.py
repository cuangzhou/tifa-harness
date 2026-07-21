from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from typing import Any

from .benchmark import run_context_benchmark, run_replay_benchmark
from .cases import CaseStore, tool_schema_digest
from .contracts import TaskContract
from .doctor import doctor
from .execution import DockerExecutionBackend, ExecutionPolicy, LocalExecutionBackend, ResourceLimits
from .models import FakeModelClient
from .live_eval import run_live_eval
from .eval_suite import evaluate_provider, suite_manifest
from .harness_ab import evaluate_harness_ab
from .evaluation import load_evaluation_result, paired_comparison, validate_evaluation_result
from .operations import collect_garbage, continuation_lineage
from .performance import benchmark_workspace
from .providers import create_model_client
from .replay import ReplayDiffReport, ReplayResult, ReplayRunner, ReplaySpec
from .runtime import Tifa, build_agent
from .reporting import load_run, render_report
from .semantic_index import SemanticIndex
from .workspace import WorkspaceContext
from .tools import ToolRegistry


def _approver(name: str, arguments: dict[str, Any]) -> bool:
    answer = input(f"Approve {name} {json.dumps(arguments, ensure_ascii=False)}? [y/N] ")
    return answer.strip().lower() in {"y", "yes"}


def command_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="tifa")
    commands = root.add_subparsers(dest="command")
    replay = commands.add_parser("replay"); replay.add_argument("bundle", type=Path); replay.add_argument("--mode", choices=["offline", "forked", "counterfactual"], default="offline"); replay.add_argument("--output", type=Path)
    replay.add_argument("--spec", type=Path); replay.add_argument("--workspace", type=Path)
    replay.add_argument("--provider", choices=["fake", "openai", "anthropic", "ollama"], default="fake"); replay.add_argument("--model")
    diff = commands.add_parser("replay-diff"); diff.add_argument("original", type=Path); diff.add_argument("replayed", type=Path)
    resume = commands.add_parser("resume-run"); resume.add_argument("run_id"); resume.add_argument("prompt", nargs="?"); resume.add_argument("--checkpoint"); resume.add_argument("--cwd", default="."); resume.add_argument("--provider", choices=["fake", "openai", "anthropic", "ollama"], default="fake"); resume.add_argument("--model"); resume.add_argument("--approval", choices=["never", "on-risk", "always"], default="on-risk")
    cases = commands.add_parser("cases"); csub = cases.add_subparsers(dest="cases_command")
    propose = csub.add_parser("propose"); propose.add_argument("run_id"); propose.add_argument("--cwd", default=".")
    verify_case = csub.add_parser("verify"); verify_case.add_argument("case_id"); verify_case.add_argument("--override", required=True); verify_case.add_argument("--provider", choices=["fake", "openai", "anthropic", "ollama"], default="fake"); verify_case.add_argument("--model"); verify_case.add_argument("--cwd", default=".")
    promote = csub.add_parser("promote"); promote.add_argument("case_id"); promote.add_argument("--cwd", default=".")
    inspect_case = csub.add_parser("inspect"); inspect_case.add_argument("case_id"); inspect_case.add_argument("--cwd", default=".")
    search = csub.add_parser("search"); search.add_argument("category"); search.add_argument("--top-k", type=int, default=3); search.add_argument("--cwd", default=".")
    listing = csub.add_parser("list"); listing.add_argument("--cwd", default=".")
    reject = csub.add_parser("reject"); reject.add_argument("case_id"); reject.add_argument("--reason"); reject.add_argument("--cwd", default=".")
    benchmark = commands.add_parser("benchmark"); bsub = benchmark.add_subparsers(dest="benchmark_command"); breplay = bsub.add_parser("replay"); breplay.add_argument("--mode", choices=["smoke", "full", "innovation"], default="smoke"); breplay.add_argument("--output", type=Path)
    bworkspace = bsub.add_parser("workspace"); bworkspace.add_argument("--output", type=Path); bworkspace.add_argument("--repeats", type=int, default=3)
    bcontext = bsub.add_parser("context"); bcontext.add_argument("--output", type=Path)
    doctor_command = commands.add_parser("doctor"); doctor_command.add_argument("--cwd", default="."); doctor_command.add_argument("--model", default="qwen2.5-coder:3b"); doctor_command.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    evaluation = commands.add_parser("eval"); esub = evaluation.add_subparsers(dest="eval_command"); live = esub.add_parser("live"); live.add_argument("--provider", choices=["ollama"], default="ollama"); live.add_argument("--model", default="qwen2.5-coder:3b"); live.add_argument("--repetitions", type=int, default=1); live.add_argument("--output", type=Path, required=True); live.add_argument("--code-version")
    local_eval = esub.add_parser("local"); local_eval.add_argument("--model", default="qwen2.5-coder:3b"); local_eval.add_argument("--repetitions", type=int, default=1); local_eval.add_argument("--output", type=Path, required=True)
    provider_eval = esub.add_parser("provider"); provider_eval.add_argument("--provider", choices=["openai", "anthropic"], required=True); provider_eval.add_argument("--model"); provider_eval.add_argument("--repetitions", type=int, default=3); provider_eval.add_argument("--output", type=Path, required=True); provider_eval.add_argument("--limit", type=int)
    all_eval = esub.add_parser("all"); all_eval.add_argument("--output-dir", type=Path, required=True); all_eval.add_argument("--openai-model"); all_eval.add_argument("--anthropic-model")
    compare_eval = esub.add_parser("compare"); compare_eval.add_argument("left", type=Path); compare_eval.add_argument("right", type=Path); compare_eval.add_argument("--output", type=Path)
    harness_ab = esub.add_parser("harness-ab"); harness_ab.add_argument("--provider", choices=["openai"], default="openai"); harness_ab.add_argument("--model", default="deepseek-v4-flash"); harness_ab.add_argument("--cases", type=int, choices=[1, 30, 100], default=30); harness_ab.add_argument("--repetitions", type=int, choices=[1, 3], default=1); harness_ab.add_argument("--seed", type=int, default=20260721); harness_ab.add_argument("--output", type=Path, required=True); harness_ab.add_argument("--ablations", choices=["none", "standard"], default="none")
    validate_eval = esub.add_parser("validate"); validate_eval.add_argument("artifact", type=Path)
    run = commands.add_parser("run"); run.add_argument("--contract", type=Path, required=True); run.add_argument("--cwd", default="."); run.add_argument("--provider", choices=["fake", "openai", "anthropic", "ollama"], default="fake"); run.add_argument("--model"); run.add_argument("--case-assistance", choices=["off", "verified"], default="off"); run.add_argument("--case-category")
    inspect = commands.add_parser("inspect"); inspect_sub = inspect.add_subparsers(dest="inspect_command"); inspect_run = inspect_sub.add_parser("run"); inspect_run.add_argument("run_id"); inspect_run.add_argument("--cwd", default="."); inspect_run.add_argument("--lineage", action="store_true")
    gc = commands.add_parser("gc"); gc.add_argument("--cwd", default="."); gc_mode = gc.add_mutually_exclusive_group(); gc_mode.add_argument("--dry-run", action="store_true"); gc_mode.add_argument("--apply", action="store_true"); gc.add_argument("--retain-days", type=int, default=30); gc.add_argument("--retain-latest", type=int, default=20)
    index = commands.add_parser("index"); index_sub = index.add_subparsers(dest="index_command"); index_build = index_sub.add_parser("build"); index_build.add_argument("--cwd", default="."); index_status = index_sub.add_parser("status"); index_status.add_argument("--cwd", default=".")
    report = commands.add_parser("report"); report.add_argument("run_id"); report.add_argument("--cwd", default="."); report.add_argument("--format", choices=["json", "html"], default="json"); report.add_argument("--output", type=Path, required=True)
    return root


def agent_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="tifa")
    root.add_argument("prompt", nargs="?"); root.add_argument("--cwd", default="."); root.add_argument("--provider", choices=["fake", "openai", "anthropic", "ollama"], default="fake")
    root.add_argument("--model"); root.add_argument("--max-steps", type=int, default=8); root.add_argument("--approval", choices=["never", "on-risk", "always"], default="on-risk"); root.add_argument("--resume")
    root.add_argument("--case-assistance", choices=["off", "verified"], default="off"); root.add_argument("--case-category")
    root.add_argument("--execution-backend", choices=["docker", "local"], default="local"); root.add_argument("--network-policy", choices=["deny", "allow"], default="deny"); root.add_argument("--cpu-limit", type=float, default=1.0); root.add_argument("--memory-limit", type=int, default=512); root.add_argument("--timeout", type=int, default=30)
    return root


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    is_command = bool(raw and raw[0] in {"replay", "replay-diff", "resume-run", "cases", "benchmark", "doctor", "eval", "run", "inspect", "gc", "index", "report"})
    args = (command_parser() if is_command else agent_parser()).parse_args(raw)
    if getattr(args, "command", None) == "replay":
        spec = ReplaySpec(**json.loads(args.spec.read_text(encoding="utf-8"))) if args.spec else None
        replay_client = None if args.mode == "offline" else (FakeModelClient() if args.provider == "fake" else create_model_client(args.provider, args.model))
        result = ReplayRunner().replay(args.bundle, args.mode, spec=spec, workspace=args.workspace, model_client=replay_client)
        payload: Any = asdict(result) if isinstance(result, (ReplayDiffReport, ReplayResult)) else result
        if args.output: args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        consistent = payload.get("replay_consistent", payload.get("report", {}).get("replay_consistent", False))
        print(json.dumps(payload, ensure_ascii=False, indent=2)); return 0 if consistent else 1
    if getattr(args, "command", None) == "replay-diff":
        original = json.loads(args.original.read_text(encoding="utf-8")); replayed = json.loads(args.replayed.read_text(encoding="utf-8")); print(json.dumps(ReplayRunner.diff(original, replayed), ensure_ascii=False, indent=2)); return 0
    if getattr(args, "command", None) == "resume-run":
        client = FakeModelClient() if args.provider == "fake" else create_model_client(args.provider, args.model)
        agent = Tifa.resume_run(args.cwd, client, args.run_id, args.checkpoint, approval_policy=args.approval, approver=_approver)
        request = args.prompt or (agent.history[-1].get("content", "Continue the interrupted task") if agent.history else "Continue the interrupted task")
        print(json.dumps(asdict(agent.ask(request)), ensure_ascii=False, indent=2)); return 0
    if getattr(args, "command", None) == "cases":
        store = CaseStore(Path(args.cwd) / ".tifa" / "cases")
        if args.cases_command == "propose": payload = asdict(store.propose_from_run(Path(args.cwd).resolve(), args.run_id))
        elif args.cases_command == "verify":
            card = store.load(args.case_id); key, separator, raw_value = args.override.partition("=")
            if not separator or key not in {"provider", "context_policy", "memory_enabled"}: raise ValueError("override must be provider=..., context_policy=..., or memory_enabled=...")
            value: Any = raw_value.lower() == "true" if raw_value.lower() in {"true", "false"} else raw_value
            selected_provider = raw_value if key == "provider" and raw_value in {"fake", "openai", "anthropic", "ollama"} else args.provider
            client = FakeModelClient() if selected_provider == "fake" else create_model_client(selected_provider, args.model)
            if key == "provider": value = client.provider
            if card.minimal_delta.get("variable") == "provider" and card.minimal_delta.get("after") == "review-required-provider": card.minimal_delta["after"] = value; card.suggested_override = dict(card.minimal_delta); store.save(card)
            spec = ReplaySpec(card.source_run_id or "", mode="counterfactual", workspace_policy="snapshot_copy", overrides={key: value})
            bundle = Path(args.cwd) / ".tifa" / "runs" / str(card.source_run_id) / "evidence_bundle.json"
            replay_result = ReplayRunner().replay(bundle, "counterfactual", spec=spec, workspace=Path(args.cwd).resolve(), model_client=client)
            payload = asdict(replay_result); store.record_verification(card.case_id, payload)
        elif args.cases_command == "promote": payload = asdict(store.promote(store.load(args.case_id)))
        elif args.cases_command == "inspect":
            card = store.load(args.case_id); context = WorkspaceContext.build(args.cwd); registry = ToolRegistry(context, approval_policy="never", delegate=lambda *_: "")
            store.assess_freshness(card, repo_snapshot=context.fingerprint(), dependency_digests={path: context.index.get(path, {}).get("digest", "") for path in card.dependency_digests}, tool_schema_digest=tool_schema_digest(registry.schemas()), context_policy_version=card.context_policy_version); payload = asdict(card)
        elif args.cases_command == "search": payload = [asdict(c) for c in store.search(args.category, top_k=args.top_k)]
        elif args.cases_command == "list": payload = [asdict(c) for c in store.list()]
        elif args.cases_command == "reject": payload = asdict(store.reject(args.case_id, args.reason))
        else: command_parser().error("cases requires a command")
        print(json.dumps(payload, ensure_ascii=False, indent=2)); return 0
    if getattr(args, "command", None) == "benchmark":
        payload = run_replay_benchmark(args.mode, args.output) if args.benchmark_command == "replay" else benchmark_workspace(args.output, repeats=args.repeats) if args.benchmark_command == "workspace" else run_context_benchmark(args.output) if args.benchmark_command == "context" else None
        if payload is None: command_parser().error("benchmark requires replay or workspace")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        benchmark_passed = payload.get("decision", {}).get("passed") if "decision" in payload else payload.get("status", "passed") == "passed"
        return 0 if benchmark_passed else 1
    if getattr(args, "command", None) == "doctor":
        payload = doctor(args.cwd, args.model, args.ollama_url); print(json.dumps(payload, ensure_ascii=False, indent=2)); return 0 if payload["status"] == "healthy" else 1
    if getattr(args, "command", None) == "eval":
        if args.eval_command == "live": payload = run_live_eval(args.output, args.provider, args.model, args.repetitions, args.code_version)
        elif args.eval_command == "local": payload = evaluate_provider("ollama", args.model, args.output, args.repetitions, limit=30)
        elif args.eval_command == "provider": payload = evaluate_provider(args.provider, args.model, args.output, args.repetitions, args.limit)
        elif args.eval_command == "all":
            args.output_dir.mkdir(parents=True, exist_ok=True)
            openai_result = evaluate_provider("openai", args.openai_model, args.output_dir / "openai.json") if os.getenv("OPENAI_API_KEY") else {"status": "NOT_RUN_MISSING_CREDENTIALS", "required": "OPENAI_API_KEY"}
            anthropic_result = evaluate_provider("anthropic", args.anthropic_model, args.output_dir / "anthropic.json") if os.getenv("ANTHROPIC_API_KEY") else {"status": "NOT_RUN_MISSING_CREDENTIALS", "required": "ANTHROPIC_API_KEY"}
            payload = {"manifest": suite_manifest(), "openai": openai_result, "anthropic": anthropic_result, "local": evaluate_provider("ollama", "qwen2.5-coder:3b", args.output_dir / "ollama.json", 3, 30)}
        elif args.eval_command == "compare":
            left = load_evaluation_result(args.left); right = load_evaluation_result(args.right)
            left_payload = left.get("original", left); right_payload = right.get("original", right)
            payload = paired_comparison(left_payload, right_payload)
            if args.output: args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        elif args.eval_command == "harness-ab": payload = evaluate_harness_ab(args.provider, args.model, args.output, case_count=args.cases, repetitions=args.repetitions, seed=args.seed, ablations=args.ablations)
        elif args.eval_command == "validate":
            loaded = load_evaluation_result(args.artifact)
            if not loaded.get("legacy"): validate_evaluation_result(loaded)
            payload = {"valid": True, "legacy": bool(loaded.get("legacy")), "schema_version": loaded["schema_version"]}
        else: command_parser().error("eval requires live, local, provider, all, compare, harness-ab, or validate")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        decision = payload.get("decision") if isinstance(payload, dict) else None
        return 1 if isinstance(decision, dict) and decision.get("passed") is False else 0
    if getattr(args, "command", None) == "run":
        contract = TaskContract.from_dict(json.loads(args.contract.read_text(encoding="utf-8"))); client = FakeModelClient() if args.provider == "fake" else create_model_client(args.provider, args.model); case_store = CaseStore(Path(args.cwd) / ".tifa" / "cases") if args.case_assistance == "verified" else None; run_result = build_agent(args.cwd, client, approval_policy="never").ask(contract.goal, contract=contract, case_store=case_store, case_category=args.case_category or contract.contract_id); print(json.dumps(asdict(run_result), ensure_ascii=False, indent=2)); return 0 if run_result.stop_reason == "final_answer_returned" else 1
    if getattr(args, "command", None) == "inspect":
        if args.inspect_command != "run": command_parser().error("inspect requires run")
        payload = load_run(Path(args.cwd), args.run_id); payload["lineage"] = continuation_lineage(Path(args.cwd).resolve(), args.run_id) if args.lineage else []; print(json.dumps(payload, ensure_ascii=False, indent=2)); return 0
    if getattr(args, "command", None) == "gc":
        print(json.dumps(collect_garbage(Path(args.cwd).resolve(), apply=args.apply, retain_days=args.retain_days, retain_latest=args.retain_latest), ensure_ascii=False, indent=2)); return 0
    if getattr(args, "command", None) == "index":
        if not args.index_command: command_parser().error("index requires build or status")
        semantic = SemanticIndex(WorkspaceContext.build(args.cwd)); payload = semantic.build() if args.index_command == "build" else {"exists": semantic.path.exists(), **(semantic.load(optional=True) if semantic.path.exists() else {})}; print(json.dumps(payload, ensure_ascii=False, indent=2)); return 0
    if getattr(args, "command", None) == "report":
        print(render_report(Path(args.cwd), args.run_id, args.output, args.format)); return 0
    client = FakeModelClient() if args.provider == "fake" else create_model_client(args.provider, args.model)
    backend = DockerExecutionBackend() if args.execution_backend == "docker" else LocalExecutionBackend()
    kwargs = {"max_steps": args.max_steps, "approval_policy": args.approval, "approver": _approver, "execution_backend": backend, "execution_policy": ExecutionPolicy(network=args.network_policy), "resource_limits": ResourceLimits(cpus=args.cpu_limit, memory_mb=args.memory_limit, timeout_seconds=args.timeout)}
    agent = Tifa.from_session(args.cwd, client, args.resume, **kwargs) if args.resume else build_agent(args.cwd, client, **kwargs)
    if args.prompt:
        case_store = CaseStore(Path(args.cwd) / ".tifa" / "cases") if args.case_assistance == "verified" else None
        print(json.dumps(asdict(agent.ask(args.prompt, case_store=case_store, case_category=args.case_category)), ensure_ascii=False, indent=2)); return 0
    print("Tifa interactive mode. Use /exit to quit.")
    while True:
        try: request = input("tifa> ").strip()
        except (EOFError, KeyboardInterrupt): break
        if request in {"/exit", "/quit"}: break
        if request: print(agent.ask(request).answer)
    return 0
